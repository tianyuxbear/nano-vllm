from collections import deque

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    """
    推理调度器 (Scheduler)。

    核心职责：
    1. 显存管理：通过 BlockManager 协调物理显存块的分配。
    2. 策略调度：采用 FCFS (先来先服务) 原则，优先处理 Prefill (新任务)，其次处理 Decode (旧任务)。
    3. 动态抢占：当显存不足以让当前任务生成下一个 Token 时，暂时挂起 (Preempt) 其他任务以腾出空间。
    """

    def __init__(self, config: Config):
        # 限制单次推理的最大并发序列数 (例如 256)
        self.max_num_seqs = config.max_num_seqs
        # 限制单次推理的最大 Token 处理量 (例如 4096)，防止 Prefill 阶段 OOM
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos

        # 初始化显存块管理器，它是 PagedAttention 的核心组件
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )

        # 等待队列：存放刚提交但未开始执行，或被抢占的任务
        self.waiting: deque[Sequence] = deque()
        # 运行队列：存放正在 GPU 上进行生成的任务
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """检查调度器是否空闲 (没有待处理或正在处理的任务)"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """将新请求加入等待队列"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        [核心逻辑] 执行一次调度决策。

        Returns:
            tuple[list[Sequence], bool]:
                - 本次被选中执行的序列列表
                - bool 值: True 表示本次是 Prefill 阶段，False 表示 Decode 阶段
        """
        # =================================================================
        # 阶段 1: Prefill (预填充) 调度
        # 策略：只要有显存且未达到 Batch 限制，优先让新任务进场
        # =================================================================
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0  # 统计本次 Batch 累积的 Token 数

        # 遍历等待队列
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]

            # [资源检查]
            # 1. Token 数量限制: 防止 Prompt 太长导致 OOM
            # 2. 显存块限制: 检查是否有足够的物理块存 Prompt 的 KV Cache
            if num_batched_tokens + len(
                seq
            ) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break  # 资源不足，停止从等待队列取任务

            # --- 准许入场 ---
            num_seqs += 1
            # 分配物理显存 (可能复用 Prefix Caching 的旧块)
            self.block_manager.allocate(seq)

            # 统计计算量：减去 seq.num_cached_tokens 是因为命中的缓存不需要重算
            num_batched_tokens += len(seq) - seq.num_cached_tokens

            # 状态转移：Waiting -> Running
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)

        # 如果选中了新任务，立即返回执行 Prefill，暂停 Decode 任务
        if scheduled_seqs:
            return scheduled_seqs, True

        # =================================================================
        # 阶段 2: Decode (解码) 调度
        # 策略：如果没新任务，就让旧任务继续跑。如果显存不够，触发抢占。
        # =================================================================
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()  # 取出一个正在跑的任务

            # [显存危机处理 / 抢占循环]
            # 检查：如果要让这个 seq 再生成 1 个 Token，显存够不够？
            # 只有当 block 刚好填满需要开新 block 时，can_append 才可能返回 False
            while not self.block_manager.can_append(seq):
                if self.running:
                    # 牺牲策略：踢掉 Running 队列尾部的任务 (最近最少被调度或优先级最低)
                    # 腾出它的显存给当前 seq 使用
                    victim = self.running.pop()
                    self.preempt(victim)
                else:
                    # 极端情况：队列只剩我自己了，还是不够资源
                    # 只能牺牲自己，回炉重造
                    self.preempt(seq)
                    break  # 跳出 while 循环，seq 不会被加入 scheduled_seqs
            else:
                # [Python while-else 语法]
                # 只有当 while 循环条件变为 False (即 can_append 成功) 且未触发 break 时执行
                # 说明显存足够 (或者通过抢占腾出了空间)
                num_seqs += 1

                # 真正执行显存分配/更新 (分配新块或更新哈希)
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        # 确保至少调度了一个任务 (除非队列本身就是空的)
        # 如果触发了自我抢占导致 scheduled_seqs 为空，外层的 Engine 会处理空转
        assert scheduled_seqs
        # [关键操作] 保持队列顺序
        # 刚才 popleft 出来的任务，现在按原顺序放回队列头部 (extendleft + reversed)
        # 这样保证了下一轮调度时，任务的优先级顺序不变
        self.running.extendleft(reversed(scheduled_seqs))

        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """
        [抢占机制]
        将一个 Running 状态的任务强制暂停，释放其显存，并放回 Waiting 队列。
        注意：这意味着该任务下次执行时，可能需要重新计算 KV Cache (Recompute)。
        """
        seq.status = SequenceStatus.WAITING

        # 释放该序列占用的所有物理显存块，救急！
        self.block_manager.deallocate(seq)

        # 插队到等待队列的最前面 (LIFO)，保证它下次能最先被调度回来 (也就是 Recompute)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        """
        在模型执行完一步 (Step) 后调用。
        负责更新序列内容、检查结束条件并回收已完成任务的资源。
        """
        for seq, token_id in zip(seqs, token_ids):
            # 1. 将新生成的 Token ID 追加到序列中
            seq.append_token(token_id)

            # 2. 检查停止条件
            # 条件 A: 生成了 EOS Token 且配置未忽略 EOS
            # 条件 B: 序列长度达到了预设的 max_tokens
            if (
                not seq.ignore_eos and token_id == self.eos
            ) or seq.num_completion_tokens == seq.max_tokens:
                # --- 任务完成 ---
                seq.status = SequenceStatus.FINISHED

                # 立即释放显存资源 (归还 Block 给 BlockManager)
                self.block_manager.deallocate(seq)

                # 从运行队列中移除，不再参与调度
                self.running.remove(seq)
