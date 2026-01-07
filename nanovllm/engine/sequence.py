from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """
    序列的状态枚举。
    调度器 (Scheduler) 根据这些状态决定将序列放在哪个队列中。
    """

    WAITING = auto()  # 等待中：刚被加入引擎，尚未开始计算 (在 Waiting 队列)
    RUNNING = auto()  # 运行中：正在 GPU 上进行 Prefill 或 Decode 计算 (在 Running 队列)
    FINISHED = auto()  # 已完成：生成结束 (遇到 EOS 或达到最大长度)


class Sequence:
    """
    单个推理请求的封装类。
    管理该请求的 Token 数据、生成状态以及 KV Cache 的显存映射。
    """

    # [核心参数] PagedAttention 的逻辑块大小 (Logical Block Size)。
    # 表示一个逻辑块能存储多少个 Token 的 KV Cache。
    # 这里硬编码为 256，实际 vLLM 中通常可配置 (如 16 或 32)。
    block_size = 256

    # 全局计数器，用于给每个新创建的 Sequence 分配唯一的 ID。
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params=SamplingParams()):
        """
        初始化一个序列对象。

        Args:
            token_ids: 初始输入的 Prompt Token ID 列表。
            sampling_params: 采样参数 (temperature, top_p, max_tokens 等)。
        """
        # 自动获取唯一 ID
        self.seq_id = next(Sequence.counter)
        # 初始状态设为等待
        self.status = SequenceStatus.WAITING

        # 存储完整的 Token 序列 (Prompt + Generated)。使用 copy 防止外部修改影响内部。
        self.token_ids = copy(token_ids)
        # 记录最新的一个 Token，用作下一次模型推理的输入 (Input ID)
        self.last_token = token_ids[-1]

        # --- 长度统计 ---
        self.num_tokens = len(self.token_ids)  # 当前总长度
        self.num_prompt_tokens = len(token_ids)  # Prompt 部分的长度 (固定)

        # --- KV Cache 管理 (PagedAttention 关键) ---
        # 记录已经在 GPU 显存中计算并缓存了 KV 的 Token 数量。
        # 调度器会对比 num_tokens 和 num_cached_tokens 来决定需要计算多少新 Token 的 KV。
        self.num_cached_tokens = 0

        # 物理显存块索引表 (Page Table)。
        # 列表中的每个整数代表 GPU 上一个物理 Block 的索引 ID。
        # 例如 [5, 12, 0] 表示逻辑块 0 在物理块 5，逻辑块 1 在物理块 12...
        self.block_table = []

        # --- 采样与停止条件 ---
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens  # 最大生成长度
        self.ignore_eos = sampling_params.ignore_eos  # 是否忽略 EOS Token

    def __len__(self):
        """返回当前序列的总 Token 数"""
        return self.num_tokens

    def __getitem__(self, key):
        """允许像访问列表一样访问 sequence[i]"""
        return self.token_ids[key]

    @property
    def is_finished(self):
        """判断序列是否已经结束"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """返回新生成的 Token 数量 (不含 Prompt)"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """获取 Prompt 部分的 Token IDs"""
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """获取生成部分的 Token IDs"""
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_cached_blocks(self):
        """
        计算当前已缓存的 KV Cache 占满了多少个完整块。
        注意是整除：如果 cached=260, block_size=256，则结果为 1。
        """
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """
        计算当前序列总共需要多少个逻辑块 (Logical Blocks)。
        公式 (N + B - 1) // B 用于向上取整。
        """
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """
        计算最后一个逻辑块中实际包含的 Token 数量。
        这在 Attention Kernel 计算掩码 (Mask) 或边界时非常重要。
        例如：BlockSize=256, Total=260 -> LastBlock 有 4 个 Token。
        """
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """
        获取第 i 个逻辑块中的 Token ID 数据切片。
        用于调试或某些 CPU 端的验证逻辑。
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    def append_token(self, token_id: int):
        """
        [核心操作] 向序列追加一个新生成的 Token。
        通常在 step() 的后处理阶段被调用。
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # --- 序列化优化 (Pickle Support) ---
    # 下面两个方法是为了在多进程 (Multiprocessing) 通信时减少数据传输量。
    # 主进程 (Scheduler) -> 子进程 (ModelRunner)

    def __getstate__(self):
        """
        自定义序列化逻辑。
        """
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            # [关键优化]
            # 如果是 Prefill 阶段 (num_completion_tokens == 0):
            #   需要发送完整的 token_ids 给模型进行首次计算。
            # 如果是 Decode 阶段:
            #   之前的 KV 都在 GPU 显存 (block_table) 里了，
            #   模型只需要知道上一步生成的那个 Token (last_token) 即可。
            #   这避免了在生成长文本时反复传输巨大的列表。
            self.token_ids if self.num_completion_tokens == 0 else self.last_token,
        )

    def __setstate__(self, state):
        """
        自定义反序列化逻辑 (在子进程中重建对象)。
        """
        (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
        ) = state[:-1]

        # 根据状态恢复数据
        if self.num_completion_tokens == 0:
            # Prefill: 恢复完整列表
            self.token_ids = state[-1]
        else:
            # Decode: 只恢复 last_token，token_ids 列表在子进程不需要完整维护
            self.last_token = state[-1]
