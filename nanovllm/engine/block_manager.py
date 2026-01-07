from collections import deque

import numpy as np
import xxhash

from nanovllm.engine.sequence import Sequence


class Block:
    """
    物理显存块的元数据封装类。
    代表 GPU 上实际分配的一个 KV Cache 块。
    """

    def __init__(self, block_id):
        self.block_id = block_id  # 物理块的唯一 ID (对应 GPU 内存池中的索引)
        self.ref_count = 0  # 引用计数：有多少个 Sequence 正在共享这个块
        self.hash = -1  # 块内容的哈希值，用于前缀缓存查找 (-1 表示未计算/未定型)
        self.token_ids = []  # 存储该块包含的逻辑 Token IDs (用于哈希冲突时的二次校验)

    def update(self, hash: int, token_ids: list[int]):
        """当块被填满且计算出哈希后，更新其元数据"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """重置块状态，使其变为可用状态 (用于新分配时)"""
        self.ref_count = 1  # 新分配时，引用计数初始化为 1
        self.hash = -1  # 哈希重置无效
        self.token_ids = []


class BlockManager:
    """
    显存块管理器。
    负责物理块的分配、释放、复用 (缓存命中) 以及动态增长管理。
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size  # 每个块能存多少个 Token (例如 256)

        # 初始化所有物理块对象池
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]

        # [核心] 哈希映射表: hash_value -> block_id
        # 用于实现 Prefix Caching，快速查找是否存在包含相同数据的块
        self.hash_to_block_id: dict[int, int] = dict()

        # 空闲块队列 (双端队列)，用于 O(1) 获取空闲块
        self.free_block_ids: deque[int] = deque(range(num_blocks))

        # 已使用的块 ID 集合，用于快速判断状态
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算 Token 序列的链式哈希值。

        Args:
            token_ids: 当前块内的 Token 列表。
            prefix: 前一个块的哈希值。

        Returns:
            int: 当前块结合前缀后的唯一哈希值。


        机制说明：当前块的哈希值依赖于前一个块的哈希。
        这保证了只有在 "相同的前文 + 相同的当前内容" 时，才能命中缓存。
        """
        h = xxhash.xxh64()
        if prefix != -1:
            # 将前缀哈希混入当前计算
            h.update(prefix.to_bytes(8, "little"))
        # 混入当前 Token 数据
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """[内部方法] 从空闲池中取出一个指定 ID 的块并初始化"""
        block = self.blocks[block_id]
        assert block.ref_count == 0  # 确保取出的块确实是没人在用的
        block.reset()  # 重置状态 (ref_count=1)
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int):
        """[内部方法] 将块归还给空闲池"""
        assert self.blocks[block_id].ref_count == 0  # 确保没人用了才回收
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """检查是否有足够的空闲块来满足序列的首次分配需求"""
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """
        [Prefill 阶段核心] 为一个新的序列分配所需的物理块。
        会尝试利用 Prefix Caching 复用已有块。
        """
        assert not seq.block_table  # 确保是首次分配

        h = -1
        cache_miss = False  # 标记是否发生过缓存未命中

        # 遍历序列需要的每一个逻辑块
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)

            # 1. 计算哈希: 只有当块是满的 (Full Block) 时才计算哈希用于缓存查找
            # 不满的块 (通常是最后一个) 不具备稳定的哈希特征，不参与缓存
            h = (
                self.compute_hash(token_ids, h)
                if len(token_ids) == self.block_size
                else -1
            )

            # 2. 查表: 尝试在全局哈希表中寻找是否已存在相同的块
            block_id = self.hash_to_block_id.get(h, -1)

            # 3. 校验:
            # - 如果 block_id == -1: 没查到
            # - 如果 token_ids 不匹配: 发生了哈希碰撞 (极小概率)，不能复用
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True

            # 4. 分配策略分支
            if cache_miss:
                # [Cache Miss]: 必须分配一个新的物理块
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # [Cache Hit]: 命中缓存！复用旧块
                seq.num_cached_tokens += (
                    self.block_size
                )  # 记录已缓存 Token 数，跳过计算

                if block_id in self.used_block_ids:
                    # 场景 A: 该块正在被其他序列使用 (Shared Block)
                    block = self.blocks[block_id]
                    block.ref_count += 1  # 增加引用计数
                else:
                    # 场景 B: 该块之前被释放了，但恰好没人覆盖它 (Resurrected Block)
                    # 这种情况在 free_block_ids 还没轮转完一圈时可能发生
                    block = self._allocate_block(block_id)

            # 5. 更新哈希表 (注册新块信息，供后续请求复用)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id

            # 将分配到的 block_id 加入序列的页表
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """
        释放序列占用的资源。
        利用引用计数机制，只有当块的 ref_count 降为 0 时才真正回收物理内存。
        """
        # 反向遍历 (虽不强制，但通常符合栈习惯)
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        # 清理序列状态
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """
        检查是否有足够资源追加一个 Token。
        只有当 '当前块已满，需要开新块' (len % block_size == 1) 时才需要检查空闲池。
        """
        # len(seq) 已经是加了新 token 后的长度
        # 如果 len % size == 1，说明刚跨过边界，需要分配新块
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """
        [Decode 阶段核心] 处理追加 Token 后的显存管理。
        包括：分配新块、更新旧块哈希等。
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]  # 获取当前序列正在写入的最后一个块

        # 情况 1: 刚刚跨过边界，需要分配一个新的物理块
        # (例如 block_size=16, 之前长 16, 现在长 17)
        if len(seq) % self.block_size == 1:
            # 此时前一个块 (last_block) 肯定是满的，且必须已经计算过哈希
            assert last_block.hash != -1

            # 分配新块并追加到页表
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)

        # 情况 2: 刚刚填满当前块
        # (例如 block_size=16, 之前长 15, 现在长 16)
        elif len(seq) % self.block_size == 0:
            # 之前没满时，哈希应该是 -1
            assert last_block.hash == -1

            # 现在满了，计算它的哈希并注册到全局缓存表
            # 这样未来的请求如果生成了相同的内容，就可以复用这个块了
            token_ids = seq.block(seq.num_blocks - 1)

            # 获取前一个块的哈希作为 prefix (如果是第一个块则为 -1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)

            # 更新块元数据，存入哈希表
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id

        # 情况 3: 块还没满
        # (例如 block_size=16, 之前长 10, 现在长 11)
        else:
            # 还在持续写入中，状态不稳定，不计算哈希
            assert last_block.hash == -1
