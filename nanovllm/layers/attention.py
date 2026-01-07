import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from torch import nn

from nanovllm.utils.context import get_context


# =============================================================================
# Triton Kernel: 高效 KV Cache 存储算子
# =============================================================================
@triton.jit
def store_kvcache_kernel(
    key_ptr,  # 当前计算出的 Key 张量指针
    key_stride,  # Key 在 Token 维度上的步长 (跳到下一个 Token 需要跨越的内存距离)
    value_ptr,  # 当前计算出的 Value 张量指针
    value_stride,  # Value 在 Token 维度上的步长
    k_cache_ptr,  # 物理 K 缓存池指针 (Paged Cache)
    v_cache_ptr,  # 物理 V 缓存池指针 (Paged Cache)
    slot_mapping_ptr,  # 槽位映射指针：Token 索引 -> 物理槽位索引
    D: tl.constexpr,  # 每个 Token 的总维度 (num_heads * head_dim)
):
    # 每个程序实例 (Program) 处理一个 Token 的 K 和 V 搬运
    idx = tl.program_id(0)

    # 1. 加载当前 Token 对应的物理槽位 (Slot)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:  # 如果槽位无效，说明无需缓存，直接返回
        return

    # 2. 计算输入张量的偏移量 (逻辑连续内存)
    # tl.arange(0, D) 创建向量化偏移，实现一次性搬运整个 D 维向量
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)

    # 从寄存器加载当前 Token 的 K, V
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    # 3. 计算物理缓存池的偏移量 (Paged 离散内存)
    cache_offsets = slot * D + tl.arange(0, D)

    # 将 K, V 存入物理缓存对应的槽位
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """
    Python 侧调度函数：将当前计算出的 K, V 散播(Scatter)到分页缓存池中。
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim

    # 内存对齐与连续性校验，确保 Triton 内核可以进行向量化读写 (LDG.128/STG.128)
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N

    # 启动 N 个线程实例，并行搬运 N 个 Token 的数据
    store_kvcache_kernel[(N,)](
        key,
        key.stride(0),  # 动态获取步长，支持 QKV 合并计算后的切片输入
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        D,  # ty: ignore
    )


# =============================================================================
# Attention 模块：支持 Prefill 与 Decode 两种模式
# =============================================================================
class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # 初始时缓存为空，后续由引擎外部注入(Late Binding) 预分配的大张量
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # 1. 获取全局推理上下文元数据
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # 2. 如果已挂载缓存池，则将当前新算出的 k, v 写入对应的物理槽位
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        # 3. 根据推理阶段选择最优算子
        if context.is_prefill:
            # --- Prefill (预填充阶段) ---

            # 如果命中前缀缓存 (Prefix Cache)，则 k, v 需指向完整的历史缓存池
            if context.block_tables is not None:
                k, v = k_cache, v_cache

            # 使用变长 FlashAttention，利用 cu_seqlens 消除 Batch 中的 Padding 浪费
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,  # 若非 None 则启用 Paged 模式寻址 K/V
            )
        else:
            # --- Decode (解码生成阶段) ---

            # 使用专为分页缓存优化的 FlashAttention
            # q 为当前单 Token [Batch, 1, H, D]
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                cache_seqlens=context.context_lens,  # 告诉算子每个序列当前的有效长度
                block_table=context.block_tables,  # 逻辑块到物理块的映射表
                softmax_scale=self.scale,
                causal=True,
            )

        return o
