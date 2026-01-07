from dataclasses import dataclass

import torch


@dataclass
class Context:
    """
    推理上下文元数据 (Inference Context Metadata)。

    该数据结构承载了当前 Batch 的逻辑拓扑与物理存储映射关系，是 FlashAttention、
    PagedAttention 等高性能算子进行变长计算和显存寻址的核心依据。
    """

    # 推理阶段标识：
    # True (Prefill): 预填充阶段，全量处理 Prompt 序列，属于计算密集型。
    # False (Decode): 解码阶段，逐 Token 生成，属于访存密集型，依赖 KV Cache 寻址。
    is_prefill: bool = False

    # 变长序列偏移量 (Varlen Offsets)：
    # 用于在无 Padding 的 Flatten 张量 [total_tokens, hidden_size] 中定位各序列边界。
    # 例如：Batch 长度 [3, 1, 4] 对应 cu_seqlens 为 [0, 3, 4, 8]。
    cu_seqlens_q: torch.Tensor | None = None  # Query 的累积序列长度起点坐标
    cu_seqlens_k: torch.Tensor | None = (
        None  # Key/Value 的累积序列长度起点坐标 (在 Prefix Cache 命中时可能大于 Q)
    )

    # 批次特征值：
    # 当前 Batch 中 Q 和 K 的最大有效序列长度。
    # 决定了 GPU Kernel 调度时的 Thread Block 规模和 Shared Memory 分配。
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0

    # 物理槽位寻址 (Slot Mapping)：
    # [形状: total_num_tokens] 记录每个 Token 对应 KV Cache 显存池中的绝对物理偏移量。
    # 主要用于 store_kvcache 算子将新算的 K、V 写入“散落在各处”的物理坑位。
    slot_mapping: torch.Tensor | None = None

    # 历史长度记录 (KV History)：
    # [形状: batch_size] 记录每个序列目前已生成的总 Token 数（含当前步）。
    # 在 Decode 阶段用于告诉算子每个请求在 KV Cache 中有多少有效“存货”。
    context_lens: torch.Tensor | None = None

    # 分页块表 (Block Tables)：
    # [形状: batch_size, max_blocks_per_seq] PagedAttention 的核心索引。
    # 记录每个请求所占用的物理显存块 ID（Block ID），用于在计算 Attention 时跨块检索历史 KV 信息。
    block_tables: torch.Tensor | None = None


# --- 全局单例对象 ---
# 模块级别私有变量，存储当前的上下文状态。
# 所有的模型算子都会访问这同一个对象，从而实现状态共享。
_CONTEXT = Context()


def get_context():
    """
    【读取者】获取当前的全局上下文。

    为什么使用函数而不是直接访问变量？
    1. 确保获取的是最新的对象引用。
    2. 提供统一的访问入口，便于未来增加调试、监控或检查逻辑。
    """
    return _CONTEXT


def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
):
    """
    【更新者】每一轮推理（Iteration）开始前，由引擎调用此函数设置新的状态。
    """
    global _CONTEXT  # 使用 global 关键字，明确表示我们要修改模块级别的全局变量指针

    # 创建一个新的 Context 实例并覆盖旧的单例
    # 这样所有通过 get_context() 访问的地方都会看到更新后的数据
    _CONTEXT = Context(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
    )


def reset_context():
    """
    【清理者】将全局上下文重置为默认值。
    通常在 Batch 推理结束或出现异常时调用，防止上一轮的数据干扰下一轮。
    """
    global _CONTEXT
    _CONTEXT = Context()
