from functools import lru_cache

import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    对输入张量应用旋转位置编码 (RoPE)。
    数学原理基于二维平面复数旋转：
    [y1]   [cos  -sin] [x1]
    [y2] = [sin   cos] [x2]
    """
    # 1. 维度切分 (Chunking):
    # 将 Head_Dim 维度切分为两半。
    # 假设输入 x 形状为 [Total_Tokens, Num_Heads, Head_Dim]
    # x1, x2 形状均为 [Total_Tokens, Num_Heads, Head_Dim/2]
    # 注意：强制转为 .float() (FP32) 是为了保证长序列下的旋转精度，防止累积误差。
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)

    # 2. 复数旋转 (Rotation):
    # 对应公式: Re(new) = x1*cos - x2*sin
    y1 = x1 * cos - x2 * sin
    # 对应公式: Im(new) = x2*cos + x1*sin
    y2 = x2 * cos + x1 * sin

    # 3. 拼接与恢复 (Concat & Cast):
    # 将旋转后的两部分拼回，并恢复到输入时的精度 (如 BF16/FP16)。
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """
    RoPE 模块：负责预计算频率表 (sin/cos cache) 并管理前向传播。
    """

    def __init__(
        self,
        head_size: int,  # 单个注意力头的维度 (Dim)
        rotary_dim: int,  # 实际参与旋转的维度 (通常等于 head_size)
        max_position_embeddings: int,  # 预计算的最大位置长度 (例如 4096, 8192)
        base: float,  # 频率基底 (Base Frequency, 如 10000.0)
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size, "目前仅支持全维度旋转"

        # ================= 步骤 1: 计算逆频率 (Inverse Frequencies) =================
        # 根据公式 theta_i = 1 / base^(2i/d)
        # torch.arange(0, rotary_dim, 2) 生成偶数索引 [0, 2, 4, ... d-2]
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        )

        # 生成位置索引 t = [0, 1, 2, ..., max_pos-1]
        t = torch.arange(max_position_embeddings, dtype=torch.float)

        # ================= 步骤 2: 生成全位置频率表 =================
        # 计算外积 (Outer Product): positions * frequencies
        # freqs 形状: [max_pos, rotary_dim/2]
        freqs = torch.einsum("i,j -> ij", t, inv_freq)

        # 计算 Cos 和 Sin
        cos = freqs.cos()
        sin = freqs.sin()

        # ================= 步骤 3: 构建缓存 Cache =================
        # 这里的 cache 结构设计非常紧凑:
        # 1. torch.cat((cos, sin), dim=-1):
        #    将 cos 和 sin 拼在一起，形状变为 [max_pos, rotary_dim]。
        #    (注意：前一半是 cos，后一半是 sin)
        #
        # 2. .unsqueeze_(1):
        #    关键的一步！在中间插入一个维度。
        #    最终形状: [Max_Pos, 1, Rotary_Dim]
        #    含义: [位置索引, 广播用的Head维度, 特征维度]
        #    这个 '1' 是为了在 forward 时自动广播到所有 Attention Heads。
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)

        # 注册为 buffer，persistent=False 表示该 Tensor 不会保存到 checkpoint 权重文件中
        # (因为它是确定性的数学计算结果，随时可以重新算)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile  # 启用 PyTorch 2.0 编译器加速
    def forward(
        self,
        positions: torch.Tensor,  # Token 的绝对位置索引。Flatten 模式下形状为 [Total_Tokens]
        query: torch.Tensor,  # [Total_Tokens, Num_Heads, Head_Dim]
        key: torch.Tensor,  # [Total_Tokens, Num_KV_Heads, Head_Dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. 查表 (Look up):
        # 使用高级索引 (Fancy Indexing) 取出当前 batch 中每个 token 对应的旋转参数。
        # positions 形状: [Total_Tokens]
        # cache 形状:     [Max_Pos, 1, Dim]
        # 取出结果 cos_sin: [Total_Tokens, 1, Dim]
        cos_sin = self.cos_sin_cache[positions]  # ty: ignore

        # 2. 拆解缓存 (Split):
        # 将拼在一起的 cos 和 sin 分开。
        # cos, sin 形状均为: [Total_Tokens, 1, Dim/2]
        cos, sin = cos_sin.chunk(2, dim=-1)

        # 3. 应用旋转 (Apply):
        # 利用广播机制 (Broadcasting):
        # Query: [Total_Tokens, Num_Heads, Dim/2]
        # Cos:   [Total_Tokens, 1,         Dim/2]
        # 中间的 '1' 会自动扩展匹配 Num_Heads，实现所有头共享同一旋转角度。
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)

        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    """
    RoPE 工厂函数 (单例模式)。

    使用 @lru_cache(1) 装饰器:
    - 作用: 确保对于相同的参数配置（如 Llama3 的配置），整个进程中只创建一个 RotaryEmbedding 实例。
    - 收益:
      1. 节省显存: 所有 Transformer Layer 共享同一份 sin/cos table，避免每层都存一份冗余数据。
      2. 加速初始化: 避免重复计算昂贵的 sin/cos 数学运算。
    """
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
