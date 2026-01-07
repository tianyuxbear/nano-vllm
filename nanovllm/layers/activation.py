import torch
import torch.nn.functional as F
from torch import nn


class SiluAndMul(nn.Module):
    """
    SwiGLU (Swish-Gated Linear Unit) 激活函数层的实现。
    常用于 Llama, Mistral 等现代 Transformer 架构的前馈网络 (FFN) 中。
    """

    def __init__(self):
        super().__init__()

    @torch.compile  # 核心优化：利用 PyTorch 2.0 编译技术进行算子融合
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数 x 的形状通常为: [total_num_tokens, intermediate_size * 2]

        背景说明：
        1. 在推理框架(如 nano-vllm)中，为了处理变长序列并消除 Padding 浪费，输入 x 已被 Flatten。
        2. 之前的线性层已经将 Gate 和 Up 两个投影矩阵合并计算，因此输入 x 在最后一个维度包含了两个部分。
        """

        # x.chunk(2, -1): 将最后一个维度一分为二
        # x: 对应 Gate 路径，形状为 [total_num_tokens, intermediate_size]
        # y: 对应 Up 路径，形状为 [total_num_tokens, intermediate_size]
        # 注意：chunk 是 view 操作，不产生额外的显存拷贝
        x, y = x.chunk(2, -1)

        # F.silu(x) * y:
        # 1. 对 Gate 部分应用 SiLU (x * sigmoid(x)) 激活函数。
        # 2. 将激活结果与 Up 部分进行元素级乘法 (Element-wise Multiplication)。
        # @torch.compile 会将这一系列操作融合进一个 CUDA Kernel，避免中间变量反复读写显存。
        return F.silu(x) * y
