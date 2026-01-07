import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,  # 隐藏层维度，即特征向量的长度
        eps: float = 1e-6,  # 防止除以 0 的极小值（epsilon）
    ) -> None:
        super().__init__()
        self.eps = eps
        # 学习参数：缩放因子（Scale），初始化为全 1
        # 与 LayerNorm 不同，RMSNorm 通常不使用偏置（Bias）
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile  # 使用 PyTorch 2.0 编译优化，将多个算子融合为一个内核，提升执行速度
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # 记录原始数据类型（例如可能是 float16 或 bfloat16）
        orig_dtype = x.dtype

        # 核心步骤 1：提升精度到 float32
        # 归一化涉及平方和累加，低精度（FP16）容易发生溢出，使用 FP32 保证数值稳定性
        x = x.float()

        # 核心步骤 2：计算均方根（RMS）
        # var = mean(x^2)，在最后一个维度进行计算
        var = x.pow(2).mean(dim=-1, keepdim=True)

        # 核心步骤 3：归一化计算 x = x / sqrt(var + eps)
        # torch.rsqrt 是平方根倒数，比先计算 sqrt 再计算除法更快
        x.mul_(torch.rsqrt(var + self.eps))

        # 核心步骤 4：恢复精度并进行可学习的缩放
        # x.to(orig_dtype) 将数据转回模型原始精度
        # .mul_(self.weight) 对应公式中的 g_i * x_i
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        这个函数实现了常见的 'Add & Norm' 模式，即将残差连接和归一化合并在一起处理。
        """
        orig_dtype = x.dtype

        # 步骤 1：将当前输入与残差相加
        # 同样转换为 float 计算以保证精度，并执行 inplace 操作节省内存
        x = x.float().add_(residual.float())

        # 步骤 2：将相加后的结果作为新的残差传出
        # 这是为了下一层的残差连接做准备
        residual = x.to(orig_dtype)

        # 步骤 3：对相加后的结果进行 RMS 归一化
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))

        # 步骤 4：缩放并转回原始精度
        x = x.to(orig_dtype).mul_(self.weight)

        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播入口：
        - 如果没有提供残差，只做 RMS 归一化。
        - 如果提供了残差，先做 Add 再做归一化，并返回新的残差。
        """
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
