import torch
from torch import nn


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    # 【性能关键】使用 PyTorch 2.0+ 编译器装饰器
    # 作用：算子融合 (Kernel Fusion)。
    # 它可以将下面的 div, softmax, exponential, argmax 等多个操作合并成一个 CUDA Kernel。
    # 这避免了多次读写显存 (减少 Memory Bandwidth 瓶颈)，对于这种纯数学计算的函数，加速效果极其明显。
    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        输入 Shapes 标注:
        logits:       [Batch_Size, Vocab_Size]  (例如: [2, 128000])
        temperatures: [Batch_Size]              (例如: [2])
        """

        # ================= 步骤 1: 精度转换与温度缩放 =================
        # logits.float():
        #   将数据转为 FP32。FP16/BF16 在 Softmax 和指数运算中容易溢出，FP32 保证数值稳定性。
        # temperatures.unsqueeze(dim=1):
        #   利用 'unsqueeze' 改变形状: [B] -> [B, 1]。
        #   这是为了利用广播机制 (Broadcasting)，让每行的所有 Token (Vocab维度) 都除以同一个温度值。
        # .div_(...):
        #   原地执行除法。
        #   - T > 1: 缩小 Logits 差异，概率分布变平 -> 输出更多样 (Random)。
        #   - T < 1: 放大 Logits 差异，强者越强 -> 输出更确定 (Greedy)。
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        # 此时 logits 形状: [B, V]

        # ================= 步骤 2: 转为概率分布 =================
        # dim=-1: 沿着最后一个维度 (词表维度) 进行归一化，使得每行的概率之和为 1。
        probs = torch.softmax(logits, dim=-1)
        # 此时 probs 形状: [B, V] (这就是我们的“权重”)

        # ================= 步骤 3: 高性能 Gumbel-Max 随机采样 =================
        # 这里的逻辑等价于 torch.multinomial(probs, 1)，是全词表的“加权随机采样”。
        # 核心公式: Token = argmax( Probability / Exponential_Noise )

        sample_tokens = probs.div_(
            # 3.1 极速内存分配:
            # empty_like 比 zeros_like 快，因为它只申请显存而不进行初始化（不写0）。
            torch.empty_like(probs)
            # 3.2 注入随机性 (In-place):
            # 填充服从指数分布 Exp(1) 的噪声。这是实现加权采样的数学基础。
            .exponential_(1)
            # 3.3 数值安全兜底 (In-place):
            # 防止生成极小的噪声 (接近0)，导致除法结果出现 inf 或 NaN。
            .clamp_min_(1e-10)
        ).argmax(dim=-1)  # 3.4 归约选择: 选出“得分”最大的索引。

        # 原理解析:
        # - 分子 (probs) 大的 Token，抗干扰能力强，容易被选中。
        # - 分子 (probs) 小的 Token，只有分母 (noise) 极小(运气极好)时才能被选中。
        # - .argmax(dim=-1) 会消掉最后一维，形状从 [B, V] 变为 [B]。

        # 返回形状: [Batch_Size] -> 包含了每一句话采样出的下一个 Token ID
        return sample_tokens
