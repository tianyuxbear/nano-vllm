import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


def divide(numerator, denominator):
    """
    辅助函数：确保能够整除，常用于计算每个 GPU 分到的头数或特征维度。
    """
    assert numerator % denominator == 0, f"无法整除: {numerator} / {denominator}"
    return numerator // denominator


class LinearBase(nn.Module):
    """
    所有并行线性层的基类。

    主要职责：
    1. 管理分布式环境信息 (rank, world_size)。
    2. 分配未初始化的权重内存 (Lazy Loading)。
    3. 定义统一的权重加载接口 (weight_loader)。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        # 获取当前 GPU 的编号 (Rank) 和 GPU 总数 (World Size)
        self.tp_rank = dist.get_rank()  # ty: ignore
        self.tp_size = dist.get_world_size()  # ty: ignore

        # 核心设计：使用 torch.empty 分配权重但不立即初始化数值。
        # 这样做是为了节省内存，权重数值将通过 weight_loader 从磁盘流式加载。
        self.weight = nn.Parameter(torch.empty(output_size, input_size))

        # 将 weight_loader 方法绑定到 weight 参数对象上，
        # 方便外部加载器直接调用 model.layers[i].weight.weight_loader(...)
        self.weight.weight_loader = self.weight_loader  # ty: ignore

        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader  # ty: ignore
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """
    【复制线性层】
    不进行切分。每个 GPU 上都持有一份完整的权重副本。
    通常用于 embedding 层或者模型中参数量较小的层。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # 直接拷贝完整权重，不进行任何切分
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 标准的矩阵乘法
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """
    【列并行线性层】

    切分策略：切分输出维度 (Output Size)。
    数学原理：Y = X * W^T。切分 W 的行（对应输出维度），相当于每个 GPU 计算输出向量的一部分特征。
    适用场景：Transformer 中的 MLP 第一层 (Gate/Up Proj) 或 Attention 的 QKV Proj。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()  # ty: ignore
        # output_size 被 tp_size 除，表示每个 GPU 只负责一部分输出特征
        # tp_dim=0 表示切分权重的第 0 维（即 PyTorch Linear 权重 [Out, In] 中的 Out）
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        加载逻辑：
        loaded_weight 是完整的权重 [Total_Out, In]。
        param 是本地的权重分片 [Total_Out/TP, In]。
        """
        param_data = param.data
        # shard_size = 本地分片的大小 (Total_Out / TP)
        shard_size = param_data.size(self.tp_dim)
        # 计算当前 GPU 应该从完整权重的哪个位置开始读
        start_idx = self.tp_rank * shard_size

        # narrow(dim, start, length): 从完整权重中切出属于当前 Rank 的那一段
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)  # ty: ignore

        # 拷贝到本地显存
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输出是分片的 (Sharded Output)，后续通常不需要 All-Reduce，
        # 而是直接传给下一个 Linear (如 RowParallel) 或激活函数。
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    【合并列并行线性层】

    场景：Llama 等模型中，Gate Proj 和 Up Proj 通常被合并成一个大矩阵计算以提高效率。
    难点：需要处理多个逻辑层（如 Gate 和 Up）合并后的权重加载。
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],  # 例如 [gate_size, up_size]
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        # 初始化时，总输出大小是所有子层大小之和
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int
    ):  # ty: ignore
        """
        loaded_shard_id: 指示当前加载的是第几个逻辑子层（0=Gate, 1=Up）。
        """
        param_data = param.data

        # 1. 计算本地偏移 (Local Offset):
        # sum(...) 计算该子层在全局矩阵中的起始位置。
        # // self.tp_size 将全局坐标转换为本地分片内的坐标（因为本地容器变小了）。
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size

        # 2. 计算本地分片大小:
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size

        # 3. 在本地容器中“框选”出目标区域 (View)
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)  # ty: ignore

        # 4. 从源权重中切分
        # loaded_weight.chunk(...) 将原始完整的子层权重按 TP 切分
        # [self.tp_rank] 取出属于当前 GPU 的那一份
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]  # ty: ignore

        # 5. 填空
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """
    【QKV 专用并行层】

    场景：Attention 层的 Q、K、V 投影。
    特殊性：支持 GQA (Grouped Query Attention)，即 Q 头数多，K/V 头数少。
    目标：确保每个 GPU 拿到的 Q, K, V 头在内存中是连续排列的，且属于同一个 Attention Group。
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()  # ty: ignore
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size

        # 计算每个 GPU 应持有的头数量
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)

        # 本地总输出大小 = (本地Q头 + 本地K头 + 本地V头) * 头维度
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str
    ):  # ty: ignore
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]

        # 根据 Q/K/V 类型，计算在本地容器中的偏移量和大小
        # 内存布局顺序：[Q_local | K_local | V_local]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            # K 放在 Q 后面
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            # V 放在 Q 和 K 后面
            shard_offset = (
                self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            )

        # 1. 在本地容器中定位
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)  # ty: ignore

        # 2. 从源权重中切取当前 Rank 对应的部分
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]  # ty: ignore

        # 3. 拷贝
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """
    【行并行线性层】

    切分策略：切分输入维度 (Input Size)。
    数学原理：将矩阵 W 按列切分（[Out, In/TP]）。输入 x 也按最后维度切分。
             每个 GPU 计算局部结果，最后通过 All-Reduce 求和。
    适用场景：Transformer 中的 MLP 第二层 (Down Proj) 或 Attention 的 Output Proj。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()  # ty: ignore
        # tp_dim=1 表示切分权重的第 1 维（即 PyTorch Linear 权重 [Out, In] 中的 In）
        # 注意：这里 input_size 被除以了 tp_size
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        # 获取本地分片后的宽度 (Input / TP)
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size

        # 在第 1 维 (In) 上进行切分加载
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)  # ty: ignore
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 本地计算：x_shard * W_shard
        # 注意：只有 rank 0 加 bias，避免重复累加
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)

        # 2. 分布式同步：All-Reduce (Sum)
        # 将所有 GPU 的局部结果相加，得到最终完整的输出
        if self.tp_size > 1:
            dist.all_reduce(y)  # ty: ignore

        return y
