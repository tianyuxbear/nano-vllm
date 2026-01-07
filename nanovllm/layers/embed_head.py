import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,  # 词表总大小 (例如 151936)
        embedding_dim: int,  # 每个词向量的维度 (例如 4096)
    ):
        super().__init__()
        # 获取分布式环境下的身份：我是第几个 GPU (rank)，总共有多少个 GPU (world_size)
        self.tp_rank = dist.get_rank()  # ty: ignore
        self.tp_size = dist.get_world_size()  # ty: ignore

        # 确保词表能被 GPU 数量平分，以便进行切片
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings

        # 计算当前卡需要负责的词表行数
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size

        # 计算当前卡负责词表的全局索引起始点和终点 (左闭右开)
        # 比如 rank 0 负责 0-49, rank 1 负责 50-99
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition

        # 定义本地权重：只申请自己负责的那一部分内存空间 [分片大小, 维度]
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )

        # 绑定自定义加载器，用于从完整权重中提取属于当前卡的分片
        self.weight.weight_loader = self.weight_loader  # ty: ignore

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        这个函数在加载模型时被调用。
        loaded_weight: 是从磁盘读取的完整词表权重 [总大小, 维度] (通常在 CPU 上)
        """
        param_data = param.data
        shard_size = param_data.size(0)  # 拿当前卡分片的大小

        # 计算在原始大矩阵中的起始行号
        start_idx = self.tp_rank * shard_size

        # 使用 narrow(维度, 起点, 长度) 切出属于当前卡的那一整块词向量
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)

        # 原地拷贝到 GPU 显存中的参数空间
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """
        前向传播：将 Token ID 转换为 Embedding 向量
        x: 输入的 Token 序列，形状通常为 [total_tokens]
        x 的形状为什么不是 [batch_size, seq_len]?
        为了处理变长序列，框架通常会进行‘Flatten"（拉平）操作：
        - 把Batch里的所有序列拼在一起，变成一个一维长向量。
        - 输入 x 的形状其实是 [total_num_tokens］
        """
        if self.tp_size > 1:
            # 1. 制作掩码：判断输入中哪些词是“我这块卡负责的”
            # 只有在 [vocab_start_idx, vocab_end_idx) 范围内的才设为 True
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)

            # 2. 坐标映射：将全局 ID 转换为本地局部 ID (0 ~ shard_size-1)
            # 比如全局 ID 5005，在 rank 1 (负责 5000-9999) 看来就是本地索引 5
            # 不属于我的 ID 会变成 0 (配合 mask 处理)
            x = mask * (x - self.vocab_start_idx)

        # 3. 局部查表：所有 GPU 同时在自己的分片里查表
        # 每个词在每张卡上都会查出一个向量，但只有负责该词的卡查到的是有意义的
        y = F.embedding(x, self.weight)

        if self.tp_size > 1:
            # 4. 掩码清理：不属于我负责的词，将其向量全部刷成 0
            # unsqueeze(1) 将 [N] 的 mask 变成 [N, 1] 从而广播到 [N, HiddenDim]
            y = mask.unsqueeze(1) * y

            # 5. 全局同步 (核心)：
            # 所有 GPU 把结果 y 累加同步。因为每张卡负责不同的词，
            # 最终 y = [向量1, 0, 0] + [0, 向量2, 0] + [0, 0, 向量3] ...
            # 运行完这一行，所有 GPU 上的 y 都变成了包含所有词的完整结果
            dist.all_reduce(y)  # ty: ignore

        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    并行语言模型输出头 (LM Head)。
    继承自 VocabParallelEmbedding，复用了其词表按行切分（Row Parallel）和权重加载的逻辑。
    该层的作用是将 Transformer 的隐藏状态映射回词表大小的 Logits。
    """

    def __init__(
        self,
        num_embeddings: int,  # 总词表大小
        embedding_dim: int,  # 隐藏层维度 (hidden_size)
        bias: bool = False,  # 是否使用偏置，Qwen/Llama 等现代模型通常为 False
    ):
        # 内部逻辑限制：该分布式实现暂不支持偏置项
        assert not bias
        # 调用父类初始化：完成 tp_rank/tp_size 的获取，并按 GPU 数量对权重进行切片存储
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        """
        前向传播计算 Logits。
        x: 输入的隐藏状态 [Total_Tokens, Hidden_Dim]
        """
        # 从全局单例中获取当前请求的上下文信息（包含是否为 Prefill 阶段及序列长度信息）
        context = get_context()

        # --- 优化步骤：Prefill 阶段的计算裁剪 ---
        if context.is_prefill:
            # context.cu_seqlens_q[1:] 是各个序列结束位置的偏移量
            # 减 1 得到每个序列最后一个 Token 在平铺向量中的索引（即我们真正关心的预测位）
            last_indices = context.cu_seqlens_q[1:] - 1

            # 高级索引操作：只取出每个序列最后一个 Token 的向量
            # x 从 [Total_Tokens, Hidden_Dim] 裁剪为 [Batch_Size, Hidden_Dim]
            # .contiguous() 确保内存连续，以便后续高效执行矩阵乘法 (C 语言层面的指针顺序)
            x = x[last_indices].contiguous()

        # --- 分布式矩阵乘法 ---
        # F.linear(x, self.weight) 执行 y = x * W^T
        # 由于 W (self.weight) 是按行切分的 [Vocab_per_GPU, Hidden_Dim]
        # 每个 GPU 此时只算出了自己负责的那部分词表对应的 Logits 片段
        logits = F.linear(x, self.weight)

        # --- 分布式结果汇总 ---
        if self.tp_size > 1:
            # 在 CPU 侧定义一个容器列表，仅在 Rank 0 节点上分配空间（减少非主卡显存占用）
            # 每个元素都是一个形状如 [Batch, 局部词表大小] 的张量句柄
            # 只有 Rank 0 会触发 GPU 显存申请指令。
            # 其他进程的 CPU 只是简单地把变量设为 NULL (None)，根本不去碰 GPU 显存。
            all_logits = (
                [torch.empty_like(logits) for _ in range(self.tp_size)]
                if self.tp_rank == 0
                else None
            )

            # 调用分布式集合通信 API: Gather (收集)
            # 指令流：所有 GPU 将各自显存中的 logits 发送到 Rank 0
            # 数据流：由 NCCL 库通过 NVLink/PCIe 总线在 GPU 间直接搬运，不经过 CPU 中转
            dist.gather(logits, all_logits, 0)  # 目标位置为 Rank 0 # ty: ignore

            # 结果拼接（仅在 Rank 0 执行）
            if self.tp_rank == 0:
                # 将收集到的多个局部张量在词表维度 (-1) 上横向拼接
                # [Batch, Part_1] + [Batch, Part_2] -> [Batch, Full_Vocab]
                logits = torch.cat(all_logits, -1)
            else:
                # 非 Rank 0 的 GPU 在此之后不再持有 Logits 的数据指针，返回 None
                # 后续的采样逻辑（Sampling）将由 Rank 0 负责决策
                logits = None

        # 返回的是一个指向 GPU 显存地址的 Tensor 句柄（指针封装）
        return logits
