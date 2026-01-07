import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event

import torch
import torch.distributed as dist

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.context import get_context, reset_context, set_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event  # 用于多进程同步的事件对象

        # 初始化分布式通信组 (NCCL后端)
        dist.init_process_group(  # ty: ignore
            "nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank
        )

        # 设置当前进程使用的 GPU 设备
        torch.cuda.set_device(rank)

        # 设置默认数据类型 (如 bfloat16 或 float16)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)  # ty: ignore
        torch.set_default_device("cuda")

        # 加载模型架构并加载权重
        self.model = Qwen3ForCausalLM(hf_config)  # ty: ignore
        load_model(self.model, config.model)

        # 初始化采样器
        self.sampler = Sampler()

        # 模型预热 (分配显存，避免首次推理抖动)
        self.warmup_model()

        # 分配 KV Cache (根据剩余显存大小计算可容纳的 Block 数量)
        self.allocate_kv_cache()

        # 如果未强制使用 Eager 模式，则录制 CUDA Graph 以加速 Decode 阶段
        if not self.enforce_eager:
            self.capture_cudagraph()

        # 恢复默认设置
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 多卡并行模式下的控制逻辑
        if self.world_size > 1:
            if rank == 0:
                # Rank 0 (主进程/Controller): 创建共享内存，用于向其他 Rank 发送指令
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                # 同步点：等待 SharedMemory 创建完成
                dist.barrier()  # ty: ignore
            else:
                # Rank > 0 (工作进程/Worker): 同步等待 Rank 0 创建好 SharedMemory
                dist.barrier()  # ty: ignore
                # 连接到已创建的共享内存
                self.shm = SharedMemory(name="nanovllm")
                # 进入无限循环，等待 Rank 0 的指令
                self.loop()

    def exit(self):
        """退出清理函数"""
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()  # ty: ignore
            # 只有 Rank 0 负责销毁共享内存文件
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            # 清理 CUDA Graph 资源
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()  # ty: ignore

    def loop(self):
        """Rank > 0 的工作循环：从共享内存读取指令并执行"""
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)  # 执行对应的方法
            if method_name == "exit":
                break

    def read_shm(self):
        """从共享内存读取数据 (反序列化)"""
        assert self.world_size > 1 and self.rank > 0
        # 等待事件信号，表示 SharedMemory 中已有新数据
        self.event.wait()  # ty: ignore

        # 读取数据长度 (前4字节)
        n = int.from_bytes(self.shm.buf[0:4], "little")  # ty: ignore
        # 读取并反序列化数据 (方法名和参数)
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])  # ty: ignore

        # 清除事件状态，准备下一次等待
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """Rank 0 向共享内存写入数据 (序列化)"""
        assert self.world_size > 1 and self.rank == 0
        # 序列化方法名和参数
        data = pickle.dumps([method_name, *args])
        n = len(data)
        # 写入长度和数据
        self.shm.buf[0:4] = n.to_bytes(4, "little")  # ty: ignore
        self.shm.buf[4 : n + 4] = data  # ty: ignore

        # 触发事件，通知所有 Worker 进程读取
        for event in self.event:  # ty: ignore
            event.set()

    def call(self, method_name, *args):
        """
        统一调用入口。
        如果是 Rank 0，先通过 SharedMemory 将调用分发给其他 Ranks，然后自己执行。
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)  # ty: ignore

    def warmup_model(self):
        """执行一次虚拟推理，触发 Lazy Loading 或初始化"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # 创建最大长度的 Dummy Sequence
        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        num_seqs = min(
            max_num_batched_tokens // max_model_len, self.config.max_num_seqs
        )
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)  # 运行一次 Prefill
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """根据显存占用情况，计算并分配 KV Cache"""
        config = self.config
        hf_config = config.hf_config

        # 获取当前显存状态
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        # 计算本地（当前 GPU）需要负责的 KV Heads 数量
        num_kv_heads = hf_config.num_key_value_heads // self.world_size  # ty: ignore
        head_dim = getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,  # ty: ignore
        )

        # 计算一个 Block 所占用的字节数
        # 2 表示 K 和 V
        # layers * block_size * heads * dim * dtype_size
        block_bytes = (
            2
            * hf_config.num_hidden_layers  # ty: ignore
            * self.block_size
            * num_kv_heads
            * head_dim
            * hf_config.torch_dtype.itemsize  # ty: ignore
        )

        # 计算剩余可用显存可以容纳多少个 Blocks
        # 保留一部分显存 (gpu_memory_utilization) 并减去已使用的和峰值
        config.num_kvcache_blocks = (
            int(total * config.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )
        assert config.num_kvcache_blocks > 0

        # 分配全局 KV Cache Tensor
        # 形状: [2, layers, num_blocks, block_size, heads, head_dim]
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,  # ty: ignore
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
        )

        # 将 KV Cache 的不同 Layer 切片分配给模型对应的 Module
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """
        准备 Block Table Tensor。
        Block Table 记录了每个 Sequence 逻辑 Block 到物理 Block 的映射。
        需要进行 Padding 以对齐 Batch 中最长的序列。
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        # 用 -1 填充较短的序列
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        准备 Prefill 阶段所需的输入数据和元数据。
        """
        input_ids = []
        positions = []
        # FlashAttention 需要的 cumulative sequence lengths (累积序列长度)
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None

        for seq in seqs:
            seqlen = len(seq)
            # 提取新生成的 tokens (未被缓存的部分) 作为 input_ids
            input_ids.extend(seq[seq.num_cached_tokens :])
            # 生成对应的位置索引
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))

            # Query 长度：本次需要计算的新 token 数量
            seqlen_q = seqlen - seq.num_cached_tokens
            # Key 长度：总历史长度 (Prefix Cache + New Tokens)
            seqlen_k = seqlen

            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:  # warmup 阶段可能没有 block table
                continue

            # 计算 slot_mapping: 告诉 Kernel 每个 Token 的 KV 应该填入物理内存的哪个位置
            # 遍历该序列分配到的所有 Block
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                # 物理 Block ID * Block Size = 该 Block 起始的物理 Token 索引
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    # 最后一个 Block 可能没填满
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))

        # 如果 Key 长度 > Query 长度，说明利用了 Prefix Cache (前缀缓存)，
        # 此时 Attention 计算需要读取旧的 Block，因此必须准备 Block Table
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            block_tables = self.prepare_block_tables(seqs)

        # 转换为 Tensor 并移动到 GPU
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        # 设置全局 Context，传递给 Model 内部的 Attention 层使用
        set_context(
            True,  # is_prefill
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,  # context_lens 在 prefill 阶段通常不需要 (由 cu_seqlens 决定几何形状)
            block_tables,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """
        准备 Decode 阶段所需的输入数据。
        Decode 阶段 Batch 中的每个 Sequence 只生成 1 个 Token。
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        for seq in seqs:
            # Decode 阶段输入只是最后一个生成的 token
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            # 记录当前总长度，用于 RoPE 计算 (确定旋转角度) 和 Attention Mask
            context_lens.append(len(seq))

            # 计算当前这个新 Token 应该存放在 KV Cache 的哪个物理槽位
            # 最后一个 Block 的 ID * Size + 最后一个 Block 已有的 Token 数 - 1 (变为索引)
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        # 设置 Context，Decode 阶段需要 block_tables 来检索历史 KV，
        # 需要 context_lens 来计算 RoPE
        set_context(
            False,  # is_decode
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """准备采样参数 (如 temperature)"""
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ):
        """
        执行模型的前向计算。
        策略：
        1. Prefill 阶段或 Batch Size 过大时，直接使用 PyTorch Eager 模式。
        2. Decode 阶段且 Batch Size 较小时，使用 CUDA Graph 加速 (消除 Kernel 启动开销)。
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # Eager 模式直接计算
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # CUDA Graph 模式
            bs = input_ids.size(0)
            context = get_context()
            # 找到能够容纳当前 Batch Size 的最小 Graph (例如 bs=3 会用 graph_bs=4 的图)
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]

            # 将动态输入数据拷贝到 CUDA Graph 预分配的静态内存地址 (graph_vars)
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, : context.block_tables.size(1)] = (
                context.block_tables
            )

            # 重放 CUDA Graph (极快)
            graph.replay()

            # 获取输出并计算 Logits
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        推理的主入口函数。
        1. 准备数据 (Prepare)
        2. 运行模型 (Run)
        3. 采样 Token (Sample) - 仅 Rank 0 需要做
        """
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )
        # 仅 Rank 0 需要温度参数进行采样，其他 Rank 只需要负责计算 Logits/HiddenStates
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        logits = self.run_model(input_ids, positions, is_prefill)

        # 只有 Rank 0 进行采样操作
        token_ids = (
            self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        )
        reset_context()
        return token_ids  # ty: ignore

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        捕获 CUDA Graphs。
        针对不同的常用 Batch Size (1, 2, 4, 8, 16...) 预先录制计算图。
        这是减少 Decode 阶段小 Batch 情况下 CPU Launch Overhead 的关键技术。
        """
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)

        # 计算最大可能的 Block 数量 (用于预分配 Block Table buffer)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        # 预分配静态显存 Buffer (作为 Graph 的输入/输出占位符)
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)  # ty: ignore

        # 定义需要捕获的 Batch Size 档位
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None  # 共享内存池

        # 从大到小遍历捕获 (通常先捕获大的有助于内存复用，但这里用了共享 pool)
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()

            # 设置 Context 指向静态 Buffer 的切片
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )

            # Warmup: 先跑一次，确保分配和初始化完成
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

            # Capture: 开始录制 CUDA 核心执行流
            # 使用共享的 graph_pool，这样不同 Batch Size 的 Graph 可以复用显存
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

            if self.graph_pool is None:
                self.graph_pool = graph.pool()

            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 保存对静态 Buffer 的引用，以便 run_model 时拷贝数据
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
