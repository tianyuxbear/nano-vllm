import atexit
from dataclasses import fields
from time import perf_counter

import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class LLMEngine:
    """
    LLM 推理引擎核心类。
    负责协调模型加载、多进程并行计算、请求调度 (Scheduler) 以及 Token 生成的主循环。
    """

    def __init__(self, model, **kwargs):
        """
        初始化引擎。

        Args:
            model: 模型路径或名称 (用于加载 Config 和 Tokenizer)。
            **kwargs: 覆盖 Config 默认值的参数 (例如 tensor_parallel_size, enforce_eager 等)。
        """
        # --- 1. 配置处理 ---
        # 动态获取 Config 类中定义的所有字段名
        config_fields = {field.name for field in fields(Config)}
        # 筛选出 kwargs 中属于 Config 的参数
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # 实例化配置对象，统一管理模型参数
        config = Config(model, **config_kwargs)

        # --- 2. 多进程上下文 (Multiprocessing Context) ---
        self.ps = []  # 存储子进程对象 (Rank 1 ~ N-1)
        self.events = []  # 存储进程间同步用的 Event 对象

        # 关键点：使用 "spawn" 启动方式。
        # 在涉及 CUDA 的多进程编程中，必须用 spawn，否则 fork 会复制父进程的 CUDA Context 导致崩溃。
        ctx = mp.get_context("spawn")

        # --- 3. 启动并行工作进程 (Worker Processes) ---
        # 如果 tensor_parallel_size > 1，启动 rank 1 到 rank N-1 的子进程。
        # Rank 0 (主进程) 将在当前线程中运行。
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            # 每个子进程运行 ModelRunner，负责加载和计算模型权重的切片
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # --- 4. 初始化主进程组件 ---
        # 在主进程 (Rank 0) 实例化 ModelRunner。
        # 它是“指挥官”，除了负责自己的计算外，还负责协调通信。
        self.model_runner = ModelRunner(config, 0, self.events)

        # 加载 HuggingFace Tokenizer (用于文本 <-> Token ID 转换)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        # 将 Tokenizer 的 EOS Token ID 同步给 Config，用于判断生成结束
        config.eos = self.tokenizer.eos_token_id

        # 初始化调度器 (Scheduler)
        # 负责管理 KV Cache 显存块、决定哪些请求进入 Prefill 或 Decode 阶段
        self.scheduler = Scheduler(config)

        # 注册退出钩子：确保程序结束 (无论正常还是异常) 时调用 self.exit 清理子进程
        atexit.register(self.exit)

    def exit(self):
        """
        清理资源，优雅关闭。
        防止程序结束后留下僵尸进程或占用显存。
        """
        # 向所有 ModelRunner (包括子进程) 发送 "exit" 指令
        self.model_runner.call("exit")

        # 显式删除主进程的 ModelRunner 实例，触发析构清理
        del self.model_runner

        # 等待所有子进程安全退出
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        向引擎添加一个推理请求。
        请求会被加入等待队列，等待 step() 方法调度执行。

        Args:
            prompt: 输入提示词，可以是字符串或已编码的 token list。
            sampling_params: 该请求的采样参数 (temperature, top_p, max_tokens 等)。
        """
        # 如果输入是字符串，先进行 Tokenize 编码
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)

        # 将 Token IDs 和采样参数封装成 Sequence 对象
        seq = Sequence(prompt, sampling_params)

        # 提交给调度器 (Scheduler) 的等待队列
        self.scheduler.add(seq)

    def step(self):
        """
        执行一步推理的核心原子操作。
        包含：调度 -> 模型前向计算 -> 后处理。

        Returns:
            outputs: 本次 step 中**刚刚完成** (Finished) 的序列列表 [(seq_id, token_ids), ...]。
            num_tokens: 用于吞吐量统计。
                        正数表示 Prefill 阶段处理的 Token 总数；
                        负数表示 Decode 阶段生成的 Token 总数 (即 Batch Size)。
        """
        # 1. 调度 (Schedule)
        # 调度器决定当前运行哪些序列，分配 KV Cache，并返回批次数据。
        # is_prefill=True: 处理新进来的 Prompt (并行计算量大)
        # is_prefill=False: 处理正在生成的序列 (逐个 Token 生成)
        seqs, is_prefill = self.scheduler.schedule()

        # 2. 模型执行 (Model Execution)
        # 调用 ModelRunner 执行 GPU 计算。
        # 如果是多卡环境，Rank 0 会通过内部机制同步控制其他子进程。
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        # 3. 后处理 (Postprocess)
        # 将新生成的 token_ids 追加到 Sequence 中。
        # 检查停止条件 (EOS, Max Length)，并在 Scheduler 中释放已完成序列的资源。
        self.scheduler.postprocess(seqs, token_ids)

        # 4. 收集结果
        # 筛选出状态变为 "Finished" 的序列
        outputs = [
            (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]

        # 5. 统计 Token 数量 (用于性能监控)
        # 这里的负数设计是一个 trick，方便外部区分是 Prefill 还是 Decode 阶段
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)

        return outputs, num_tokens

    def is_finished(self):
        """检查引擎中是否还有未完成的任务 (包括等待队列和正在运行的队列)"""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        用户调用的主接口，执行完整的推理生成过程。
        实现了 Continuous Batching 的主循环。

        Args:
            prompts: 输入提示词列表。
            sampling_params: 采样参数 (可以是单个对象广播给所有 prompt，也可以是列表)。
            use_tqdm: 是否显示进度条。

        Returns:
            list[dict]: 包含生成文本和 token_ids 的结果列表。
        """
        # --- 1. 准备工作 ---
        # 初始化进度条，总数为请求的数量
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        # 广播采样参数
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # 将所有请求批量加入引擎
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}  # 存储最终结果 {seq_id: token_ids}
        prefill_throughput = decode_throughput = 0.0  # 吞吐量计数器

        # --- 2. 主生成循环 (Event Loop) ---
        # 只要调度器里还有活儿没干完，就一直 step
        while not self.is_finished():
            t = perf_counter()

            # >>> 关键：执行一步推理 <<<
            output, num_tokens = self.step()

            # --- 3. 性能监控与进度更新 ---
            if use_tqdm:
                dt = perf_counter() - t
                # 根据 num_tokens 的正负判断当前是 Prefill 还是 Decode
                if num_tokens > 0:
                    # Prefill 阶段：吞吐量 = Prompt Tokens / 时间
                    prefill_throughput = num_tokens / dt
                else:
                    # Decode 阶段：吞吐量 = Generated Tokens (Batch Size) / 时间
                    # num_tokens 为负数，取反
                    decode_throughput = -num_tokens / dt

                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )

            # --- 4. 收集本步完成的请求 ---
            # output 仅包含本步刚刚变为 Finished 的 seq
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)  # 更新进度条 (按请求数)

        # --- 5. 结果整理与解码 ---
        # 确保输出顺序与输入 Prompts 顺序一致 (通过 seq_id 排序)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]

        # Detokenize: 将生成的 Token IDs 转回文本
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]

        if use_tqdm:
            pbar.close()

        return outputs  # ty: ignore
