import os
from glob import glob

import torch
from safetensors import safe_open
from torch import nn


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """
    默认权重加载器：最直接的内存拷贝。
    使用 .copy_() 原地操作，将 CPU 上的张量数值灌入模型参数所在的内存地址。
    """
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """
    通用的模型权重加载函数，支持 Safetensors 格式及参数打包（Packing）逻辑。

    path: 存放 .safetensors 权重文件的文件夹路径。
    """
    # 获取模型定义的打包映射表（例如将 q_proj, k_proj, v_proj 映射到 qkv_proj）。
    # 如果模型没有定义该属性，则默认为空字典。
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    # 遍历路径下所有的 .safetensors 文件（支持分片权重加载）
    for file in glob(os.path.join(path, "*.safetensors")):
        # 使用 pt (PyTorch) 模式打开文件，将张量加载到 CPU 内存
        # safe_open 采用内存映射，能够高效读取
        with safe_open(file, "pt", "cpu") as f:
            # 遍历当前文件中的每一个权重名称（key）
            for weight_name in f.keys():
                # --- 核心逻辑 A：处理打包/合并模块 ---
                # 检查当前权重名是否命中映射表（例如判断 "q_proj" 是否在 "layer.0.q_proj.weight" 中）
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # 命中映射规则：取出对应的目标参数名(v)和分片索引(shard_id)
                        # 例如：k="q_proj" -> v="qkv_proj", shard_id="q"
                        v, shard_id = packed_modules_mapping[k]

                        # 通过字符串替换，将磁盘名称转换为模型内部真实的参数路径
                        param_name = weight_name.replace(k, v)
                        # 根据路径获取模型中真实的 nn.Parameter 对象（类似于获取指针）
                        param = model.get_parameter(param_name)

                        # 获取该参数特有的加载器（例如 QKVParallelLinear 内部定义的加载逻辑）
                        weight_loader = getattr(param, "weight_loader")

                        # 执行专用加载：将分片数据塞入目标参数的特定位置
                        weight_loader(param, f.get_tensor(weight_name), shard_id)

                        # 匹配成功并处理完后，跳出当前循环，处理下一个 weight_name
                        break

                # --- 核心逻辑 B：处理普通/非打包模块 ---
                # 如果上述 for 循环完整跑完且没有触发 break（即没有匹配到任何打包规则）
                else:
                    # 直接按名称查找模型参数
                    param = model.get_parameter(weight_name)

                    # 尝试获取参数自定义的加载器，如果没有，则使用默认的 copy_ 逻辑
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )

                    # 执行加载：将从文件读取的 tensor 拷贝给模型参数
                    weight_loader(param, f.get_tensor(weight_name))
