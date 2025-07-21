from utils.loraPathConfigure import pathConfigurator
import os
import json
from utils.loraTrain.buildandloadData import TextDataset,collate_fn
from torch.utils.data import DataLoader
# from benchmark.config.code.config import CORPUS_PATH
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from tqdm import tqdm
import torch
from benchmark.config.code.config_lora import MODEL_SOURCE
if MODEL_SOURCE == "modelscope":
    from modelscope import AutoTokenizer
    from modelscope import Model
else:
    from transformers import AutoTokenizer
    from transformers import AutoModelForCausalLM as Model
from accelerate import load_checkpoint_and_dispatch
from benchmark.config.code.config_lora import LORA_CONFIG_PATH,load_config
from utils.loraTrain.buildandloadData import QADataset
import traceback
import requests
import time
from together import Together
import logging
from safetensors.torch import load_file
from utils.memoryDebug.memoryCheck import GPUMemoryProfiler
profiler = GPUMemoryProfiler()
# 保存 LoRA 模型
def save_lora_model(lora_model, save_path):
    """保存 LoRA 模型"""
    lora_model.save_pretrained(save_path)

# 加载 LoRA 模型
def load_lora_model_withPeft(base_model, load_path):
    """加载 LoRA 模型"""
    return PeftModel.from_pretrained(base_model, load_path)
# 加载基础模型和分词器
# 训练 LoRA 模型
# 创建 LoRA 配置
def create_lora_config(target_modules, layers_to_transform=[0,1,2], r=8, alpha=16):
    """创建 LoRA 配置"""
    # 构建 target_modules 列表，包含特定层的特定模块
    # 对于 Mistral MoE，我们需要针对专家层进行配置
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        layers_to_transform=layers_to_transform,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        modules_to_save=None,
    )
def train_lora_model(lora_model, dataloader, num_epochs=10, learning_rate=1e-3, train_config=None,output_adaptor_path=None):
    """
    训练因果语言模型的LoRA适配器
    
    Args:
        lora_model: LoRA模型实例
        dataloader: 训练数据加载器
        num_epochs: 训练轮数
        learning_rate: 学习率
        config: 训练配置字典，包含以下可选参数:
            - target_batch_size (int): 目标批次大小，用于梯度累积计算，默认16
            - precision (str): 训练精度，可选["float32", "float16", "bfloat16"]，默认"float32"
            - save_path_base (str): 保存路径基础目录，默认"/datanfs2/chenrongyi/models/versiBCB"
            - 其他参数会被忽略，不影响训练过程
        output_adaptor_path: 输出适配器路径
        
    Returns:
        lora_model: 训练好的LoRA模型
        
    Note:
        此函数专注于训练过程，config中只使用训练相关的参数。
        路径相关的参数(如model_name, knowledge_type等)应在调用此函数前处理好。
    """
    from llmfoundry.optim import DecoupledLionW
    import os
    
    # 尝试导入matplotlib，如果失败则禁用绘图功能
    try:
        import matplotlib
        matplotlib.use('Agg')  # 设置为非交互式后端
        import matplotlib.pyplot as plt
        plotting_enabled = True
    except ImportError:
        print("警告：无法导入matplotlib，损失曲线绘图功能将被禁用。")
        print("可以尝试运行 'pip install matplotlib' 来启用损失曲线绘图功能。")
        plotting_enabled = False
    
    # 使用 DecoupledLionW 优化器
    optimizer = DecoupledLionW(
        lora_model.parameters(),
        lr=float(learning_rate),
        betas=(0.9, 0.95),
        weight_decay=1e-6
    )
    
    # 简化设备检查 - 只打印基本信息
    print("模型设备信息:")
    if hasattr(lora_model, 'hf_device_map'):
        print(f"设备映射: {lora_model.hf_device_map}")
    else:
        first_param_device = next(lora_model.parameters()).device
        print(f"模型设备: {first_param_device}")
    
    # 梯度累积设置
    actual_batch_size = dataloader.batch_size
    target_batch_size = 16  # 默认目标批次大小
    
    # 如果提供了config，从中读取目标批次大小
    if train_config and "target_batch_size" in train_config:
        target_batch_size = train_config["target_batch_size"]
        print(f"从配置中读取目标批次大小: {target_batch_size}")
    
    accumulation_steps = max(1, target_batch_size // actual_batch_size)
    print(f"使用梯度累积: 实际批次大小={actual_batch_size}, 目标批次大小={target_batch_size}, 累积步数={accumulation_steps}")
    
    # 创建保存日志和模型的目录
    save_path_base = train_config.get("save_path_base", "/datanfs2/chenrongyi/models/versiBCB") if train_config else "/datanfs2/chenrongyi/models/versiBCB"
    log_dir = os.path.join(output_adaptor_path, "training_logs")
    checkpoint_dir = os.path.join(output_adaptor_path, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 为本次训练创建唯一的时间戳
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建日志文件路径和checkpoint路径基础
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.csv")
    log_plot = os.path.join(log_dir, f"loss_curve_{timestamp}.png")
    checkpoint_base = os.path.join(checkpoint_dir, f"checkpoint_{timestamp}")
    
    # 初始化损失记录
    epoch_losses = []
    step_losses = []
    
    # 初始化batch级别的损失记录
    batch_losses = []  # 记录每个batch的loss
    
    # 写入日志文件头
    with open(log_file, "w") as f:
        f.write("epoch,step,batch,loss\n")
    
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        batch_count = 0
        valid_batch_count = 0  # Count of batches with valid (non-NaN) losses
        step_count = 0
        epoch_step_losses = []  # 记录当前epoch的每个step的loss
        
        # 在每个epoch开始前清零梯度
        optimizer.zero_grad()
        
        for batch in tqdm(dataloader):
            profiler.record("训练模型")
            # 提取batch数据
            
            inputs = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]
            
            # 打印输入数据维度信息
            print(f"📊 Epoch {epoch+1}, Batch {batch_count+1} 输入维度:")
            print(f"  input_ids: {inputs.shape} (dtype: {inputs.dtype})")
            print(f"  labels: {labels.shape} (dtype: {labels.dtype})")
            print(f"  attention_mask: {attention_mask.shape} (dtype: {attention_mask.dtype})")
            print(f"  设备: {inputs.device}")
            
            # 简化设备处理 - 让模型自己处理设备分配
            # 不手动移动数据，让transformers的device_map自动处理
            
            # 获取精度设置
            # precision = train_config.get("precision")
            # torch_dtype = torch.float32
            # if precision == "float16":
            #     torch_dtype = torch.float16
            # elif precision == "bfloat16":
            #     torch_dtype = torch.bfloat16
            
            try:
                profiler.record("前向传播之前")
                # 前向传播 - 使用autocast但让模型自己处理设备
                # with torch.amp.autocast(device_type='cuda', dtype=torch_dtype, enabled=torch.cuda.is_available()):
                    
                outputs = lora_model(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    labels=labels
                )
                print(f"📊 Epoch {epoch+1}, Batch {batch_count+1} 输入维度:")
                print(f"  input_ids: {inputs.shape} (dtype: {inputs.dtype})")
                print(f"  labels: {labels.shape} (dtype: {labels.dtype})")
                print(f"  attention_mask: {attention_mask.shape} (dtype: {attention_mask.dtype})")
                print(f"  设备: {inputs.device}")
                            
                profiler.record("前向传播之后")
                loss = outputs.loss
                loss = loss / accumulation_steps
                
                # 反向传播
                profiler.record("反向传播之前")
                loss.backward()
                profiler.record("反向传播之后")
                # 更新累积损失（用于日志记录）
                current_loss = loss.item() * accumulation_steps  # 恢复原始损失大小用于记录
                
                # 检查loss是否为NaN，只有非NaN值才计入平均值
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += current_loss
                    valid_batch_count += 1
                    # 记录每个batch的loss
                    batch_losses.append(current_loss)
                else:
                    print(f"Warning: NaN or Inf loss detected in Epoch {epoch+1}, Batch {batch_count+1}. Ignoring this batch for average calculation.")
                
                batch_count += 1
                step_count += 1
                
                # 每个batch都输出当前batch的loss
                if batch_count % 10 == 0:  # 每10个batch输出一次，减少输出频率
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_count}/{len(dataloader)}, Current Batch Loss: {current_loss:.4f}")
                
                # 每accumulation_steps步更新一次参数
                if step_count % accumulation_steps == 0 or batch_count == len(dataloader):
                    # 梯度裁剪，防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(lora_model.parameters(), max_norm=1.0)
                    
                    # 参数更新
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # 释放未使用的GPU内存缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # 计算当前步骤的平均损失，只使用有效的batch
                    if valid_batch_count > 0:
                        avg_loss = total_loss / valid_batch_count
                    else:
                        avg_loss = float('nan')  # 如果没有有效batch，设置为NaN
                    
                    if not torch.isnan(torch.tensor(avg_loss)) and not torch.isinf(torch.tensor(avg_loss)):
                        epoch_step_losses.append(avg_loss)
                        step_losses.append(avg_loss)
                    
                    # 记录到日志文件
                    with open(log_file, "a") as f:
                        f.write(f"{epoch+1},{step_count // accumulation_steps},{batch_count},{avg_loss:.6f}\n")
                    
                    # 每2个更新步骤输出一次当前avg loss
                    if (step_count // accumulation_steps) % 2 == 0:
                        print(f"Epoch {epoch+1}/{num_epochs}, Step {step_count // accumulation_steps}, Batch {batch_count}/{len(dataloader)}, Avg Loss: {avg_loss:.4f}")
                
            except RuntimeError as e:
                if "expected all tensors to be on the same device" in str(e).lower():
                    print(f"设备不一致错误: {e}")
                    print("跳过此batch并继续训练...")
                    
                    # 跳过这个batch
                    print(f"跳过 Epoch {epoch+1}, Batch {batch_count+1} 由于设备不一致错误")
                    batch_count += 1
                    step_count += 1
                    
                    # 清理梯度
                    optimizer.zero_grad()
                    
                    # 继续下一个batch
                    continue
                elif "out of memory" in str(e).lower():
                    print(f"GPU内存不足: {e}")
                    print("清理缓存并跳过此batch...")
                    torch.cuda.empty_cache()
                    
                    batch_count += 1
                    step_count += 1
                    optimizer.zero_grad()
                    continue
                else:
                    # 其他RuntimeError，添加traceback并重新抛出
                    print(f"训练过程中出现RuntimeError: {e}")
                    traceback.print_exc()
                    raise e
        
        # 计算并存储此epoch的平均损失
        avg_epoch_loss = total_loss / max(1, valid_batch_count)  # 使用有效batch数量
        epoch_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss:.4f} (基于 {valid_batch_count} 个有效batch)")
        
        # 每2个epoch保存一次模型
        if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
            checkpoint_path = f"{checkpoint_base}_epoch{epoch+1}"
            os.makedirs(checkpoint_path, exist_ok=True)
            print(f"保存中间检查点到 {checkpoint_path}")
            lora_model.save_pretrained(checkpoint_path)
            
            # 保存最新的损失曲线图
            if plotting_enabled:
                try:
                    plt.figure(figsize=(12, 8))
                    
                    # 绘制Epoch损失曲线
                    plt.subplot(2, 2, 1)
                    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
                    plt.title('Epoch Average Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.grid(True)
                    
                    # 绘制Step损失曲线
                    plt.subplot(2, 2, 2)
                    plt.plot(step_losses, marker='.', linestyle='-', alpha=0.5)
                    plt.title('Training Loss per Step')
                    plt.xlabel('Step')
                    plt.ylabel('Loss')
                    plt.grid(True)
                    
                    # 绘制Batch损失曲线
                    plt.subplot(2, 2, 3)
                    plt.plot(batch_losses, marker='.', linestyle='-', alpha=0.3, color='red')
                    plt.title('Training Loss per Batch')
                    plt.xlabel('Batch')
                    plt.ylabel('Loss')
                    plt.grid(True)
                    
                    # 平滑的Batch损失曲线（使用移动平均）
                    if len(batch_losses) > 10:
                        window_size = min(10, len(batch_losses) // 5)
                        if window_size > 0:
                            import numpy as np
                            smooth_losses = np.convolve(batch_losses, np.ones(window_size)/window_size, mode='valid')
                            plt.subplot(2, 2, 4)
                            plt.plot(smooth_losses, marker='', linestyle='-', color='green')
                            plt.title(f'Smoothed Batch Loss (Window={window_size})')
                            plt.xlabel('Batch')
                            plt.ylabel('Loss')
                            plt.grid(True)
                    
                    plt.tight_layout()
                    plt.savefig(log_plot)
                    plt.close()
                    
                    print(f"损失曲线已保存到 {log_plot}")
                except Exception as e:
                    print(f"绘制损失曲线时出错: {e}")
    
    # 训练结束，保存最终的损失曲线
    if plotting_enabled:
        try:
            plt.figure(figsize=(12, 8))
            
            # 绘制Epoch损失曲线
            plt.subplot(2, 2, 1)
            plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
            plt.title('Epoch Average Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            
            # 绘制Step损失曲线
            plt.subplot(2, 2, 2)
            plt.plot(step_losses, marker='.', linestyle='-', alpha=0.5)
            plt.title('Training Loss per Step')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.grid(True)
            
            # 绘制Batch损失曲线
            plt.subplot(2, 2, 3)
            plt.plot(batch_losses, marker='.', linestyle='-', alpha=0.3, color='red')
            plt.title('Training Loss per Batch')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.grid(True)
            
            # 平滑的Batch损失曲线（使用移动平均）
            if len(batch_losses) > 10:
                window_size = min(10, len(batch_losses) // 5)
                if window_size > 0:
                    import numpy as np
                    smooth_losses = np.convolve(batch_losses, np.ones(window_size)/window_size, mode='valid')
                    plt.subplot(2, 2, 4)
                    plt.plot(smooth_losses, marker='', linestyle='-', color='green')
                    plt.title(f'Smoothed Batch Loss (Window={window_size})')
                    plt.xlabel('Batch')
                    plt.ylabel('Loss')
                    plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(log_plot)
            plt.close()
            
            print(f"训练完成。最终损失曲线已保存到 {log_plot}")
        except Exception as e:
            print(f"绘制最终损失曲线时出错: {e}")
    else:
        print("训练完成。由于matplotlib不可用，损失曲线未绘制。")
    
    return lora_model

def loadTokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return tokenizer
def str2dtype(precision):
    """根据精度字符串返回对应的torch数据类型"""
    if precision == 'fp16':
        return torch.float16
    elif precision == 'fp32':
        return torch.float32
    elif precision == 'bf16':
        return torch.bfloat16
    else:
        print(f"警告: 不支持的精度 '{precision}'，使用默认精度 fp16")
        return torch.float16
def load_base_model(model_name_or_path, device_map="auto", precision_from_arg='fp16', force_gpu=False):
    """加载基础模型和分词器
    
    Args:
        model_name_or_path: str, 模型名称或路径
        device_map: str or dict, 设备映射策略。直接使用"auto"进行自动分配
        precision: str, 模型精度 ('fp16', 'fp32', 'bf16')
        force_gpu: bool, 强制将整个模型加载到GPU上，优先选择内存最多的GPU
    
    Returns:
        model: 加载的模型
        tokenizer: 加载的分词器
    """

    
    def get_best_gpu():
        """获取可用内存最多的GPU"""
        if not torch.cuda.is_available():
            return None
            
        best_gpu = 0
        max_free_memory = 0
        
        for i in range(torch.cuda.device_count()):
            try:
                # 清理GPU缓存以获得准确的内存信息
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                
                total_memory = torch.cuda.get_device_properties(i).total_memory
                allocated_memory = torch.cuda.memory_allocated(i)
                reserved_memory = torch.cuda.memory_reserved(i)
                free_memory = total_memory - max(allocated_memory, reserved_memory)
                
                print(f"GPU {i}: {free_memory/(1024**3):.1f}GB free / {total_memory/(1024**3):.1f}GB total")
                
                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    best_gpu = i
                    
            except Exception as e:
                print(f"检查GPU {i} 时出错: {e}")
                continue
                
        print(f"选择GPU {best_gpu}，可用内存: {max_free_memory/(1024**3):.1f}GB")
        return best_gpu
    
    config = load_config(LORA_CONFIG_PATH)
    if precision_from_arg:
        precision = precision_from_arg
    else:
        precision = config.get("precison","fp16")

    print(f"加载基础模型: {model_name_or_path}")
    print(f"强制GPU模式: {force_gpu}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    # 为 Mistral/CodeGemma 等模型设置 padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("设置 pad_token = eos_token")
    
    # 获取 GPU 信息
    gpu_count = torch.cuda.device_count()
    print(f"可用GPU数量: {gpu_count}")
    
    # 确定设备映射策略
    if force_gpu and torch.cuda.is_available():
        best_gpu = get_best_gpu()
        if best_gpu is not None:
            # 强制使用特定GPU
            optimized_device_map = {"": best_gpu}
            print(f"强制GPU模式：将整个模型加载到GPU {best_gpu}")
        else:
            print("警告：无法找到合适的GPU，回退到auto模式")
            optimized_device_map = device_map
    else:
        optimized_device_map = device_map
        print(f"使用设备映射策略: {optimized_device_map}")
    
    try:
        # 检查内存情况
        for i in range(gpu_count):
            free_mem = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
            free_mem_gb = free_mem / (1024**3)
            print(f"GPU {i} 可用内存: {free_mem_gb:.2f} GB")
    except Exception as e:
        print(f"获取GPU内存信息时出错: {e}")
    
    print(f"使用模型精度: {precision}")
    
    # 简化的模型加载方式 - 直接使用AutoModelForCausalLM的device_map
    torch_dtype = str2dtype(precision)
    print(f"使用精度 {precision} (torch_dtype: {torch_dtype}) 加载模型")
    
    # 模型加载参数
    model_kwargs = {
        "device_map": optimized_device_map,
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }
    
    # 如果强制GPU模式，添加额外参数确保模型完全加载到GPU
    if force_gpu and isinstance(optimized_device_map, dict) and "" in optimized_device_map:
        target_gpu = optimized_device_map[""]
        # 计算可用内存并设置max_memory以确保模型能加载
        try:
            torch.cuda.set_device(target_gpu)
            torch.cuda.empty_cache()
            
            total_memory = torch.cuda.get_device_properties(target_gpu).total_memory
            allocated_memory = torch.cuda.memory_allocated(target_gpu)
            free_memory = total_memory - allocated_memory
            
            # 为模型预留90%的可用内存
            reserved_memory_gb = (free_memory * 0.9) / (1024**3)
            model_kwargs["max_memory"] = {target_gpu: f"{reserved_memory_gb:.1f}GB"}
            print(f"为GPU {target_gpu} 设置最大内存限制: {reserved_memory_gb:.1f}GB")
            
        except Exception as e:
            print(f"设置内存限制时出错: {e}")
    
    try:
        print("开始加载模型...")
        model = Model.from_pretrained(
            model_name_or_path,
            **model_kwargs
        )
        print("模型加载成功!")
        
        # 打印模型分布情况
        if hasattr(model, 'hf_device_map'):
            print(f"模型层分布: {model.hf_device_map}")
            
            # 检查是否有层在CPU上
            devices_used = set(model.hf_device_map.values())
            cpu_layers = [layer for layer, device in model.hf_device_map.items() if device == 'cpu']
            
            if cpu_layers and force_gpu:
                print(f"警告：尽管启用了强制GPU模式，仍有 {len(cpu_layers)} 层在CPU上:")
                print(f"CPU层: {cpu_layers[:5]}{'...' if len(cpu_layers) > 5 else ''}")
            elif not cpu_layers:
                print("✓ 所有模型层都在GPU上")
            
            # 统计设备分布
            device_count = {}
            for device in devices_used:
                device_count[device] = sum(1 for d in model.hf_device_map.values() if d == device)
            
            print("设备分布统计:")
            for device, count in device_count.items():
                print(f"  {device}: {count} 层")
                
        else:
            print("模型未使用设备映射")
            
    except Exception as e:
        print(f"模型加载失败: {e}")
        if force_gpu:
            print("尝试回退到非强制GPU模式...")
            model_kwargs["device_map"] = "auto"
            if "max_memory" in model_kwargs:
                del model_kwargs["max_memory"]
            
            try:
                model = Model.from_pretrained(
                    model_name_or_path,
                    **model_kwargs
                )
                print("使用回退策略成功加载模型")
            except Exception as fallback_error:
                print(f"回退策略也失败: {fallback_error}")
                raise fallback_error
        else:
            raise e
    
    return model, tokenizer
def getDataExistence(corpus_path,dependency):
    '''
    Description:
        判断数据是否存在
    Args:
        data: dict,数据
    '''
    for pkg,version in dependency.items():
        files_info = []
        try:
            with open(f"{corpus_path}/{pkg}/{version}.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    line_data = json.loads(line)
                    files_info.append(str(line_data))    
        except Exception as e:
            return False
        if len(files_info) == 0:
            return False
    return True
# def load_config(config_path):
#     with open(config_path, "r") as f:
#         config = json.load(f)
#     return config
def loraModelExists(pkg,version,model_name,config,knowledge_type=None):
    if config is None:
        raise ValueError("config is None")
    pathConfig = pathConfigurator()
    if model_name is None:
        model_name = config["model_name"].split("/")[-1]
    path = pathConfig.getPath(config,pkg,version,model_name,knowledge_type=knowledge_type)
    # 接着检查对应文件夹下是否有safetensors
    if not os.path.exists(path):
        return False
    
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path) and file.endswith(".safetensors"):
            return True
    return False
    # return os.path.exists(path)
def load_lora_model(pkg, version, config, knowledge_type, pred_args):
    '''
    Description:
        加载lora模型，支持根据adopt_IFT_checkpoint参数选择加载IFT checkpoint
        增强版：使用IFT模型管理器智能选择最佳匹配的IFT模型
        新增：支持包级别的IFT启用控制，只对指定的包启用IFT
    Args:
        pkg: str,包名
        version: str,版本号
        config: dict,配置
        knowledge_type: str,知识类型
        pred_args: object,预测参数对象，包含adopt_IFT_checkpoint等参数
    Returns:
        str: 加载的lora模型路径
    '''
    pathConfig = pathConfigurator()
    model_name = config["model_name"].split("/")[-1]
    # 用于兼容直接使用llama3.1-8B的checkpoint
    # if model_name == "Llama-3.1-8B-Instruct":
    #     model_name = "Llama-3.1-8B"
    # 检查是否全局启用IFT checkpoint
    adopt_ift_global = getattr(pred_args, 'adopt_IFT_checkpoint', False) if pred_args else False
    
    # 🎯 新增：检查包级别的IFT启用控制
    ift_enabled_packages = getattr(pred_args, 'ift_enabled_packages', None) if pred_args else None
    
    # 决定是否为当前包启用IFT
    adopt_ift_for_this_pkg = False
    ift_control_mode = "default"
    
    if ift_enabled_packages is not None:
        # 如果指定了包级别控制，只对指定的包启用IFT
        adopt_ift_for_this_pkg = pkg in ift_enabled_packages
        ift_control_mode = "package_selective"
        if adopt_ift_for_this_pkg:
            logging.info(f"📦 {pkg}-{version}: 包在IFT启用列表中，将尝试加载IFT模型")
        else:
            logging.info(f"📝 {pkg}-{version}: 包不在IFT启用列表中，直接使用LoRA模型")
    elif adopt_ift_global:
        # 如果没有指定包级别控制，但全局启用了IFT，则所有包都尝试IFT
        adopt_ift_for_this_pkg = True
        ift_control_mode = "global_enabled"
        logging.info(f"🚀 {pkg}-{version}: 全局IFT模式，将尝试加载IFT模型")
    else:
        # 既没有包级别控制，也没有全局启用，使用普通LoRA
        adopt_ift_for_this_pkg = False
        ift_control_mode = "global_disabled"
        logging.info(f"📝 {pkg}-{version}: IFT未启用，使用普通LoRA模型")
    
    # 如果决定为当前包启用IFT，则尝试加载IFT模型
    if adopt_ift_for_this_pkg:
        logging.info(f"🔍 尝试通过IFT模型管理器加载IFT模型: {pkg}-{version} (控制模式: {ift_control_mode})")
        
        try:
            # 使用IFT模型管理器查找最佳匹配的IFT模型
            from utils.iftModelManager import get_default_manager as get_ift_manager
            ift_manager = get_ift_manager()
            
            # 从pred_args获取IFT选择偏好
            preferred_data_strategy = getattr(pred_args, 'ift_data_strategy', None)
            preferred_ift_type = getattr(pred_args, 'ift_type', None)
            
            # 查找最佳匹配的IFT模型
            best_match = ift_manager.get_best_match(
                pkg=pkg,
                version=version,
                base_model=config["model_name"],
                knowledge_type=knowledge_type,
                data_strategy=preferred_data_strategy,
                ift_type=preferred_ift_type
            )
            
            if best_match:
                ift_path = best_match["ift_model_path"]
                model_id = best_match["model_id"]
                actual_strategy = best_match["data_strategy"]
                actual_ift_type = best_match["ift_type"]
                actual_knowledge_type = best_match.get("knowledge_type")
                
                # 🔧 新增：严格验证knowledge_type约束
                if knowledge_type and actual_knowledge_type != knowledge_type:
                    logging.warning(f"❌ IFT模型knowledge_type不匹配: 期望'{knowledge_type}', 实际'{actual_knowledge_type}', 跳过此模型")
                    best_match = None
                elif knowledge_type and knowledge_type not in ift_path:
                    logging.warning(f"❌ IFT模型路径不包含knowledge_type '{knowledge_type}': {ift_path}, 跳过此模型")
                    best_match = None
                else:
                    logging.info(f"✅ 找到匹配的IFT模型 ({pkg}-{version}):")
                    logging.info(f"  模型ID: {model_id}")
                    logging.info(f"  知识类型: {actual_knowledge_type}")
                    logging.info(f"  数据策略: {actual_strategy}")
                    logging.info(f"  IFT类型: {actual_ift_type or 'default'}")
                    logging.info(f"  模型路径: {ift_path}")
                    
                    # 验证IFT模型文件完整性
                    if os.path.exists(ift_path):
                        ift_model_path = os.path.join(ift_path, "adapter_model.safetensors")
                        if os.path.exists(ift_model_path):
                            try:
                                _ = load_file(ift_model_path)
                                logging.info(f"✅ 成功验证并加载IFT模型: {ift_path}")
                                
                                # 如果偏好与实际不匹配，给出提示
                                if preferred_data_strategy and preferred_data_strategy != actual_strategy:
                                    logging.info(f"注意: 期望数据策略 '{preferred_data_strategy}'，实际使用 '{actual_strategy}'")
                                if preferred_ift_type and preferred_ift_type != actual_ift_type:
                                    logging.info(f"注意: 期望IFT类型 '{preferred_ift_type}'，实际使用 '{actual_ift_type or 'default'}'")
                                    
                                return ift_path
                            except Exception as e:
                                logging.warning(f"IFT模型文件损坏 {ift_model_path}: {e}")
                                best_match = None
                        else:
                            logging.warning(f"IFT模型文件不存在: {ift_model_path}")
                            best_match = None
                    else:
                        logging.warning(f"IFT模型目录不存在: {ift_path}")
                        best_match = None
            
            # 如果IFT模型管理器查找失败或验证失败，尝试传统方式
            if not best_match:
                logging.info(f"IFT模型管理器未找到匹配的IFT模型 {pkg}-{version}")
                
                # 如果指定了偏好但没找到，给出提示
                if preferred_data_strategy or preferred_ift_type:
                    logging.info(f"未找到满足偏好的IFT模型:")
                    if preferred_data_strategy:
                        logging.info(f"  期望数据策略: {preferred_data_strategy}")
                    if preferred_ift_type:
                        logging.info(f"  期望IFT类型: {preferred_ift_type}")
                
                # 回退到传统的IFT检查方式
                logging.info(f"回退到传统IFT检查方式...")
                ift_exists, ift_path = pathConfig.checkIFTModelExists(config, pkg, version, model_name, knowledge_type, pred_args)
                
                if ift_exists:
                    # 🔧 新增：传统方式也要验证knowledge_type约束
                    if knowledge_type and knowledge_type not in ift_path:
                        logging.warning(f"❌ 传统IFT模型路径不包含knowledge_type '{knowledge_type}': {ift_path}, 跳过此模型")
                        ift_exists = False
                    else:
                        # 验证IFT模型文件完整性
                        ift_model_path = os.path.join(ift_path, "adapter_model.safetensors")
                        if os.path.exists(ift_model_path):
                            try:
                                _ = load_file(ift_model_path)
                                logging.info(f"✅ 通过传统方式找到并验证IFT模型: {ift_path}")
                                return ift_path
                            except Exception as e:
                                logging.warning(f"IFT模型文件损坏 {ift_model_path}: {e}")
                        else:
                            logging.warning(f"IFT模型文件不存在: {ift_model_path}")
                
        except Exception as e:
            logging.error(f"使用IFT模型管理器时发生错误 {pkg}-{version}: {e}")
            logging.info(f"回退到普通LoRA模型")
    
    # 加载普通LoRA模型（默认行为或IFT加载失败时的回退）
    if ift_control_mode == "package_selective" and not adopt_ift_for_this_pkg:
        logging.info(f"📦 {pkg}-{version}: 按包级别控制直接加载普通LoRA模型")
    else:
        logging.info(f"📝 {pkg}-{version}: 加载普通LoRA模型 (原因: {ift_control_mode})")
    
    path = pathConfig.getPath(config, pkg, version, model_name, knowledge_type, pred_args)
    
    # 🔧 新增：验证普通LoRA模型路径也必须包含knowledge_type
    if knowledge_type and knowledge_type not in path:
        raise ValueError(f"LoRA模型路径不包含knowledge_type '{knowledge_type}': {path}")
    
    if os.path.exists(path):
        # 加载safetensors格式的模型
        model_path = os.path.join(path, "adapter_model.safetensors")
        if os.path.exists(model_path):
            try:
                _ = load_file(model_path)  # 验证文件完整性
                logging.info(f"✅ 成功找到并验证LoRA模型: {path}")
                return path
            except Exception as e:
                raise ValueError(f"LoRA模型文件损坏: {model_path}, 错误: {e}")
        else:
            raise ValueError(f"LoRA模型文件不存在: {model_path}")
    else:
        raise ValueError(f"LoRA模型目录不存在: {path}")
def get_dataloader(config,corpus_path,pkg,version,tokenizer):
    files_info = []
    with open(f"{corpus_path}/{pkg}/{version}.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line_data = json.loads(line)
            files_info.append(str(line_data))    
    files_info = files_info[:int(len(files_info)*config["traindata_percentage"])]
    if len(files_info) == 0:
        return None
    # print(f"{pkg} {version} {len(files_info)}")
    dataset = TextDataset(files_info, tokenizer, block_size=128,pkg=pkg,version=version)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,collate_fn=collate_fn)
    return dataloader

def getEquipAdaptorModel(config):
    '''
    Description:
        由foundation model获取对应的peft模型
    Args:
        config: dict, 配置信息
            - use_balanced_device_map: bool, 是否使用均衡设备映射
            - force_balance: bool, 是否强制均衡分配
            - exclude_cpu: bool, 是否排除CPU设备
            - check_r_consistency: bool, 是否检查r值一致性
    Returns:
        lora_model: 训练好的LoRA模型
    '''
    # 记录传入的config配置
    logging.info("=" * 60)
    logging.info("getEquipAdaptorModel - 接收到的配置信息:")
    logging.info("=" * 60)
    
    # 定义train_lora.py中的默认参数用于比较
    train_lora_defaults = {
        "use_balanced_device_map": True,
        "force_balance": True,
        "exclude_cpu": True,
        "check_r_consistency": True,
        "strict_r_check": False,
        "precision": "bf16",
        "device_map": "auto"
    }
    
    # 记录所有config参数
    logging.info("传入的config参数:")
    for key, value in config.items():
        if key in train_lora_defaults:
            default_value = train_lora_defaults[key]
            if value != default_value:
                logging.warning(f"  {key}: {value} [与train_lora默认值 {default_value} 不同]")
            else:
                logging.info(f"  {key}: {value} [与train_lora默认值一致]")
        else:
            logging.info(f"  {key}: {value}")
    
    # 检查是否缺少重要参数
    missing_params = []
    for key, default_value in train_lora_defaults.items():
        if key not in config:
            missing_params.append(key)
            logging.warning(f"  {key}: 未设置 [train_lora默认值: {default_value}]")
    
    if missing_params:
        logging.warning(f"缺少以下参数，将使用函数内部默认值: {missing_params}")
    
    # 记录LoRA相关的核心参数
    logging.info("\nLoRA核心参数:")
    lora_core_params = ["model_name", "r", "alpha", "target_modules", "target_layers"]
    for param in lora_core_params:
        if param in config:
            logging.info(f"  {param}: {config[param]}")
        else:
            logging.error(f"  {param}: 未设置 [必需参数]")
    
    # 记录设备映射相关参数
    logging.info("\n设备映射相关参数:")
    device_params = ["use_balanced_device_map", "force_balance", "exclude_cpu", "device_map"]
    for param in device_params:
        value = config.get(param, train_lora_defaults.get(param, "未设置"))
        logging.info(f"  {param}: {value}")
    
    # 记录r值检查相关参数
    logging.info("\nr值检查相关参数:")
    r_check_params = ["check_r_consistency", "strict_r_check"]
    for param in r_check_params:
        value = config.get(param, train_lora_defaults.get(param, "未设置"))
        logging.info(f"  {param}: {value}")
    
    logging.info("=" * 60)
    
    # 清理现有的GPU缓存
    torch.cuda.empty_cache()
    
    # 检查设备映射策略
    use_balanced_device_map = config.get("use_balanced_device_map", False)
    use_dynamic_device_map = config.get("use_dynamic_device_map", False)
    
    logging.info(f"均衡设备映射启用状态: {use_balanced_device_map}")
    logging.info(f"动态设备映射启用状态: {use_dynamic_device_map}")
    
    # 获取精度设置
    precision = config.get("precision", "fp16")
    logging.info(f"使用精度: {precision}")
    print(f"使用精度: {precision}")
    
    # 检查GPU可用性
    num_gpus = torch.cuda.device_count()
    logging.info(f"检测到 {num_gpus} 个GPU设备")
    print(f"检测到 {num_gpus} 个GPU设备")
    
    # 根据配置选择设备映射策略
    if use_dynamic_device_map:
        # 动态策略：先auto，检查均衡性，不均衡则重新平衡
        balance_threshold = config.get("balance_threshold", 0.3)
        force_balance = config.get("force_balance", False)
        exclude_cpu = config.get("exclude_cpu", True)
        
        logging.info(f"使用动态设备映射策略")
        logging.info(f"动态映射参数: balance_threshold={balance_threshold}, force_balance={force_balance}, exclude_cpu={exclude_cpu}")
        
        base_model, tokenizer, device_map_info = create_dynamic_device_map(
            config["model_name"],
            balance_threshold=balance_threshold,
            force_balance=force_balance,
            exclude_cpu=exclude_cpu
        )
        
        if base_model is None:
            logging.error("动态设备映射失败")
            raise RuntimeError("动态设备映射失败")
        
        logging.info(f"动态设备映射完成: 策略={device_map_info['strategy']}, 原因={device_map_info['reason']}")
        print(f"动态设备映射完成: 策略={device_map_info['strategy']}")
        
        # 记录均衡性信息
        if 'balance_info' in device_map_info and device_map_info['balance_info']:
            balance_info = device_map_info['balance_info']
            logging.info(f"设备分布: {balance_info['device_distribution']}")
            logging.info(f"不均衡系数: {balance_info['imbalance_ratio']:.3f}")
        
    elif use_balanced_device_map:
        # 均衡策略：直接使用均衡设备映射
        force_balance = config.get("force_balance", False)
        exclude_cpu = config.get("exclude_cpu", True)
        
        logging.info(f"使用均衡设备映射策略")
        logging.info(f"均衡映射参数: force_balance={force_balance}, exclude_cpu={exclude_cpu}")
        print("启用均衡设备映射...")
        
        device_map = create_balanced_device_map(
            config["model_name"],
            force_balance=force_balance,
            exclude_cpu=exclude_cpu
        )
        if device_map is None:
            logging.warning("均衡设备映射失败，回退到auto模式")
            print("⚠️  均衡设备映射失败，回退到auto模式")
            device_map = "auto"
        else:
            logging.info(f"均衡设备映射创建成功: {type(device_map)}")
        
        print(f"使用设备映射: {device_map}")
        
        # 加载基础模型和tokenizer
        logging.info(f"开始加载基础模型: {config['model_name']}")
        print(f"加载基础模型: {config['model_name']}")
        try:
            base_model, tokenizer = load_base_model(config["model_name"], device_map, precision)
            logging.info("基础模型加载成功")
            print("基础模型加载成功")
        except Exception as e:
            logging.error(f"加载基础模型失败: {e}")
            print(f"加载基础模型失败: {e}")
            raise e
    
    else:
        # 标准策略：使用auto或用户指定的设备映射
        device_map = config.get("device_map", "auto")
        logging.info(f"使用标准设备映射: {device_map}")
        print(f"使用设备映射: {device_map}")
        
        # 加载基础模型和tokenizer
        logging.info(f"开始加载基础模型: {config['model_name']}")
        print(f"加载基础模型: {config['model_name']}")
        try:
            base_model, tokenizer = load_base_model(config["model_name"], device_map, precision)
            logging.info("基础模型加载成功")
            print("基础模型加载成功")
        except Exception as e:
            logging.error(f"加载基础模型失败: {e}")
            print(f"加载基础模型失败: {e}")
            raise e
    
    # 创建LoRA配置
    logging.info("创建LoRA配置...")
    logging.info(f"LoRA配置参数: target_modules={config['target_modules']}, target_layers={config['target_layers']}, r={config['r']}, alpha={config['alpha']}")
    print("创建LoRA配置...")
    lora_config = create_lora_config(
        config["target_modules"], 
        config["target_layers"], 
        config["r"], 
        config["alpha"]
    )
    
    # 创建LoRA模型
    logging.info("创建LoRA模型...")
    print("创建LoRA模型...")
    lora_model = get_peft_model(base_model, lora_config)
    
    # 简化的模型信息输出
    if hasattr(lora_model, 'hf_device_map'):
        device_map_info = str(lora_model.hf_device_map)
        logging.info(f"LoRA模型设备映射: {device_map_info}")
        print(f"LoRA模型设备映射: {lora_model.hf_device_map}")
    else:
        try:
            first_param_device = next(lora_model.parameters()).device
            logging.info(f"LoRA模型设备: {first_param_device}")
            print(f"LoRA模型设备: {first_param_device}")
        except:
            logging.warning("无法确定模型设备")
            print("无法确定模型设备")
    
    # 打印GPU内存使用情况
    if torch.cuda.is_available():
        logging.info("当前GPU内存使用情况:")
        print("\n当前GPU内存使用情况:")
        for i in range(torch.cuda.device_count()):
            try:
                usage = torch.cuda.memory_allocated(i)
                total = torch.cuda.get_device_properties(i).total_memory
                usage_percent = (usage / total) * 100
                gpu_info = f"cuda:{i}: {usage / (1024**2):.2f} MB / {total / (1024**3):.2f} GB ({usage_percent:.1f}%)"
                logging.info(f"- {gpu_info}")
                print(f"- {gpu_info}")
            except:
                error_info = f"cuda:{i}: 无法获取内存信息"
                logging.warning(f"- {error_info}")
                print(f"- {error_info}")
    
    # 检查r值一致性（如果配置中启用了检查）
    check_r_consistency = config.get("check_r_consistency", False)
    logging.info(f"r值一致性检查启用状态: {check_r_consistency}")
    
    if check_r_consistency:
        strict_r_check = config.get("strict_r_check", False)
        logging.info(f"严格r值检查模式: {strict_r_check}")
        logging.info("开始检查LoRA模型r值一致性...")
        print("\n检查LoRA模型r值一致性...")
        
        consistency_result = check_lora_r_consistency(lora_model, config)
        
        if not consistency_result['is_consistent']:
            logging.warning("LoRA模型r值与配置不一致")
            logging.warning(f"不匹配的层: {consistency_result['mismatched_layers']}")
            logging.warning(f"期望r值: {consistency_result['expected_r']}")
            logging.warning(f"实际r值: {consistency_result['actual_r_values']}")
            
            print("⚠️  警告：LoRA模型r值与配置不一致")
            
            if strict_r_check:
                logging.error(f"严格模式下r值不一致，中止执行: {consistency_result['mismatched_layers']}")
                raise ValueError(f"LoRA模型r值不一致: {consistency_result['mismatched_layers']}")
        else:
            logging.info("✅ LoRA模型r值一致性检查通过")
    
    logging.info("LoRA模型创建完成")
    print("LoRA模型创建完成")
    return lora_model
def train_lora_model_withPEFT(lora_model, dataloader, config, output_adaptor_path):
    '''
        对于peft模型进行训练
    Args:
        lora_model: 获取到的LoRA模型
        dataloader: DataLoader, 训练数据加载器
        config: 训练配置字典，包含以下可选参数:
            - target_batch_size (int): 目标批次大小，用于梯度累积计算，默认16
            - precision (str): 训练精度，可选["float32", "float16", "bfloat16"]，默认"float32"
            - save_path_base (str): 保存路径基础目录，默认"/datanfs2/chenrongyi/models/versiBCB"
            - num_epochs
            - learning_rate
        output_adaptor_path: 输出适配器路径

    Returns:
        lora_model: 训练好的LoRA模型
    '''
    # 检查CUDA可用性和设备情况
    print("检查模型设备分布...")
    if hasattr(lora_model, 'hf_device_map'):
        print(f"模型当前device_map: {lora_model.hf_device_map}")

    # 分析模型的设备分布
    device_distribution = {}
    for name, param in lora_model.named_parameters():
        device = param.device
        if device not in device_distribution:
            device_distribution[device] = 0
        device_distribution[device] += 1
    
    print("训练前模型参数设备分布:")
    for device, count in device_distribution.items():
        print(f"- {device}: {count} parameters")
    
    # 检查模型是否分布在多个设备上
    is_multi_device = len(device_distribution) > 1
    
    if is_multi_device:
        print("检测到多设备分布模型，训练将使用模型自身的设备管理")
        print("如果遇到设备不一致错误，相关batch将被自动跳过")
    else:
        model_device = list(device_distribution.keys())[0]
        print(f"单设备模型，设备: {model_device}")
    
    # 训练模型
    try:
        # 使用config中的参数训练LoRA模型
        num_epochs = config.get("num_epochs", 5)
        learning_rate = config.get("learning_rate", 1e-3)
        # 将str转为float
        learning_rate = float(learning_rate)
        print(f"开始训练: epochs={num_epochs}, lr={learning_rate}, target_batch_size={config.get('target_batch_size', 16)}")
        
        lora_model = train_lora_model(
            lora_model, 
            dataloader, 
            num_epochs, 
            learning_rate,
            config,
            output_adaptor_path
        )
        
        # 将模型移到CPU并分离计算图
        print("训练完成，将模型移动到CPU...")
        lora_model = lora_model.cpu()
        for param in lora_model.parameters():
            param.requires_grad = False
            
        # 清理GPU缓存
        torch.cuda.empty_cache()
        return lora_model
        
    except Exception as e:
        print(f"训练过程中出错: {e}")
        # 确保清理GPU缓存
        torch.cuda.empty_cache()
        raise e

def buildandTrainLoraModel(config, dataloader, precision='fp16',pkg=None,version=None,knowledge_type=None):
    '''
    Description:
        训练lora模型，并确保适当的内存管理
    Args:
        config: dict, 配置信息
            - use_balanced_device_map: bool, 是否使用均衡设备映射
            - force_balance: bool, 是否强制均衡分配
            - exclude_cpu: bool, 是否排除CPU设备
            - check_r_consistency: bool, 是否检查r值一致性
        dataloader: DataLoader, 训练数据加载器
        precision: str, 模型精度 ('fp16', 'fp32', 'bf16')
        pkg: str, 包名
        version: str, 版本号  
        knowledge_type: str, 知识类型
    Returns:
        lora_model: 训练好的LoRA模型
    '''
    # 记录训练开始和配置信息
    logging.info("=" * 60)
    logging.info(f"buildandTrainLoraModel - 开始训练: {pkg}-{version}")
    logging.info("=" * 60)
    
    # 定义train_lora.py中的默认参数用于比较
    train_lora_defaults = {
        "use_balanced_device_map": True,
        "force_balance": True,
        "exclude_cpu": True,
        "check_r_consistency": True,
        "strict_r_check": False,
        "precision": "bf16",
        "device_map": "auto"
    }
    
    # 记录传入的precision参数
    logging.info(f"传入的precision参数: {precision}")
    config_precision = config.get("precision", "fp16")
    if precision != config_precision:
        logging.warning(f"precision参数不一致: 函数参数={precision}, config中={config_precision}")
    
    # 记录包和版本信息
    logging.info(f"训练目标: pkg={pkg}, version={version}, knowledge_type={knowledge_type}")
    
    model_name = config["model_name"].split("/")[-1]
    if pkg and version:
        pathConfig = pathConfigurator()
        output_adaptor_path = pathConfig.getPath(config, pkg, version,model_name,knowledge_type=knowledge_type)
        logging.info(f"输出路径: {output_adaptor_path}")
    else:
        logging.error("pkg和version必须提供")
        raise ValueError("pkg and version must be provided")
    
    try:
        # 清理现有的GPU缓存
        torch.cuda.empty_cache()
        
        # 检查是否启用均衡设备映射
        use_balanced_device_map = config.get("use_balanced_device_map", False)
        logging.info(f"均衡设备映射启用状态: {use_balanced_device_map}")
        
        if use_balanced_device_map:
            force_balance = config.get("force_balance", False)
            exclude_cpu = config.get("exclude_cpu", True)
            logging.info(f"均衡设备映射参数: force_balance={force_balance}, exclude_cpu={exclude_cpu}")
            print("启用均衡设备映射...")
            
            device_map = create_balanced_device_map(
                config["model_name"],
                force_balance=force_balance,
                exclude_cpu=exclude_cpu
            )
            if device_map is None:
                logging.warning("均衡设备映射失败，回退到auto模式")
                print("⚠️  均衡设备映射失败，回退到auto模式")
                device_map = "auto"
            else:
                logging.info(f"均衡设备映射创建成功: {type(device_map)}")
        else:
            # 使用配置中的设备映射，支持多GPU分布
            device_map = config.get("device_map", "auto")
            logging.info(f"使用标准设备映射: {device_map}")
        
        print(f"使用设备映射: {device_map}")
        
        # 加载基础模型和tokenizer，使用指定的精度和设备映射
        logging.info(f"开始加载基础模型: {config['model_name']}")
        logging.info(f"使用精度: {precision}")


        # base_model, tokenizer = load_base_model(config["model_name"], device_map, precision)
        base_model = Model.from_pretrained(config["model_name"],torch_dtype=str2dtype(precision),device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        logging.info("创建LoRA配置...")
        logging.info(f"LoRA配置参数: target_modules={config['target_modules']}, target_layers={config['target_layers']}, r={config['r']}, alpha={config['alpha']}")
        lora_config = create_lora_config(
            config["target_modules"], 
            config["target_layers"], 
            config["r"], 
            config["alpha"]
        )
        
        # 创建LoRA模型
        logging.info("创建LoRA模型...")
        lora_model = get_peft_model(base_model, lora_config)
        profiler.record("创建LoRA模型")
        print("memory_usage")
        # 分析模型参数分布
        devices_found = {}
        for name, param in lora_model.named_parameters():
            device = param.device
            if device not in devices_found:
                devices_found[device] = []
            devices_found[device].append(name)
        
        logging.info("训练模型参数分布:")
        print("训练模型参数分布:")
        for device, params in devices_found.items():
            device_info = f"{device}: {len(params)} parameters"
            logging.info(f"- {device_info}")
            print(f"- {device_info}")
        
        # 确定主设备
        if len(devices_found) > 1:
            main_device = max(devices_found.keys(), key=lambda d: len(devices_found[d]))
            logging.info(f"训练将使用主设备: {main_device}")
            print(f"训练将使用主设备: {main_device}")
        else:
            main_device = list(devices_found.keys())[0]
            logging.info(f"训练将在设备: {main_device}")
            print(f"训练将在设备: {main_device}")
        
        # 训练前检查r值一致性
        check_r_consistency = config.get("check_r_consistency", False)
        logging.info(f"训练前r值一致性检查启用状态: {check_r_consistency}")
        
        if check_r_consistency:
            strict_r_check = config.get("strict_r_check", False)
            logging.info(f"严格r值检查模式: {strict_r_check}")
            logging.info("开始训练前LoRA模型r值一致性检查...")
            print("\n训练前检查LoRA模型r值一致性...")
            
            consistency_result = check_lora_r_consistency(lora_model, config)
            
            if not consistency_result['is_consistent']:
                logging.warning("训练前LoRA模型r值与配置不一致")
                logging.warning(f"不匹配的层: {consistency_result['mismatched_layers']}")
                logging.warning(f"期望r值: {consistency_result['expected_r']}")
                logging.warning(f"实际r值: {consistency_result['actual_r_values']}")
                
                print("⚠️  警告：LoRA模型r值与配置不一致")
                
                if strict_r_check:
                    logging.error(f"严格模式下训练前r值不一致，中止执行: {consistency_result['mismatched_layers']}")
                    raise ValueError(f"LoRA模型r值不一致: {consistency_result['mismatched_layers']}")
            else:
                logging.info("✅ 训练前LoRA模型r值一致性检查通过")
        
        # 训练模型
        logging.info("开始训练模型（支持多GPU分布）")
        logging.info(f"训练参数: epochs={config['num_epochs']}, lr={config['learning_rate']}")
        print(f"开始训练模型（支持多GPU分布）")
        profiler.record("开始训练模型")
        
        lora_model = train_lora_model(
            lora_model, 
            dataloader, 
            config["num_epochs"], 
            config["learning_rate"],
            config,
            output_adaptor_path
        )
        
        # 训练后再次检查r值一致性
        if check_r_consistency:
            logging.info("开始训练后LoRA模型r值一致性检查...")
            print("\n训练后检查LoRA模型r值一致性...")
            
            consistency_result = check_lora_r_consistency(lora_model, config)
            
            if not consistency_result['is_consistent']:
                logging.warning("训练后LoRA模型r值与配置不一致")
                logging.warning(f"不匹配的层: {consistency_result['mismatched_layers']}")
                logging.warning(f"期望r值: {consistency_result['expected_r']}")
                logging.warning(f"实际r值: {consistency_result['actual_r_values']}")
                
                print("⚠️  警告：训练后LoRA模型r值与配置不一致")
                
                # 保存检查结果到日志
                import json
                consistency_log = os.path.join(output_adaptor_path, "r_consistency_check.json")
                with open(consistency_log, 'w') as f:
                    json.dump(consistency_result, f, indent=2)
                    
                logging.info(f"r值一致性检查结果已保存到: {consistency_log}")
                print(f"r值一致性检查结果已保存到: {consistency_log}")
            else:
                logging.info("✅ 训练后LoRA模型r值一致性检查通过")
        
        # 将模型移到CPU并分离计算图
        logging.info("训练完成，将模型移动到CPU...")
        lora_model = lora_model.cpu()
        for param in lora_model.parameters():
            param.requires_grad = False
            
        # 清理不需要的对象
        logging.info("清理训练资源...")
        del base_model
        del tokenizer
        torch.cuda.empty_cache()
        
        logging.info(f"训练完成: {pkg}-{version}")
        logging.info("=" * 60)
        
        return lora_model
        
    except Exception as e:
        logging.error(f"训练过程中出错: {str(e)}")
        logging.error("错误详情:")
        logging.error(traceback.format_exc())
        
        print(f"创建LoRA模型时出错: {str(e)}")
        traceback.print_exc()
        
        # 确保清理所有可能的资源
        if 'base_model' in locals():
            del base_model
        if 'tokenizer' in locals():
            del tokenizer
        if 'lora_model' in locals():
            del lora_model
        torch.cuda.empty_cache()
        raise e
    
def merge_lora_weights(base_model, lora_models_paths, weights=None):
    """合并多个 LoRA 模型的权重"""
    if weights is None:
        weights = [1.0/len(lora_models_paths)] * len(lora_models_paths)
    if len(lora_models_paths) != len(weights):
        raise ValueError("模型路径和权重数量必须相同")
    
    # 加载第一个模型作为基础
    merged_model = load_lora_model_withPeft(base_model, lora_models_paths[0])
    
    # 获取第一个模型的权重并应用权重
    state_dict = merged_model.state_dict()
    for key in state_dict:
        if "lora" in key:
            state_dict[key] = state_dict[key] * weights[0]
    
    # 加载其他模型并合并权重
    for i in range(1, len(lora_models_paths)):
        print(f"加载第{i}个模型{lora_models_paths[i]}")
        model_path = lora_models_paths[i]
        weight = weights[i]
        
        # 临时加载模型
        temp_model = load_lora_model_withPeft(base_model, model_path)
        temp_state_dict = temp_model.state_dict()
        
        # 合并权重
        for key in temp_state_dict:
            if "lora" in key and key in state_dict:
                state_dict[key] += temp_state_dict[key] * weight
    
    # 将合并后的权重加载到模型中
    merged_model.load_state_dict(state_dict)
    return merged_model

# 推理


def inference(
        model, 
        tokenizer, 
        prompt,
        max_new_tokens=30,
        temperature=0.2,
        top_p=0.95,
        truncate=False,
        truncate_length=2048,
        inference_type="local",
        api_key=None,
        model_name=None,
        api_base_url=None,
        stop_tokens=None,
):
    """使用模型进行推理，支持本地、HuggingFace API和TogetherAI API
    Args:
        model: 本地模型（仅在inference_type="local"时使用）
        tokenizer: 分词器（仅在inference_type="local"时使用）
        prompt: 输入提示文本
        max_new_tokens: 最大生成token数
        temperature: 温度参数
        top_p: top_p参数
        truncate: 是否截断输入
        truncate_length: 截断长度
        inference_type: 推理类型 ("local", "huggingface", "togetherai")
        api_key: API密钥（用于远程API）
        model_name: 模型名称（用于远程API）
        api_base_url: API基础URL（用于HuggingFace API）
        stop_tokens: 停止词列表，支持token序列匹配和字符串匹配（仅用于本地推理）
    Returns:
        str: 生成的文本（不包含原始提示）
    """

    if inference_type == "local":
        return _local_inference(model, tokenizer, prompt, max_new_tokens, temperature, top_p, truncate, truncate_length, stop_tokens)
    else:
        from utils.callapi import _qdd_api_inference,_huggingface_api_inference,_togetherai_api_inference
        if inference_type == "huggingface":
            return _huggingface_api_inference(prompt, max_new_tokens, temperature, top_p, api_key, model_name, api_base_url)
        elif inference_type == "togetherai":
            return _togetherai_api_inference(prompt, max_new_tokens, temperature, top_p, api_key, model_name)
        elif inference_type == "qdd":
            return _qdd_api_inference(prompt, max_new_tokens, temperature, top_p, api_key, model_name, api_base_url)
        else:
            raise ValueError(f"Unsupported inference_type: {inference_type}")
from transformers import StoppingCriteria, StoppingCriteriaList
class MultiTokenStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_tokens, tokenizer, original_input_length=None, check_last_chars=15, min_generated_length=30):
        """
        停止条件类，支持token序列匹配和字符串匹配
        
        Args:
            stop_tokens: list of strings, 停止词列表
            tokenizer: 分词器
            original_input_length: int, 原始输入的长度（用于字符串匹配）
            check_last_chars: int, 检查最后几个字符（用于字符串匹配）
            min_generated_length: int, 最小生成长度（避免过早停止）
        """
        self.stop_tokens = stop_tokens
        self.tokenizer = tokenizer
        self.original_input_length = original_input_length
        self.check_last_chars = check_last_chars
        self.min_generated_length = min_generated_length
        
        # 提前编码所有停词（支持多token停词如 `<end>`）
        self.encoded_stop_tokens = [
            self.tokenizer.encode(stop, add_special_tokens=False) 
            for stop in self.stop_tokens
        ]

    def __call__(self, input_ids, scores, **kwargs):
        """
        检查是否应该停止生成
        同时支持token序列匹配和字符串匹配策略
        """
        current_tokens = input_ids[0].tolist()
        
        # 如果设置了原始输入长度，检查是否达到最小生成长度
        if self.original_input_length is not None:
            generated_length = len(current_tokens) - self.original_input_length
            if generated_length < self.min_generated_length:
                return False
        
        # 策略1：Token序列匹配（原有逻辑）
        for stop_seq in self.encoded_stop_tokens:
            if len(stop_seq) > 0 and len(current_tokens) >= len(stop_seq):
                if current_tokens[-len(stop_seq):] == stop_seq:
                    return True
        
        # 策略2：字符串匹配（兼容greedy_search策略）
        if self.original_input_length is not None and len(current_tokens) > self.original_input_length:
            # 获取生成的部分
            generated_tokens = current_tokens[self.original_input_length:]
            if len(generated_tokens) > 20:
                '''
                    仅当生成的token长度大于20时，才进行字符串匹配，防止开头的```python直接被截断
                '''
                try:
                    # Decode生成的部分
                    generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    
                    # 检查最后N个字符是否包含停词
                    text_to_check = generated_text[-self.check_last_chars:] if len(generated_text) > self.check_last_chars else generated_text
                    
                    for stopword in self.stop_tokens:
                        if isinstance(stopword, str) and stopword in text_to_check:
                            return True
                            
                except Exception as e:
                    # 如果解码失败，跳过字符串检查
                    pass
        
        return False

# 定义需要停止的标记（支持多个）
def _local_inference(model, tokenizer, prompt, max_new_tokens, temperature, top_p, truncate, truncate_length, stop_tokens=None):
    """本地推理实现
    
    Args:
        model: 模型
        tokenizer: 分词器
        prompt: 输入提示
        max_new_tokens: 最大新token数
        temperature: 温度参数
        top_p: top_p参数
        truncate: 是否截断
        truncate_length: 截断长度
        stop_tokens: 停止词列表，默认为["<end>","```","###"]
    """
    from utils.loraTrain.buildandloadData import encode_with_left_truncation    
    
    # 确保输入不会太长
    if truncate:
        max_input_length = truncate_length # 设置合理的最大输入长度
        inputs = encode_with_left_truncation(tokenizer,prompt,max_input_length,only_tokenize=False).to(model.device)
    else:
        inputs = tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=False,
        ).to(model.device)

    # 获取原始输入长度
    original_input_length = len(inputs["input_ids"][0])
    
    # 设置默认停止词
    if stop_tokens is None:
        # ###是lora中常出现的问题
        stop_tokens = ["<end>","```","###"]  # 默认停止词
    # 创建停止条件实例，支持token序列匹配和字符串匹配
    stopping_criteria = StoppingCriteriaList([
        MultiTokenStoppingCriteria(
            stop_tokens=stop_tokens, 
            tokenizer=tokenizer,
            original_input_length=original_input_length,
            check_last_chars=20,  # 检查最后20个字符
            min_generated_length=30  # 最小生成长度
        )
    ])

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,     # 调整温度
                top_p=top_p,         # 调整 top_p
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                
                num_beams=1,        # 使用简单采样而不是束搜索
                early_stopping=True,
                stopping_criteria=stopping_criteria,
                # min_length=10,      # 设置最小长度
                # max_length=2048,    # 设置最大长度
            )
        
        # 只返回新生成的内容（去除输入提示）
        prompt_length = len(inputs["input_ids"][0])
        generated_sequence = outputs[0][prompt_length:]
        
        return tokenizer.decode(generated_sequence, skip_special_tokens=True)
    
    except RuntimeError as e:
        traceback.print_exc()
        raise 


def check_lora_params(model):
    """
    检查模型是否包含LoRA参数
    Args:
        model: 需要检查的模型
    Returns:
        has_lora: bool, 是否包含LoRA参数
        lora_params: list, 所有LoRA参数的名称和形状
    """
    has_lora = False
    lora_params = []
    
    # 遍历所有参数
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            has_lora = True
            lora_params.append({
                'name': name,
                'shape': tuple(param.shape),
                'requires_grad': param.requires_grad
            })
    
    if has_lora:
        print("\nFound LoRA parameters:")
        for param in lora_params:
            print(f"Parameter: {param['name']}")
            print(f"Shape: {param['shape']}")
            print(f"Requires gradient: {param['requires_grad']}")
            print("-" * 40)
    else:
        print("\nNo LoRA parameters found in the model.")
    
    return has_lora, lora_params

def check_lora_r_consistency(model, config):
    """
    检查加载到设备的模型的LoRA参数的r值是否与config["r"]一致
    
    Args:
        model: 已加载的LoRA模型
        config: 配置字典，包含预期的r值
    
    Returns:
        dict: 包含检查结果的字典
            - is_consistent: bool, 是否一致
            - expected_r: int, 期望的r值
            - actual_r_values: dict, 实际的r值映射
            - mismatched_layers: list, 不匹配的层
    """
    expected_r = config.get("r", 8)  # 默认r值为8
    actual_r_values = {}
    mismatched_layers = []
    
    print(f"检查LoRA模型的r值一致性...")
    print(f"期望的r值: {expected_r}")
    
    # 检查是否有LoRA配置
    if hasattr(model, 'peft_config') and model.peft_config:
        print(f"从PEFT配置中获取r值:")
        for adapter_name, peft_config in model.peft_config.items():
            if hasattr(peft_config, 'r'):
                actual_r = peft_config.r
                actual_r_values[adapter_name] = actual_r
                print(f"  适配器 '{adapter_name}': r = {actual_r}")
                
                if actual_r != expected_r:
                    mismatched_layers.append(adapter_name)
                    print(f"  ❌ 不匹配: 期望 {expected_r}, 实际 {actual_r}")
                else:
                    print(f"  ✅ 匹配")
    
    # 通过参数名称和形状推断r值
    print(f"\n通过参数形状推断r值:")
    lora_a_params = {}
    lora_b_params = {}
    
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            if 'lora_a' in name.lower():
                # lora_A的形状通常是 (r, input_dim)
                layer_name = name.replace('lora_A', '').replace('lora_a', '').replace('.weight', '')
                lora_a_params[layer_name] = param.shape[0]  # r值
            elif 'lora_b' in name.lower():
                # lora_B的形状通常是 (output_dim, r)
                layer_name = name.replace('lora_B', '').replace('lora_b', '').replace('.weight', '')
                lora_b_params[layer_name] = param.shape[1]  # r值
    
    # 验证lora_A和lora_B的r值一致性
    for layer_name in lora_a_params:
        if layer_name in lora_b_params:
            r_a = lora_a_params[layer_name]
            r_b = lora_b_params[layer_name]
            
            if r_a == r_b:
                actual_r_values[layer_name] = r_a
                print(f"  层 '{layer_name}': r = {r_a}")
                
                if r_a != expected_r:
                    mismatched_layers.append(layer_name)
                    print(f"    ❌ 不匹配: 期望 {expected_r}, 实际 {r_a}")
                else:
                    print(f"    ✅ 匹配")
            else:
                print(f"  ❌ 层 '{layer_name}' 的lora_A和lora_B的r值不一致: A={r_a}, B={r_b}")
                mismatched_layers.append(layer_name)
    
    # 判断总体一致性
    is_consistent = len(mismatched_layers) == 0 and len(actual_r_values) > 0
    
    # 结果总结
    print(f"\n=== LoRA r值一致性检查结果 ===")
    print(f"期望r值: {expected_r}")
    print(f"实际检测到的r值: {set(actual_r_values.values()) if actual_r_values else '无'}")
    print(f"总体一致性: {'✅ 一致' if is_consistent else '❌ 不一致'}")
    
    if mismatched_layers:
        print(f"不匹配的层: {mismatched_layers}")
    
    if not actual_r_values:
        print("⚠️  警告: 未检测到任何LoRA参数")
    
    return {
        'is_consistent': is_consistent,
        'expected_r': expected_r,
        'actual_r_values': actual_r_values,
        'mismatched_layers': mismatched_layers
    }

def create_balanced_device_map(model_name_or_path, force_balance=False, exclude_cpu=True):
    """
    创建均衡的设备映射，将模型层平均分配到所有可用的GPU上
    
    Args:
        model_name_or_path: 模型名称或路径
        force_balance: bool, 是否强制均衡分配（即使某些GPU内存不足）
        exclude_cpu: bool, 是否排除CPU设备
    
    Returns:
        dict: 均衡的设备映射字典
    """
    print("创建均衡的设备映射...")
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，无法创建GPU设备映射")
        return "cpu" if not exclude_cpu else None
    
    # 获取GPU数量和信息
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个GPU设备")
    
    if num_gpus == 0:
        print("❌ 没有可用的GPU设备")
        return "cpu" if not exclude_cpu else None
    
    # 获取每个GPU的内存信息
    gpu_memory_info = []
    for i in range(num_gpus):
        try:
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory
            allocated_memory = torch.cuda.memory_allocated(i)
            reserved_memory = torch.cuda.memory_reserved(i)
            free_memory = total_memory - max(allocated_memory, reserved_memory)
            
            gpu_info = {
                'device_id': i,
                'name': props.name,
                'total_memory_gb': total_memory / (1024**3),
                'free_memory_gb': free_memory / (1024**3),
                'utilization_percent': (allocated_memory / total_memory) * 100
            }
            gpu_memory_info.append(gpu_info)
            
            print(f"GPU {i} ({props.name}): "
                  f"{gpu_info['free_memory_gb']:.1f}GB free / "
                  f"{gpu_info['total_memory_gb']:.1f}GB total "
                  f"({gpu_info['utilization_percent']:.1f}% used)")
            
        except Exception as e:
            print(f"❌ 获取GPU {i} 信息时出错: {e}")
            continue
    
    if not gpu_memory_info:
        print("❌ 无法获取任何GPU信息")
        return "cpu" if not exclude_cpu else None
    
    # 过滤出可用的GPU（内存足够）
    min_memory_gb = 2.0  # 最小内存要求
    available_gpus = [gpu for gpu in gpu_memory_info if gpu['free_memory_gb'] >= min_memory_gb]
    
    if not available_gpus:
        print(f"❌ 没有GPU有足够的内存（需要至少{min_memory_gb}GB）")
        if not force_balance:
            return "cpu" if not exclude_cpu else None
        else:
            print("⚠️  强制均衡模式：将使用所有GPU")
            available_gpus = gpu_memory_info
    
    print(f"可用的GPU: {[gpu['device_id'] for gpu in available_gpus]}")
    
    # 获取模型层信息
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        # 尝试获取层数
        num_layers = None
        if hasattr(config, 'num_hidden_layers'):
            num_layers = config.num_hidden_layers
        elif hasattr(config, 'num_layers'):
            num_layers = config.num_layers
        elif hasattr(config, 'n_layer'):
            num_layers = config.n_layer
        elif hasattr(config, 'n_layers'):
            num_layers = config.n_layers
        
        if num_layers is None:
            print("⚠️  无法自动检测模型层数，使用默认分配策略")
            # 使用简单的自动设备映射
            device_map = "auto"
        else:
            print(f"检测到模型层数: {num_layers}")
            device_map = create_layer_balanced_mapping(num_layers, available_gpus, config)
            
    except Exception as e:
        print(f"⚠️  获取模型配置时出错: {e}")
        print("使用简单的自动设备映射")
        device_map = "auto"
    
    return device_map

def create_layer_balanced_mapping(num_layers, available_gpus, model_config):
    """
    创建层级均衡的设备映射，确保每个GPU分配到尽可能相等的层数
    
    Args:
        num_layers: 模型层数
        available_gpus: 可用的GPU列表
        model_config: 模型配置
    
    Returns:
        dict: 层级设备映射
    """
    num_gpus = len(available_gpus)
    device_map = {}
    
    # 计算每个GPU应该分配的层数 - 使用简单的均匀分配
    base_layers_per_gpu = num_layers // num_gpus
    extra_layers = num_layers % num_gpus
    
    print(f"层数均衡分配策略:")
    print(f"  总层数: {num_layers}")
    print(f"  可用GPU: {num_gpus}")
    print(f"  基础层数/GPU: {base_layers_per_gpu}")
    print(f"  额外层数: {extra_layers}")
    
    # 显示GPU内存信息（仅供参考）
    print(f"\nGPU内存信息:")
    for gpu in available_gpus:
        print(f"  GPU {gpu['device_id']}: {gpu['free_memory_gb']:.1f}GB 可用")
    
    # 均匀分配层数，额外的层分配给前几个GPU
    layer_assignments = []
    for i in range(num_gpus):
        # 前 extra_layers 个GPU多分配一层
        if i < extra_layers:
            assigned_layers = base_layers_per_gpu + 1
        else:
            assigned_layers = base_layers_per_gpu
        
        layer_assignments.append(assigned_layers)
        gpu_id = available_gpus[i]['device_id']
        print(f"  GPU {gpu_id}: {assigned_layers} 层")
    
    # 验证分配是否正确
    total_assigned = sum(layer_assignments)
    assert total_assigned == num_layers, f"层数分配错误: {total_assigned} != {num_layers}"
    print(f"✅ 验证通过: 总分配层数 = {total_assigned}")
    
    # 创建设备映射 - 只映射确实存在的组件
    current_layer = 0
    
    # 将嵌入层分配给第一个GPU，输出层分配给最后一个GPU
    first_gpu = available_gpus[0]['device_id']
    last_gpu = available_gpus[-1]['device_id']
    
    # 只分配模型实际使用的层结构
    # 基于模型配置和架构推断层名称
    # model_arch = getattr(model_config, 'architectures', [''])
    # arch_name = model_arch[0] if model_arch else ''
    arch_name = 'Llama'
    print(f"\n检测到模型架构: {arch_name}")
    
    # 根据架构确定层命名模式
    layer_patterns = []
    embedding_patterns = []
    output_patterns = []
    norm_patterns = []
    rotary_patterns = []
    
    if 'Llama' in arch_name or 'llama' in arch_name.lower():
        layer_patterns = [f"model.layers.{i}" for i in range(num_layers)]
        embedding_patterns = ["model.embed_tokens"]
        output_patterns = ["lm_head"]
        norm_patterns = ["model.norm"]
        rotary_patterns = ["model.rotary_emb"]
    else:
        raise ValueError(f"不支持的模型架构: {arch_name}")
    
    print(f"使用层命名模式: {layer_patterns[:3]}...{layer_patterns[-3:] if len(layer_patterns) > 6 else layer_patterns[3:]}")
    
    # 分配嵌入层到第一个GPU
    for pattern in embedding_patterns:
        device_map[pattern] = first_gpu
    
    # 分配输出层到最后一个GPU
    for pattern in output_patterns:
        device_map[pattern] = last_gpu
    
    # 分配标准化层到最后一个GPU
    for pattern in norm_patterns:
        device_map[pattern] = last_gpu
    for pattern in rotary_patterns:
        device_map[pattern] = first_gpu
    
    print(f"\n层分配详情:")
    
    # 分配transformer层
    for gpu_idx, num_assigned_layers in enumerate(layer_assignments):
        device_id = available_gpus[gpu_idx]['device_id']
        
        start_layer = current_layer
        end_layer = current_layer + num_assigned_layers - 1
        
        if num_assigned_layers > 0:
            print(f"  GPU {device_id}: 层 {start_layer}-{end_layer} ({num_assigned_layers} 层)")
            
            for layer_idx in range(current_layer, current_layer + num_assigned_layers):
                if layer_idx < len(layer_patterns):
                    device_map[layer_patterns[layer_idx]] = device_id
            
            current_layer += num_assigned_layers
        else:
            print(f"  GPU {device_id}: 无层分配")
    
    # 统计每个设备的组件数量
    device_counts = {}
    for component, device in device_map.items():
        if device not in device_counts:
            device_counts[device] = 0
        device_counts[device] += 1
    
    print(f"\n设备映射统计:")
    for device_id in sorted(device_counts.keys()):
        print(f"  GPU {device_id}: {device_counts[device_id]} 个组件")
    
    print(f"\n创建的设备映射样本:")
    # 按层序号排序显示样本
    layer_keys = []
    other_keys = []
    
    for key in device_map.keys():
        if any(layer_pattern in key for layer_pattern in ["layers.", "h.", "block."]):
            layer_keys.append(key)
        else:
            other_keys.append(key)
    
    # 显示一些层映射
    if layer_keys:
        layer_keys.sort()
        sample_layer_keys = layer_keys[:3] + layer_keys[-3:] if len(layer_keys) > 6 else layer_keys
        
        print("  层映射:")
        for key in sample_layer_keys[:3]:
            print(f"    {key}: cuda:{device_map[key]}")
        
        if len(layer_keys) > 6:
            print(f"    ... 中间 {len(layer_keys) - 6} 层 ...")
            for key in sample_layer_keys[-3:]:
                print(f"    {key}: cuda:{device_map[key]}")
    
    if other_keys:
        print("  其他组件:")
        for key in other_keys:
            print(f"    {key}: cuda:{device_map[key]}")
    
    print(f"\n总映射数量: {len(device_map)} 个组件")
    
    return device_map

def apply_balanced_device_map(model_name_or_path, device_map_config=None):
    """
    应用均衡的设备映射来加载模型
    
    Args:
        model_name_or_path: 模型路径
        device_map_config: 设备映射配置字典
            - force_balance: bool, 是否强制均衡
            - exclude_cpu: bool, 是否排除CPU
            - min_memory_gb: float, 最小内存要求
    
    Returns:
        tuple: (model, tokenizer, device_map)
    """
    config = device_map_config or {}
    
    # 创建均衡的设备映射
    device_map = create_balanced_device_map(
        model_name_or_path,
        force_balance=config.get('force_balance', False),
        exclude_cpu=config.get('exclude_cpu', True)
    )
    
    if device_map is None:
        raise RuntimeError("无法创建有效的设备映射")
    
    print(f"使用设备映射加载模型: {model_name_or_path}")
    
    # 使用均衡的设备映射加载模型
    model, tokenizer = load_base_model(
        model_name_or_path,
        device_map=device_map,
        precision_from_arg=config.get('precision', 'fp16')
    )
    
    return model, tokenizer, device_map

def create_dynamic_device_map(model_name_or_path, balance_threshold=0.3, force_balance=False, exclude_cpu=True):
    """
    创建动态设备映射：先尝试auto加载，检查是否均衡，如果不均衡则重新平衡
    
    Args:
        model_name_or_path: 模型名称或路径
        balance_threshold: 不均衡阈值 (0.0-1.0)，超过此值认为需要重新平衡
        force_balance: 是否强制执行平衡
        exclude_cpu: 是否排除CPU设备
    
    Returns:
        tuple: (model, tokenizer, device_map_info)
    """
    print("🔄 启动动态设备映射策略...")
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，使用CPU")
        model, tokenizer = load_base_model(model_name_or_path, device_map="cpu")
        return model, tokenizer, {"strategy": "cpu", "reason": "cuda_unavailable"}
    
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        print(f"🔧 只有 {num_gpus} 个GPU，使用auto策略")
        model, tokenizer = load_base_model(model_name_or_path, device_map="auto")
        return model, tokenizer, {"strategy": "auto", "reason": "single_gpu"}
    
    # 第一步：尝试auto加载
    print("📥 第1步：尝试auto策略加载模型...")
    try:
        model, tokenizer = load_base_model(model_name_or_path, device_map="auto")
        print("✅ auto策略加载成功")
        
        # 检查模型是否有设备映射
        if hasattr(model, 'hf_device_map'):
            device_map = model.hf_device_map
            print(f"📊 检测到设备映射: {len(device_map)} 个组件")
            
            # 分析分配均衡性
            balance_info = analyze_device_balance(device_map)
            print(f"⚖️  分配均衡性分析:")
            print(f"  不均衡系数: {balance_info['imbalance_ratio']:.3f}")
            print(f"  设备分布: {balance_info['device_distribution']}")
            
            # 判断是否需要重新平衡
            needs_rebalance = balance_info['imbalance_ratio'] > balance_threshold
            
            if not needs_rebalance and not force_balance:
                print("✅ 分配已经足够均衡，使用auto策略结果")
                return model, tokenizer, {
                    "strategy": "auto", 
                    "reason": "already_balanced",
                    "balance_info": balance_info
                }
            else:
                print(f"⚠️  分配不均衡 (阈值: {balance_threshold:.2f})，需要重新平衡")
                
                # 释放当前模型
                del model
                torch.cuda.empty_cache()
                
        else:
            print("⚠️  模型没有设备映射信息，尝试重新平衡")
            del model
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"❌ auto策略加载失败: {e}")
        print("🔄 尝试使用balanced策略...")
    
    # 第二步：使用balanced策略重新加载
    print("📥 第2步：使用balanced策略重新加载...")
    try:
        balanced_device_map = create_balanced_device_map(
            model_name_or_path, 
            force_balance=force_balance, 
            exclude_cpu=exclude_cpu
        )
        
        if balanced_device_map is None:
            print("❌ 无法创建balanced设备映射")
            return None, None, {"strategy": "failed", "reason": "no_balanced_map"}
        
        model, tokenizer = load_base_model(model_name_or_path, device_map=balanced_device_map)
        print("✅ balanced策略加载成功")
        
        # 分析新的分配
        balance_info = None
        if hasattr(model, 'hf_device_map'):
            balance_info = analyze_device_balance(model.hf_device_map)
            print(f"📊 重新平衡后的分配:")
            print(f"  不均衡系数: {balance_info['imbalance_ratio']:.3f}")
            print(f"  设备分布: {balance_info['device_distribution']}")
        
        return model, tokenizer, {
            "strategy": "balanced", 
            "reason": "rebalanced",
            "balance_info": balance_info
        }
        
    except Exception as e:
        print(f"❌ balanced策略也失败: {e}")
        
        # 第三步：回退到auto策略
        print("📥 第3步：回退到auto策略...")
        try:
            model, tokenizer = load_base_model(model_name_or_path, device_map="auto")
            print("✅ 回退到auto策略成功")
            
            balance_info = None
            if hasattr(model, 'hf_device_map'):
                balance_info = analyze_device_balance(model.hf_device_map)
            
            return model, tokenizer, {
                "strategy": "auto_fallback", 
                "reason": "balanced_failed",
                "balance_info": balance_info
            }
            
        except Exception as fallback_error:
            print(f"❌ 所有策略都失败: {fallback_error}")
            return None, None, {"strategy": "failed", "reason": "all_failed"}

def analyze_device_balance(device_map):
    """
    分析设备映射的均衡性
    
    Args:
        device_map: 设备映射字典
    
    Returns:
        dict: 均衡性分析结果
    """
    # 统计每个设备的组件数量
    device_counts = {}
    for component, device in device_map.items():
        device_str = str(device)
        if device_str not in device_counts:
            device_counts[device_str] = 0
        device_counts[device_str] += 1
    
    # 计算总组件数
    total_components = sum(device_counts.values())
    
    # 计算理想的每个设备组件数
    num_devices = len(device_counts)
    ideal_per_device = total_components / num_devices
    
    # 计算不均衡系数
    max_deviation = 0
    for device, count in device_counts.items():
        deviation = abs(count - ideal_per_device) / ideal_per_device
        max_deviation = max(max_deviation, deviation)
    
    # 分析层分布 (只考虑transformer层)
    layer_distribution = {}
    for component, device in device_map.items():
        if any(pattern in component for pattern in ['layers.', 'h.', 'block.']):
            device_str = str(device)
            if device_str not in layer_distribution:
                layer_distribution[device_str] = 0
            layer_distribution[device_str] += 1
    
    return {
        "imbalance_ratio": max_deviation,
        "device_distribution": device_counts,
        "layer_distribution": layer_distribution,
        "total_components": total_components,
        "ideal_per_device": ideal_per_device
    }

def get_precision_info():
    """
    获取支持的精度信息和配置示例
    """
    info = """
    支持的模型精度选项:
    
    1. 'fp16' (默认): 半精度浮点，节省内存，推理速度快
    2. 'fp32': 全精度浮点，精度最高，占用内存最多
    3. 'bf16': Brain float 16，在某些GPU上性能更好
    
    配置示例：
    在您的配置文件中添加：
    {
        "precision": "fp16",  // 或 "fp32", "bf16"
        "model_name": "your_model_path",
        ...
    }
    
    或在函数调用中传递：
    load_base_model(model_path, device_map="auto", precision="fp16")
    """
    print(info)
    return info

# 使用示例和文档
"""
增强的停止条件使用示例：

1. 基础使用（默认停止词）：
   result = inference(model, tokenizer, prompt, max_new_tokens=100)

2. 自定义停止词（token序列匹配）：
   stop_tokens = ["<end>", "</s>", "<|endoftext|>"]
   result = inference(model, tokenizer, prompt, max_new_tokens=100, stop_tokens=stop_tokens)

3. 自定义停止词（字符串匹配）：
   stop_tokens = ["END", "STOP", "##"]
   result = inference(model, tokenizer, prompt, max_new_tokens=100, stop_tokens=stop_tokens)

4. 混合停止词（同时支持token序列和字符串匹配）：
   stop_tokens = ["<end>", "END", "STOP", "</s>"]
   result = inference(model, tokenizer, prompt, max_new_tokens=100, stop_tokens=stop_tokens)

停止条件策略说明：
- 策略1：Token序列匹配 - 检查生成的token序列是否以停止词的token序列结尾
- 策略2：字符串匹配 - 将生成的token解码为文本，检查最后N个字符是否包含停止词
- 两种策略并行工作，任一策略匹配都会触发停止
- 设置最小生成长度（默认30个token）以避免过早停止
- 兼容 greedy_search 中的停止策略

参数说明：
- stop_tokens: 停止词列表，支持多种格式的停止词
- check_last_chars: 字符串匹配时检查的最后字符数（默认15）
- min_generated_length: 最小生成长度，避免过早停止（默认30）

=====================================================================

新增功能使用示例：

1. 检查LoRA模型的r值一致性：
   ```python
   # 加载LoRA模型
   lora_model = getEquipAdaptorModel(config)
   
   # 检查r值一致性
   result = check_lora_r_consistency(lora_model, config)
   
   if result['is_consistent']:
       print("✅ LoRA r值一致")
   else:
       print(f"❌ LoRA r值不一致: {result['mismatched_layers']}")
       print(f"期望: {result['expected_r']}")
       print(f"实际: {result['actual_r_values']}")
   ```

2. 创建均衡的设备映射：
   ```python
   # 创建均衡的设备映射
   device_map = create_balanced_device_map(
       model_name_or_path="your_model_path",
       force_balance=False,  // 是否强制均衡
       exclude_cpu=True      // 是否排除CPU
   )
   
   # 使用均衡的设备映射加载模型
   model, tokenizer, actual_device_map = apply_balanced_device_map(
       model_name_or_path="your_model_path",
       device_map_config={
           'force_balance': False,
           'exclude_cpu': True,
           'precision': 'fp16'
       }
   )
   ```

3. 在现有训练流程中集成：
   ```python
   # 在配置中添加均衡设备映射选项
   config = {
       "model_name": "your_model_path",
       "r": 16,
       "alpha": 32,
       "target_modules": ["q_proj", "v_proj"],
       "target_layers": [0, 1, 2, 3],
       "use_balanced_device_map": True,  // 启用均衡设备映射
       "force_balance": False,
       "exclude_cpu": True
   }
   
   # 训练前检查
   if config.get("use_balanced_device_map", False):
       print("使用均衡设备映射...")
       device_map = create_balanced_device_map(
           config["model_name"],
           force_balance=config.get("force_balance", False),
           exclude_cpu=config.get("exclude_cpu", True)
       )
       config["device_map"] = device_map
   
   # 训练LoRA模型
   lora_model = buildandTrainLoraModel(config, dataloader, precision='fp16')
   
   # 训练后检查r值一致性
   consistency_result = check_lora_r_consistency(lora_model, config)
   if not consistency_result['is_consistent']:
       print("⚠️  警告：训练后的模型r值与配置不一致")
   ```

4. 设备映射配置选项：
   - force_balance: 强制均衡分配，即使某些GPU内存不足
   - exclude_cpu: 排除CPU设备，只使用GPU
   - min_memory_gb: 最小内存要求（默认2GB）
   - precision: 模型精度（fp16/fp32/bf16）

5. 高级用法 - 自定义内存权重：
   ```python
   # 获取GPU信息并自定义分配策略
   device_map = create_balanced_device_map(
       model_name_or_path="your_model_path",
       force_balance=True,    // 强制使用所有GPU
       exclude_cpu=True       // 排除CPU
   )
   
   # 手动调整设备映射（如果需要）
   if isinstance(device_map, dict):
       # 可以手动调整特定层的设备分配
       device_map["model.layers.0"] = 0  // 将第0层强制分配给GPU 0
   ```

功能特点：
- 🔍 自动检查LoRA参数的r值一致性
- ⚖️ 根据GPU内存容量智能分配模型层
- 🎯 支持多种模型架构的层命名模式
- 🛡️ 包含错误处理和回退机制
- 📊 详细的设备使用情况报告
- 🔧 灵活的配置选项

注意事项：
- 均衡设备映射主要针对多GPU环境优化
- 在单GPU环境中会自动回退到标准的device_map="auto"
- 强制均衡模式可能会导致某些GPU过载，请谨慎使用
- r值检查支持PEFT配置和参数形状两种检查方式
"""