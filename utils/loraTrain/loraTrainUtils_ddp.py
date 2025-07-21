"""
分布式训练工具函数模块
支持DistributedDataParallel (DDP) 的LoRA训练
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import logging
import os
import traceback
from datetime import datetime
from tqdm import tqdm
import numpy as np

from peft import get_peft_model, LoraConfig
from utils.loraTrain.loraTrainUtils import create_lora_config, load_base_model, create_balanced_device_map


def setup_distributed_training():
    """设置分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # 设置CUDA设备
        torch.cuda.set_device(local_rank)
        
        # 初始化分布式进程组
        dist.init_process_group(backend='nccl', init_method='env://')
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def create_ddp_model(model, local_rank, find_unused_parameters=False):
    """创建DDP模型"""
    model = model.to(f'cuda:{local_rank}')
    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused_parameters
    )
    return ddp_model


def create_distributed_dataloader(dataset, batch_size, rank, world_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=None):
    """创建分布式数据加载器"""
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            num_workers=num_workers
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            num_workers=num_workers
        )
    
    return dataloader


def get_device_map_for_ddp(config, world_size, local_rank):
    """获取分布式训练的设备映射"""
    if world_size > 1:
        # 分布式训练时，每个进程使用一个GPU
        return local_rank
    else:
        # 单GPU训练，使用原有的设备映射策略
        use_balanced_device_map = config.get("use_balanced_device_map", False)
        
        if use_balanced_device_map:
            force_balance = config.get("force_balance", False)
            exclude_cpu = config.get("exclude_cpu", True)
            
            device_map = create_balanced_device_map(
                config["model_name"],
                force_balance=force_balance,
                exclude_cpu=exclude_cpu
            )
            
            if device_map is None:
                device_map = "auto"
        else:
            device_map = config.get("device_map", "auto")
        
        return device_map


def train_lora_model_ddp(
    model, 
    dataloader, 
    optimizer, 
    num_epochs=10, 
    rank=0, 
    world_size=1, 
    local_rank=0,
    accumulation_steps=1,
    max_grad_norm=1.0,
    precision="fp16",
    log_file=None,
    checkpoint_dir=None,
    save_every_n_epochs=2
):
    """
    分布式训练LoRA模型的核心函数
    """
    # 获取精度设置
    torch_dtype = torch.float32
    if precision == "float16":
        torch_dtype = torch.float16
    elif precision == "bfloat16":
        torch_dtype = torch.bfloat16
    
    # 初始化损失记录
    epoch_losses = []
    step_losses = []
    batch_losses = []
    
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        batch_count = 0
        valid_batch_count = 0
        step_count = 0
        
        # 设置分布式采样器的epoch
        if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 进度条（仅在rank 0上显示）
        if rank == 0:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        else:
            pbar = dataloader
        
        for batch in pbar:
            # 提取batch数据
            inputs = batch["input_ids"].to(f'cuda:{local_rank}', non_blocking=True)
            labels = batch["labels"].to(f'cuda:{local_rank}', non_blocking=True)
            attention_mask = batch["attention_mask"].to(f'cuda:{local_rank}', non_blocking=True)
            
            try:
                # 前向传播
                with torch.amp.autocast(device_type='cuda', dtype=torch_dtype, enabled=torch.cuda.is_available()):
                    outputs = model(
                        input_ids=inputs,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / accumulation_steps
                
                # 反向传播
                loss.backward()
                
                # 记录损失
                current_loss = loss.item() * accumulation_steps
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += current_loss
                    valid_batch_count += 1
                    
                    if rank == 0:
                        batch_losses.append(current_loss)
                else:
                    if rank == 0:
                        logging.warning(f"NaN/Inf loss detected in Epoch {epoch+1}, Batch {batch_count+1}")
                
                batch_count += 1
                step_count += 1
                
                # 梯度更新
                if step_count % accumulation_steps == 0 or batch_count == len(dataloader):
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    
                    # 参数更新
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # 计算平均损失
                    if valid_batch_count > 0:
                        avg_loss = total_loss / valid_batch_count
                        
                        if rank == 0:
                            step_losses.append(avg_loss)
                            
                            # 记录到日志文件
                            if log_file:
                                with open(log_file, "a") as f:
                                    f.write(f"{epoch+1},{step_count // accumulation_steps},{batch_count},{avg_loss:.6f}\n")
                            
                            # 更新进度条
                            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if rank == 0:
                        logging.error(f"GPU内存不足: {e}")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                else:
                    if rank == 0:
                        logging.error(f"训练过程中出现RuntimeError: {e}")
                    raise e
        
        # 同步所有进程
        if world_size > 1:
            dist.barrier()
        
        # 计算epoch平均损失
        if rank == 0:
            avg_epoch_loss = total_loss / max(1, valid_batch_count)
            epoch_losses.append(avg_epoch_loss)
            logging.info(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_epoch_loss:.4f}")
            
            # 保存检查点
            if checkpoint_dir and ((epoch + 1) % save_every_n_epochs == 0 or epoch == num_epochs - 1):
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{timestamp}_epoch{epoch+1}")
                os.makedirs(checkpoint_path, exist_ok=True)
                
                # 从DDP中提取原始模型
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(checkpoint_path)
                
                logging.info(f"Checkpoint saved to {checkpoint_path}")
    
    return model, {
        'epoch_losses': epoch_losses,
        'step_losses': step_losses,
        'batch_losses': batch_losses
    }


def save_training_plots(losses_dict, output_path, rank=0):
    """保存训练损失图表"""
    if rank != 0:
        return
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        epoch_losses = losses_dict['epoch_losses']
        step_losses = losses_dict['step_losses']
        batch_losses = losses_dict['batch_losses']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Epoch损失
        axes[0, 0].plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
        axes[0, 0].set_title('Epoch Average Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Step损失
        axes[0, 1].plot(step_losses, marker='.', linestyle='-', alpha=0.7)
        axes[0, 1].set_title('Training Loss per Step')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # Batch损失
        axes[1, 0].plot(batch_losses, marker='.', linestyle='-', alpha=0.5, color='red')
        axes[1, 0].set_title('Training Loss per Batch')
        axes[1, 0].set_xlabel('Batch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
        
        # 平滑Batch损失
        if len(batch_losses) > 10:
            window_size = min(20, len(batch_losses) // 10)
            smooth_losses = np.convolve(batch_losses, np.ones(window_size)/window_size, mode='valid')
            axes[1, 1].plot(smooth_losses, linestyle='-', color='green')
            axes[1, 1].set_title(f'Smoothed Batch Loss (Window={window_size})')
            axes[1, 1].set_xlabel('Batch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Training plots saved to {output_path}")
        
    except ImportError:
        logging.warning("matplotlib not available, skipping plot generation")
    except Exception as e:
        logging.error(f"Error generating training plots: {e}")


def cleanup_distributed():
    """清理分布式训练资源"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_model_device_info(model, rank=0):
    """获取模型设备信息"""
    if rank != 0:
        return
    
    devices_found = {}
    model_for_analysis = model.module if hasattr(model, 'module') else model
    
    for name, param in model_for_analysis.named_parameters():
        device = param.device
        if device not in devices_found:
            devices_found[device] = []
        devices_found[device].append(name)
    
    logging.info("模型参数设备分布:")
    for device, params in devices_found.items():
        logging.info(f"  {device}: {len(params)} parameters")
    
    return devices_found


def synchronize_loss_across_ranks(loss_tensor, world_size):
    """同步所有rank的损失值"""
    if world_size > 1:
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor /= world_size
    return loss_tensor


def save_distributed_checkpoint(model, optimizer, epoch, loss, checkpoint_path, rank=0):
    """保存分布式训练检查点"""
    if rank == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Checkpoint saved to {checkpoint_path}")


def load_distributed_checkpoint(model, optimizer, checkpoint_path, local_rank):
    """加载分布式训练检查点"""
    map_location = f'cuda:{local_rank}'
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # 加载模型状态
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    logging.info(f"Checkpoint loaded from {checkpoint_path}, epoch={epoch}, loss={loss}")
    return epoch, loss


def estimate_memory_usage(model, batch_size, sequence_length, rank=0):
    """估计内存使用量"""
    if rank != 0:
        return
    
    try:
        # 估算模型参数内存
        total_params = sum(p.numel() for p in model.parameters())
        param_memory = total_params * 4  # 假设float32，每个参数4字节
        
        # 估算梯度内存
        grad_memory = param_memory  # 梯度与参数同样大小
        
        # 估算激活内存 (粗略估计)
        activation_memory = batch_size * sequence_length * 4 * 1000  # 粗略估计
        
        total_memory = param_memory + grad_memory + activation_memory
        
        logging.info(f"Memory usage estimation:")
        logging.info(f"  Parameters: {param_memory / 1e9:.2f} GB")
        logging.info(f"  Gradients: {grad_memory / 1e9:.2f} GB")
        logging.info(f"  Activations: {activation_memory / 1e9:.2f} GB")
        logging.info(f"  Total: {total_memory / 1e9:.2f} GB")
        
    except Exception as e:
        logging.error(f"Error estimating memory usage: {e}")


def print_distributed_info(rank, world_size, local_rank):
    """打印分布式训练信息"""
    if rank == 0:
        logging.info(f"Distributed training setup:")
        logging.info(f"  World size: {world_size}")
        logging.info(f"  Rank: {rank}")
        logging.info(f"  Local rank: {local_rank}")
        logging.info(f"  Backend: {dist.get_backend() if dist.is_initialized() else 'Not initialized'}")
        logging.info(f"  Master addr: {os.environ.get('MASTER_ADDR', 'Not set')}")
        logging.info(f"  Master port: {os.environ.get('MASTER_PORT', 'Not set')}") 