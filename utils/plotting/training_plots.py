"""
训练损失曲线绘图工具模块

提供训练过程中损失曲线的绘制功能，包括：
- Epoch平均损失曲线
- Step损失曲线  
- Batch损失曲线
- 平滑的Batch损失曲线
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import traceback
from typing import List, Optional, Tuple


def setup_matplotlib_backend(backend: str = 'Agg') -> bool:
    """
    设置matplotlib后端
    
    Args:
        backend (str): 后端类型，默认为'Agg'（非交互式）
        
    Returns:
        bool: 是否成功设置
    """
    try:
        matplotlib.use(backend)
        return True
    except Exception as e:
        print(f"警告：无法设置matplotlib后端 '{backend}': {e}")
        return False


def plot_training_losses(
    epoch_losses: List[float], 
    step_losses: List[float], 
    batch_losses: List[float], 
    save_path: str, 
    title_suffix: str = "",
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 100
) -> bool:
    """
    绘制训练损失曲线并保存到文件
    
    Args:
        epoch_losses (List[float]): 每个epoch的平均损失
        step_losses (List[float]): 每个step的损失
        batch_losses (List[float]): 每个batch的损失
        save_path (str): 图片保存路径
        title_suffix (str): 标题后缀，用于区分不同的绘图
        figsize (Tuple[int, int]): 图片尺寸
        dpi (int): 图片分辨率
        
    Returns:
        bool: 是否成功绘制和保存
    """
    try:
        # 确保matplotlib后端已设置
        if not setup_matplotlib_backend():
            return False
            
        plt.figure(figsize=figsize, dpi=dpi)
        
        # 绘制Epoch损失曲线
        plt.subplot(2, 2, 1)
        if epoch_losses:
            plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', linewidth=2, markersize=6)
        plt.title(f'Epoch Average Loss{title_suffix}', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 绘制Step损失曲线
        plt.subplot(2, 2, 2)
        if step_losses:
            plt.plot(step_losses, marker='.', linestyle='-', alpha=0.7, linewidth=1)
        plt.title(f'Training Loss per Step{title_suffix}', fontsize=12, fontweight='bold')
        plt.xlabel('Step', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 绘制Batch损失曲线
        plt.subplot(2, 2, 3)
        if batch_losses:
            plt.plot(batch_losses, marker='.', linestyle='-', alpha=0.5, color='red', linewidth=1)
        plt.title(f'Training Loss per Batch{title_suffix}', fontsize=12, fontweight='bold')
        plt.xlabel('Batch', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 平滑的Batch损失曲线
        plt.subplot(2, 2, 4)
        if len(batch_losses) > 10:
            window_size = min(10, len(batch_losses) // 5)
            if window_size > 0:
                smooth_losses = np.convolve(batch_losses, np.ones(window_size)/window_size, mode='valid')
                plt.plot(smooth_losses, marker='', linestyle='-', color='green', linewidth=2)
                plt.title(f'Smoothed Batch Loss (Window={window_size}){title_suffix}', fontsize=12, fontweight='bold')
            else:
                plt.title(f'Smoothed Batch Loss{title_suffix}', fontsize=12, fontweight='bold')
        else:
            plt.title(f'Smoothed Batch Loss{title_suffix}', fontsize=12, fontweight='bold')
        plt.xlabel('Batch', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存图片
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"损失曲线已保存到 {save_path}")
        return True
        
    except Exception as e:
        print(f"绘制损失曲线时出错: {e}")
        traceback.print_exc()
        return False


def save_training_plots(
    epoch_losses: List[float], 
    step_losses: List[float], 
    batch_losses: List[float], 
    log_plot: str, 
    is_final: bool = False,
    verbose: bool = True
) -> bool:
    """
    保存训练损失曲线图
    
    Args:
        epoch_losses (List[float]): 每个epoch的平均损失
        step_losses (List[float]): 每个step的损失
        batch_losses (List[float]): 每个batch的损失
        log_plot (str): 图片保存路径
        is_final (bool): 是否为最终绘图（训练结束时）
        verbose (bool): 是否输出详细信息
        
    Returns:
        bool: 是否成功保存
    """
    title_suffix = " (Final)" if is_final else ""
    success = plot_training_losses(epoch_losses, step_losses, batch_losses, log_plot, title_suffix)
    
    if verbose:
        if is_final:
            if success:
                print(f"训练完成。最终损失曲线已保存到 {log_plot}")
            else:
                print("训练完成。由于绘图错误，损失曲线未保存。")
        else:
            if not success:
                print("中间损失曲线绘制失败。")
    
    return success


def plot_loss_comparison(
    loss_data: dict,
    save_path: str,
    title: str = "Training Loss Comparison",
    figsize: Tuple[int, int] = (15, 10)
) -> bool:
    """
    绘制多个训练过程的损失对比图
    
    Args:
        loss_data (dict): 损失数据字典，格式为 {name: {'epoch': [...], 'step': [...], 'batch': [...]}}
        save_path (str): 图片保存路径
        title (str): 图片标题
        figsize (Tuple[int, int]): 图片尺寸
        
    Returns:
        bool: 是否成功绘制和保存
    """
    try:
        if not setup_matplotlib_backend():
            return False
            
        plt.figure(figsize=figsize)
        
        # 绘制Epoch损失对比
        plt.subplot(2, 2, 1)
        for name, data in loss_data.items():
            if 'epoch' in data and data['epoch']:
                plt.plot(range(1, len(data['epoch']) + 1), data['epoch'], 
                        marker='o', label=name, linewidth=2, markersize=6)
        plt.title('Epoch Average Loss Comparison', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 绘制Step损失对比
        plt.subplot(2, 2, 2)
        for name, data in loss_data.items():
            if 'step' in data and data['step']:
                plt.plot(data['step'], marker='.', linestyle='-', alpha=0.7, 
                        label=name, linewidth=1)
        plt.title('Training Loss per Step Comparison', fontsize=12, fontweight='bold')
        plt.xlabel('Step', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 绘制Batch损失对比
        plt.subplot(2, 2, 3)
        for name, data in loss_data.items():
            if 'batch' in data and data['batch']:
                plt.plot(data['batch'], marker='.', linestyle='-', alpha=0.5, 
                        label=name, linewidth=1)
        plt.title('Training Loss per Batch Comparison', fontsize=12, fontweight='bold')
        plt.xlabel('Batch', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 绘制平滑Batch损失对比
        plt.subplot(2, 2, 4)
        for name, data in loss_data.items():
            if 'batch' in data and len(data['batch']) > 10:
                window_size = min(10, len(data['batch']) // 5)
                if window_size > 0:
                    smooth_losses = np.convolve(data['batch'], np.ones(window_size)/window_size, mode='valid')
                    plt.plot(smooth_losses, marker='', linestyle='-', 
                            label=f"{name} (smooth)", linewidth=2)
        plt.title('Smoothed Batch Loss Comparison', fontsize=12, fontweight='bold')
        plt.xlabel('Batch', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存图片
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"损失对比图已保存到 {save_path}")
        return True
        
    except Exception as e:
        print(f"绘制损失对比图时出错: {e}")
        traceback.print_exc()
        return False


def create_loss_summary(
    epoch_losses: List[float], 
    step_losses: List[float], 
    batch_losses: List[float]
) -> dict:
    """
    创建损失统计摘要
    
    Args:
        epoch_losses (List[float]): 每个epoch的平均损失
        step_losses (List[float]): 每个step的损失
        batch_losses (List[float]): 每个batch的损失
        
    Returns:
        dict: 损失统计摘要
    """
    summary = {
        'epoch_count': len(epoch_losses),
        'step_count': len(step_losses),
        'batch_count': len(batch_losses)
    }
    
    if epoch_losses:
        summary['epoch_loss'] = {
            'min': min(epoch_losses),
            'max': max(epoch_losses),
            'mean': np.mean(epoch_losses),
            'std': np.std(epoch_losses),
            'final': epoch_losses[-1] if epoch_losses else None
        }
    
    if step_losses:
        summary['step_loss'] = {
            'min': min(step_losses),
            'max': max(step_losses),
            'mean': np.mean(step_losses),
            'std': np.std(step_losses),
            'final': step_losses[-1] if step_losses else None
        }
    
    if batch_losses:
        summary['batch_loss'] = {
            'min': min(batch_losses),
            'max': max(batch_losses),
            'mean': np.mean(batch_losses),
            'std': np.std(batch_losses),
            'final': batch_losses[-1] if batch_losses else None
        }
    
    return summary


def print_loss_summary(summary: dict, title: str = "Training Loss Summary"):
    """
    打印损失统计摘要
    
    Args:
        summary (dict): 损失统计摘要
        title (str): 标题
    """
    print(f"\n=== {title} ===")
    print(f"训练轮次: {summary['epoch_count']}")
    print(f"训练步数: {summary['step_count']}")
    print(f"训练批次: {summary['batch_count']}")
    
    if 'epoch_loss' in summary:
        epoch = summary['epoch_loss']
        print(f"\nEpoch损失统计:")
        print(f"  最小值: {epoch['min']:.6f}")
        print(f"  最大值: {epoch['max']:.6f}")
        print(f"  平均值: {epoch['mean']:.6f}")
        print(f"  标准差: {epoch['std']:.6f}")
        print(f"  最终值: {epoch['final']:.6f}")
    
    if 'step_loss' in summary:
        step = summary['step_loss']
        print(f"\nStep损失统计:")
        print(f"  最小值: {step['min']:.6f}")
        print(f"  最大值: {step['max']:.6f}")
        print(f"  平均值: {step['mean']:.6f}")
        print(f"  标准差: {step['std']:.6f}")
        print(f"  最终值: {step['final']:.6f}")
    
    if 'batch_loss' in summary:
        batch = summary['batch_loss']
        print(f"\nBatch损失统计:")
        print(f"  最小值: {batch['min']:.6f}")
        print(f"  最大值: {batch['max']:.6f}")
        print(f"  平均值: {batch['mean']:.6f}")
        print(f"  标准差: {batch['std']:.6f}")
        print(f"  最终值: {batch['final']:.6f}")
    
    print("=" * 50) 