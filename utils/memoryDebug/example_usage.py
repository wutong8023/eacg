#!/usr/bin/env python3
"""
内存分析器使用示例
展示如何在LoRA训练中使用GPUMemoryProfiler进行内存监控
"""

import torch
import torch.nn as nn
from memoryCheck import GPUMemoryProfiler
import time

def example_lora_training():
    """示例：在LoRA训练中使用内存分析器"""
    
    # 创建内存分析器
    profiler = GPUMemoryProfiler(
        log_dir="logs/example_lora_training",
        enable_file_logging=True
    )
    
    # 记录训练开始
    profiler.record("训练开始")
    
    # 模拟模型创建
    print("🔧 创建模型...")
    hidden_size = 4096
    r = 16
    
    # 创建LoRA参数
    lora_A = torch.randn(r, hidden_size, device='cuda', requires_grad=True)
    lora_B = torch.randn(hidden_size, r, device='cuda', requires_grad=True)
    
    # 记录模型创建
    tensor_info = {
        'lora_A': lora_A,
        'lora_B': lora_B
    }
    model_info = {
        'total_params': lora_A.numel() + lora_B.numel(),
        'trainable_params': lora_A.numel() + lora_B.numel(),
        'model_size_mb': (lora_A.numel() + lora_B.numel()) * lora_A.element_size() / (1024**2)
    }
    profiler.record_detailed("模型创建完成", tensor_info=tensor_info, model_info=model_info)
    
    # 创建优化器
    print("🔧 创建优化器...")
    optimizer = torch.optim.AdamW([lora_A, lora_B], lr=1e-4)
    profiler.record("优化器创建完成")
    
    # 模拟训练循环
    print("🚀 开始训练循环...")
    num_epochs = 3
    batch_size = 8
    seq_length = 512
    
    for epoch in range(num_epochs):
        profiler.record(f"Epoch {epoch+1} 开始")
        
        for batch in range(5):  # 模拟5个batch
            # 创建输入数据
            input_tensor = torch.randn(batch_size, seq_length, hidden_size, device='cuda')
            
            # 前向传播
            output = torch.matmul(input_tensor, lora_A.T)
            output = torch.matmul(output, lora_B.T)
            loss = output.sum()
            
            profiler.record(f"Epoch {epoch+1} Batch {batch+1} 前向传播", 
                          f"loss={loss.item():.4f}, output_shape={output.shape}")
            
            # 反向传播
            loss.backward()
            profiler.record(f"Epoch {epoch+1} Batch {batch+1} 反向传播")
            
            # 参数更新
            optimizer.step()
            optimizer.zero_grad()
            profiler.record(f"Epoch {epoch+1} Batch {batch+1} 参数更新")
            
            # 清理中间变量
            del input_tensor, output, loss
            torch.cuda.empty_cache()
        
        profiler.record(f"Epoch {epoch+1} 完成")
    
    # 记录训练完成
    profiler.record("训练完成")
    
    # 生成报告
    print("\n📊 生成训练报告...")
    profiler.print_report()
    summary = profiler.generate_summary_report()
    print(summary)
    
    # 保存数据
    json_file = profiler.save_to_json()
    print(f"📄 训练数据已保存到: {json_file}")
    
    # 清理
    profiler.cleanup()
    del lora_A, lora_B, optimizer
    
    print("✅ 示例训练完成!")

def example_memory_monitoring():
    """示例：持续内存监控"""
    
    profiler = GPUMemoryProfiler(
        log_dir="logs/example_monitoring",
        enable_file_logging=True
    )
    
    print("🔍 开始持续内存监控...")
    
    # 模拟长时间运行的任务
    for i in range(10):
        # 创建一些张量
        tensor = torch.randn(1000, 1000, device='cuda')
        
        # 执行一些计算
        result = torch.matmul(tensor, tensor.T)
        
        # 记录内存使用
        profiler.record(f"步骤 {i+1}", f"tensor_shape={tensor.shape}, result_shape={result.shape}")
        
        # 清理
        del tensor, result
        torch.cuda.empty_cache()
        
        # 等待一段时间
        time.sleep(0.5)
    
    # 生成报告
    profiler.print_report()
    profiler.generate_summary_report()
    profiler.save_to_json()
    profiler.cleanup()
    
    print("✅ 持续监控完成!")

def example_memory_analysis():
    """示例：内存分析"""
    
    profiler = GPUMemoryProfiler(
        log_dir="logs/example_analysis",
        enable_file_logging=True
    )
    
    print("🔬 开始内存分析...")
    
    # 测试不同大小的张量
    sizes = [100, 500, 1000, 2000]
    
    for size in sizes:
        # 创建张量
        tensor = torch.randn(size, size, device='cuda')
        
        # 记录详细信息
        tensor_info = {f'tensor_{size}x{size}': tensor}
        profiler.record_detailed(f"创建 {size}x{size} 张量", tensor_info=tensor_info)
        
        # 执行计算
        result = torch.matmul(tensor, tensor.T)
        profiler.record(f"计算 {size}x{size} 矩阵乘法", f"result_shape={result.shape}")
        
        # 清理
        del tensor, result
        torch.cuda.empty_cache()
    
    # 分析内存使用模式
    profiler.print_report()
    summary = profiler.generate_summary_report()
    print(summary)
    
    # 保存分析结果
    json_file = profiler.save_to_json()
    print(f"📄 分析结果已保存到: {json_file}")
    
    profiler.cleanup()
    print("✅ 内存分析完成!")

def main():
    """主函数"""
    print("🚀 内存分析器使用示例")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，无法运行示例")
        return
    
    # 运行示例
    print("\n1️⃣ LoRA训练示例")
    example_lora_training()
    
    print("\n2️⃣ 持续监控示例")
    example_memory_monitoring()
    
    print("\n3️⃣ 内存分析示例")
    example_memory_analysis()
    
    print("\n🎉 所有示例完成!")
    print("📁 查看 logs/ 目录下的日志文件")

if __name__ == "__main__":
    main() 