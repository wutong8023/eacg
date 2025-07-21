# GPU内存分析器 (GPUMemoryProfiler)

一个功能强大的GPU内存监控和分析工具，专门为深度学习训练（特别是LoRA微调）设计。

## 🚀 主要功能

### 1. 实时内存监控
- 跟踪GPU内存分配和保留情况
- 记录最大内存使用量
- 时间戳记录和耗时统计

### 2. 详细日志记录
- 自动生成带时间戳的日志文件
- 同时输出到控制台和文件
- 支持额外信息记录

### 3. 张量和模型信息记录
- 记录张量的形状、数据类型、大小
- 统计模型参数数量和大小
- 分析可训练参数

### 4. 多种报告格式
- 控制台时间线报告
- 详细的摘要报告
- JSON格式数据导出

## 📦 安装和使用

### 基本使用

```python
from memoryCheck import GPUMemoryProfiler

# 创建分析器
profiler = GPUMemoryProfiler(
    log_dir="logs/my_training",  # 日志目录
    enable_file_logging=True     # 启用文件日志
)

# 记录事件
profiler.record("模型创建")
profiler.record("前向传播", "loss=0.1234")

# 记录详细信息
tensor_info = {'input': input_tensor, 'output': output_tensor}
model_info = {'total_params': 1000000, 'trainable_params': 500000}
profiler.record_detailed("训练步骤", tensor_info=tensor_info, model_info=model_info)

# 生成报告
profiler.print_report()
summary = profiler.generate_summary_report()
json_file = profiler.save_to_json()

# 清理资源
profiler.cleanup()
```

### 在LoRA训练中使用

```python
def train_lora_with_profiling():
    profiler = GPUMemoryProfiler(log_dir="logs/lora_training")
    
    # 记录训练开始
    profiler.record("训练开始")
    
    # 模型创建
    lora_A = torch.randn(r, hidden_size, device='cuda', requires_grad=True)
    lora_B = torch.randn(hidden_size, r, device='cuda', requires_grad=True)
    
    tensor_info = {'lora_A': lora_A, 'lora_B': lora_B}
    model_info = {
        'total_params': lora_A.numel() + lora_B.numel(),
        'trainable_params': lora_A.numel() + lora_B.numel()
    }
    profiler.record_detailed("LoRA模型创建", tensor_info=tensor_info, model_info=model_info)
    
    # 训练循环
    for epoch in range(num_epochs):
        profiler.record(f"Epoch {epoch+1} 开始")
        
        for batch in dataloader:
            # 前向传播
            loss = model(batch)
            profiler.record("前向传播", f"loss={loss.item():.4f}")
            
            # 反向传播
            loss.backward()
            profiler.record("反向传播")
            
            # 参数更新
            optimizer.step()
            optimizer.zero_grad()
            profiler.record("参数更新")
    
    # 生成最终报告
    profiler.print_report()
    profiler.generate_summary_report()
    profiler.save_to_json()
    profiler.cleanup()
```

## 📊 输出示例

### 控制台输出
```
==== GPU Memory Timeline =====
初始化               | Allocated:    0.0MB | Reserved:    0.0MB
模型创建             | Allocated:  128.0MB | Reserved:  256.0MB
前向传播             | Allocated:  512.0MB | Reserved:  768.0MB
反向传播             | Allocated:  768.0MB | Reserved: 1024.0MB
参数更新             | Allocated:  512.0MB | Reserved:  768.0MB
```

### 日志文件内容
```
2024-01-15 10:30:15 - MemoryProfiler_20240115_103015 - INFO - ================================================================================
2024-01-15 10:30:15 - MemoryProfiler_20240115_103015 - INFO - GPU内存分析器初始化
2024-01-15 10:30:15 - MemoryProfiler_20240115_103015 - INFO - ================================================================================
2024-01-15 10:30:15 - MemoryProfiler_20240115_103015 - INFO - 日志文件: logs/lora_training/memory_profiler_20240115_103015.log
2024-01-15 10:30:15 - MemoryProfiler_20240115_103015 - INFO - CUDA可用: True
2024-01-15 10:30:15 - MemoryProfiler_20240115_103015 - INFO - GPU数量: 4
2024-01-15 10:30:15 - MemoryProfiler_20240115_103015 - INFO - GPU 0: NVIDIA A100-SXM4-40GB (40.0GB)
2024-01-15 10:30:16 - MemoryProfiler_20240115_103015 - INFO - 事件: 训练开始 | 分配:    0.0MB | 保留:    0.0MB | 最大分配:    0.0MB | 最大保留:    0.0MB | 耗时: 0.00s
2024-01-15 10:30:17 - MemoryProfiler_20240115_103015 - INFO - 事件: 模型创建 | 分配:  128.0MB | 保留:  256.0MB | 最大分配:  128.0MB | 最大保留:  256.0MB | 耗时: 1.23s
```

### JSON数据格式
```json
{
  "训练开始": [
    {
      "time": 0.0,
      "allocated": 0.0,
      "reserved": 0.0,
      "max_allocated": 0.0,
      "max_reserved": 0.0,
      "timestamp": "2024-01-15T10:30:16.123456"
    }
  ],
  "模型创建": [
    {
      "time": 1.23,
      "allocated": 128.0,
      "reserved": 256.0,
      "max_allocated": 128.0,
      "max_reserved": 256.0,
      "timestamp": "2024-01-15T10:30:17.456789"
    }
  ]
}
```

## 🔧 高级功能

### 1. 内存泄漏检测
```python
# 监控内存增长
initial_memory = torch.cuda.memory_allocated()
# ... 执行操作 ...
final_memory = torch.cuda.memory_allocated()
growth = final_memory - initial_memory

if growth > threshold:
    profiler.record("内存泄漏警告", f"增长={growth:.2f}MB")
```

### 2. 峰值内存分析
```python
# 获取峰值内存使用
peak_allocated = torch.cuda.max_memory_allocated()
peak_reserved = torch.cuda.max_memory_reserved()

profiler.record("峰值内存", f"分配={peak_allocated:.2f}MB, 保留={peak_reserved:.2f}MB")
```

### 3. 自定义内存清理
```python
def custom_memory_cleanup():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    profiler.record("自定义清理")
```

## 📈 性能优化建议

### 1. 内存使用优化
- 使用较小的批次大小
- 启用梯度检查点
- 使用混合精度训练
- 及时清理中间变量

### 2. 监控策略
- 在关键节点记录内存使用
- 定期生成摘要报告
- 设置内存使用阈值警告
- 保存历史数据用于分析

### 3. 日志管理
- 定期清理旧日志文件
- 使用有意义的日志目录结构
- 为不同实验使用不同的日志目录

## 🐛 故障排除

### 常见问题

1. **CUDA不可用**
   ```python
   if not torch.cuda.is_available():
       print("CUDA不可用，使用CPU模式")
   ```

2. **内存不足**
   ```python
   try:
       # 执行内存密集型操作
       pass
   except torch.cuda.OutOfMemoryError:
       profiler.record("内存不足错误")
       torch.cuda.empty_cache()
   ```

3. **日志文件过大**
   ```python
   # 定期清理日志
   import os
   if os.path.getsize(log_file) > max_size:
       # 压缩或删除旧日志
       pass
   ```

## 📝 测试

运行测试套件：
```bash
python tests/testProfiler.py
```

运行使用示例：
```bash
python utils/memoryDebug/example_usage.py
```

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个工具！

## �� 许可证

MIT License 