# 内存调试工具使用指南

这是一套完整的内存调试工具，用于分析模型训练时各个参数矩阵的显存占用情况，帮助您发现和解决内存相关的问题。

## 📁 工具组件

### 1. 核心组件
- **`memoryDebugger.py`** - 核心内存调试器，提供详细的内存分析功能
- **`trainWithMemoryDebug.py`** - 集成训练的内存调试工具
- **`quickMemoryDebug.py`** - 快速内存调试工具，提供简单易用的接口

### 2. 使用示例
- **`benchmark/debug_memory_usage.py`** - 完整的使用示例脚本
- **`README_Memory_Debug.md`** - 本使用文档

## 🚀 快速开始

### 1. 基础使用 - 快速内存调试

```python
from utils.loraTrain.quickMemoryDebug import (
    memory_snapshot, analyze_model, monitor_function, 
    check_memory_status, cleanup_memory, generate_report
)

# 获取内存快照
snapshot = memory_snapshot("before_training")

# 分析模型内存使用
model_analysis = analyze_model(your_model, "my_model")

# 监控函数内存变化
result, comparison = monitor_function(your_training_function, args...)

# 检查内存状态
check_memory_status(threshold_mb=500.0)

# 清理内存
cleanup_memory()

# 生成报告
report = generate_report()
```

### 2. 高级使用 - 完整训练监控

```python
from utils.loraTrain.trainWithMemoryDebug import debug_lora_training

# 完整的训练过程内存调试
trained_model, debug_results = debug_lora_training(
    config=config,
    dataloader=dataloader,
    precision='fp16',
    pkg='matplotlib',
    version='3.8.0',
    knowledge_type='code',
    log_dir='logs/training_debug'
)

# 查看调试结果
print("模型创建内存占用:", debug_results['creation_analysis']['summary']['total_memory_mb'])
print("训练后内存占用:", debug_results['training_analysis']['summary']['total_memory_mb'])
print("检测到的内存问题:", debug_results['memory_issues'])
```

### 3. 命令行使用

```bash
# 仅调试模型创建
python benchmark/debug_memory_usage.py --debug_level model_only --pkg matplotlib --version 3.8.0

# 完整训练调试
python benchmark/debug_memory_usage.py --debug_level full --pkg matplotlib --version 3.8.0 --precision fp16

# 自定义阶段调试
python benchmark/debug_memory_usage.py --debug_level custom --pkg matplotlib --version 3.8.0

# 配置比较调试
python benchmark/debug_memory_usage.py --debug_level comparison --pkg matplotlib --version 3.8.0
```

## 📊 主要功能

### 1. 内存监控功能

#### 实时内存监控
- **GPU内存使用情况**：分配内存、保留内存、利用率
- **系统内存监控**：总内存、可用内存、使用率
- **内存历史追踪**：记录内存使用变化历史

#### 参数级别分析
- **参数矩阵详情**：形状、数据类型、设备分布
- **内存占用统计**：每个参数的内存占用量
- **层级分组分析**：按层级分析内存分布

### 2. 异常检测功能

#### 内存问题检测
- **内存泄漏检测**：检测内存持续增长
- **过度使用检测**：检测GPU利用率过高
- **异常增长检测**：检测内存突然大幅增长
- **内存碎片化检测**：检测内存碎片化问题

#### 参数异常检测
- **大参数检测**：检测异常大的参数矩阵
- **NaN/Inf检测**：检测参数中的异常值
- **梯度异常检测**：检测梯度中的异常情况

### 3. 分析报告功能

#### 详细报告
- **内存使用报告**：完整的内存使用分析
- **参数分析报告**：详细的参数分析结果
- **训练过程报告**：训练各阶段的内存变化

#### 比较分析
- **检查点比较**：比较不同时间点的内存状态
- **配置比较**：比较不同配置的内存使用
- **阶段比较**：比较训练各阶段的内存变化

## 📋 使用场景

### 1. 模型开发阶段
```python
# 分析模型参数内存占用
from utils.loraTrain.quickMemoryDebug import analyze_model

# 创建模型
model = create_your_model()

# 分析内存使用
analysis = analyze_model(model, "development_model")
print(f"模型内存占用: {analysis['memory_mb']:.2f} MB")
print(f"参数效率: {analysis['param_efficiency']:.2f}%")
```

### 2. 训练过程调试
```python
# 使用上下文管理器监控训练
from utils.loraTrain.trainWithMemoryDebug import memory_profiled_training

with memory_profiled_training() as profiler:
    # 模型创建阶段
    model = profiler.profile_stage("model_creation", create_model)
    
    # 数据加载阶段
    dataloader = profiler.profile_stage("data_loading", create_dataloader)
    
    # 训练阶段
    trained_model = profiler.profile_stage("training", train_model, model, dataloader)
    
    # 自动生成训练报告
    report = profiler.generate_training_report()
```

### 3. 内存问题诊断
```python
# 检测内存问题
from utils.loraTrain.memoryDebugger import create_memory_debugger

with create_memory_debugger() as debugger:
    # 创建多个检查点
    debugger.create_memory_checkpoint("start")
    
    # 执行可能有问题的代码
    problematic_function()
    
    debugger.create_memory_checkpoint("after_problem")
    
    # 比较检查点
    comparison = debugger.compare_checkpoints("start", "after_problem")
    
    # 生成分析报告
    report = debugger.generate_memory_report()
```

### 4. 配置优化
```python
# 比较不同配置的内存使用
from utils.loraTrain.trainWithMemoryDebug import debug_memory_with_different_configs

base_config = load_config()
variations = {
    "small_r": {"r": 8, "alpha": 16},
    "medium_r": {"r": 16, "alpha": 32},
    "large_r": {"r": 32, "alpha": 64}
}

results = debug_memory_with_different_configs(base_config, variations)

# 找到最优配置
best_config = min(results.items(), key=lambda x: x[1]['memory_mb'])
print(f"最优配置: {best_config[0]}, 内存占用: {best_config[1]['memory_mb']:.2f} MB")
```

## 🔧 配置选项

### 1. 日志配置
```python
# 自定义日志目录
debugger = create_memory_debugger(
    log_dir="custom_logs/memory_debug",
    enable_real_time=True
)

# 设置日志级别
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. 监控配置
```python
# 自定义监控间隔
debugger.start_real_time_monitoring(interval=0.5)  # 0.5秒间隔

# 设置内存阈值
debugger._check_memory_anomalies(threshold_mb=200.0)
```

### 3. 报告配置
```python
# 自定义报告输出
report = debugger.generate_memory_report(
    output_file="custom_report.txt"
)

# 设置检查点历史长度
debugger.memory_history = debugger.memory_history[-500:]  # 保留最近500条记录
```

## 📈 输出示例

### 1. 内存快照输出
```
[INFO] 获取内存快照: training_start
[INFO]   GPU 0: 1234.5MB / 11264.0MB (10.9%)
[INFO]   GPU 1: 987.3MB / 11264.0MB (8.8%)
```

### 2. 模型分析输出
```
[INFO] 快速分析模型: lora_model
[INFO]   总参数: 7,241,728
[INFO]   可训练参数: 131,072
[INFO]   内存占用: 27.64 MB
[INFO]   参数效率: 1.81%
```

### 3. 内存变化输出
```
[INFO] 比较内存快照: before_training vs after_training
[INFO]   GPU 0: 内存变化 +456.7MB, 利用率变化 +4.1%
[INFO]   GPU 1: 内存变化 +234.5MB, 利用率变化 +2.1%
```

### 4. 异常检测输出
```
[WARNING] GPU 0 内存使用超过阈值: 1456.7MB > 1000.0MB
[WARNING] 大参数检测: model.lm_head.weight (145.67 MB)
[ERROR] 参数包含NaN: model.layers.0.attention.q_proj.weight
```

## 🎯 最佳实践

### 1. 训练前检查
```python
# 训练前进行内存预检查
from utils.loraTrain.quickMemoryDebug import analyze_model, check_memory_status

# 分析模型
model_analysis = analyze_model(model, "pre_training")

# 检查可用内存
check_memory_status(threshold_mb=model_analysis['memory_mb'] * 3)  # 预留3倍内存

# 如果内存不足，进行清理
if model_analysis['memory_mb'] > 1000:
    cleanup_memory()
```

### 2. 训练中监控
```python
# 在训练循环中添加内存监控
for epoch in range(num_epochs):
    # 每个epoch开始时检查内存
    epoch_start_snapshot = memory_snapshot(f"epoch_{epoch}_start")
    
    # 训练代码
    train_epoch(model, dataloader)
    
    # 每个epoch结束时检查内存
    epoch_end_snapshot = memory_snapshot(f"epoch_{epoch}_end")
    
    # 比较内存变化
    comparison = compare_snapshots(epoch_start_snapshot, epoch_end_snapshot)
    
    # 如果内存增长过快，进行清理
    for change in comparison['gpu_changes']:
        if change['memory_change_mb'] > 100:
            cleanup_memory()
```

### 3. 问题排查
```python
# 系统性的内存问题排查
def debug_memory_issue():
    # 1. 获取基准内存状态
    baseline = memory_snapshot("baseline")
    
    # 2. 逐步执行可能有问题的代码
    model = monitor_function(create_model)[0]
    dataloader = monitor_function(create_dataloader)[0]
    
    # 3. 分析每个步骤的内存变化
    model_analysis = analyze_model(model, "debug_model")
    
    # 4. 检查是否有内存问题
    issues = check_memory_status(threshold_mb=500.0)
    
    # 5. 生成详细报告
    report = generate_report()
    
    return report
```

## 📝 注意事项

### 1. 性能影响
- 内存调试工具会带来一定的性能开销
- 建议在开发和调试阶段使用，生产环境可以关闭
- 实时监控功能对性能影响较大，可根据需要开启

### 2. 内存使用
- 调试工具本身也会占用一定内存
- 历史记录会占用内存，建议定期清理
- 大模型分析时可能需要额外内存

### 3. 平台兼容性
- 主要针对CUDA环境优化
- CPU环境也支持，但功能有限
- 不同GPU架构可能有差异

## 🔄 更新日志

### v1.0.0
- 初始版本发布
- 支持基本的内存监控和分析
- 支持参数级别的内存分析
- 支持异常检测和报告生成

### 未来计划
- 增加更多模型架构支持
- 优化内存分析算法
- 增加可视化界面
- 支持分布式训练监控

## 📞 支持

如果您在使用过程中遇到问题，请：

1. 查看日志文件中的详细信息
2. 检查配置是否正确
3. 确认GPU驱动和PyTorch版本兼容
4. 提供完整的错误日志和环境信息

---

希望这个内存调试工具能够帮助您更好地分析和优化模型的内存使用情况！🚀 