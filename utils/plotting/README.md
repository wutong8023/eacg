# 训练损失曲线绘图工具

这个模块提供了训练过程中损失曲线的绘制功能，包括Epoch平均损失、Step损失、Batch损失以及平滑的Batch损失曲线。

## 功能特性

- **多种损失曲线**: 支持Epoch、Step、Batch三种不同粒度的损失曲线
- **平滑处理**: 自动对Batch损失进行平滑处理，减少噪声
- **统计摘要**: 提供详细的损失统计信息
- **对比分析**: 支持多个训练过程的损失对比
- **灵活配置**: 支持自定义图片尺寸、分辨率等参数
- **错误处理**: 完善的异常处理和错误提示

## 主要函数

### `plot_training_losses()`
绘制训练损失曲线并保存到文件

```python
from utils.plotting.training_plots import plot_training_losses

success = plot_training_losses(
    epoch_losses=[1.5, 1.2, 0.9, 0.7],  # Epoch平均损失
    step_losses=[1.6, 1.4, 1.1, 0.8],   # Step损失
    batch_losses=[1.7, 1.5, 1.2, 0.9],  # Batch损失
    save_path="loss_curve.png",          # 保存路径
    title_suffix=" (Final)",             # 标题后缀
    figsize=(12, 8),                     # 图片尺寸
    dpi=100                              # 分辨率
)
```

### `save_training_plots()`
保存训练损失曲线图（推荐使用）

```python
from utils.plotting.training_plots import save_training_plots

success = save_training_plots(
    epoch_losses=epoch_losses,
    step_losses=step_losses,
    batch_losses=batch_losses,
    log_plot="training_logs/loss_curve.png",
    is_final=True,    # 是否为最终绘图
    verbose=True      # 是否输出详细信息
)
```

### `create_loss_summary()`
创建损失统计摘要

```python
from utils.plotting.training_plots import create_loss_summary

summary = create_loss_summary(epoch_losses, step_losses, batch_losses)
print(summary)
# 输出示例:
# {
#     'epoch_count': 10,
#     'step_count': 500,
#     'batch_count': 2000,
#     'epoch_loss': {
#         'min': 0.123456,
#         'max': 2.345678,
#         'mean': 1.234567,
#         'std': 0.567890,
#         'final': 0.123456
#     },
#     ...
# }
```

### `print_loss_summary()`
打印格式化的损失统计摘要

```python
from utils.plotting.training_plots import print_loss_summary

print_loss_summary(summary, "训练完成 - 模型A")
```

### `plot_loss_comparison()`
绘制多个训练过程的损失对比图

```python
from utils.plotting.training_plots import plot_loss_comparison

loss_data = {
    "模型A": {
        'epoch': epoch_losses_a,
        'step': step_losses_a,
        'batch': batch_losses_a
    },
    "模型B": {
        'epoch': epoch_losses_b,
        'step': step_losses_b,
        'batch': batch_losses_b
    }
}

success = plot_loss_comparison(
    loss_data=loss_data,
    save_path="comparison.png",
    title="模型A vs 模型B 训练损失对比"
)
```

## 在训练代码中的使用

### 基本使用

```python
from utils.plotting.training_plots import save_training_plots, create_loss_summary, print_loss_summary

# 在训练循环中
for epoch in range(num_epochs):
    # ... 训练代码 ...
    
    # 每N个epoch保存一次中间损失曲线
    if (epoch + 1) % 2 == 0:
        save_training_plots(epoch_losses, step_losses, batch_losses, 
                          log_plot, is_final=False)

# 训练结束后保存最终损失曲线
save_training_plots(epoch_losses, step_losses, batch_losses, 
                   log_plot, is_final=True)

# 打印训练统计摘要
summary = create_loss_summary(epoch_losses, step_losses, batch_losses)
print_loss_summary(summary, "训练完成")
```

### 在分布式训练中的使用

```python
# 仅在rank 0上执行绘图操作
if rank == 0:
    if plotting_enabled:
        save_training_plots(epoch_losses, step_losses, batch_losses, 
                          log_plot, is_final=True)
    else:
        print("训练完成。由于matplotlib不可用，损失曲线未绘制。")
    
    # 打印训练损失统计摘要
    if epoch_losses or step_losses or batch_losses:
        summary = create_loss_summary(epoch_losses, step_losses, batch_losses)
        print_loss_summary(summary, f"训练完成 - {pkg}-{version}")
```

## 输出图片说明

生成的图片包含4个子图：

1. **Epoch Average Loss**: 每个epoch的平均损失曲线
2. **Training Loss per Step**: 每个训练步骤的损失曲线
3. **Training Loss per Batch**: 每个batch的损失曲线
4. **Smoothed Batch Loss**: 平滑处理后的batch损失曲线

## 配置选项

### matplotlib后端设置

```python
from utils.plotting.training_plots import setup_matplotlib_backend

# 设置非交互式后端（推荐用于服务器环境）
setup_matplotlib_backend('Agg')

# 设置交互式后端（推荐用于本地开发）
setup_matplotlib_backend('TkAgg')
```

### 图片质量设置

```python
# 高分辨率图片
plot_training_losses(..., figsize=(15, 10), dpi=300)

# 标准质量图片
plot_training_losses(..., figsize=(12, 8), dpi=100)
```

## 错误处理

模块包含完善的错误处理机制：

- **matplotlib导入失败**: 自动禁用绘图功能并给出提示
- **文件保存失败**: 捕获异常并输出错误信息
- **数据异常**: 处理空数据、NaN值等异常情况

## 测试

运行测试脚本验证功能：

```bash
cd tests
python test_plotting.py
```

测试包括：
- 基本绘图功能
- 保存训练绘图功能
- 损失统计摘要功能
- 损失对比绘图功能
- 边界情况处理

## 依赖要求

- matplotlib >= 3.0
- numpy >= 1.19
- Python >= 3.7

## 注意事项

1. **服务器环境**: 建议使用'Agg'后端，避免GUI依赖
2. **内存使用**: 大量数据时注意内存使用情况
3. **文件权限**: 确保有写入目标目录的权限
4. **并发安全**: 在分布式训练中仅在主进程执行绘图操作 