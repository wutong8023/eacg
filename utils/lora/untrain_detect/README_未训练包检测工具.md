# 未训练包检测工具使用指南

## 概述

这套工具用于检测哪些包还没有对应的LoRA模型，并生成批量训练脚本，帮助规划和自动化训练任务。

## 工具组成

### 1. `detect_untrained_packages.py` - 包检测工具
用于扫描所有需要的包版本，检查哪些还没有对应的LoRA模型。

### 2. `generate_training_script.py` - 训练脚本生成器  
基于检测结果生成自动化的批量训练脚本。

## 使用流程

### 步骤1: 检测未训练的包

```bash
# 基本使用 - 检测默认数据集的所有知识类型
python detect_untrained_packages.py

# 指定特定的知识类型
python detect_untrained_packages.py --knowledge_types docstring

# 指定输出文件和显示详细信息
python detect_untrained_packages.py \
    --knowledge_types docstring srccodes \
    --output_file my_detection_report.json \
    --show_missing \
    --verbose

# 使用不同的benchmark数据
python detect_untrained_packages.py \
    --benchmark_data_path "benchmark/data/VersiBCB_Benchmark/vace_datas_for_warning.json" \
    --model_name "/datanfs2/chenrongyi/models/Llama-3.1-8B" \
    --knowledge_types docstring srccodes
```

#### 检测工具参数说明
- `--benchmark_data_path`: benchmark数据文件路径
- `--model_name`: 模型名称  
- `--knowledge_types`: 要检查的知识类型 (docstring, srccodes)
- `--output_file`: 输出结果文件路径 (JSON格式)
- `--show_missing`: 显示所有缺失的包列表
- `--verbose`: 显示详细信息

### 步骤2: 生成训练脚本

```bash
# 基于检测报告生成训练脚本
python generate_training_script.py \
    --report_path untrained_packages_report_20241201_143022.json \
    --output_script batch_train_lora.sh \
    --priority_strategy version_count \
    --gpu_devices "0,1,2"

# 自定义配置
python generate_training_script.py \
    --report_path my_detection_report.json \
    --output_script my_training_script.sh \
    --output_summary my_training_summary.json \
    --model_name "/datanfs2/chenrongyi/models/Llama-3.1-8B" \
    --precision bf16 \
    --gpu_devices "0,1" \
    --base_corpus_path "/datanfs2/chenrongyi/data"
```

#### 脚本生成器参数说明
- `--report_path`: 检测报告文件路径 (必需)
- `--output_script`: 输出训练脚本路径
- `--output_summary`: 输出摘要文件路径
- `--priority_strategy`: 包优先级策略 (version_count, alphabetical)
- `--model_name`: 模型名称
- `--precision`: 训练精度 (fp16, fp32, bf16)
- `--gpu_devices`: GPU设备配置
- `--base_corpus_path`: 语料库基础路径

### 步骤3: 执行批量训练

生成的训练脚本支持灵活的执行选项：

```bash
# 给脚本执行权限
chmod +x batch_train_lora.sh

# 训练所有docstring类型的包
./batch_train_lora.sh docstring

# 训练指定范围的包 (索引0-9，共10个包)
./batch_train_lora.sh docstring 0 9

# 训练所有srccodes类型的包
./batch_train_lora.sh srccodes

# 训练特定范围的srccodes包
./batch_train_lora.sh srccodes 10 20
```

## 输出文件说明

### 检测报告 (`untrained_packages_report_*.json`)
包含以下信息：
- `timestamp`: 检测时间
- `summary`: 各知识类型的统计摘要
- `detailed_results`: 详细的检测结果
- `training_plan`: 训练计划

### 训练计划摘要 (`training_plan_summary.json`)
包含：
- 按优先级排序的包列表
- 各知识类型的训练统计
- 完整的包版本信息

## 示例工作流

### 完整的检测和训练流程

```bash
# 1. 检测未训练的包
echo "=== 步骤1: 检测未训练的包 ==="
python detect_untrained_packages.py \
    --knowledge_types docstring srccodes \
    --show_missing \
    --verbose

# 2. 生成训练脚本 (假设生成的报告文件名为 untrained_packages_report_20241201_143022.json)
echo "=== 步骤2: 生成训练脚本 ==="
python generate_training_script.py \
    --report_path untrained_packages_report_20241201_143022.json \
    --output_script batch_train_lora.sh \
    --gpu_devices "0,1,2" \
    --precision bf16

# 3. 执行训练
echo "=== 步骤3: 开始批量训练 ==="
chmod +x batch_train_lora.sh

# 先训练版本较少的docstring包 (前5个)
./batch_train_lora.sh docstring 0 4

# 然后训练srccodes包
./batch_train_lora.sh srccodes 0 9
```

### 渐进式训练策略

```bash
# 小批量测试训练
./batch_train_lora.sh docstring 0 2    # 训练前3个包测试

# 检查训练结果，如果成功则继续
./batch_train_lora.sh docstring 3 9    # 训练接下来7个包

# 大批量训练
./batch_train_lora.sh docstring        # 训练所有剩余的docstring包
./batch_train_lora.sh srccodes         # 训练所有srccodes包
```

## 高级功能

### 1. 自定义优先级策略

```bash
# 按版本数量排序 (默认，版本少的优先)
python generate_training_script.py \
    --report_path report.json \
    --priority_strategy version_count

# 按字母顺序排序
python generate_training_script.py \
    --report_path report.json \
    --priority_strategy alphabetical
```

### 2. 多GPU配置

```bash
# 使用特定GPU
python generate_training_script.py \
    --report_path report.json \
    --gpu_devices "0,2,3"

# 生成的脚本会自动设置 CUDA_VISIBLE_DEVICES=0,2,3
```

### 3. 监控训练进度

生成的训练脚本包含详细的进度信息：
- 当前训练的包和版本
- 训练成功/失败统计
- 总耗时和预估剩余时间
- 自动GPU缓存清理

### 4. 错误处理

脚本包含robust的错误处理：
- 单个包训练失败不会影响其他包
- 详细的错误日志记录
- 最终统计报告包含成功/失败数量

## 故障排除

### 常见问题

1. **检测工具报告配置加载失败**
   ```bash
   # 检查配置文件路径
   python -c "from benchmark.config.code.config_lora import LORA_CONFIG_PATH; print(LORA_CONFIG_PATH)"
   ```

2. **模型路径检查失败**
   ```bash
   # 确认模型路径正确
   ls -la /datanfs2/chenrongyi/models/Llama-3.1-8B
   ```

3. **生成的脚本权限问题**
   ```bash
   # 添加执行权限
   chmod +x batch_train_lora.sh
   ```

4. **GPU内存不足**
   ```bash
   # 在训练脚本中会自动清理GPU缓存
   # 或手动清理：
   python -c "import torch; torch.cuda.empty_cache()"
   ```

### 调试模式

```bash
# 启用详细日志
python detect_untrained_packages.py --verbose

# 只显示缺失的包，不显示详细检查过程
python detect_untrained_packages.py --show_missing
```

## 性能优化

### 1. 批量大小调整

在生成的训练脚本中，可以调整批量处理的包数量：
- 每5个包清理一次GPU缓存 (默认)
- 可根据GPU内存情况调整

### 2. 并行训练

如果有多个GPU节点，可以：
1. 生成多个不同索引范围的训练任务
2. 在不同节点上并行执行

```bash
# 节点1执行
./batch_train_lora.sh docstring 0 24

# 节点2执行  
./batch_train_lora.sh docstring 25 49

# 节点3执行
./batch_train_lora.sh srccodes 0 24
```

## 最佳实践

1. **定期检测**: 建议在每次大规模训练前运行检测工具
2. **增量训练**: 使用索引范围进行增量训练，便于监控和调试
3. **备份策略**: 重要的检测报告应该备份保存
4. **资源监控**: 训练过程中监控GPU使用情况和内存占用
5. **日志管理**: 定期清理和归档训练日志 