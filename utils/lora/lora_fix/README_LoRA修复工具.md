# LoRA适配器文件修复工具使用指南

## 概述

这套工具用于检查和修复LoRA适配器模型中缺失的关键文件，特别是`adapter_model.safetensors`和`adapter_config.json`。当主目录缺少这些文件时，工具会从`checkpoints`目录中的最佳checkpoint（优先epoch4）恢复这些文件。

## 工具组成

### 1. `fix_lora_adapters.py` - 完整版修复工具
功能全面的修复工具，支持详细配置、干运行模式、详细报告等。

### 2. `quick_fix_lora.py` - 快速修复工具
简化版工具，专注于快速检查和修复，使用简单。

## 快速使用

### 方法1: 使用快速修复工具（推荐）

```bash
# 使用默认路径
python quick_fix_lora.py

# 指定自定义路径
python quick_fix_lora.py /path/to/your/loraadaptors/

# 给予执行权限并直接运行
chmod +x quick_fix_lora.py
./quick_fix_lora.py
```

### 方法2: 使用完整版工具

```bash
# 基本使用 - 修复默认路径下的所有模型
python fix_lora_adapters.py

# 干运行模式 - 只检查不修复
python fix_lora_adapters.py --dry_run

# 只处理特定模型
python fix_lora_adapters.py --model_name "Llama-3.1-8B"

# 只处理特定包
python fix_lora_adapters.py --package_filter "sklearn"

# 详细日志模式
python fix_lora_adapters.py --verbose
```

## 工具工作原理

### 检查逻辑
1. **扫描目录**: 遍历LoRA模型基础目录，查找包含`checkpoints`子目录的模型目录
2. **检查文件**: 验证主目录是否包含必需文件：
   - `adapter_model.safetensors`
   - `adapter_config.json`
3. **查找最佳checkpoint**: 
   - 优先选择`*_epoch4`目录
   - 如果没有epoch4，选择最高epoch数的checkpoint
   - 如果都没有，使用第一个可用的checkpoint

### 修复过程
1. **复制必需文件**: 从最佳checkpoint复制缺失的关键文件
2. **复制可选文件**: 如果存在，也会复制以下文件：
   - `README.md`
   - `training_args.bin`
   - `trainer_state.json`
   - `pytorch_model.bin`

## 详细使用示例

### 快速修复示例

```bash
# 场景：检查并修复所有LoRA模型
python quick_fix_lora.py

# 输出示例：
# ============================================================
# 🚀 快速LoRA适配器文件修复工具
# ============================================================
# 🔍 扫描LoRA模型目录: /datanfs2/chenrongyi/models/loraadaptors/
# 
# 📦 检查模型: sklearn_0.21.3_srccodes_up_down_gate_1-32_64_128_1e-6_4_0.1_bf16
#   ⚠️  缺失文件: adapter_model.safetensors, adapter_config.json
#   🔧 使用checkpoint: checkpoint_20250530_122115_epoch4
#     ✅ 复制: adapter_model.safetensors
#     ✅ 复制: adapter_config.json
#   🎉 修复成功! 恢复了 2 个文件
```

### 完整版工具示例

```bash
# 干运行模式 - 查看将要进行的操作
python fix_lora_adapters.py --dry_run --verbose

# 只处理sklearn相关的模型
python fix_lora_adapters.py --package_filter sklearn --verbose

# 生成详细报告
python fix_lora_adapters.py --output_report sklearn_fix_report.json

# 只处理特定模型和包的组合
python fix_lora_adapters.py \
    --model_name "Llama-3.1-8B" \
    --package_filter "sklearn" \
    --verbose \
    --output_report sklearn_llama_fix_report.json
```

## 参数说明

### 完整版工具参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|---------|------|
| `--base_path` | str | `/datanfs2/chenrongyi/models/loraadaptors/` | LoRA模型基础路径 |
| `--model_name` | str | `Llama-3.1-8B` | 指定模型名称，只处理该模型下的适配器 |
| `--package_filter` | str | `None` | 包名过滤器，只处理匹配的包 |
| `--dry_run` | flag | `False` | 干运行模式，只检查不执行修复 |
| `--output_report` | str | `None` | 输出报告文件路径 |
| `--verbose` | flag | `False` | 显示详细日志 |

### 快速工具参数

快速工具只接受一个可选的位置参数：基础路径。

```bash
python quick_fix_lora.py [base_path]
```

## 输出报告

### 控制台输出
两个工具都会在控制台显示：
- 实时处理进度
- 每个模型的检查和修复状态
- 最终统计报告

### 详细报告文件（仅完整版）
```json
{
  "timestamp": "2024-12-01T14:30:22.123456",
  "summary": {
    "total_models": 10,
    "complete_models": 7,
    "fixed_models": 2,
    "failed_models": 1,
    "error_models": 0,
    "success_rate": "90.0%"
  },
  "detailed_results": [...]
}
```

## 使用场景

### 1. 日常维护
```bash
# 每天运行一次，确保所有模型文件完整
python quick_fix_lora.py
```

### 2. 训练后检查
```bash
# 训练完成后，检查新生成的模型
python fix_lora_adapters.py --package_filter "新包名" --verbose
```

### 3. 批量修复
```bash
# 对特定模型进行批量修复
python fix_lora_adapters.py --model_name "Llama-3.1-8B" --output_report batch_fix_report.json
```

### 4. 问题诊断
```bash
# 详细检查问题模型
python fix_lora_adapters.py --dry_run --verbose --package_filter "问题包名"
```

## 安全性和注意事项

### 安全措施
1. **非破坏性操作**: 工具只复制文件，不会删除或修改现有文件
2. **干运行模式**: 支持预览操作，确认无误后再执行
3. **详细日志**: 记录所有操作，便于问题追踪

### 注意事项
1. **权限要求**: 确保对目标目录有写权限
2. **磁盘空间**: 复制操作需要足够的磁盘空间
3. **备份建议**: 重要数据建议先备份
4. **版本一致性**: 确保checkpoint版本与期望的模型版本一致

## 故障排除

### 常见问题

1. **权限错误**
   ```bash
   # 检查目录权限
   ls -la /datanfs2/chenrongyi/models/loraadaptors/
   
   # 如果需要，修改权限
   chmod -R 755 /datanfs2/chenrongyi/models/loraadaptors/
   ```

2. **找不到checkpoint**
   ```bash
   # 手动检查checkpoint目录
   find /datanfs2/chenrongyi/models/loraadaptors/ -name "checkpoints" -type d
   ```

3. **文件复制失败**
   ```bash
   # 检查磁盘空间
   df -h /datanfs2/chenrongyi/models/loraadaptors/
   
   # 检查文件系统错误
   fsck /dev/your-device
   ```

4. **模型目录结构异常**
   ```bash
   # 使用详细模式检查具体问题
   python fix_lora_adapters.py --verbose --package_filter "问题包名"
   ```

### 调试技巧

1. **使用干运行模式**
   ```bash
   python fix_lora_adapters.py --dry_run --verbose
   ```

2. **检查特定模型**
   ```bash
   python fix_lora_adapters.py --package_filter "sklearn" --verbose
   ```

3. **生成详细报告**
   ```bash
   python fix_lora_adapters.py --output_report debug_report.json --verbose
   ```

## 性能优化

### 大规模修复
如果需要处理大量模型，建议：

1. **分批处理**
   ```bash
   # 按包名分批处理
   python fix_lora_adapters.py --package_filter "sklearn"
   python fix_lora_adapters.py --package_filter "numpy"
   ```

2. **并行处理**
   ```bash
   # 在不同终端中并行处理不同模型
   python fix_lora_adapters.py --model_name "Llama-3.1-8B" &
   python fix_lora_adapters.py --model_name "Mistral-7B" &
   ```

3. **监控资源使用**
   ```bash
   # 监控磁盘I/O和内存使用
   iostat -x 1
   htop
   ```

## 最佳实践

1. **定期检查**: 建议每周运行一次完整检查
2. **训练后验证**: 每次训练完成后立即检查新模型
3. **备份重要模型**: 对重要的LoRA模型进行定期备份
4. **监控日志**: 关注修复工具的输出，及时发现问题
5. **版本管理**: 记录每次修复操作的时间和范围 