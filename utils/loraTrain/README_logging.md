# LoRA训练完整日志系统使用说明

## 概述

新的日志系统可以捕获train_lora.py训练过程中的所有终端输出，包括：
- 所有print语句的输出
- 所有logging模块的输出
- 标准输出(stdout)和标准错误(stderr)
- 异常信息和错误信息

所有输出都会同时显示在终端和保存到一个完整的.log文件中，方便调试和追踪问题。

## 主要功能

### 1. 完整输出捕获
- ✅ 捕获所有print语句
- ✅ 捕获所有logging输出
- ✅ 捕获标准输出和标准错误
- ✅ 捕获异常和错误信息
- ✅ 同时在终端显示

### 2. 智能日志格式
- 自动添加时间戳前缀
- 避免重复时间戳
- 会话开头包含详细配置信息
- 支持多worker模式标记

### 3. 文件管理
- 自动创建带时间戳的日志目录
- 统一的日志文件命名规则
- 程序退出时自动清理

## 使用方法

### 基本使用

```python
# 在train_lora.py中已自动集成
python benchmark/train_lora.py --dataset_type docstring --precision bf16
```

### 日志文件位置

日志文件保存在：
```
logs/
└── train_lora_docstring_20241213_143022/
    └── train_lora_docstring_complete.log
```

格式说明：
- `logs/`: 日志根目录
- `train_lora_{dataset_type}_{timestamp}/`: 带时间戳的会话目录
- `train_lora_{dataset_type}_complete.log`: 完整日志文件

### 多Worker模式

在多worker模式下，每个worker都会创建自己的日志文件：
```
logs/
├── train_lora_docstring_20241213_143022/  # Worker 0
│   └── train_lora_docstring_complete.log
├── train_lora_docstring_20241213_143025/  # Worker 1
│   └── train_lora_docstring_complete.log
└── ...
```

## 日志文件格式

### 文件头信息
```
================================================================================
LoRA训练完整日志 - 2024-12-13 14:30:22
================================================================================
训练配置:
  - dataset_type: docstring
  - precision: bf16
  - corpus_path: /datanfs4/chenrongyi/data/docs
  - model_name: /datanfs2/chenrongyi/models/Llama-3.1-8B
  - loraadaptor_save_path_base: /datanfs2/chenrongyi/models/loraadaptors/
  - benchmark_paths: ['data/VersiBCB_Benchmark/vace_datas.json']
  - 多worker模式: rank=0, world_size=2
================================================================================
```

### 日志内容格式
```
[2024-12-13 14:30:22] 🚀 日志系统已启动
[2024-12-13 14:30:22] 📝 完整日志保存到: logs/train_lora_docstring_20241213_143022/train_lora_docstring_complete.log
[2024-12-13 14:30:22] 📊 训练配置: precision=bf16, dataset_type=docstring
2024-12-13 14:30:23 - INFO - === CUDA环境诊断 ===
2024-12-13 14:30:23 - INFO - CUDA_VISIBLE_DEVICES=0,1,2,3
[2024-12-13 14:30:23] 正在加载包 numpy-1.24.0 的训练数据...
[2024-12-13 14:30:24] 模型加载完成，开始训练...
```

## 高级功能

### 自定义日志级别
```python
import logging

# 设置更详细的日志级别
logging.getLogger().setLevel(logging.DEBUG)

# 添加自定义日志
logging.debug("调试信息")
logging.info("普通信息")
logging.warning("警告信息")
logging.error("错误信息")
```

### 手动清理日志系统
```python
from utils.loraTrain.log import cleanup_logging

# 手动清理（通常不需要，程序退出时自动清理）
cleanup_logging()
```

## 调试技巧

### 1. 查看实时日志
```bash
# 实时查看日志输出
tail -f logs/train_lora_docstring_20241213_143022/train_lora_docstring_complete.log
```

### 2. 搜索特定信息
```bash
# 搜索错误信息
grep -i "error\|exception\|failed" logs/train_lora_docstring_20241213_143022/train_lora_docstring_complete.log

# 搜索特定包的训练信息
grep -i "numpy-1.24.0" logs/train_lora_docstring_20241213_143022/train_lora_docstring_complete.log

# 搜索GPU信息
grep -i "gpu\|cuda" logs/train_lora_docstring_20241213_143022/train_lora_docstring_complete.log
```

### 3. 分析训练进度
```bash
# 查看训练统计
grep -i "训练\|跳过\|完成" logs/train_lora_docstring_20241213_143022/train_lora_docstring_complete.log

# 查看最后几行（训练结果）
tail -n 20 logs/train_lora_docstring_20241213_143022/train_lora_docstring_complete.log
```

## 常见问题解决

### Q: 日志文件过大怎么办？
A: 可以使用logrotate或手动清理旧日志：
```bash
# 清理7天前的日志
find logs/ -name "*.log" -mtime +7 -delete
```

### Q: 如何禁用日志记录？
A: 修改train_lora.py中的setup_logging调用：
```python
# 临时禁用（不推荐）
# log_dir = setup_logging(args)
```

### Q: 日志文件权限问题？
A: 确保有足够的磁盘空间和写入权限：
```bash
# 检查磁盘空间
df -h

# 检查权限
ls -la logs/
```

### Q: 多worker模式日志混乱？
A: 每个worker会创建独立的日志文件，避免混乱。查看特定worker的日志：
```bash
# 查看所有worker的日志目录
ls -la logs/

# 根据时间戳找到对应的worker日志
```

## 注意事项

1. **性能影响**: 日志系统会轻微影响性能，但影响很小
2. **磁盘空间**: 长时间训练会产生大量日志，注意清理
3. **编码问题**: 日志使用UTF-8编码，支持中文
4. **异常处理**: 即使出现异常，日志系统也会正常工作
5. **程序退出**: 程序正常或异常退出时会自动清理日志系统

## 技术实现细节

### TeeOutput类
- 实现stdout/stderr的双重输出
- 自动添加时间戳
- 异常安全的文件操作

### LogFileHandler类
- 自定义logging handler
- 避免重复输出
- 统一的日志格式

### 清理机制
- 使用atexit模块注册清理函数
- 自动恢复原始输出流
- 安全关闭文件句柄

这个完整的日志系统确保了训练过程中的所有信息都能被准确记录和追踪，大大提高了调试效率。 