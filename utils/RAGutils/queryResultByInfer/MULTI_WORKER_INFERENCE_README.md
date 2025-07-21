# Multi-Worker Inference System

这个系统为基于context的推理环节提供了多worker并行处理能力，支持在多个GPU上分布式执行推理任务。

## 系统架构

### 核心组件

1. **multi_worker_inference.py** - 多worker推理主程序
2. **run_multi_worker_inference.sh** - 多worker推理运行脚本
3. **run_complete_pipeline.sh** - 完整端到端流水线脚本
4. **test_multi_worker_inference.py** - 测试脚本

### 工作流程

```
Context文件 (JSONL) → 数据切片 → 多Worker推理 → 结果合并 → 最终输出
```

## 主要特性

### 1. 智能数据切片
- 自动将contexts按worker数量均匀分配
- 处理不能整除的情况，确保所有数据都被处理
- 避免数据重复和遗漏

### 2. 智能GPU分配管理
- 支持GPU独占分配：8个GPU，4个worker，每个worker占用2个GPU
- 自动计算每个worker的GPU分配：`gpus_per_worker = total_gpus // num_workers`
- 支持单GPU和多GPU配置
- 自动设置`CUDA_VISIBLE_DEVICES`为每个worker进程

### 3. 分布式处理
- 使用PyTorch的分布式训练框架
- 支持torchrun启动多worker
- 自动同步和结果合并

### 4. 容错处理
- 处理失败的context项目
- 记录详细的错误信息
- 统计成功率和性能指标

## 使用方法

### 1. 单Worker模式

```bash
python utils/RAGutils/queryResultByInfer/multi_worker_inference.py \
    --contexts_file data/temp/contexts/batch_contexts.jsonl \
    --output_file data/temp/inference/results.jsonl \
    --model_path /path/to/model \
    --inference_type local \
    --max_new_tokens 512 \
    --temperature 0.1
```

### 2. 多Worker模式

```bash
torchrun --nproc_per_node=4 utils/RAGutils/queryResultByInfer/multi_worker_inference.py \
    --contexts_file data/temp/contexts/batch_contexts.jsonl \
    --output_file data/temp/inference/results.jsonl \
    --model_path /path/to/model \
    --inference_type local \
    --max_new_tokens 512 \
    --temperature 0.1
```

### 3. 使用运行脚本

```bash
# 编辑脚本中的参数
vim utils/RAGutils/queryResultByInfer/run_multi_worker_inference.sh

# 运行
bash utils/RAGutils/queryResultByInfer/run_multi_worker_inference.sh
```

### 4. 完整流水线

```bash
# 运行完整的context生成 + 推理流水线
bash utils/RAGutils/queryResultByInfer/run_complete_pipeline.sh
```

## 参数说明

### 输入输出参数
- `--contexts_file`: Context文件路径（JSONL格式）
- `--output_file`: 输出结果文件路径

### 推理参数
- `--model_path`: 模型路径或名称
- `--inference_type`: 推理类型（local/huggingface/togetherai）
- `--max_new_tokens`: 最大生成token数
- `--temperature`: 采样温度
- `--top_p`: Top-p采样参数

### 控制参数
- `--max_samples_per_worker`: 每个worker处理的最大样本数
- `--verbose`: 启用详细日志

## 输入格式

Context文件应为JSONL格式，每行包含一个context对象：

```json
{
    "id": "item_123",
    "original_item_id": "original_45",
    "query_index": 1,
    "dedup_key": "query_text_api_name",
    "query": "How to use numpy.array?",
    "target_api": "numpy.array",
    "dependencies": {"numpy": "1.16.6"},
    "retrieval_method": "exact_api_match",
    "context": "Documentation for numpy.array...",
    "context_length": 150,
    "retrieval_time": 0.12,
    "success": true
}
```

## 输出格式

推理结果为JSONL格式，每行包含一个推理结果：

```json
{
    "id": "item_123",
    "original_item_id": "original_45",
    "query_index": 1,
    "dedup_key": "query_text_api_name",
    "query": "How to use numpy.array?",
    "target_api": "numpy.array",
    "dependencies": {"numpy": "1.16.6"},
    "retrieval_method": "exact_api_match",
    "context_length": 150,
    "answer": "To use numpy.array, you can...",
    "retrieval_time": 0.12,
    "inference_time": 1.45,
    "total_time": 1.57,
    "worker_rank": 0,
    "success": true
}
```

## 性能优化

### 1. 数据分布
- 系统自动平衡各worker的工作负载
- 支持不均匀数据分布的处理

### 2. GPU利用率
- 每个worker独占一个GPU设备
- 避免GPU内存竞争

### 3. 内存管理
- 流式处理，避免一次性加载所有数据
- 及时释放不需要的内存

### 4. 并行效率
- 理论上可以实现接近线性的加速比
- 实际性能取决于模型大小和推理复杂度

## 监控和统计

### 运行时监控
- 实时显示各worker的处理进度
- 记录详细的时间统计信息
- 显示成功率和错误信息

### 结果统计
```
Worker 0 INFERENCE STATISTICS
============================================================
Total contexts processed: 250
Successful inferences: 245
Errors: 5
Success rate: 98.0%
Average inference time: 1.23s
Results saved to: output_rank0.jsonl
```

## 故障排除

### 常见问题

1. **CUDA设备不足**
   ```
   解决：减少worker数量或检查GPU可用性
   export CUDA_VISIBLE_DEVICES="0,1"
   ```

2. **内存不足**
   ```
   解决：减少max_samples_per_worker或使用更小的模型
   --max_samples_per_worker 100
   ```

3. **模型加载失败**
   ```
   解决：检查模型路径和权限
   --model_path /correct/path/to/model
   ```

4. **分布式初始化失败**
   ```
   解决：确保PyTorch版本支持分布式训练
   pip install torch torchvision torchaudio
   ```

### 调试模式

启用详细日志：
```bash
--verbose
```

测试数据切片逻辑：
```bash
python utils/RAGutils/queryResultByInfer/test_multi_worker_inference.py --skip-inference
```

## 扩展性

### 支持的推理后端
- **Local**: 本地模型推理
- **HuggingFace**: HuggingFace Hub模型
- **TogetherAI**: TogetherAI API

### 自定义扩展
可以通过修改`QueryBasedInference`类来支持新的推理后端：

```python
class CustomInference(QueryBasedInference):
    def generate_answer(self, query, context, retrieval_method):
        # 自定义推理逻辑
        return custom_inference_result
```

## GPU分配策略

### 自动分配算法
系统使用以下算法自动分配GPU：

```python
total_gpus = 8  # 可用GPU总数
num_workers = 4  # worker数量
gpus_per_worker = total_gpus // num_workers  # 每个worker的GPU数量

# Worker 0: GPUs [0, 1]
# Worker 1: GPUs [2, 3]  
# Worker 2: GPUs [4, 5]
# Worker 3: GPUs [6, 7]
```

### GPU分配规则
1. **均匀分配**：每个worker获得相等数量的GPU
2. **独占访问**：每个worker独占其分配的GPU，避免竞争
3. **自动降级**：当GPU不足时，自动回退到共享模式
4. **环境隔离**：为每个worker设置独立的`CUDA_VISIBLE_DEVICES`

### 推荐配置

| 场景 | 总GPU | Workers | GPU/Worker | 适用情况 |
|------|-------|---------|------------|----------|
| 开发测试 | 4 | 2 | 2 | 小规模测试 |
| 平衡处理 | 8 | 4 | 2 | 生产环境推荐 |
| 最大并行 | 8 | 8 | 1 | 小模型高吞吐量 |
| 大模型 | 8 | 2 | 4 | 大模型推理 |

## 最佳实践

### 1. 资源配置
- 根据GPU内存选择合适的模型大小
- 确保worker数量能被GPU总数整除
- 监控GPU利用率和内存使用

### 2. 数据准备
- 确保context文件格式正确
- 过滤掉失败的context项目
- 合理设置样本数量限制

### 3. 性能调优
- 使用适当的worker数量（通常等于GPU数量）
- 调整推理参数以平衡质量和速度
- 定期监控和分析性能指标

### 4. 错误处理
- 设置合理的超时时间
- 记录和分析失败案例
- 实现重试机制（如需要）

## 示例配置

### 小规模测试（4 GPUs）
```bash
NUM_WORKERS=2
TOTAL_GPUS=4
GPUS_PER_WORKER=2
MAX_SAMPLES_PER_WORKER=100
```

### 中等规模处理（8 GPUs）
```bash
NUM_WORKERS=4
TOTAL_GPUS=8  
GPUS_PER_WORKER=2
MAX_SAMPLES_PER_WORKER=1000
```

### 大规模生产（8 GPUs, 最大并行）
```bash
NUM_WORKERS=8
TOTAL_GPUS=8
GPUS_PER_WORKER=1
MAX_SAMPLES_PER_WORKER=5000
```

### 大模型推理（8 GPUs, 更多GPU每worker）
```bash
NUM_WORKERS=2
TOTAL_GPUS=8
GPUS_PER_WORKER=4
MAX_SAMPLES_PER_WORKER=2000
```

## 版本历史

- **v1.0**: 基础多worker推理功能
- **v1.1**: 添加数据切片优化和错误处理
- **v1.2**: 支持完整流水线和性能监控
- **v1.3**: 增强容错能力和扩展性 