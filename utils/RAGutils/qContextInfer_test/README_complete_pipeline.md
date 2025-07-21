# Complete RAG Pipeline

这是一个模块化的完整RAG（Retrieval-Augmented Generation）流水线，将query生成context和根据context生成answer的过程整合到一个统一的、易于使用的类中。

## 📋 目录

- [特性](#-特性)
- [架构设计](#-架构设计)
- [快速开始](#-快速开始)
- [配置说明](#-配置说明)
- [使用示例](#-使用示例)
- [多Worker支持](#-多worker支持)
- [API参考](#-api参考)
- [最佳实践](#-最佳实践)

## ✨ 特性

### 🔧 模块化设计
- **分离关注点**: Context生成和推理阶段完全分离
- **灵活配置**: 每个阶段都有独立的配置选项
- **可选执行**: 可以跳过任一阶段，只执行需要的部分

### 🚀 多种执行模式
- **完整流水线**: 一键执行从query到answer的完整流程
- **分步执行**: 分别控制context生成和推理阶段
- **部分执行**: 跳过某些阶段，使用已有的中间结果

### 🔄 多Worker支持
- **并行处理**: 支持多worker并行处理提高效率
- **自动分配**: 智能的数据分片和负载均衡
- **结果合并**: 自动合并多worker的结果

### 📊 统计和监控
- **实时统计**: 详细的执行统计信息
- **进度监控**: 可视化的进度条
- **错误处理**: 完善的错误处理和日志记录

### 🎯 多种推理方式
- **本地推理**: 使用本地模型进行推理
- **API推理**: 支持HuggingFace、TogetherAI等API
- **混合模式**: 可以混合使用不同的推理方式

## 🏗️ 架构设计

```
CompletePipeline
├── ContextGenerationConfig     # Context生成配置
├── InferenceConfig            # 推理配置  
├── PipelineConfig            # 流水线总配置
├── PipelineStatistics        # 统计信息
├── QueryBasedRetriever       # 查询检索器
├── BatchContextGenerator     # 批量Context生成器
└── QueryBasedInference       # 查询推理器
```

### 核心组件

1. **配置层**: 使用数据类定义清晰的配置结构
2. **执行层**: 模块化的执行引擎
3. **统计层**: 完整的性能监控和统计
4. **工具层**: 便捷的配置创建工具

## 🚀 快速开始

### 1. 基本安装

```bash
# 确保已安装必要的依赖
pip install torch transformers accelerate
pip install sentence-transformers  # 用于embedding
pip install chromadb  # 用于向量存储
```

### 2. 最简单的使用

```python
from utils.RAGutils.complete_pipeline import (
    CompletePipeline,
    create_context_config,
    create_inference_config,
    create_pipeline_config
)

# 创建配置
context_config = create_context_config(
    corpus_path="/path/to/your/docs",
    corpus_type="docstring"
)

inference_config = create_inference_config(
    model_path="/path/to/your/model",
    inference_type="local"
)

pipeline_config = create_pipeline_config(
    queries_file="data/queries.json",
    contexts_output="data/contexts.jsonl",
    final_output="data/results.jsonl",
    context_config=context_config,
    inference_config=inference_config
)

# 运行流水线
pipeline = CompletePipeline(pipeline_config)
results = pipeline.run_complete_pipeline()

# 查看统计信息
pipeline.print_statistics()
```

### 3. 分步执行

```python
# 自定义queries
queries = [
    {
        "id": "q1",
        "query": "How to use pandas DataFrame?",
        "target_api": "pandas.DataFrame",
        "dependencies": {"pandas": "latest"}
    }
]

# 步骤1: 生成contexts
contexts = pipeline.generate_contexts(queries)

# 步骤2: 运行推理
results = pipeline.run_inference(contexts)
```

## ⚙️ 配置说明

### ContextGenerationConfig

```python
@dataclass
class ContextGenerationConfig:
    corpus_path: str              # 语料库路径
    corpus_type: str = "docstring"   # 语料类型: docstring, srccodes
    embedding_source: str = "local"  # 嵌入来源: local, togetherai
    max_documents: int = 10       # 最大检索文档数
    max_tokens: int = 4000        # 最大token数
    enable_str_match: bool = True # 启用字符串匹配
    fixed_docs_per_query: int = 1 # 每个query固定的文档数
    jump_exact_match: bool = False # 跳过精确匹配
```

### InferenceConfig

```python
@dataclass
class InferenceConfig:
    model_path: str                    # 模型路径
    inference_type: str = "local"      # 推理类型: local, huggingface, togetherai
    api_key: Optional[str] = None      # API密钥
    api_model_name: Optional[str] = None # API模型名称
    max_new_tokens: int = 512          # 最大新生成token数
    temperature: float = 0.2           # 采样温度
    top_p: float = 0.95               # Top-p采样
```

### PipelineConfig

```python
@dataclass
class PipelineConfig:
    # 输入输出
    queries_file: str           # 查询文件路径
    contexts_output: str        # Context输出路径
    final_output: str          # 最终结果输出路径
    
    # 阶段配置
    context_config: ContextGenerationConfig
    inference_config: InferenceConfig
    
    # 多worker设置
    num_workers: int = 1                      # Worker数量
    max_samples_per_worker: Optional[int] = None # 每个worker最大样本数
    
    # 控制设置
    skip_context_generation: bool = False     # 跳过context生成
    skip_inference: bool = False             # 跳过推理
    verbose: bool = False                    # 详细日志
    
    # 性能设置
    enable_progress_bar: bool = True         # 启用进度条
    save_intermediate_results: bool = True   # 保存中间结果
    cleanup_worker_files: bool = True        # 清理worker文件
```

## 📚 使用示例

### 示例1: 本地模型完整流水线

```python
from utils.RAGutils.complete_pipeline import *

# 配置
context_config = create_context_config(
    corpus_path="/datanfs4/chenrongyi/data/docs",
    corpus_type="docstring",
    max_documents=3,
    max_tokens=2000
)

inference_config = create_inference_config(
    model_path="/datanfs4/chenrongyi/models/Llama-3.1-8B-Instruct",
    inference_type="local",
    max_new_tokens=512,
    temperature=0.1
)

pipeline_config = create_pipeline_config(
    queries_file="data/queries.json",
    contexts_output="data/contexts.jsonl",
    final_output="data/results.jsonl",
    context_config=context_config,
    inference_config=inference_config,
    verbose=True
)

# 执行
pipeline = CompletePipeline(pipeline_config)
results = pipeline.run_complete_pipeline()
```

### 示例2: API推理

```python
inference_config = create_inference_config(
    model_path="",  # API不需要本地模型路径
    inference_type="togetherai",
    api_key="your_api_key_here",
    api_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    max_new_tokens=512,
    temperature=0.7
)

# 其他配置相同...
```

### 示例3: 只生成Context（跳过推理）

```python
pipeline_config = create_pipeline_config(
    queries_file="data/queries.json",
    contexts_output="data/contexts.jsonl",
    final_output="data/results.jsonl",  # 不会使用
    context_config=context_config,
    inference_config=inference_config,
    skip_inference=True  # 跳过推理阶段
)

pipeline = CompletePipeline(pipeline_config)
contexts = pipeline.run_complete_pipeline()  # 只返回contexts
```

### 示例4: 只运行推理（使用已有Context）

```python
pipeline_config = create_pipeline_config(
    queries_file="",  # 不需要
    contexts_output="data/existing_contexts.jsonl",  # 已存在的contexts
    final_output="data/results.jsonl",
    context_config=context_config,
    inference_config=inference_config,
    skip_context_generation=True  # 跳过context生成
)

pipeline = CompletePipeline(pipeline_config)
results = pipeline.run_complete_pipeline()
```

### 示例5: 自定义Queries分步执行

```python
# 自定义queries
custom_queries = [
    {
        "id": "pandas_q1",
        "query": "How to read CSV with pandas?",
        "target_api": "pandas.read_csv",
        "dependencies": {"pandas": "1.5.0"}
    },
    {
        "id": "numpy_q1", 
        "query": "How to create numpy array?",
        "target_api": "numpy.array",
        "dependencies": {"numpy": "1.21.0"}
    }
]

pipeline = CompletePipeline(pipeline_config)

# 步骤1: 生成contexts
print("🔍 Generating contexts...")
contexts = pipeline.generate_contexts(custom_queries)

# 检查contexts
successful_contexts = [ctx for ctx in contexts if ctx.get('success', False)]
print(f"✅ Generated {len(successful_contexts)} successful contexts")

# 步骤2: 运行推理
print("🧠 Running inference...")
results = pipeline.run_inference(contexts)

# 查看结果
successful_results = [r for r in results if r.get('success', False)]
print(f"✅ Generated {len(successful_results)} successful results")

# 打印统计信息
pipeline.print_statistics()
```

## 🔧 多Worker支持

### 使用外部进程协调（推荐）

```bash
# 使用torchrun启动多worker
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    utils/RAGutils/complete_pipeline.py \
    --queries_file data/queries.json \
    --contexts_output data/contexts.jsonl \
    --final_output data/results.jsonl \
    --corpus_path /path/to/docs \
    --model_path /path/to/model \
    --num_workers 4
```

### 程序内部多Worker配置

```python
pipeline_config = create_pipeline_config(
    queries_file="data/queries.json",
    contexts_output="data/contexts.jsonl", 
    final_output="data/results.jsonl",
    context_config=context_config,
    inference_config=inference_config,
    num_workers=4,  # 多worker模式
    max_samples_per_worker=100,  # 限制每个worker的样本数
    verbose=True
)

# 注意：程序内部的多worker需要外部进程协调才能真正并行
pipeline = CompletePipeline(pipeline_config)
results = pipeline.run_complete_pipeline()
```

## 📊 统计信息

流水线提供详细的统计信息：

```python
# 获取统计信息
stats = pipeline.get_statistics()
print(stats)

# 或者直接打印
pipeline.print_statistics()
```

输出示例：
```
================================================================================
COMPLETE PIPELINE STATISTICS  
================================================================================

📋 CONTEXT GENERATION:
  Total queries: 100
  Successful contexts: 95
  Failed contexts: 5
  Success rate: 95.0%
  Total time: 45.23s
  Average retrieval time: 0.45s

🧠 INFERENCE:
  Total contexts: 95
  Successful inferences: 92
  Failed inferences: 3
  Success rate: 96.8%
  Total time: 123.45s
  Average inference time: 1.34s

⏱️  OVERALL PIPELINE:
  Total pipeline time: 168.68s
  End-to-end success rate: 92.0%
================================================================================
```

## 🛠️ API参考

### 主要类

#### `CompletePipeline`

**方法:**
- `__init__(config: PipelineConfig)`: 初始化流水线
- `run_complete_pipeline(queries: Optional[List[Dict]] = None) -> List[Dict]`: 运行完整流水线
- `generate_contexts(queries: List[Dict], use_multi_worker: bool = None) -> List[Dict]`: 生成contexts
- `run_inference(contexts: List[Dict], use_multi_worker: bool = None) -> List[Dict]`: 运行推理
- `get_statistics() -> Dict`: 获取统计信息
- `print_statistics()`: 打印统计信息

#### 配置创建工具

- `create_context_config(...)`: 创建context生成配置
- `create_inference_config(...)`: 创建推理配置  
- `create_pipeline_config(...)`: 创建流水线配置

### 数据格式

#### Query格式
```json
{
  "id": "unique_id",
  "query": "用户查询文本",
  "target_api": "目标API",
  "dependencies": {"package": "version"}
}
```

#### Context格式
```json
{
  "id": "unique_id",
  "query": "用户查询",
  "context": "检索到的相关文档",
  "retrieval_method": "检索方法",
  "success": true,
  "retrieval_time": 0.5
}
```

#### Result格式
```json
{
  "id": "unique_id", 
  "query": "用户查询",
  "answer": "生成的回答",
  "inference_time": 1.2,
  "total_time": 1.7,
  "success": true
}
```

## 💡 最佳实践

### 1. 配置管理
- 使用配置文件管理复杂配置
- 为不同环境创建不同的配置模板
- 使用环境变量管理敏感信息（如API密钥）

### 2. 性能优化
- 根据硬件资源调整worker数量
- 使用适当的batch size和max_tokens
- 监控GPU内存使用情况

### 3. 错误处理
- 启用verbose模式进行调试
- 保存中间结果以便错误恢复
- 定期检查统计信息

### 4. 资源管理
- 使用max_samples_per_worker限制资源使用
- 定期清理临时文件
- 监控磁盘空间使用

### 5. 扩展性
- 使用模块化配置便于扩展
- 保持接口的一致性
- 考虑向后兼容性

## 🔍 故障排除

### 常见问题

1. **导入错误**
   ```python
   # 确保路径正确
   import sys
   sys.path.append('/path/to/your/project')
   ```

2. **模型加载失败**
   - 检查模型路径是否正确
   - 确保有足够的GPU内存
   - 验证模型格式是否兼容

3. **API调用失败**
   - 验证API密钥是否有效
   - 检查网络连接
   - 确认API配额和限制

4. **内存不足**
   - 减少batch_size
   - 降低max_tokens
   - 使用更小的模型

### 调试技巧

1. **启用详细日志**
   ```python
   pipeline_config.verbose = True
   ```

2. **使用小数据集测试**
   ```python
   pipeline_config.max_samples_per_worker = 10
   ```

3. **分步执行定位问题**
   ```python
   # 先只运行context生成
   pipeline_config.skip_inference = True
   ```

## 📝 更新日志

### v1.0.0
- 初始版本发布
- 支持完整的RAG流水线
- 多worker并行处理
- 详细的统计和监控

---

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个流水线！

## 📄 许可证

本项目采用MIT许可证。 