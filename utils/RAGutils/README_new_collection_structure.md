# RAG Collection 构建和存储结构升级

## 概述

本次升级对RAG系统的collection构建和存储结构进行了重大改进，主要包括：

1. **任务导向的collection构建**：可以一次性构建特定任务所需的所有collections
2. **新的存储结构**：每个dependency组合有独立的文件夹，便于管理和查看
3. **向后兼容**：保持与现有代码的兼容性

## 新功能

### 1. 任务导向的Collection构建

新增了 `build_all_task_collections` 方法，可以：
- 分析数据集中的所有dependency组合
- 根据任务类型（VACE/VSCC）确定需要的dependency字段
- 一次性构建所有需要的collections
- 支持多进程并行构建

### 2. 新的存储结构

#### 旧结构
```
{RAG_COLLECTION_BASE}/{KNOWLEDGE_TYPE}/{EMBEDDING_SOURCE}/
├── collection_hash1/
├── collection_hash2/
└── ...
```

#### 新结构
```
{RAG_COLLECTION_BASE}/{KNOWLEDGE_TYPE}/{EMBEDDING_SOURCE}/
├── numpy_1.20.0_pandas_1.2.0/
│   └── collection_hash1/
├── numpy_1.21.0_pandas_1.3.0/
│   └── collection_hash2/
└── ...
```

**优势：**
- 文件夹名称直观显示dependency组合
- 便于用户查看和管理特定dependency的collection
- 每个collection有独立的ChromaDB实例，避免冲突

### 3. 智能回退机制

RAGRetriever现在会：
1. 首先尝试从新存储结构加载collection
2. 如果找不到，自动回退到旧结构
3. 保证现有代码的正常运行

## 使用方法

### 1. 构建特定任务的所有Collections

```python
from utils.RAGutils.buildAllTaskCollections import build_all_task_collections

# 为VACE任务构建所有collections
collections = build_all_task_collections(
    dataset_path="data/VersiBCB_Benchmark/vace_datas.json",
    dataset_name="VersiBCB",
    task_name="VACE",
    ban_deprecation=False,
    force_rebuild=False,
    num_processes=1
)
```

### 2. 命令行使用

```bash
# 为VACE任务构建collections
python utils/RAGutils/buildAllTaskCollections.py \
    --dataset VersiBCB \
    --task VACE \
    --batch-size 250 \
    --num-processes 1

# 为VSCC任务构建collections（使用ban deprecation数据）
python utils/RAGutils/buildAllTaskCollections.py \
    --dataset VersiBCB \
    --task VSCC \
    --ban-deprecation \
    --force
```

### 3. 在现有代码中使用

现有的RAGRetriever代码无需修改，会自动使用新的存储结构：

```python
# 现有代码保持不变
rag_retriever = RAGContextRetriever(chroma_client, embedding_args, knowledge_type)
context = rag_retriever.retrieve_context(data, dataset, task, max_token_length)
```

## 文件结构

```
utils/RAGutils/
├── CollectionBuilder.py              # 增强的collection构建器
├── buildAllTaskCollections.py        # 新的任务导向构建脚本
├── RAGRetriever.py                   # 更新的检索器（支持新存储结构）
├── buildCollection.py               # 原有的构建脚本（保持兼容）
└── README_new_collection_structure.md # 本文档

examples/
└── build_task_collections_example.py # 使用示例
```

## 配置参数

### CollectionBuilder新增方法

#### `build_all_task_collections`

**参数：**
- `dataset_path`: 数据集文件路径
- `dataset_name`: 数据集名称 ("VersiBCB" 或 "VersiCode")
- `task_name`: 任务名称 ("VACE" 或 "VSCC")
- `ban_deprecation`: 是否使用ban deprecation版本的数据
- `force_rebuild`: 是否强制重建已有集合
- `num_processes`: 使用的进程数量

**返回：**
- `Dict[str, str]`: collection映射 {依赖项哈希: 集合名称}

### 存储路径生成

#### `get_dependency_folder_name`
生成格式：`"_".join(sorted(dependency.items()))`

示例：
```python
dependencies = {"pandas": "1.3.0", "numpy": "1.21.0"}
folder_name = "numpy_1.21.0_pandas_1.3.0"
```

#### `get_collection_path`
完整路径：`{RAG_COLLECTION_BASE}/{KNOWLEDGE_TYPE}/{EMBEDDING_SOURCE}/{folder_name}/{collection_hash}`

## 性能优化

### 1. 独立ChromaDB实例
- 每个collection使用独立的ChromaDB客户端
- 避免大型数据库的性能问题
- 支持并行访问

### 2. 智能跳过机制
- 自动检测已存在的collections
- 支持增量构建
- 减少重复计算

### 3. 批处理优化
- 保持原有的批处理机制
- 支持可配置的批处理大小
- 内存使用优化

## 迁移指南

### 对于新项目
直接使用新的构建方法：
```bash
python utils/RAGutils/buildAllTaskCollections.py --dataset VersiBCB --task VACE
```

### 对于现有项目
1. 现有collections继续可用（向后兼容）
2. 新构建的collections会使用新结构
3. 可以选择性迁移到新结构：
   ```bash
   # 强制重建所有collections到新结构
   python utils/RAGutils/buildAllTaskCollections.py --dataset VersiBCB --task VACE --force
   ```

## 故障排除

### 1. Collection找不到
- 检查存储路径是否正确
- 确认dependency组合是否匹配
- 查看日志中的详细错误信息

### 2. 性能问题
- 调整批处理大小（`--batch-size`）
- 使用多进程构建（`--num-processes`）
- 检查磁盘空间和内存使用

### 3. 兼容性问题
- 确认配置文件路径正确
- 检查embedding模型配置
- 验证数据集格式

## 示例

完整的使用示例请参考：`examples/build_task_collections_example.py`

```bash
python examples/build_task_collections_example.py
```

## 注意事项

1. **磁盘空间**：新结构可能需要更多磁盘空间（每个collection独立存储）
2. **并发访问**：多个进程可以同时访问不同的collections
3. **备份策略**：建议定期备份collection数据
4. **版本管理**：dependency版本变化会创建新的collection

## 未来计划

1. **多进程优化**：完善新存储结构下的多进程构建
2. **自动清理**：添加过期collection的自动清理功能
3. **监控工具**：开发collection使用情况的监控工具
4. **压缩存储**：研究collection数据的压缩存储方案 