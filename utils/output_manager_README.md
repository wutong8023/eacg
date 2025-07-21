# OutputManager - 统一输出管理系统

## 概述

`OutputManager` 是一个统一的输出管理类，用于处理所有方法（MoE、RAG、Memory、LoRA）的输出路径生成和配置文件管理。它提供了标准化的输出结构，自动处理文件名冲突，并生成详细的配置文件记录实验参数。

## 主要功能

1. **统一的输出目录结构**：所有输出都保存在 `output/approach_eval/{approach}/` 目录下
2. **自动文件名生成**：根据实验参数自动生成描述性文件名
3. **文件冲突处理**：自动添加数字后缀避免覆盖已有文件
4. **配置文件生成**：为每次实验生成详细的配置文件，便于结果追溯

## 输出目录结构

```
output/
└── approach_eval/
    ├── MoE/
    │   ├── versibcb_vace_em-llm-model.jsonl
    │   └── versibcb_vace_em-llm-model_config.json
    ├── RAG/
    │   ├── versibcb_vace_srccodes_emb_local_8000_Mistral-7B-Instruct-v0.2.jsonl
    │   └── versibcb_vace_srccodes_emb_local_8000_Mistral-7B-Instruct-v0.2_config.json
    ├── Memory/
    │   ├── versibcb_vscc_deepseek-coder-6.7b-instruct.jsonl
    │   └── versibcb_vscc_deepseek-coder-6.7b-instruct_config.json
    └── LoRA/
        ├── versibcb_vace_10_appendsrcDep_CodeLlama-7b-hf.jsonl
        └── versibcb_vace_10_appendsrcDep_CodeLlama-7b-hf_config.json
```

## 使用方法

### 1. 创建OutputManager实例

```python
from utils.output_manager import OutputManager

output_manager = OutputManager()
```

### 2. 确定方法类型

```python
# MoE方法
approach = output_manager.get_approach_from_context(model_type="em-llm")

# RAG方法
approach = output_manager.get_approach_from_context(has_corpus_type=True)

# LoRA方法
approach = output_manager.get_approach_from_context(is_lora=True)

# Memory方法（默认）
approach = output_manager.get_approach_from_context()
```

### 3. 生成基础文件名

```python
# RAG示例
base_filename = output_manager.generate_base_filename(
    dataset="versibcb_vace",
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    approach="RAG",
    corpus_type="srccodes",
    embedding_source="local",
    max_tokens=8000
)

# LoRA示例
base_filename = output_manager.generate_base_filename(
    dataset="versibcb_vace",
    model_name="codellama/CodeLlama-7b-hf",
    approach="LoRA",
    max_dependency_num=10,
    append_srcDep=True
)
```

### 4. 获取输出路径

```python
output_path, config_path = output_manager.get_output_path_and_config(
    approach=approach,
    base_filename=base_filename
)
```

### 5. 生成配置文件

```python
# 生成配置数据
config_data = output_manager.generate_config(
    approach=approach,
    args=args,  # 参数对象
    rag_config=rag_config  # RAG配置（可选）
)

# 保存配置文件
output_manager.save_config(config_path, config_data)
```

## 配置文件格式

生成的配置文件包含以下信息：

```json
{
  "experiment_info": {
    "approach": "RAG",
    "timestamp": "2024-01-01 12:00:00",
    "dataset": "VersiBCB",
    "task": "VACE",
    "ban_deprecation": false
  },
  "model_config": {
    "model_path": "mistralai/Mistral-7B-Instruct-v0.2",
    "inference_type": "local",
    "precision": "fp16"
  },
  "generation_config": {
    "max_tokens": 8000,
    "max_new_tokens": 1024,
    "temperature": 0.2
  },
  "rag_config": {
    "corpus_type": "srccodes",
    "embedding_source": "local",
    "rag_document_num": 10
  },
  "processing_config": {
    "num_workers": 1,
    "skip_generation": false
  }
}
```

## 文件名格式说明

### MoE/Memory
- 格式：`{dataset}_{model_name}.jsonl`
- 示例：`versibcb_vace_Llama-2-7b-hf.jsonl`

### RAG
- 格式：`{dataset}_{corpus_type}_emb_{embedding_source}_{max_tokens}_{model_name}.jsonl`
- 示例：`versibcb_vace_srccodes_emb_local_8000_Mistral-7B-Instruct-v0.2.jsonl`

### LoRA
- 格式：`{dataset}_{max_dependency_num}[_appendsrcDep]_{model_name}.jsonl`
- 示例：`versibcb_vace_10_appendsrcDep_CodeLlama-7b-hf.jsonl`

## 优势

1. **代码复用**：所有方法使用统一的输出管理逻辑，减少重复代码
2. **可维护性**：集中管理输出逻辑，便于维护和更新
3. **可追溯性**：自动生成的配置文件记录所有实验参数
4. **防止覆盖**：自动处理文件名冲突，避免意外覆盖结果
5. **清晰的组织**：按方法分类存储，便于对比不同方法的结果

## 扩展性

OutputManager设计为易于扩展，可以：
- 添加新的方法类型
- 自定义文件名格式
- 扩展配置文件内容
- 集成新的实验参数

通过使用OutputManager，我们实现了实验输出的标准化管理，提高了代码质量和实验的可重复性。 