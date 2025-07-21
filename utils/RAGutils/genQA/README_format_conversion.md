# 格式转换使用说明

## 功能说明

这些脚本用于将concurrent处理结果转换为versibcb的统一格式，确保输出格式与现有的versibcb格式兼容。

## 脚本文件

### 1. `convertToVersiBCBFormatTest.py` - 测试版（推荐先使用）
- 使用测试数据验证格式转换
- 显示详细的转换过程和结果对比
- 用于验证转换逻辑是否正确

### 2. `convertToVersiBCBFormat.py` - 完整版
- 处理完整的concurrent结果
- 生成versibcb格式的输出文件
- 生产环境使用

## 格式转换说明

### 输入格式 (Concurrent JSONL)
```json
{
  "id": 336,
  "timestamp": "2024-01-01 12:00:00",
  "status": "success",
  "queries": [
    {
      "error_ids": ["error_0001", "error_0002"],
      "target_api_path": "matplotlib.pyplot.subplots",
      "query_content": "What is the correct way to use plt.subplots with ax parameter in seaborn functions?",
      "explanation": "The error suggests ax parameter issue with seaborn.pairplot, need to clarify correct usage pattern.",
      "original_id": 336
    }
  ],
  "query_count": 2,
  "error_count": 2,
  "processing_time": 3.45
}
```

### 输出格式 (VersiBCB)
```json
{
  "336": {
    "queries": [
      {
        "query": "What is the correct way to use plt.subplots with ax parameter in seaborn functions?",
        "target_api": "matplotlib.pyplot.subplots"
      }
    ],
    "raw_response": "Generated 2 queries for ID 336",
    "original_data": {
      "description": "原始任务描述...",
      "origin_dependency": "",
      "target_dependency": {
        "matplotlib": "3.4.3",
        "python": "3.8"
      }
    }
  }
}
```

## 使用方法

### 步骤1: 测试格式转换
```bash
# 先运行测试版，验证转换逻辑
python utils/RAGutils/genQA/convertToVersiBCBFormatTest.py
```

### 步骤2: 完整格式转换
```bash
# 功能验证无误后，运行完整版
python utils/RAGutils/genQA/convertToVersiBCBFormat.py
```

## 输出文件

### 测试版输出
- 测试结果: `data/temp/versibcb_format_test.json`
- 样本文件: `data/temp/versibcb_format_queries_sample.json`

### 完整版输出
- 完整结果: `data/temp/versibcb_format_queries.json`

## 转换逻辑

### 字段映射
| Concurrent字段 | VersiBCB字段 | 说明 |
|---------------|-------------|------|
| `id` | 字典key | 转换为字符串作为字典key |
| `queries[].query_content` | `queries[].query` | 查询内容 |
| `queries[].target_api_path` | `queries[].target_api` | 目标API路径 |
| - | `raw_response` | 生成描述信息 |
| - | `original_data` | 从vscc原始数据获取 |

### 数据过滤
- 只处理状态为 `success` 的结果
- 跳过找不到原始vscc数据的ID
- 保留所有有效的查询

### 原始数据补充
- 从 `data/VersiBCB_Benchmark/vscc_datas.json` 获取原始信息
- 包括描述、依赖关系等元数据

## 验证方法

### 1. 格式检查
```python
import json

# 读取转换结果
with open("data/temp/versibcb_format_test.json", 'r') as f:
    data = json.load(f)

# 检查格式
for id_str, entry in data.items():
    assert "queries" in entry
    assert "raw_response" in entry
    assert "original_data" in entry
    
    for query in entry["queries"]:
        assert "query" in query
        assert "target_api" in query

print("格式验证通过!")
```

### 2. 与原始格式对比
脚本会自动与原始versibcb格式进行对比，确保：
- 字段名称一致
- 数据结构兼容
- 查询格式正确

## 注意事项

1. **依赖文件**: 确保以下文件存在：
   - `data/temp/query_results_concurrent.jsonl` (或测试文件)
   - `data/VersiBCB_Benchmark/vscc_datas.json`

2. **数据完整性**: 转换过程会跳过失败的结果，确保最终输出质量

3. **格式兼容**: 输出格式与现有versibcb格式完全兼容

4. **性能**: 转换过程很快，主要是文件I/O操作

## 错误处理

- 输入文件不存在时会提示错误
- 找不到原始vscc数据时会跳过并警告
- 格式错误时会记录并继续处理其他数据

## 示例输出

转换后的文件可以直接用于versibcb相关任务，格式示例：

```json
{
  "336": {
    "queries": [
      {
        "query": "What is the correct way to use plt.subplots with ax parameter in seaborn functions?",
        "target_api": "matplotlib.pyplot.subplots"
      }
    ],
    "raw_response": "Generated 1 queries for ID 336",
    "original_data": {
      "description": "Create a visualization showing the relationship between two variables...",
      "origin_dependency": "",
      "target_dependency": {
        "matplotlib": "3.4.3",
        "seaborn": "0.11.2"
      }
    }
  }
}
``` 