# 并发Query Generation使用说明

## 功能说明

这些脚本实现了对代码错误信息的并发query generation，每个ID的处理结果以JSONL格式保存（每行一个JSON对象）。

## 脚本文件

### 1. `genQueryConcurrentTest.py` - 测试版（推荐先使用）
- 只处理前5个数据项
- 显示详细的处理过程和结果
- 用于验证API连接和功能是否正常

### 2. `genQueryConcurrent.py` - 完整版
- 处理所有77个数据项
- 提供进度显示和统计信息
- 生产环境使用

## 使用方法

### 步骤1: 测试运行
```bash
# 先运行测试版，确保功能正常
python utils/RAGutils/genQA/genQueryConcurrentTest.py
```

### 步骤2: 完整运行
```bash
# 功能验证无误后，运行完整版
python utils/RAGutils/genQA/genQueryConcurrent.py
```

## 输出格式

### JSONL文件格式
每行都是一个独立的JSON对象，包含以下字段：

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
  "query_count": 1,
  "error_count": 2,
  "processing_time": 3.45
}
```

### 状态类型
- `success`: 成功生成查询
- `no_queries`: 未生成查询（通常是简单错误或误报）
- `error`: 数据错误（找不到代码或错误信息）
- `api_error`: API调用错误
- `exception`: 处理异常

## 配置参数

### 并发控制
```python
max_workers = 3  # 可调整并发数，建议2-5之间
```

### 输出文件
- 测试版: `data/temp/query_results_test.jsonl`
- 完整版: `data/temp/query_results_concurrent.jsonl`

## 性能优化

### 并发数建议
- 测试环境: 2-3个并发
- 生产环境: 3-5个并发
- 不建议超过5个，可能触发API限制

### 估算处理时间
- 平均每个ID处理时间: 2-5秒
- 77个数据项预计总时间: 3-8分钟（取决于并发数和网络情况）

## 结果分析

脚本会自动统计和显示：
- 总处理时间
- 成功/失败数量
- 各种状态的分布
- 生成的总查询数

## 错误处理

脚本包含完善的错误处理机制：
1. API调用失败会重试
2. JSON解析失败会尝试提取有效部分
3. 单个ID处理失败不会影响其他ID
4. 所有错误都会记录在结果中

## 读取结果

```python
import json

# 读取JSONL文件
results = []
with open("data/temp/query_results_concurrent.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            results.append(json.loads(line))

# 过滤成功的结果
successful_results = [r for r in results if r['status'] == 'success']

# 获取所有查询
all_queries = []
for result in successful_results:
    if result.get('queries'):
        all_queries.extend(result['queries'])

print(f"总共生成了 {len(all_queries)} 个查询")
```

## 注意事项

1. 确保QDD API密钥配置正确
2. 网络连接稳定
3. 有足够的磁盘空间存储结果
4. 不要同时运行多个实例，避免API限制
5. 建议先运行测试版验证功能 