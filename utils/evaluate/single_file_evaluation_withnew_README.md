# Single File Evaluation with New Test Statistics

## 概述

`single_file_evaluation_withnew.py` 是一个用于对指定预测文件进行快速评估的脚本，它使用 `testPassBCB_new` 获取详细的测试统计信息，避免处理大量无关文件，并提供全面的评估结果分析。

## 主要功能

- **快速单文件评估**：针对特定预测文件进行评估，无需处理整个数据集
- **详细测试统计**：提供测试用例级别的统计信息，包括通过率、失败率等
- **多进程并行处理**：支持多进程并行评估，提高处理速度
- **智能文件名解析**：自动从文件名提取任务类型、知识类型等信息
- **多种清理函数**：支持多种代码清理策略
- **详细错误分析**：提供完整的错误追踪和调试信息
- **成比例正确性统计**：计算每个任务的部分正确性得分
## 基础环境配置和运行
```python
export PYTHONPATH=your_working_folder

conda activate /datanfs2/chenrongyi/miniconda3/envs/mvBCBbuilder

bash utils/evaluate/single_file_evaluation_new.sh
```

## 命令行参数

### 必需参数

- `--prediction_file`：要评估的预测文件路径（支持 JSON 和 JSONL 格式）

### 可选参数

#### 基本配置
- `--benchmark_dir`：基准数据目录（默认：`data/output/VersiBCB_Benchmark`）
- `--benchmark_file`：指定基准文件名（默认：`vace_datas.json`）
- `--ban_deprecation`：是否禁用过时API（可选，默认根据文件名自动确定）

#### 输出控制
- `--output_file`, `-of`：详细结果输出文件路径（JSON格式，可选）
- `--log_file`：日志文件路径（可选，默认根据预测文件名自动生成）
- `--no_save`：不保存详细结果文件

#### 性能优化
- `--num_processes`：并行进程数量（默认：4）
- `--max_samples`：最大评估样本数量（用于快速测试）

#### 代码处理
- `--clean_func`, `-cf`：清理函数选择
  - `basic`：基础清理函数
  - `loose`：宽松清理函数
  - `lora`：LoRA清理函数（默认）
  - `review`：评审清理函数
  - `errorfix`：错误修复清理函数

#### 任务筛选
- `--task_ids`：指定要检测的task ID，用逗号分隔（例如：`"1,2,3"` 或 `"task_1,task_2"`）
- `--task_ids_file`, `-tidf`：包含task IDs的文件路径，文件内容为JSON数组格式

## 使用示例

### 1. 基本使用
```bash
python utils/evaluate/single_file_evaluation_withnew.py \
    --prediction_file data/output/predictions/my_model_predictions.jsonl
```

### 2. 指定基准文件和配置
```bash
python utils/evaluate/single_file_evaluation_withnew.py \
    --prediction_file data/output/predictions/vscc_predictions.jsonl \
    --benchmark_file vscc_datas_for_warning.json \
    --ban_deprecation true \
    --num_processes 8
```

### 3. 使用特定清理函数
```bash
python utils/evaluate/single_file_evaluation_withnew.py \
    --prediction_file data/output/predictions/model_output.json \
    --clean_func review \
    --output_file results/detailed_results.json
```

### 4. 评估特定任务
```bash
# 通过命令行指定task IDs
python utils/evaluate/single_file_evaluation_withnew.py \
    --prediction_file data/output/predictions/test_predictions.jsonl \
    --task_ids "task_1,task_5,task_10"

# 通过文件指定task IDs
python utils/evaluate/single_file_evaluation_withnew.py \
    --prediction_file data/output/predictions/test_predictions.jsonl \
    --task_ids_file task_ids_to_evaluate.json
```

### 5. 快速测试（限制样本数量）
```bash
python utils/evaluate/single_file_evaluation_withnew.py \
    --prediction_file data/output/predictions/large_dataset.jsonl \
    --max_samples 100 \
    --num_processes 2
```

### 6. 完整配置示例
```bash
python utils/evaluate/single_file_evaluation_withnew.py \
    --prediction_file data/output/approach_eval/BASELINE/Llama-3.1-8B/versibcb_vace_Llama-3.1-8B.jsonl \
    --benchmark_dir data/output/VersiBCB_Benchmark \
    --benchmark_file vace_datas_for_warning.json \
    --ban_deprecation true \
    --clean_func lora \
    --num_processes 8 \
    --output_file results/llama_detailed_results.json \
    --log_file logs/llama_evaluation.log
```

## 输入文件格式

### 预测文件格式
支持 JSON 和 JSONL 格式，每个数据项需要包含：
- `id`：任务ID
- `pred` 或 `answer` 或 `model_output`：模型预测结果

#### JSON 格式示例
```json
[
    {
        "id": "task_1",
        "pred": "def function_name():\n    return result"
    },
    {
        "id": "task_2", 
        "answer": "class MyClass:\n    pass"
    }
]
```

#### JSONL 格式示例
```jsonl
{"id": "task_1", "pred": "def function_name():\n    return result"}
{"id": "task_2", "model_output": "class MyClass:\n    pass"}
```

### Task IDs 文件格式
```json
["task_1", "task_5", "task_10", "task_15"]
```

## 输出说明

### 1. 控制台输出
脚本会在控制台显示详细的评估结果，包括：

```
==============================================================
EVALUATION RESULTS WITH TEST STATISTICS
==============================================================
File Information:
  Task Type:        VACE
  Knowledge Type:   Doc
  Ban Deprecation:  True
  Benchmark File:   vace_datas_for_warning.json

Task-Level Results:
  Total Tasks:      100
  Passed Tasks:     85
  Failed Tasks:     15
  Error Tasks:      2
  Task Pass Rate:   0.8500 (85.00%)

Test Case-Level Statistics:
  Tasks with Stats: 98/100
  Total Test Cases: 450
  Passed Tests:     380
  Failed Tests:     60
  Error Tests:      8
  Skipped Tests:    2
  Test Case Pass Rate: 0.8444 (84.44%)

Proportional Correctness Statistics:
  Total Proportional Correct: 78.5000
  Avg Proportional Correct:   0.7850 (78.50%)
  Avg Tests/Task:   4.6
==============================================================
```

### 2. 详细结果文件
如果未指定 `--no_save`，脚本会生成详细的JSON结果文件，包含：

```json
{
  "evaluation_info": {
    "source_file": "path/to/prediction.jsonl",
    "knowledge_type": "Doc",
    "task_type": "VACE",
    "ban_deprecation": true,
    "benchmark_file": "vace_datas_for_warning.json",
    "evaluation_time": "2024-01-15 10:30:00"
  },
  "task_level_statistics": {
    "total_tasks": 100,
    "passed_tasks": 85,
    "failed_tasks": 15,
    "error_tasks": 2,
    "task_pass_rate": 0.85
  },
  "test_case_level_statistics": {
    "total_tasks": 100,
    "tasks_with_stats": 98,
    "total_test_cases": 450,
    "total_passed_tests": 380,
    "total_failed_tests": 60,
    "total_error_tests": 8,
    "total_skipped_tests": 2,
    "test_case_pass_rate": 0.8444,
    "total_proportional_correct": 78.5,
    "avg_proportional_correct": 0.785
  },
  "detailed_results": [
    {
      "task_id": "task_1",
      "passed": true,
      "error_type": null,
      "message": "Success",
      "prediction": "def example():\n    return 'hello'",
      "cleaned_code": "def example():\n    return 'hello'",
      "test_code": "assert example() == 'hello'",
      "test_stats": {
        "total_tests": 5,
        "passed_tests": 5,
        "failed_tests": 0,
        "error_tests": 0,
        "skipped_tests": 0,
        "success_rate": 1.0,
        "execution_time": 0.123
      }
    }
  ]
}
```

### 3. 日志文件
包含完整的执行日志，包括：
- 执行时间戳
- 文件加载信息
- 评估进度
- 错误信息
- 统计结果

## 文件名自动解析

脚本可以从预测文件名自动提取信息：

- **任务类型**：从文件名中的 `vace` 或 `vscc` 识别
- **知识类型**：从文件名中的 `docstring`/`doc` 或 `srccode` 识别
- **禁用过时API**：从文件名中的 `bd` 标识识别
- **基准文件**：根据任务类型和是否禁用过时API自动选择

示例文件名解析：
- `versibcb_vace_bd_docstring_emb_local_4000.jsonl`
  - 任务类型：VACE
  - 知识类型：Doc
  - 禁用过时API：是
  - 基准文件：vace_datas_for_warning.json

## 性能优化建议

### 1. 进程数量设置
- **CPU密集型任务**：设置进程数为CPU核心数
- **I/O密集型任务**：可以设置为CPU核心数的1.5-2倍
- **内存限制**：每个进程会占用一定内存，根据可用内存调整

### 2. 样本数量控制
- 使用 `--max_samples` 进行快速测试
- 使用 `--task_ids` 或 `--task_ids_file` 评估特定任务

### 3. 清理函数选择
- `basic`：最快，适合简单代码
- `lora`：平衡性能和准确性，推荐默认选择
- `review`：更严格的清理，适合复杂代码

## 错误处理

脚本提供多级错误处理：
1. **文件不存在**：检查预测文件和基准文件是否存在
2. **格式错误**：验证JSON/JSONL格式
3. **测试超时**：设置200秒超时时间
4. **进程错误**：捕获并报告子进程错误
5. **完整追踪**：提供详细的错误追踪信息

## 常见问题

### Q: 如何选择合适的清理函数？
A: 根据你的模型输出特点选择：
- 输出格式规范：使用 `basic`
- 输出包含额外文本：使用 `lora` 或 `review`
- 需要错误修复：使用 `errorfix`

### Q: 评估速度太慢怎么办？
A: 可以采取以下措施：
- 增加 `--num_processes` 参数
- 使用 `--max_samples` 限制样本数量
- 使用 `--task_ids` 评估关键任务

### Q: 如何理解成比例正确性？
A: 成比例正确性反映每个任务中测试用例的通过比例，即使任务整体失败，也能体现部分正确性。

### Q: 输出文件太大怎么办？
A: 使用 `--no_save` 跳过详细结果保存，只查看控制台输出。

