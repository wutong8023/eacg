# Pyright Parser

一个专门用于从字符串中直接解析Python代码的pyright分析器，返回原始诊断结果而不是格式化后的错误信息。

## 功能特性

- 🔍 **直接字符串解析**: 支持从代码字符串直接进行静态分析
- 📊 **原始结果返回**: 返回pyright的原始诊断信息，保持完整的数据结构
- 🎯 **格式化输出**: 支持JSON、文本、Markdown等多种输出格式
- 🔧 **错误信息格式化**: 将诊断信息格式化为标准的错误信息格式
- ⚡ **多线程支持**: 支持批量分析多个文件
- 🐍 **Conda环境集成**: 自动在指定的conda环境中安装和运行pyright
- 📝 **详细日志**: 支持详细的日志记录和调试信息

## 安装要求

- Python 3.7+
- pyright (会自动安装)
- conda (用于环境管理)

## 基本用法

### 1. 作为Python模块使用

```python
from pyright_parser import PyrightParser

# 创建解析器实例
parser = PyrightParser(enable_logging=True)

# 要分析的代码
code = """
import matplotlib.pyplot as plt

def plot_data(x, y):
    plt.plot(x, y, color='red', size=12)  # size不是有效参数
    plt.title(title_var)  # 未定义的变量
    return plt.gcf()
"""

# 目标依赖信息
target_dependency = {
    "matplotlib": "3.5.0",
    "seaborn": "0.11.0"
}

# 获取格式化的错误信息
error_infos = parser.get_error_info_from_pyright(code, target_dependency)

# 输出结果
for error_info in error_infos:
    print(f"错误ID: {error_info['error_id']}")
    print(f"工具: {error_info['tool']}")
    print(f"规则: {error_info['rule']}")
    print(f"错误信息: {error_info['error_info']}")
    print("---")
```

### 2. 命令行使用

```bash
# 分析单个文件
python pyright_parser.py test.py --venv /path/to/conda/env

# 分析目录中的所有Python文件
python pyright_parser.py . --venv /path/to/conda/env --format json

# 保存结果到文件
python pyright_parser.py test.py --venv /path/to/conda/env --output results.json

# 使用自定义依赖
python pyright_parser.py test.py --venv /path/to/conda/env --dependency '{"matplotlib": "3.5.0"}'

# 启用详细日志
python pyright_parser.py test.py --venv /path/to/conda/env --verbose

# 多线程处理
python pyright_parser.py . --venv /path/to/conda/env --max-workers 8
```

## 命令行参数

| 参数 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| `target` | 字符串 | 是 | - | 要分析的文件或目录路径 |
| `--venv` | 字符串 | 是 | - | conda环境目录路径 |
| `--format` | 字符串 | 否 | json | 输出格式 (json/text/markdown) |
| `--output` | 字符串 | 否 | - | 输出文件路径 |
| `--dependency` | 字符串 | 否 | {"python": "3.8"} | 目标依赖信息，JSON格式 |
| `--timeout` | 整数 | 否 | 60 | 分析超时时间，秒 |
| `--verbose` | 标志 | 否 | False | 启用详细日志输出 |
| `--max-workers` | 整数 | 否 | 4 | 最大工作线程数 |

## 输出格式

### 1. 错误信息格式

`get_error_info_from_pyright` 方法返回的每个错误信息包含以下字段：

```python
{
    "error_info": "Line 4:     plt.plot(x, y, color='red', size=12)\nError: 'size' is not a valid parameter",
    "tool": "pyright",
    "rule": "reportCallIssue",
    "error_id": "error_0004_0020"
}
```

### 2. JSON格式输出

```json
{
  "has_error": true,
  "diagnostics": [
    {
      "file": "temp.py",
      "severity": "error",
      "message": "'size' is not a valid parameter",
      "range": {
        "start": {"line": 3, "character": 20},
        "end": {"line": 3, "character": 24}
      },
      "rule": "reportCallIssue",
      "code": null
    }
  ],
  "raw_json": {...},
  "error_message": null,
  "execution_time": 1.23
}
```

### 3. 文本格式输出

```
Pyright Analysis Results:
Has Error: True
Execution Time: 1.23s
Total Diagnostics: 1

❌ Diagnostic 1: ERROR
   Message: 'size' is not a valid parameter
   File: temp.py
   Rule: reportCallIssue
   Range: {'start': {'line': 3, 'character': 20}, 'end': {'line': 3, 'character': 24}}
```

## 高级用法

### 1. 详细分析

```python
# 获取详细的分析结果
result = parser.analyze_code_string(code, venv_dir, target_dependency)

# 按严重程度过滤
errors = parser.get_diagnostics_by_severity(result, "error")
warnings = parser.get_diagnostics_by_severity(result, "warning")

# 按规则过滤
call_issues = parser.get_diagnostics_by_rule(result, "reportCallIssue")

# 格式化输出
json_output = parser.format_diagnostics_for_output(result, "json")
text_output = parser.format_diagnostics_for_output(result, "text")
markdown_output = parser.format_diagnostics_for_output(result, "markdown")
```

### 2. 批量处理

```python
import os
from concurrent.futures import ThreadPoolExecutor

def analyze_file(file_path):
    with open(file_path, 'r') as f:
        code = f.read()
    return parser.get_error_info_from_pyright(code, target_dependency)

# 获取所有Python文件
python_files = []
for root, _, files in os.walk("."):
    for file in files:
        if file.endswith('.py'):
            python_files.append(os.path.join(root, file))

# 多线程分析
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(analyze_file, python_files))
```

## 错误处理

解析器包含完整的错误处理机制：

- **超时处理**: 默认60秒超时，可自定义
- **环境检查**: 自动检查conda环境是否存在
- **依赖安装**: 自动安装pyright和必要的stubs
- **异常捕获**: 捕获所有可能的异常并返回错误信息

## 测试

运行测试：

```bash
python test_pyright_parser.py
```

运行示例：

```bash
python pyright_parser_example.py
```

## 注意事项

1. **Conda环境**: 确保指定的conda环境路径正确且存在
2. **依赖管理**: 解析器会自动安装pyright，但可能需要手动安装特定的stubs
3. **超时设置**: 对于大型代码库，可能需要增加超时时间
4. **内存使用**: 多线程处理大量文件时注意内存使用情况

## 与现有系统的集成

这个解析器设计为与现有的错误分析系统兼容，特别是：

- 返回格式与 `data/temp/combined_errors_vscc_with_ids.json` 中的格式一致
- 支持与 `testmypy_utils.py` 中的环境管理函数集成
- 可以与现有的多轮推理系统无缝集成

## 许可证

本项目遵循与主项目相同的许可证。 