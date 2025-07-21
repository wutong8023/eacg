# get_error_info_from_mypy 函数测试

本目录包含了用于测试 `get_error_info_from_mypy` 函数的测试文件。

## 测试文件说明

### 1. simple_test_mypy.py
最简单的测试文件，包含基本的测试用例：
- 正确的代码
- 有类型错误的代码
- 语法错误的代码

**运行方式：**
```bash
cd tests/pyStaticAnalysis
python simple_test_mypy.py
```

### 2. test_get_error_info_from_mypy.py
完整的测试文件，包含更多测试用例：
- 各种类型的错误代码
- 不同Python版本的测试
- 错误处理测试
- 临时文件测试

**运行方式：**
```bash
cd tests/pyStaticAnalysis
python test_get_error_info_from_mypy.py
```

### 3. test_mypy_unittest.py
使用 unittest 框架的测试文件，提供更规范的测试结构：
- 使用 unittest.TestCase 类
- 包含 setUp 方法
- 使用断言进行验证
- 支持测试套件运行

**运行方式：**
```bash
cd tests/pyStaticAnalysis
python test_mypy_unittest.py
```

或者使用 unittest 模块：
```bash
python -m unittest test_mypy_unittest.py -v
```

## 测试用例说明

### 基本测试用例

1. **正确的代码测试**
   ```python
   def add_numbers(a: int, b: int) -> int:
       return a + b
   
   result = add_numbers(5, 3)
   print(result)
   ```

2. **类型错误测试**
   ```python
   def add_numbers(a: int, b: int) -> int:
       return a + b
   
   result = add_numbers("5", 3)  # 类型错误：字符串传递给int参数
   print(result)
   ```

3. **语法错误测试**
   ```python
   def add_numbers(a: int, b: int) -> int:
       return a + b
   
   result = add_numbers(5, 3  # 缺少右括号
   print(result)
   ```

4. **未定义变量测试**
   ```python
   def add_numbers(a: int, b: int) -> int:
       return a + b
   
   result = add_numbers(5, undefined_variable)  # 未定义的变量
   print(result)
   ```

5. **复杂类型错误测试**
   ```python
   from typing import List, Dict
   
   def process_data(data: List[Dict[str, int]]) -> int:
       total = 0
       for item in data:
           total += item["value"]
       return total
   
   data = [{"value": "10"}, {"value": 20}]  # 第一个value是字符串
   result = process_data(data)
   print(result)
   ```

### 环境测试

- 测试不同的Python版本（3.7, 3.8, 3.9, 3.10）
- 测试不同的依赖配置
- 测试环境路径生成

### 边界情况测试

- 空代码测试
- None 代码测试
- 无效依赖配置测试

## 函数说明

### get_error_info_from_mypy(generated_code, target_dependency)

**参数：**
- `generated_code` (str): 要检查的Python代码
- `target_dependency` (dict): 目标依赖配置，例如 `{"python": "3.8", "numpy": "1.21"}`

**返回值：**
- `str`: 错误信息字符串，如果没有错误则返回 "No errors found"

**功能：**
1. 根据目标依赖获取对应的conda环境路径
2. 在指定环境中运行mypy检查代码
3. 解析并格式化错误信息
4. 返回标准化的错误信息

## 依赖要求

- Python 3.7+
- mypy
- conda 环境管理
- 相关的conda环境已创建并安装了mypy

## 注意事项

1. 确保conda环境路径正确配置
2. 确保目标conda环境中已安装mypy
3. 测试前请确保网络连接正常（用于安装缺失的stubs）
4. 某些测试可能需要特定的conda环境存在

## 故障排除

如果测试失败，请检查：

1. **环境路径问题**
   - 确认 `getTargetEnvPath` 函数返回的路径是否正确
   - 确认conda环境是否存在

2. **mypy安装问题**
   - 确认目标环境中是否安装了mypy
   - 检查mypy版本兼容性

3. **权限问题**
   - 确认有权限访问conda环境目录
   - 确认有权限创建临时文件

4. **网络问题**
   - 确认可以访问pip源进行stubs安装
   - 检查防火墙设置 