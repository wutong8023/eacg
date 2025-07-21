#!/usr/bin/env python3
"""
测试 get_error_info_from_mypy 函数
"""

import sys
import os
import tempfile
import json
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.pyStaticAnalysis.mypyParse.testMypy_parallel import get_error_info_from_mypy, getTargetEnvPath


def test_get_error_info_from_mypy():
    """
    测试 get_error_info_from_mypy 函数
    """
    print("开始测试 get_error_info_from_mypy 函数...")
    
    # 测试用例1: 正确的代码
    print("\n=== 测试用例1: 正确的代码 ===")
    correct_code = """
def add_numbers(a: int, b: int) -> int:
    return a + b

result = add_numbers(5, 3)
print(result)
"""
    target_dependency = {"python": "3.8", "numpy": "1.21"}
    
    try:
        result = get_error_info_from_mypy(correct_code, target_dependency)
        print(f"结果: {result}")
        print("✓ 测试用例1完成")
    except Exception as e:
        print(f"✗ 测试用例1失败: {e}")
    
    # 测试用例2: 有类型错误的代码
    print("\n=== 测试用例2: 有类型错误的代码 ===")
    error_code = """
def add_numbers(a: int, b: int) -> int:
    return a + b

result = add_numbers("5", 3)  # 类型错误：字符串传递给int参数
print(result)
"""
    
    try:
        result = get_error_info_from_mypy(error_code, target_dependency)
        print(f"结果: {result}")
        print("✓ 测试用例2完成")
    except Exception as e:
        print(f"✗ 测试用例2失败: {e}")
    
    # 测试用例3: 语法错误的代码
    print("\n=== 测试用例3: 语法错误的代码 ===")
    syntax_error_code = """
def add_numbers(a: int, b: int) -> int:
    return a + b

result = add_numbers(5, 3
print(result)  # 缺少右括号
"""
    
    try:
        result = get_error_info_from_mypy(syntax_error_code, target_dependency)
        print(f"结果: {result}")
        print("✓ 测试用例3完成")
    except Exception as e:
        print(f"✗ 测试用例3失败: {e}")
    
    # 测试用例4: 使用未定义变量的代码
    print("\n=== 测试用例4: 使用未定义变量的代码 ===")
    undefined_var_code = """
def add_numbers(a: int, b: int) -> int:
    return a + b

result = add_numbers(5, undefined_variable)  # 未定义的变量
print(result)
"""
    
    try:
        result = get_error_info_from_mypy(undefined_var_code, target_dependency)
        print(f"结果: {result}")
        print("✓ 测试用例4完成")
    except Exception as e:
        print(f"✗ 测试用例4失败: {e}")
    
    # 测试用例5: 复杂的类型错误
    print("\n=== 测试用例5: 复杂的类型错误 ===")
    complex_error_code = """
from typing import List, Dict

def process_data(data: List[Dict[str, int]]) -> int:
    total = 0
    for item in data:
        total += item["value"]
    return total

# 错误的数据类型
data = [{"value": "10"}, {"value": 20}]  # 第一个value是字符串
result = process_data(data)
print(result)
"""
    
    try:
        result = get_error_info_from_mypy(complex_error_code, target_dependency)
        print(f"结果: {result}")
        print("✓ 测试用例5完成")
    except Exception as e:
        print(f"✗ 测试用例5失败: {e}")
    
    # 测试用例6: 测试不同的Python版本环境
    print("\n=== 测试用例6: 测试不同的Python版本环境 ===")
    simple_code = """
def hello(name: str) -> str:
    return f"Hello, {name}!"

print(hello("World"))
"""
    
    # 测试不同的Python版本
    python_versions = [
        {"python": "3.7"},
        {"python": "3.8"},
        {"python": "3.9"},
        {"python": "3.10"}
    ]
    
    for version in python_versions:
        try:
            print(f"测试Python版本: {version}")
            result = get_error_info_from_mypy(simple_code, version)
            print(f"结果: {result}")
            print(f"✓ Python {version['python']} 测试完成")
        except Exception as e:
            print(f"✗ Python {version['python']} 测试失败: {e}")
    
    # 测试用例7: 测试环境路径函数
    print("\n=== 测试用例7: 测试环境路径函数 ===")
    test_dependencies = [
        {"python": "3.8", "numpy": "1.21"},
        {"python": "3.9", "pandas": "1.3"},
        {"python": "3.10", "tensorflow": "2.8"}
    ]
    
    for dep in test_dependencies:
        env_path = getTargetEnvPath(dep)
        print(f"依赖 {dep} -> 环境路径: {env_path}")
    
    print("\n=== 所有测试完成 ===")


def test_with_temp_file():
    """
    使用临时文件测试函数
    """
    print("\n=== 使用临时文件测试 ===")
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        test_code = """
def calculate_area(length: float, width: float) -> float:
    return length * width

# 测试正确的调用
area1 = calculate_area(5.0, 3.0)
print(f"Area: {area1}")

# 测试类型错误的调用
area2 = calculate_area("5", 3.0)  # 类型错误
print(f"Area: {area2}")
"""
        f.write(test_code)
        temp_file_path = f.name
    
    try:
        target_dependency = {"python": "3.8"}
        result = get_error_info_from_mypy(test_code, target_dependency)
        print(f"临时文件测试结果: {result}")
        print("✓ 临时文件测试完成")
    except Exception as e:
        print(f"✗ 临时文件测试失败: {e}")
    finally:
        # 清理临时文件
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


def test_error_handling():
    """
    测试错误处理
    """
    print("\n=== 测试错误处理 ===")
    
    # 测试空代码
    print("测试空代码:")
    try:
        result = get_error_info_from_mypy("", {"python": "3.8"})
        print(f"结果: {result}")
    except Exception as e:
        print(f"错误: {e}")
    
    # 测试None代码
    print("测试None代码:")
    try:
        result = get_error_info_from_mypy(None, {"python": "3.8"})
        print(f"结果: {result}")
    except Exception as e:
        print(f"错误: {e}")
    
    # 测试无效的依赖配置
    print("测试无效的依赖配置:")
    try:
        result = get_error_info_from_mypy("print('hello')", {})
        print(f"结果: {result}")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    print("开始运行 get_error_info_from_mypy 测试...")
    
    # 运行所有测试
    test_get_error_info_from_mypy()
    test_with_temp_file()
    test_error_handling()
    
    print("\n所有测试完成！") 