#!/usr/bin/env python3
"""
简化版测试 get_error_info_from_mypy 函数
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.pyStaticAnalysis.mypyParse.testMypy_parallel import get_error_info_from_mypy


def main():
    """
    主测试函数
    """
    print("=== 测试 get_error_info_from_mypy 函数 ===")
    
    # 测试用例1: 正确的代码
    print("\n1. 测试正确的代码:")
    correct_code = """
def add_numbers(a: int, b: int) -> int:
    return a + b

result = add_numbers(5, 3)
print(result)
"""
    target_dependency = {"psutil":"5.2.2","python": "3.5"}
    
    try:
        result = get_error_info_from_mypy(correct_code, target_dependency)
        print(f"结果: {result}")
    except Exception as e:
        print(f"错误: {e}")
    
    # 测试用例2: 有类型错误的代码
    print("\n2. 测试有类型错误的代码:")
    error_code = """
import subprocess\nimport psutil\nimport time\n\n\ndef task_func(process_name):\n    for proc in psutil.process_iter(['pid', 'name']):\n        if proc.info['name'] == process_name:\n            proc.terminate()\n            time.sleep(1)\n            subprocess.run([process_name], shell=True)\n            return f'Process found. Restarting {process_name}.'\n    subprocess.run([process_name], shell=True)\n    return f'Process not found. Starting {process_name}.'\n\n\nprint(task_func('notepad'))\n
"""
    
    try:
        result = get_error_info_from_mypy(error_code, target_dependency)
        print(f"结果: {result}")
    except Exception as e:
        print(f"错误: {e}")
    
    # 测试用例3: 语法错误的代码
    print("\n3. 测试语法错误的代码:")
    syntax_error_code = """
def add_numbers(a: int, b: int) -> int:
    return a + b

result = add_numbers(5, '3')  # 缺少右括号
print(result)
"""
    
    try:
        result = get_error_info_from_mypy(syntax_error_code, target_dependency)
        print(f"结果: {result}")
    except Exception as e:
        print(f"错误: {e}")
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    main() 