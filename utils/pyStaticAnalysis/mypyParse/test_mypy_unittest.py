#!/usr/bin/env python3
"""
使用 unittest 框架测试 get_error_info_from_mypy 函数
"""

import unittest
import sys
import os
import tempfile

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.pyStaticAnalysis.mypyParse.testMypy_parallel import get_error_info_from_mypy, getTargetEnvPath


class TestGetErrorInfoFromMypy(unittest.TestCase):
    """测试 get_error_info_from_mypy 函数的测试类"""
    
    def setUp(self):
        """测试前的设置"""
        self.target_dependency = {"python": "3.8"}
    
    def test_correct_code(self):
        """测试正确的代码"""
        correct_code = """
def add_numbers(a: int, b: int) -> int:
    return a + b

result = add_numbers(5, 3)
print(result)
"""
        result = get_error_info_from_mypy(correct_code, self.target_dependency)
        self.assertIsInstance(result, str)
        print(f"正确代码测试结果: {result}")
    
    def test_type_error_code(self):
        """测试有类型错误的代码"""
        error_code = """
def add_numbers(a: int, b: int) -> int:
    return a + b

result = add_numbers("5", 3)  # 类型错误：字符串传递给int参数
print(result)
"""
        result = get_error_info_from_mypy(error_code, self.target_dependency)
        self.assertIsInstance(result, str)
        print(f"类型错误代码测试结果: {result}")
    
    def test_syntax_error_code(self):
        """测试语法错误的代码"""
        syntax_error_code = """
def add_numbers(a: int, b: int) -> int:
    return a + b

result = add_numbers(5, 3  # 缺少右括号
print(result)
"""
        result = get_error_info_from_mypy(syntax_error_code, self.target_dependency)
        self.assertIsInstance(result, str)
        print(f"语法错误代码测试结果: {result}")
    
    def test_undefined_variable_code(self):
        """测试使用未定义变量的代码"""
        undefined_var_code = """
def add_numbers(a: int, b: int) -> int:
    return a + b

result = add_numbers(5, undefined_variable)  # 未定义的变量
print(result)
"""
        result = get_error_info_from_mypy(undefined_var_code, self.target_dependency)
        self.assertIsInstance(result, str)
        print(f"未定义变量代码测试结果: {result}")
    
    def test_complex_type_error_code(self):
        """测试复杂的类型错误"""
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
        result = get_error_info_from_mypy(complex_error_code, self.target_dependency)
        self.assertIsInstance(result, str)
        print(f"复杂类型错误代码测试结果: {result}")
    
    def test_empty_code(self):
        """测试空代码"""
        result = get_error_info_from_mypy("", self.target_dependency)
        self.assertIsInstance(result, str)
        print(f"空代码测试结果: {result}")
    
    def test_none_code(self):
        """测试None代码"""
        with self.assertRaises(Exception):
            get_error_info_from_mypy(None, self.target_dependency)
    
    def test_different_python_versions(self):
        """测试不同的Python版本"""
        simple_code = """
def hello(name: str) -> str:
    return f"Hello, {name}!"

print(hello("World"))
"""
        
        python_versions = [
            {"python": "3.7"},
            {"python": "3.8"},
            {"python": "3.9"},
            {"python": "3.10"}
        ]
        
        for version in python_versions:
            with self.subTest(version=version):
                try:
                    result = get_error_info_from_mypy(simple_code, version)
                    self.assertIsInstance(result, str)
                    print(f"Python {version['python']} 测试结果: {result}")
                except Exception as e:
                    print(f"Python {version['python']} 测试失败: {e}")
    
    def test_get_target_env_path(self):
        """测试 getTargetEnvPath 函数"""
        test_dependencies = [
            {"python": "3.8", "numpy": "1.21"},
            {"python": "3.9", "pandas": "1.3"},
            {"python": "3.10", "tensorflow": "2.8"}
        ]
        
        for dep in test_dependencies:
            with self.subTest(dependency=dep):
                env_path = getTargetEnvPath(dep)
                self.assertIsInstance(env_path, str)
                self.assertTrue(env_path.startswith("/datanfs2/chenrongyi/conda_env"))
                print(f"依赖 {dep} -> 环境路径: {env_path}")


class TestMypyWithTempFile(unittest.TestCase):
    """使用临时文件测试 mypy 功能"""
    
    def setUp(self):
        """测试前的设置"""
        self.target_dependency = {"python": "3.8"}
    
    def test_with_temp_file(self):
        """使用临时文件测试"""
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
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            temp_file_path = f.name
        
        try:
            result = get_error_info_from_mypy(test_code, self.target_dependency)
            self.assertIsInstance(result, str)
            print(f"临时文件测试结果: {result}")
        finally:
            # 清理临时文件
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_suite.addTest(unittest.makeSuite(TestGetErrorInfoFromMypy))
    test_suite.addTest(unittest.makeSuite(TestMypyWithTempFile))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("开始运行 get_error_info_from_mypy 单元测试...")
    
    success = run_tests()
    
    if success:
        print("\n✓ 所有测试通过！")
    else:
        print("\n✗ 部分测试失败！")
    
    # 也可以直接运行 unittest
    # unittest.main(verbosity=2) 