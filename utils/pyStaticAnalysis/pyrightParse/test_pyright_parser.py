#!/usr/bin/env python3
"""
Pyright Parser 测试文件

这个文件用于测试 PyrightParser 的基本功能。
"""

import unittest
import tempfile
import os
import json
from unittest.mock import patch, MagicMock
from pyright_parser import PyrightParser, PyrightDiagnostic, PyrightResult

class TestPyrightParser(unittest.TestCase):
    """PyrightParser 测试类"""
    
    def setUp(self):
        """测试前的设置"""
        self.parser = PyrightParser(enable_logging=False)
        
        # 测试用的代码
        self.test_code = """
import matplotlib.pyplot as plt

def plot_data(x, y):
    plt.plot(x, y, color='red', size=12)  # size不是有效参数
    plt.title(title_var)  # 未定义的变量
    return plt.gcf()
"""
        
        # 测试用的依赖
        self.test_dependency = {
            "matplotlib": "3.5.0"
        }
    
    def test_format_diagnostic_to_error_info(self):
        """测试诊断信息格式化"""
        # 创建模拟的诊断信息
        diagnostic = PyrightDiagnostic(
            file="test.py",
            severity="error",
            message="'size' is not a valid parameter",
            range={
                "start": {"line": 4, "character": 20},
                "end": {"line": 4, "character": 24}
            },
            rule="reportCallIssue",
            code="test-code"
        )
        
        code_lines = self.test_code.splitlines()
        
        # 测试格式化
        error_info = self.parser._format_diagnostic_to_error_info(diagnostic, code_lines)
        
        # 验证结果
        self.assertIn("error_info", error_info)
        self.assertEqual(error_info["tool"], "pyright")
        self.assertEqual(error_info["rule"], "reportCallIssue")
        self.assertIn("error_id", error_info)
        self.assertIn("Line 5:", error_info["error_info"])  # 注意：行号是1-based
    
    def test_get_error_info_from_pyright_mock(self):
        """测试获取错误信息（使用模拟）"""
        # 模拟诊断结果
        mock_diagnostic = PyrightDiagnostic(
            file="temp.py",
            severity="error",
            message="'size' is not a valid parameter",
            range={
                "start": {"line": 3, "character": 20},
                "end": {"line": 3, "character": 24}
            },
            rule="reportCallIssue"
        )
        
        mock_result = PyrightResult(
            has_error=True,
            diagnostics=[mock_diagnostic],
            execution_time=1.0
        )
        
        # 模拟 analyze_code_string 方法
        with patch.object(self.parser, 'analyze_code_string', return_value=mock_result):
            error_infos = self.parser.get_error_info_from_pyright(self.test_code, self.test_dependency)
            
            # 验证结果
            self.assertEqual(len(error_infos), 1)
            self.assertEqual(error_infos[0]["tool"], "pyright")
            self.assertEqual(error_infos[0]["rule"], "reportCallIssue")
            self.assertIn("error_id", error_infos[0])
            self.assertIn("Line 4:", error_infos[0]["error_info"])
    
    def test_get_diagnostics_by_severity(self):
        """测试按严重程度过滤诊断信息"""
        # 创建测试结果
        error_diagnostic = PyrightDiagnostic(
            file="test.py",
            severity="error",
            message="Error message",
            range={"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 0}}
        )
        
        warning_diagnostic = PyrightDiagnostic(
            file="test.py",
            severity="warning",
            message="Warning message",
            range={"start": {"line": 1, "character": 0}, "end": {"line": 1, "character": 0}}
        )
        
        result = PyrightResult(
            has_error=True,
            diagnostics=[error_diagnostic, warning_diagnostic]
        )
        
        # 测试过滤
        errors = self.parser.get_diagnostics_by_severity(result, "error")
        warnings = self.parser.get_diagnostics_by_severity(result, "warning")
        all_diagnostics = self.parser.get_diagnostics_by_severity(result)
        
        self.assertEqual(len(errors), 1)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(len(all_diagnostics), 2)
    
    def test_get_diagnostics_by_rule(self):
        """测试按规则过滤诊断信息"""
        # 创建测试结果
        diagnostic1 = PyrightDiagnostic(
            file="test.py",
            severity="error",
            message="Message 1",
            range={"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 0}},
            rule="rule1"
        )
        
        diagnostic2 = PyrightDiagnostic(
            file="test.py",
            severity="error",
            message="Message 2",
            range={"start": {"line": 1, "character": 0}, "end": {"line": 1, "character": 0}},
            rule="rule2"
        )
        
        result = PyrightResult(
            has_error=True,
            diagnostics=[diagnostic1, diagnostic2]
        )
        
        # 测试过滤
        rule1_diagnostics = self.parser.get_diagnostics_by_rule(result, "rule1")
        rule2_diagnostics = self.parser.get_diagnostics_by_rule(result, "rule2")
        unknown_rule_diagnostics = self.parser.get_diagnostics_by_rule(result, "unknown")
        
        self.assertEqual(len(rule1_diagnostics), 1)
        self.assertEqual(len(rule2_diagnostics), 1)
        self.assertEqual(len(unknown_rule_diagnostics), 0)
    
    def test_format_diagnostics_for_output(self):
        """测试格式化输出"""
        # 创建测试结果
        diagnostic = PyrightDiagnostic(
            file="test.py",
            severity="error",
            message="Test error message",
            range={"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 0}},
            rule="testRule"
        )
        
        result = PyrightResult(
            has_error=True,
            diagnostics=[diagnostic],
            execution_time=1.5
        )
        
        # 测试JSON格式
        json_output = self.parser.format_diagnostics_for_output(result, "json")
        json_data = json.loads(json_output)
        
        self.assertEqual(json_data["has_error"], True)
        self.assertEqual(len(json_data["diagnostics"]), 1)
        self.assertEqual(json_data["diagnostics"][0]["severity"], "error")
        
        # 测试文本格式
        text_output = self.parser.format_diagnostics_for_output(result, "text")
        self.assertIn("Pyright Analysis Results:", text_output)
        self.assertIn("Test error message", text_output)
        
        # 测试Markdown格式
        markdown_output = self.parser.format_diagnostics_for_output(result, "markdown")
        self.assertIn("# Pyright Analysis Results", markdown_output)
        self.assertIn("**Severity:** error", markdown_output)
    
    def test_empty_result_formatting(self):
        """测试空结果的格式化"""
        result = PyrightResult(
            has_error=False,
            diagnostics=[],
            execution_time=0.5
        )
        
        # 测试各种格式的空结果
        json_output = self.parser.format_diagnostics_for_output(result, "json")
        text_output = self.parser.format_diagnostics_for_output(result, "text")
        markdown_output = self.parser.format_diagnostics_for_output(result, "markdown")
        
        json_data = json.loads(json_output)
        self.assertEqual(json_data["has_error"], False)
        self.assertEqual(len(json_data["diagnostics"]), 0)
        
        self.assertIn("No issues found", text_output)
        self.assertIn("No issues found", markdown_output)

class TestPyrightParserIntegration(unittest.TestCase):
    """集成测试类"""
    
    def setUp(self):
        """测试前的设置"""
        self.parser = PyrightParser(enable_logging=False)
    
    @patch('subprocess.run')
    def test_run_pyright_analysis_success(self, mock_run):
        """测试成功的pyright分析"""
        # 模拟成功的pyright输出
        mock_output = {
            "generalDiagnostics": [
                {
                    "file": "test.py",
                    "severity": "error",
                    "message": "Test error",
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 0}
                    },
                    "rule": "testRule"
                }
            ]
        }
        
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout=json.dumps(mock_output),
            stderr=""
        )
        
        # 测试分析
        result = self.parser.run_pyright_analysis("/fake/venv", "test.py", {"python": "3.8"})
        
        self.assertTrue(result.has_error)
        self.assertEqual(len(result.diagnostics), 1)
        self.assertEqual(result.diagnostics[0].message, "Test error")
    
    @patch('subprocess.run')
    def test_run_pyright_analysis_timeout(self, mock_run):
        """测试pyright分析超时"""
        mock_run.side_effect = TimeoutError("Command timed out")
        
        result = self.parser.run_pyright_analysis("/fake/venv", "test.py", {"python": "3.8"})
        
        self.assertTrue(result.has_error)
        self.assertIn("timed out", result.error_message)

def run_tests():
    """运行所有测试"""
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_suite.addTest(unittest.makeSuite(TestPyrightParser))
    test_suite.addTest(unittest.makeSuite(TestPyrightParserIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("运行 Pyright Parser 测试...")
    success = run_tests()
    
    if success:
        print("\n✅ 所有测试通过！")
    else:
        print("\n❌ 部分测试失败！")
    
    exit(0 if success else 1) 