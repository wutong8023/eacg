#!/usr/bin/env python3
"""
Pyright Parser 使用示例

这个示例展示了如何使用 PyrightParser 类来分析代码并获取格式化的错误信息。
"""

import json
import tempfile
import os
from pyright_parser import PyrightParser

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 创建解析器实例
    parser = PyrightParser(enable_logging=False)
    
    # 示例代码（包含一些错误）
    sample_code = """
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_plots(df):
    n_plots = 3
    fig, axs = plt.subplots(n_plots, figsize=(10, 6 * n_plots))
    
    for i in range(n_plots):
        sns.pairplot(df, ax=axs[i])  # 错误：pairplot没有ax参数
    
    return fig

# 未定义的变量
undefined_var = some_undefined_variable
"""
    
    # 目标依赖信息
    target_dependency = {
        "matplotlib": "3.5.0",
        "seaborn": "0.11.0",
        "pandas": "1.3.0"
    }
    
    print("分析代码...")
    print("代码内容:")
    print(sample_code)
    print("\n" + "="*50)
    
    # 获取错误信息
    error_infos = parser.get_error_info_from_pyright(sample_code, target_dependency)
    
    print(f"发现 {len(error_infos)} 个错误:")
    for i, error_info in enumerate(error_infos, 1):
        print(f"\n错误 {i}:")
        print(f"  ID: {error_info['error_id']}")
        print(f"  工具: {error_info['tool']}")
        print(f"  规则: {error_info['rule']}")
        print(f"  信息: {error_info['error_info']}")

def example_detailed_analysis():
    """详细分析示例"""
    print("\n=== 详细分析示例 ===")
    
    parser = PyrightParser(enable_logging=False)
    
    # 更复杂的示例代码
    complex_code = """
from typing import List, Dict, Any
import numpy as np

class DataProcessor:
    def __init__(self, data: List[float]):
        self.data = data
        self.processed = False
    
    def process_data(self) -> Dict[str, Any]:
        if not self.data:
            return {}
        
        # 类型错误：numpy.mean期望array-like，但data是List[float]
        mean_val = np.mean(self.data)
        
        # 未定义的变量
        result = {
            'mean': mean_val,
            'count': len(self.data),
            'status': status  # 未定义
        }
        
        self.processed = True
        return result

# 使用示例
processor = DataProcessor([1.0, 2.0, 3.0])
results = processor.process_data()
print(results)
"""
    
    target_dependency = {
        "numpy": "1.21.0",
        "typing-extensions": "3.10.0"
    }
    
    print("分析复杂代码...")
    
    # 获取详细分析结果
    result = parser.analyze_code_string(complex_code, "/path/to/venv", target_dependency)
    
    print(f"分析完成:")
    print(f"  执行时间: {result.execution_time:.2f}秒")
    print(f"  是否有错误: {result.has_error}")
    print(f"  诊断数量: {len(result.diagnostics)}")
    
    # 按严重程度过滤
    errors = parser.get_diagnostics_by_severity(result, "error")
    warnings = parser.get_diagnostics_by_severity(result, "warning")
    
    print(f"  错误数量: {len(errors)}")
    print(f"  警告数量: {len(warnings)}")
    
    # 格式化输出
    print("\n格式化输出 (JSON):")
    json_output = parser.format_diagnostics_for_output(result, "json")
    print(json_output[:500] + "..." if len(json_output) > 500 else json_output)

def example_error_info_format():
    """错误信息格式示例"""
    print("\n=== 错误信息格式示例 ===")
    
    parser = PyrightParser(enable_logging=False)
    
    # 包含多种错误的代码
    error_code = """
import matplotlib.pyplot as plt

def plot_data(x, y):
    # 未导入的模块
    import seaborn as sns
    
    # 错误的函数调用
    plt.plot(x, y, color='red', size=12)  # size不是有效参数
    
    # 未定义的变量
    plt.title(title_var)
    
    # 类型错误
    plt.xlabel(123)  # 应该是字符串
    
    return plt.gcf()

# 调用函数
result = plot_data([1, 2, 3], [4, 5, 6])
"""
    
    target_dependency = {
        "matplotlib": "3.5.0",
        "seaborn": "0.11.0"
    }
    
    print("获取格式化的错误信息...")
    error_infos = parser.get_error_info_from_pyright(error_code, target_dependency)
    
    print(f"发现 {len(error_infos)} 个错误:")
    
    # 模拟JSON文件格式
    result_data = {
        "id": 123,
        "error_infos": error_infos
    }
    
    print("\nJSON格式输出:")
    print(json.dumps(result_data, indent=2, ensure_ascii=False))

def example_command_line_usage():
    """命令行使用示例"""
    print("\n=== 命令行使用示例 ===")
    
    print("命令行参数示例:")
    print("1. 分析单个文件:")
    print("   python pyright_parser.py test.py --venv /path/to/venv")
    
    print("\n2. 分析目录中的所有Python文件:")
    print("   python pyright_parser.py . --venv /path/to/venv --format json")
    
    print("\n3. 保存结果到文件:")
    print("   python pyright_parser.py test.py --venv /path/to/venv --output results.json")
    
    print("\n4. 使用自定义依赖:")
    print("   python pyright_parser.py test.py --venv /path/to/venv --dependency '{\"matplotlib\": \"3.5.0\"}'")
    
    print("\n5. 启用详细日志:")
    print("   python pyright_parser.py test.py --venv /path/to/venv --verbose")
    
    print("\n6. 多线程处理:")
    print("   python pyright_parser.py . --venv /path/to/venv --max-workers 8")

def main():
    """主函数"""
    print("Pyright Parser 使用示例")
    print("=" * 50)
    
    try:
        example_basic_usage()
        example_detailed_analysis()
        example_error_info_format()
        example_command_line_usage()
        
        print("\n" + "=" * 50)
        print("示例完成！")
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        print("请确保已正确安装pyright和相关依赖")

if __name__ == "__main__":
    main() 