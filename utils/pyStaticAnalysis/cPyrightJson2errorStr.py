import json
import os
from typing import List, Dict, Any

def load_test_results(test_results_file: str) -> Dict[int, Dict[str, Any]]:
    '''
    加载测试结果文件，建立id到代码的映射
    
    Args:
        test_results_file: 测试结果文件路径
    
    Returns:
        Dict[int, Dict[str, Any]]: id到测试结果的映射
    '''
    id_to_result = {}
    try:
        with open(test_results_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'id' in data and 'code' in data:
                        id_to_result[data['id']] = data
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error loading test results: {str(e)}")
    return id_to_result

def format_error_info(diagnostic: Dict[str, Any], code: str) -> str:
    '''
    将单个诊断信息转换为标准格式，包含错误行的代码
    
    Args:
        diagnostic: 单个诊断信息字典
        code: 完整的代码内容
    
    Returns:
        str: 格式化后的错误信息
    '''
    file_path = diagnostic['file']
    severity = diagnostic['severity']
    message = diagnostic['message']
    start_line = diagnostic['range']['start']['line'] + 1  # 转换为1-based行号
    start_char = diagnostic['range']['start']['character']
    
    # 获取错误行的代码
    code_lines = code.split('\n')
    if 0 <= start_line - 1 < len(code_lines):
        error_line = code_lines[start_line - 1]
    else:
        error_line = "Line not found"
    
    return f"Line {start_line}: {error_line}\nError: {message}"

def process_raw_diagnostics(raw_file: str, test_results_file: str, output_file: str):
    '''
    处理原始诊断信息文件，转换为标准格式
    
    Args:
        raw_file: 原始诊断信息文件路径
        test_results_file: 测试结果文件路径
        output_file: 输出文件路径
    '''
    # 加载测试结果
    id_to_result = load_test_results(test_results_file)
    print(f"Loaded {len(id_to_result)} test results")
    
    # 用于存储每个id的错误信息
    id_to_errors = {}
    
    # 读取原始文件
    with open(raw_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                item_id = data['id']
                diagnostics = data['generalDiagnostics']
                
                # 获取对应的代码
                if item_id not in id_to_result:
                    print(f"Warning: No test result found for id {item_id}")
                    continue
                
                code = id_to_result[item_id]['code']
                
                # 处理每个诊断信息
                errors = []
                for diagnostic in diagnostics:
                    if diagnostic['severity'] == 'error':  # 只处理error级别的诊断
                        error_dict = {
                            'error_info': format_error_info(diagnostic, code),
                            'rule': diagnostic['rule']
                        }
                        errors.append(error_dict)
                
                if errors:  # 只保存有错误的项
                    id_to_errors[item_id] = errors
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line: {str(e)}")
                continue
            except Exception as e:
                print(f"Error processing line: {str(e)}")
                continue
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(id_to_errors, f, ensure_ascii=False, indent=2)
    
    print(f"Processed {len(id_to_errors)} items with errors")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    raw_file = "data/temp/pyright_test_results_raw_vscc.jsonl"
    test_results_file = "data/temp/pyright_test_results_vscc.jsonl"
    output_file = "data/temp/pyright_error_summary_vscc.json"
    
    if not os.path.exists(raw_file):
        print(f"Error: Raw file {raw_file} not found")
        exit(1)
    
    if not os.path.exists(test_results_file):
        print(f"Error: Test results file {test_results_file} not found")
        exit(1)
    
    process_raw_diagnostics(raw_file, test_results_file, output_file)
