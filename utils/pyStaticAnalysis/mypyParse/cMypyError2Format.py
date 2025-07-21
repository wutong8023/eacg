import json
import os
import re
from typing import List, Dict, Any

def parse_mypy_error(error_info: str, code: str) -> List[Dict[str, str]]:
    '''
    解析mypy的错误信息，转换为标准格式
    
    Args:
        error_info: mypy的错误信息,为str
        code: 完整的代码内容
    
    Returns:
        List[Dict[str, str]]: 格式化后的错误信息列表
    '''
    errors = []
    code_lines = code.split('\n')
    # 匹配错误行模式
    error_pattern = r'([^:]+):(\d+):\s+error:\s+(.+)'
    matches = re.finditer(error_pattern, error_info)
    
    for match in matches:
        file_path, line_num, message = match.groups()
        line_num = int(line_num)
        
        # 获取错误行的代码
        if 0 <= line_num - 1 < len(code_lines):
            error_line = code_lines[line_num - 1]
        else:
            error_line = "Line not found"
        
        # 创建标准格式的错误信息
        error_dict = {
            'error_info': f"Line {line_num}: {error_line}\nError: {message}"
        }
        errors.append(error_dict)
    
    return errors

def process_mypy_results(mypy_file: str, output_file: str):
    '''
    处理mypy测试结果文件，转换为标准格式
    
    Args:
        mypy_file: mypy测试结果文件路径
        output_file: 输出文件路径
    '''
    # 用于存储每个id的错误信息
    id_to_errors = {}
    
    # 读取mypy结果文件
    with open(mypy_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                item_id = data['id']
                
                # 只处理有错误的项
                if data.get('has_error', False) and data.get('error_info', '').strip():
                    errors = parse_mypy_error(data['error_info'], data['code'])
                    if errors:
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
    mypy_file = "data/temp/mypy_test_results_vscc.jsonl"
    output_file = "data/temp/mypy_error_summary_vscc.json"
    
    if not os.path.exists(mypy_file):
        print(f"Error: Mypy results file {mypy_file} not found")
        exit(1)
    
    process_mypy_results(mypy_file, output_file)
