import json
import os
from typing import Dict, List, Any

def load_error_file(file_path: str) -> Dict[str, List[Dict[str, str]]]:
    '''
    加载错误信息文件
    
    Args:
        file_path: 错误信息文件路径
    
    Returns:
        Dict[str, List[Dict[str, str]]]: 错误信息字典
    '''
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        return {}

def combine_errors(mypy_errors: Dict[str, List[Dict[str, str]]], 
                  pyright_errors: Dict[str, List[Dict[str, str]]]) -> List[Dict[str, Any]]:
    '''
    合并mypy和pyright的错误信息
    
    Args:
        mypy_errors: mypy错误信息字典
        pyright_errors: pyright错误信息字典
    
    Returns:
        List[Dict[str, Any]]: 合并后的错误信息列表，每个元素包含id和error_infos
    '''
    combined_errors = []
    
    # 处理所有ID
    all_ids = set(list(mypy_errors.keys()) + list(pyright_errors.keys()))
    
    for item_id in all_ids:
        error_infos = []
        
        # 添加mypy错误
        if item_id in mypy_errors:
            for error in mypy_errors[item_id]:
                error_info = error['error_info']
                if "Format strings are only supported in Python 3.6 and greater" in error_info:
                    error_info += "\nIn python3.5, as an example, should use \"{} {}\".format(x,y) to replace f\"{x} {y}\""
                error_infos.append({
                    'error_info': error_info,
                    'tool': 'mypy'
                })
        
        # 添加pyright错误
        if item_id in pyright_errors:
            for error in pyright_errors[item_id]:
                error_info = error['error_info']
                if "Format strings are only supported in Python 3.6 and greater" in error_info:
                    error_info += "\nIn python3.5, as an example, should use \"{} {}\".format(x,y) to replace f\"{x} {y}\""
                if 'rule' in error:
                    error_infos.append({
                        'error_info': error_info,
                        'tool': 'pyright',
                        'rule': error['rule']
                    })
                else:
                    error_infos.append({
                        'error_info': error_info,
                        'tool': 'pyright'
                    })
        
        if error_infos:
            combined_errors.append({
                'id': int(item_id),
                'error_infos': error_infos
            })
    
    return combined_errors

def save_combined_errors(combined_errors: List[Dict[str, Any]], output_file: str):
    '''
    保存合并后的错误信息
    
    Args:
        combined_errors: 合并后的错误信息列表
        output_file: 输出文件路径
    '''
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_errors, f, ensure_ascii=False, indent=2)
        print(f"Combined errors saved to {output_file}")
    except Exception as e:
        print(f"Error saving combined errors: {str(e)}")

def main():
    # 文件路径
    mypy_file = "data/temp/mypy_error_summary_vscc.json"
    pyright_file = "data/temp/pyright_error_summary_vscc.json"
    output_file = "data/temp/combined_errors_vscc.json"
    
    # 检查文件是否存在
    if not os.path.exists(mypy_file):
        print(f"Error: Mypy error file {mypy_file} not found")
        return
    
    if not os.path.exists(pyright_file):
        print(f"Error: Pyright error file {pyright_file} not found")
        return
    
    # 加载错误信息
    print("Loading error files...")
    mypy_errors = load_error_file(mypy_file)
    pyright_errors = load_error_file(pyright_file)
    
    print(f"Loaded {len(mypy_errors)} items from mypy")
    print(f"Loaded {len(pyright_errors)} items from pyright")
    
    # 合并错误信息
    print("Combining errors...")
    combined_errors = combine_errors(mypy_errors, pyright_errors)
    
    print(f"Combined {len(combined_errors)} items with errors")
    
    # 保存合并后的错误信息
    save_combined_errors(combined_errors, output_file)

if __name__ == "__main__":
    main()
