import json
import re
from typing import Dict, List, Tuple, Optional
import ast
from pathlib import Path
from collections import defaultdict

def extract_error_api(error_info: str) -> Optional[Tuple[str, str, str, Optional[str]]]:
    """
    从错误信息中提取API名称和其所属的模块/类
    
    Args:
        error_info: 错误信息字符串
        
    Returns:
        Optional[Tuple[str, str, str, Optional[str]]]: (api_name, module/class_name, error_type, full_module_path) 或 None
    """
    # 处理 f-string 错误
    if "Format strings are only supported in Python 3.6 and greater" in error_info:
        fstring_match = re.search(r'f["\'](.*?)["\']', error_info)
        if fstring_match:
            return "f-string", "string", "syntax", None
    
    # 处理属性访问错误
    attr_pattern = r'Cannot access attribute "([^"]+)" for class "([^"]+)"'
    attr_match = re.search(attr_pattern, error_info)
    if attr_match:
        return attr_match.group(1), attr_match.group(2), "attribute", None
    
    # 处理方法调用错误
    method_pattern = r'Cannot call method "([^"]+)" on type "([^"]+)"'
    method_match = re.search(method_pattern, error_info)
    if method_match:
        return method_match.group(1), method_match.group(2), "method", None
    
    # 处理导入错误
    import_pattern = r'Import "([^"]+)" could not be resolved'
    import_match = re.search(import_pattern, error_info)
    if import_match:
        return import_match.group(1), import_match.group(1), "import", import_match.group(1)
    
    # 处理模块属性错误
    module_attr_pattern = r'"([^"]+)" is not a known attribute of module "([^"]+)"'
    module_attr_match = re.search(module_attr_pattern, error_info)
    if module_attr_match:
        return module_attr_match.group(1), module_attr_match.group(2), "module_attribute", module_attr_match.group(2)
    
    # 处理参数错误
    param_pattern = r'No parameter named "([^"]+)"'
    param_match = re.search(param_pattern, error_info)
    if param_match:
        # 从错误信息中提取函数调用行
        line_pattern = r'Line \d+: (.*?)\n'
        line_match = re.search(line_pattern, error_info)
        if line_match:
            # 提取函数调用部分
            call_line = line_match.group(1)
            # 匹配函数调用，包括可能的链式调用
            func_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\('
            func_match = re.search(func_pattern, call_line)
            if func_match:
                func_name = func_match.group(1)
                return param_match.group(1), func_name, "parameter", None
    
    return None

def parse_code_for_api(code: str, api_name: str, module_class: str, full_module_path: Optional[str] = None) -> Tuple[Optional[str], List[int]]:
    """
    从代码中静态解析API的完整路径
    
    Args:
        code: Python代码字符串
        api_name: 要查找的API名称
        module_class: API所属的模块/类名
        full_module_path: 完整的模块路径（如果已知）
        
    Returns:
        Tuple[Optional[str], List[int]]: (API的完整路径, 已识别的部分索引列表) 或 (None, [])
    """
    try:
        # 如果已经知道完整的模块路径，直接使用
        if full_module_path:
            if full_module_path == api_name:  # 处理导入错误的情况
                return full_module_path, []  # 导入错误返回空列表
            return f"{full_module_path}.{api_name}", list(range(len(full_module_path.split('.'))))
        
        # 处理参数错误的情况
        if module_class == "function":
            tree = ast.parse(code)
            imports = defaultdict(list)
            
            # 收集所有导入语句
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports[name.asname or name.name].append(name.name)
                elif isinstance(node, ast.ImportFrom):
                    for name in node.names:
                        full_name = f"{node.module}.{name.name}"
                        imports[name.asname or name.name].append(full_name)
            
            # 从导入语句中查找函数
            for imported_names in imports.values():
                for imported_name in imported_names:
                    if imported_name.split('.')[-1] == module_class.split('.')[-1]:
                        return imported_name, list(range(len(imported_name.split('.'))))
            
            # 如果找不到，返回None
            return None, []
        
        tree = ast.parse(code)
        imports = defaultdict(list)
        class_imports = {}  # 存储类名到完整导入路径的映射
        
        # 收集所有导入语句
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports[name.asname or name.name].append(name.name)
            elif isinstance(node, ast.ImportFrom):
                for name in node.names:
                    full_name = f"{node.module}.{name.name}"
                    imports[name.asname or name.name].append(full_name)
                    if name.name == module_class:
                        class_imports[module_class] = full_name
        
        # 查找类的完整路径
        class_path = None
        if module_class in class_imports:
            class_path = class_imports[module_class]
        else:
            # 尝试从导入语句中查找
            for imported_names in imports.values():
                for imported_name in imported_names:
                    if module_class in imported_name:
                        class_path = imported_name
                        break
                if class_path:
                    break
        
        if class_path:
            full_path = f"{class_path}.{api_name}"
            identified_parts = list(range(len(class_path.split('.'))))
            return full_path, identified_parts
        
        return None, []
    except:
        return None, []

def process_errors(combined_errors_file: str, code_file: str) -> List[Dict]:
    """
    处理错误信息并关联代码中的API使用
    
    Args:
        combined_errors_file: 合并后的错误信息文件路径
        code_file: 代码文件路径
        
    Returns:
        List[Dict]: 处理后的错误信息列表
    """
    # 读取错误信息
    with open(combined_errors_file, 'r') as f:
        errors = json.load(f)
    
    # 读取代码
    code_data = []
    with open(code_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            code_data.append(item)
    
    # 创建task_id到代码的映射
    code_map = {item['id']: item['answer'] for item in code_data}
    
    processed_errors = []
    for error_item in errors:
        task_id = error_item['id']
        if task_id not in code_map:
            continue
            
        code = code_map[task_id]
        error_infos = []
        
        for error in error_item['error_infos']:
            error_info = error['error_info']
            api_info = extract_error_api(error_info)
            
            if api_info:
                api_name, module_class, error_type, full_module_path = api_info
                full_api_path, identified_parts = parse_code_for_api(code, api_name, module_class, full_module_path)
                
                error_infos.append({
                    'error_info': error_info,
                    'tool': error['tool'],
                    'api_name': api_name,
                    'module_class': module_class,
                    'error_type': error_type,
                    'full_api_path': full_api_path,
                    'identified_parts': identified_parts
                })
            else:
                error_infos.append({
                    'error_info': error_info,
                    'tool': error['tool']
                })
        
        if error_infos:
            processed_errors.append({
                'id': task_id,
                'error_infos': error_infos
            })
    
    return processed_errors

def main():
    # 设置文件路径
    combined_errors_file = "data/temp/combined_errors.json"
    code_file = "output/approach_eval/BASELINE/Llama-3.1-8B/versibcb_vace_Llama-3.1-8B_maxdep10.jsonl"
    
    # 处理错误信息
    processed_errors = process_errors(combined_errors_file, code_file)
    
    # 保存处理后的结果
    output_file = "data/temp/processed_errors.json"
    with open(output_file, 'w') as f:
        json.dump(processed_errors, f, indent=2)
    
    print(f"处理完成，结果已保存到 {output_file}")

if __name__ == "__main__":
    main() 