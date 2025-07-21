import subprocess
import os
import sys
import json
import tempfile
import datetime
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.pyStaticAnalysis.clean_utils import clean_return_annotations,clean_annotations

def get_existing_results(output_file):
    '''
    获取已经处理过的结果
    
    Args:
        output_file: 输出文件路径
    
    Returns:
        set: 已处理的id集合
    '''
    existing_ids = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'id' in data:
                            existing_ids.add(data['id'])
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: Error reading existing results: {str(e)}")
    return existing_ids
from utils.pyStaticAnalysis.testmypy_utils import getTargetEnvPath


def ensure_pyright_installed(venv_dir, target_dependency):
    '''
    确保在指定conda环境中安装了pyright和必要的stubs
    
    Args:
        venv_dir: conda环境目录
        target_dependency: 目标依赖信息
    
    Returns:
        bool: 是否成功安装或已存在
    '''
    # 获取conda环境中的pip路径
    pip_executable = os.path.join(venv_dir, 'bin', 'pip') if os.name != 'nt' else os.path.join(venv_dir, 'Scripts', 'pip.exe')
    pyright_executable = os.path.join(venv_dir, 'bin', 'pyright') if os.name != 'nt' else os.path.join(venv_dir, 'Scripts', 'pyright.exe')
    
    if os.path.exists(pyright_executable):
        # 检查是否需要安装matplotlib-stubs
        if 'matplotlib' in target_dependency:
            try:
                # 使用conda run命令在指定环境中安装matplotlib-stubs
                conda_run_cmd = ['conda', 'run', '-p', venv_dir, 'pip', 'install', 'matplotlib-stubs']
                result = subprocess.run(conda_run_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Warning: Failed to install matplotlib-stubs: {result.stderr}")
            except Exception as e:
                print(f"Warning: Error installing matplotlib-stubs: {str(e)}")
        return True
        
    try:
        # 使用conda run命令在指定环境中安装pyright
        conda_run_cmd = ['conda', 'run', '-p', venv_dir, 'pip', 'install', 'pyright']
        result = subprocess.run(conda_run_cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and 'matplotlib' in target_dependency:
            # 安装matplotlib-stubs
            conda_run_cmd = ['conda', 'run', '-p', venv_dir, 'pip', 'install', 'matplotlib-stubs']
            result = subprocess.run(conda_run_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Warning: Failed to install matplotlib-stubs: {result.stderr}")
        
        return True
    except Exception as e:
        print(f"安装pyright时出错: {str(e)}")
        return False

def format_error_info(error_json):
    '''
    格式化pyright的JSON输出，提取关键错误信息
    
    Args:
        error_json: pyright输出的JSON字符串
    
    Returns:
        str: 格式化后的错误信息
    '''
    try:
        data = json.loads(error_json)
        if not data.get('generalDiagnostics'):
            return "No errors found"
            
        formatted_errors = []
        for diagnostic in data['generalDiagnostics']:
            file_path = diagnostic['file']
            severity = diagnostic['severity']
            message = diagnostic['message']
            start_line = diagnostic['range']['start']['line'] + 1  # 转换为1-based行号
            start_char = diagnostic['range']['start']['character']
            
            # 读取错误行内容
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if start_line <= len(lines):
                        error_line = lines[start_line - 1].rstrip()
                        formatted_errors.append(
                            f"Line {start_line}: {error_line}\n"
                            f"Error: {message}\n"
                        )
            except Exception as e:
                formatted_errors.append(f"Error reading file {file_path}: {str(e)}")
        
        return "\n".join(formatted_errors)
    except json.JSONDecodeError:
        return error_json
    except Exception as e:
        return f"Error processing pyright output: {str(e)}"

def run_pyright_in_env(venv_dir, file_path, target_dependency,return_list=False):
    '''
    在指定conda环境中运行pyright检查指定文件
    
    Args:
        venv_dir: conda环境目录
        file_path: 要检查的文件路径
        target_dependency: 目标依赖信息
    
    Returns:
        tuple: (has_error: bool, error_info: str, raw_diagnostics: list)
    '''
    # 确保pyright已安装
    if not ensure_pyright_installed(venv_dir, target_dependency):
        return True, "Failed to install pyright in the conda environment", None
    
    # 获取conda环境中的pyright路径
    pyright_executable = os.path.join(venv_dir, 'bin', 'pyright') if os.name != 'nt' else os.path.join(venv_dir, 'Scripts', 'pyright.exe')
    
    if not os.path.exists(pyright_executable):
        return True, f"Pyright executable not found in {venv_dir} after installation attempt", None
    
    try:
        # 使用conda run命令在指定环境中运行pyright
        conda_run_cmd = ['conda', 'run', '-p', venv_dir, 'pyright', '--outputjson', file_path]
        
        result = subprocess.run(
            conda_run_cmd,
            capture_output=True,
            text=True
        )
        
        has_error = result.returncode != 0
        if has_error:
            try:
                error_info = format_error_info(result.stdout)
                print(error_info)
            except Exception as e:
                error_info = result.stdout
                print(f"Error formatting error info: {str(e)}")

        else:
            error_info = ""
        
        # 解析原始诊断信息
        raw_diagnostics = None
        if has_error:
            try:
                data = json.loads(result.stdout)
                if 'generalDiagnostics' in data:
                    raw_diagnostics = data['generalDiagnostics']
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse pyright output: {str(e)}")
        
        if return_list:
            error_info = error_info.split("\n")
        return has_error, error_info, raw_diagnostics
        
    except subprocess.CalledProcessError as e:
        return True, f"Error running pyright: {str(e)}", None
    except Exception as e:
        return True, f"Unexpected error: {str(e)}", None
from utils.pyStaticAnalysis.fstring35detect import getFStringErrorInfo
def get_error_info_from_pyright(generated_code,target_dependency,return_list=True):
    '''
    return:
        error_info: 错误信息,list[str]
    '''
    venv_dir = getTargetEnvPath(target_dependency)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=True) as temp_file:
        temp_file.write(generated_code)
    has_error, error_info, raw_diagnostics = run_pyright_in_env(venv_dir, temp_file.name, target_dependency,return_list)
    fstring_errors = getFStringErrorInfo(generated_code,target_dependency)
    if fstring_errors:
        error_info.extend(fstring_errors)
    return error_info

def clean_model_output(output):
    '''
    清理模型输出，移除markdown标记等
    '''
    if "<start>" in output:
        start_index = output.find("<start>") + len("<start>")
        if "<end>" in output:
            end_index = output.find("<end>")
            content = output[start_index:end_index].replace('```python', '').replace('```', '')
        else:
            content = output[start_index:].replace('```python', '').replace('```', '')
    else:
        # 目前不要求严格start和end
        content = output.replace('```python', '').replace('```', '')
    
    return content

def read_benchmark_file(benchmark_file_path):
    '''
    读取benchmark文件，返回id到target_dependency的映射
    '''
    try:
        with open(benchmark_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        id_to_dependency = {}
        for item in data:
            if 'id' in item and ('target_dependency' in item or 'dependency' in item):
                id_to_dependency[item['id']] = item['target_dependency'] if 'target_dependency' in item else item['dependency']
        
        return id_to_dependency
    except Exception as e:
        print(f"Error reading benchmark file {benchmark_file_path}: {str(e)}")
        return {}

def process_single_item(item_data, id_to_dependency, output_file, raw_output_file, lock):
    '''
    处理单个测试项
    
    Args:
        item_data: 包含id和answer的数据
        id_to_dependency: id到target_dependency的映射
        output_file: 输出文件路径
        raw_output_file: 原始错误信息输出文件路径
        lock: 文件写入锁
    
    Returns:
        bool: 是否成功处理
    '''
    try:
        item_id = item_data['id']
        answer = item_data['answer']
        
        # 获取对应的依赖信息
        if item_id not in id_to_dependency:
            print(f"Warning: No dependency found for id {item_id}")
            return False
        
        target_dependency = id_to_dependency[item_id]
        venv_dir = getTargetEnvPath(target_dependency)
        
        # 清理模型输出
        clean_code = clean_model_output(answer)
        clean_code = clean_annotations(clean_code)
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(clean_code)
            temp_file_path = temp_file.name
        
        try:
            # 测试代码
            has_error, error_info_str, raw_diagnostics = run_pyright_in_env(venv_dir, temp_file_path, target_dependency)
            from utils.pyStaticAnalysis.fstring35detect import getFStringErrorInfo
            fstring_errors = getFStringErrorInfo(clean_code,target_dependency)
            if fstring_errors:
                fstring_errors_str = "\n".join(fstring_errors)
                error_info_str = fstring_errors_str + "\n" + error_info_str
            # 准备输出数据
            result = {
                "id": item_id,
                "code": clean_code,
                "target_dependency": target_dependency,
                "has_error": has_error,
                "error_info": error_info_str,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # 使用锁保护文件写入
            with lock:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                # 如果有错误，保存原始诊断信息
                if has_error and raw_diagnostics:
                    raw_result = {
                        "id": item_id,
                        "target_dependency": target_dependency,
                        "generalDiagnostics": raw_diagnostics,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    with open(raw_output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(raw_result, ensure_ascii=False) + '\n')
            
            return True
            
        finally:
            # 清理临时文件
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
    except Exception as e:
        print(f"Error processing item {item_data.get('id', 'unknown')}: {str(e)}")
        return False

def process_jsonl_with_benchmark(jsonl_file_path, benchmark_file_path, output_file, max_workers=4):
    '''
    从JSONL文件读取代码，从benchmark文件读取依赖，使用多线程测试代码并保存结果
    
    Args:
        jsonl_file_path: JSONL文件路径，包含id和answer字段
        benchmark_file_path: benchmark文件路径，包含target_dependency
        output_file: 输出JSONL文件路径
        max_workers: 最大线程数
    
    Returns:
        tuple: (成功测试数量, 总数量)
    '''
    # 设置原始错误信息输出文件
    raw_output_file = output_file.replace('.jsonl', '_raw.jsonl')
    
    # 读取benchmark文件获取依赖映射
    id_to_dependency = read_benchmark_file(benchmark_file_path)
    
    # 获取已处理的结果
    existing_ids = get_existing_results(output_file)
    print(f"Found {len(existing_ids)} existing results")
    
    # 读取所有测试项
    test_items = []
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    if 'id' not in data or 'answer' not in data:
                        continue
                    # 跳过已处理的项目
                    if data['id'] in existing_ids:
                        continue
                    test_items.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line: {line[:100]}... Error: {str(e)}")
                    continue
    except Exception as e:
        print(f"Error reading JSONL file {jsonl_file_path}: {str(e)}")
        return 0, 0
    
    total_count = len(test_items)
    if total_count == 0:
        print("No new items to process")
        return 0, 0
    
    print(f"Processing {total_count} new items...")
    
    # 创建文件写入锁
    file_lock = threading.Lock()
    
    # 使用线程池处理测试项
    success_count = 0
    start_time = time.time()
    
    print(f"Starting parallel processing with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_item = {
            executor.submit(
                process_single_item, 
                item, 
                id_to_dependency, 
                output_file,
                raw_output_file,
                file_lock
            ): item for item in test_items
        }
        
        # 处理完成的任务
        for i, future in enumerate(as_completed(future_to_item), 1):
            item = future_to_item[future]
            try:
                if future.result():
                    success_count += 1
                # 打印进度
                if i % 10 == 0 or i == total_count:
                    elapsed = time.time() - start_time
                    print(f"Progress: {i}/{total_count} ({(i/total_count*100):.1f}%) - "
                          f"Success: {success_count}/{i} - "
                          f"Time elapsed: {elapsed:.1f}s")
            except Exception as e:
                print(f"Error processing item {item.get('id', 'unknown')}: {str(e)}")
    
    total_time = time.time() - start_time
    print(f"\nProcessing completed in {total_time:.1f} seconds")
    print(f"Successfully tested {success_count} out of {total_count} code snippets")
    print(f"Raw error diagnostics saved to {raw_output_file}")
    
    return success_count, total_count

if __name__ == "__main__":
    # Example: Process JSONL file with benchmark
    jsonl_file = "output/approach_eval/BASELINE/Llama-3.1-8B-Instruct/versibcb_vscc_Llama-3.1-8B-Instruct_maxdep10.jsonl"
    benchmark_file = "data/VersiBCB_Benchmark/vscc_datas.json"
    output_file = "data/temp/pyright_test_results_vscc.jsonl"
    
    print("Processing JSONL file with benchmark data...")
    success_count, total_count = process_jsonl_with_benchmark(
        jsonl_file, 
        benchmark_file, 
        output_file,
        max_workers=8  # 可以根据CPU核心数调整
    )
    print(f"Results saved to {output_file}") 