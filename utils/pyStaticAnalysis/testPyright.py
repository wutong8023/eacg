import subprocess
import os
import sys
import json
import tempfile
import datetime

def getTargetEnvPath(version_combination):
    '''
    获取对应的测试环境路径，方便subprocess调用测试
    '''
    venv_dir = os.path.join(
        "/datanfs2/chenrongyi/conda_env",
        '_'.join([f"{pkg}{ver}" for pkg, ver in sorted(version_combination.items())])
        
    )
    return venv_dir

def run_tests_in_virtualenv(venv_dir,test_file_path,outputTestInfo=True):
    '''
    在虚拟环境中运行测试，可选择返回测试统计信息
    
    Args:
        venv_dir: 虚拟环境目录
        test_file_path: 测试文件路径
        outputTestInfo: 是否输出测试信息
        return_test_stats: 是否返回测试统计信息
    
    Returns:
        如果return_test_stats=True，返回包含test_stats的result对象
        否则返回原来的result对象
    '''
    python_executable = os.path.join(venv_dir, 'bin', 'python') if os.name != 'nt' else os.path.join(venv_dir, 'Scripts', 'python.exe')

    if not os.path.exists(python_executable):
        print(f"未找到虚拟环境的 Python 可执行文件: {python_executable}")
        sys.exit(1)

    print("运行测试中...")
    # 使用 subprocess.run 替代 check_call，并捕获输出
    # test_dir = os.path.join(os.path.dirname(__file__), "test")  # 获取test目录绝对路径
    result = subprocess.run(
        [python_executable, test_file_path],
        capture_output=True,  # 捕获stdout和stderr
        text=True,  # 将输出转换为字符串
    )
    
    
    if result.returncode != 0:
        if outputTestInfo:
            print("测试运行失败。")
            print("错误输出：")
            print(result.stderr)
        # 创建一个包含完整输出信息的CalledProcessError
        raise subprocess.CalledProcessError(
            returncode=result.returncode,
            cmd=result.args,
            output=result.stdout,
            stderr=result.stderr
        )
    if outputTestInfo:
        print(result.stdout)
    return result

def ensure_pyright_installed(venv_dir):
    '''
    确保在指定conda环境中安装了pyright
    
    Args:
        venv_dir: conda环境目录
    
    Returns:
        bool: 是否成功安装或已存在
    '''
    # 获取conda环境中的pip路径
    pip_executable = os.path.join(venv_dir, 'bin', 'pip') if os.name != 'nt' else os.path.join(venv_dir, 'Scripts', 'pip.exe')
    pyright_executable = os.path.join(venv_dir, 'bin', 'pyright') if os.name != 'nt' else os.path.join(venv_dir, 'Scripts', 'pyright.exe')
    
    if os.path.exists(pyright_executable):
        return True
        
    try:
        # 使用conda run命令在指定环境中安装pyright
        conda_run_cmd = ['conda', 'run', '-p', venv_dir, 'pip', 'install', 'pyright']
        
        result = subprocess.run(
            conda_run_cmd,
            capture_output=True,
            text=True
        )
        return result.returncode == 0
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
                        # 添加错误位置标记
                        # pointer = ' ' * start_char + '^' * 3
                        formatted_errors.append(
                            # f"File: {file_path}\n"
                            f"Line {start_line}: {error_line}\n"
                            # f"      {pointer}\n"
                            f"Error: {message}\n"
                        )
            except Exception as e:
                formatted_errors.append(f"Error reading file {file_path}: {str(e)}")
        
        return "\n".join(formatted_errors)
    except json.JSONDecodeError:
        return f"Failed to parse pyright output: {error_json}"
    except Exception as e:
        return f"Error processing pyright output: {str(e)}"

def get_python_version(venv_dir):
    '''
    获取conda环境中的Python版本
    
    Args:
        venv_dir: conda环境目录
    
    Returns:
        tuple: (major, minor) 版本号
    '''
    try:
        conda_run_cmd = ['conda', 'run', '-p', venv_dir, 'python', '-c', 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")']
        result = subprocess.run(conda_run_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            version_str = result.stdout.strip()
            major, minor = map(int, version_str.split('.'))
            return major, minor
    except Exception as e:
        print(f"Error getting Python version: {str(e)}")
    return None, None

def run_pyright_in_env(venv_dir, file_path):
    '''
    在指定conda环境中运行pyright检查指定文件
    
    Args:
        venv_dir: conda环境目录
        file_path: 要检查的文件路径
    
    Returns:
        tuple: (has_error: bool, error_info: str)
    '''
    # 检查Python版本
    major, minor = get_python_version(venv_dir)
    if major == 3 and minor == 5:
        return False, "Skipped pyright check for Python 3.5 environment"
    
    # 确保pyright已安装
    if not ensure_pyright_installed(venv_dir):
        return True, "Failed to install pyright in the conda environment"
    
    # 获取conda环境中的pyright路径
    pyright_executable = os.path.join(venv_dir, 'bin', 'pyright') if os.name != 'nt' else os.path.join(venv_dir, 'Scripts', 'pyright.exe')
    
    if not os.path.exists(pyright_executable):
        return True, f"Pyright executable not found in {venv_dir} after installation attempt"
    
    try:
        # 使用conda run命令在指定环境中运行pyright
        conda_run_cmd = ['conda', 'run', '-p', venv_dir, 'pyright', '--outputjson', file_path]
        
        result = subprocess.run(
            conda_run_cmd,
            capture_output=True,
            text=True
        )
        
        has_error = result.returncode != 0
        error_info = format_error_info(result.stdout) if has_error else "No errors found"
        
        return has_error, error_info
        
    except subprocess.CalledProcessError as e:
        return True, f"Error running pyright: {str(e)}"
    except Exception as e:
        return True, f"Unexpected error: {str(e)}"

def test_code_content(venv_dir, code_content, output_file):
    '''
    测试代码内容并将结果保存到JSONL文件
    
    Args:
        venv_dir: 虚拟环境目录
        code_content: 要测试的代码内容
        output_file: 输出JSONL文件路径
    
    Returns:
        bool: 是否成功完成测试和保存
    '''
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(code_content)
        temp_file_path = temp_file.name
    
    try:
        # 运行pyright检查
        has_error, error_info = run_pyright_in_env(venv_dir, temp_file_path)
        
        # 准备输出数据
        result = {
            "code": code_content,
            "has_error": has_error,
            "error_info": error_info,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # 写入JSONL文件
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
        return True
        
    except Exception as e:
        print(f"Error testing code: {str(e)}")
        return False
    finally:
        # 清理临时文件
        try:
            os.unlink(temp_file_path)
        except:
            pass

def batch_test_code_contents(venv_dir, code_contents, output_file):
    '''
    批量测试多个代码内容
    
    Args:
        venv_dir: 虚拟环境目录
        code_contents: 代码内容列表
        output_file: 输出JSONL文件路径
    
    Returns:
        int: 成功测试的代码数量
    '''
    success_count = 0
    for code in code_contents:
        if test_code_content(venv_dir, code, output_file):
            success_count += 1
    return success_count

def clean_model_output(output):
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
    
    Args:
        benchmark_file_path: benchmark文件路径
    
    Returns:
        dict: id到target_dependency的映射
    '''
    try:
        with open(benchmark_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        id_to_dependency = {}
        for item in data:
            if 'id' in item and 'target_dependency' in item:
                id_to_dependency[item['id']] = item['target_dependency']
        
        return id_to_dependency
    except Exception as e:
        print(f"Error reading benchmark file {benchmark_file_path}: {str(e)}")
        return {}

def process_jsonl_with_benchmark(jsonl_file_path, benchmark_file_path, output_file):
    '''
    从JSONL文件读取代码，从benchmark文件读取依赖，测试代码并保存结果
    
    Args:
        jsonl_file_path: JSONL文件路径，包含id和answer字段
        benchmark_file_path: benchmark文件路径，包含target_dependency
        output_file: 输出JSONL文件路径
    
    Returns:
        tuple: (成功测试数量, 总数量)
    '''
    # 读取benchmark文件获取依赖映射
    id_to_dependency = read_benchmark_file(benchmark_file_path)
    
    success_count = 0
    total_count = 0
    
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
                    
                    item_id = data['id']
                    answer = data['answer']
                    
                    # 获取对应的依赖信息
                    if item_id not in id_to_dependency:
                        print(f"Warning: No dependency found for id {item_id}")
                        continue
                    
                    target_dependency = id_to_dependency[item_id]
                    venv_dir = getTargetEnvPath(target_dependency)
                    
                    # 清理模型输出
                    clean_code = clean_model_output(answer)
                    
                    # 测试代码
                    if test_code_content_with_id(venv_dir, clean_code, output_file, item_id, target_dependency):
                        success_count += 1
                    
                    total_count += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line: {line[:100]}... Error: {str(e)}")
                    continue
                except Exception as e:
                    print(f"Error processing item: {str(e)}")
                    continue
    
    except Exception as e:
        print(f"Error reading JSONL file {jsonl_file_path}: {str(e)}")
        return success_count, total_count
    
    return success_count, total_count

def test_code_content_with_id(venv_dir, code_content, output_file, item_id, target_dependency):
    '''
    测试代码内容并将结果保存到JSONL文件（包含ID和依赖信息）
    
    Args:
        venv_dir: 虚拟环境目录
        code_content: 要测试的代码内容
        output_file: 输出JSONL文件路径
        item_id: 项目ID
        target_dependency: 目标依赖信息
    
    Returns:
        bool: 是否成功完成测试和保存
    '''
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(code_content)
        temp_file_path = temp_file.name
    
    try:
        # 运行pyright检查
        has_error, error_info = run_pyright_in_env(venv_dir, temp_file_path)
        
        # 准备输出数据
        result = {
            "id": item_id,
            "code": code_content,
            "target_dependency": target_dependency,
            "has_error": has_error,
            "error_info": error_info,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # 写入JSONL文件
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
        return True
        
    except Exception as e:
        print(f"Error testing code for id {item_id}: {str(e)}")
        return False
    finally:
        # 清理临时文件
        try:
            os.unlink(temp_file_path)
        except:
            pass

# Example usage
if __name__ == "__main__":
    import datetime
    
    # Example: Process JSONL file with benchmark
    jsonl_file = "output/approach_eval/BASELINE/Llama-3.1-8B-Instruct/versibcb_vscc_Llama-3.1-8B-Instruct_maxdep10.jsonl"
    benchmark_file = "data/VersiBCB_Benchmark/vscc_datas.json"
    output_file = "data/temp/pyright_test_results_vscc.jsonl"
    
    print("Processing JSONL file with benchmark data...")
    success_count, total_count = process_jsonl_with_benchmark(
        jsonl_file, 
        benchmark_file, 
        output_file
    )
    print(f"Successfully tested {success_count} out of {total_count} code snippets")
    print(f"Results saved to {output_file}")    
    # Alternative: Direct testing with specific environment (for debugging)
    # test_env = getTargetEnvPath({"matplotlib": "3.7.0", "pandas": "2.0.3", "python": "3.8", "seaborn": "0.13.2"})
    # test_code = clean_model_output("```python\nimport pandas as pd\n```")
    # has_error, error_info = run_pyright_in_env(test_env, "temp_test.py")
    # print(f"Direct test - Has error: {has_error}, Info: {error_info}")
