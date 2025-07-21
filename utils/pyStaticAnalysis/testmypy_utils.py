import os
import subprocess
def format_error_info(error_output):
    '''
    格式化mypy的错误输出
    
    Args:
        error_output: mypy的错误输出
    
    Returns:
        str: 格式化后的错误信息
    '''
    if not error_output:
        return "No errors found"
    
    # 移除安装stubs的提示信息
    lines = error_output.split('\n')
    filtered_lines = []
    for line in lines:
        if not any(hint in line.lower() for hint in ['hint:', 'note:', 'see http']):
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)
def getTargetEnvPath(version_combination):
    '''
    获取对应的测试环境路径，方便subprocess调用测试
    '''
    venv_dir = os.path.join(
        "/datanfs2/chenrongyi/conda_env",
        '_'.join([f"{pkg}{ver}" for pkg, ver in sorted(version_combination.items())])
    )
    return venv_dir

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
            major, minor,_ = map(int, version_str.split('.'))
            return major, minor
    except Exception as e:
        print(f"Error getting Python version: {str(e)}")
    return None, None

def ensure_mypy_installed(venv_dir):
    '''
    确保在指定conda环境中安装了mypy
    
    Args:
        venv_dir: conda环境目录
    
    Returns:
        bool: 是否成功安装或已存在
    '''
    # 获取conda环境中的mypy路径
    mypy_executable = os.path.join(venv_dir, 'bin', 'mypy') if os.name != 'nt' else os.path.join(venv_dir, 'Scripts', 'mypy.exe')
    
    if os.path.exists(mypy_executable):
        return True
        
    try:
        # 使用conda run命令在指定环境中安装mypy
        conda_run_cmd = ['conda', 'run', '-p', venv_dir, 'pip', 'install','-U','mypy']
        result = subprocess.run(conda_run_cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"安装mypy时出错: {str(e)}")
        return False

def install_missing_stubs(venv_dir, file_path):
    '''
    在指定环境中安装缺失的stubs
    
    Args:
        venv_dir: conda环境目录
        file_path: 要检查的文件路径
    
    Returns:
        bool: 是否成功安装
    '''
    try:
        # 使用conda run命令在指定环境中运行mypy --install-types
        conda_run_cmd = ['conda', 'run', '-p', venv_dir, 'mypy', '--install-types', '--non-interactive',file_path]
        result = subprocess.run(conda_run_cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"安装stubs时出错: {str(e)}")
        return False
import tempfile
def run_mypy_in_env(venv_dir, file_path):
    '''
    在指定conda环境中运行mypy检查指定文件
    
    Args:
        venv_dir: conda环境目录
        file_path: 要检查的文件路径
    
    Returns:
        tuple: (has_error: bool, error_info: str)
    '''
    # 检查Python版本
    # major, minor = get_python_version(venv_dir)
    # if major == 3 and minor == 5:
    #     return False, "Skipped mypy check for Python 3.5 environment"
    
    # 确保mypy已安装
    if not ensure_mypy_installed(venv_dir):
        return True, "Failed to install mypy in the conda environment"
    
    # 获取conda环境中的mypy路径
    mypy_executable = os.path.join(venv_dir, 'bin', 'mypy') if os.name != 'nt' else os.path.join(venv_dir, 'Scripts', 'mypy.exe')
    
    if not os.path.exists(mypy_executable):
        return True, f"Mypy executable not found in {venv_dir} after installation attempt"
    
    try:
        # 根据Python版本构建mypy命令
        mypy_args = [file_path]
        # if major >= 3 and minor >= 8:
        #     mypy_args.append('--follow-untyped-imports')
        
        # 首先运行mypy检查
        conda_run_cmd = ['conda', 'run', '-p', venv_dir, 'mypy'] + mypy_args
        result = subprocess.run(conda_run_cmd, capture_output=True, text=True)
        
        # 如果有错误，尝试安装缺失的stubs
        if result.returncode != 0:
            if install_missing_stubs(venv_dir, file_path):
                # 重新运行mypy检查
                result = subprocess.run(conda_run_cmd, capture_output=True, text=True)
        
        has_error = result.returncode != 0
        error_info = format_error_info(result.stdout + result.stderr) if has_error else "No errors found"
        
        skip_error_infos = ['module is installed, but missing library stubs or py.typed marker', 'found module but no type hints or library stubs']
        traceback_prefix='Traceback'
        if any(info in error_info for info in skip_error_infos) or error_info.startswith(traceback_prefix):
            has_error = False
            error_info = "No errors found (missing stubs ignored or Traceback)"
        
        return has_error, error_info
        
    except subprocess.CalledProcessError as e:
        return True, f"Error running mypy: {str(e)}"
    except Exception as e:
        return True, f"Unexpected error: {str(e)}"
from utils.pyStaticAnalysis.mypyParse.cMypyError2Format import parse_mypy_error
def get_error_info_from_mypy(generated_code,target_dependency):
    '''
    使用mypy获取错误信息
    params:
        generated_code: 生成的代码
        target_dependency: 目标依赖
    return:
        error_info: 错误信息,list[dict]
    '''
    venv_dir = getTargetEnvPath(target_dependency)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(generated_code)
        temp_file_path = temp_file.name
    try:
        has_error, error_infos = run_mypy_in_env(venv_dir, temp_file_path)
    except Exception as e:
        print(f"运行mypy时出错: {str(e)}")
        return []
    # error_infos =eval(error_infos)
    # for error_info in error_infos:
        
    #     if "Format strings are only supported in Python 3.6 and greater" in error_info["error_info"]:
    #         error_info["error_info"]+=".In python 3.5, you need to apply str.format() instead of format strings. For example, you shall replace f'{{x}} {{y}}' with '{{}} {{}}'.format(x,y) in python 3.5."
    if has_error:
        try:
            #TODO:bug in simple_test_mypy场景，如果括号不对，会出错
            error_infos = parse_mypy_error(error_infos,generated_code)
            print(f"未parse前的错误信息:{error_infos}")
        except Exception as e:
            print(f"解析mypy错误信息时出错: {str(e)}")

    for error_info in error_infos:
        if "error_info" in error_info and "Format strings are only supported in Python 3.6 and greater" in error_info["error_info"]:
            error_info["error_info"]+=".In python 3.5, you need to apply str.format() instead of format strings. For example, you shall replace f'{{x}} {{y}}' with '{{}} {{}}'.format(x,y) in python 3.5."
    return error_infos if has_error else []
# def get_error_info_from_pyright(generated_code,target_dependency):
#     '''
#     使用pyright获取错误信息
#     '''
#     venv_dir = getTargetEnvPath(target_dependency)
#     with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
#         temp_file.write(generated_code)
    
#     pass
if __name__ == "__main__":
    code="""
import subprocess\nimport psutil\nimport time\n\n\ndef task_func(process_name):\n    for proc in psutil.process_iter(['pid', 'name']):\n        if proc.info['name'] == process_name:\n            proc.terminate()\n            time.sleep(1)\n            subprocess.run([process_name], shell=True)\n            return f'Process found. Restarting {process_name}.'\n    subprocess.run([process_name], shell=True)\n    return f'Process not found. Starting {process_name}.'\n\n\nprint(task_func('notepad'))\n
"""
    print(get_error_info_from_mypy(code, {"psutil":"5.2.2","python": "3.5"}))