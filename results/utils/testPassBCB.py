
import os
import subprocess
import traceback
import sys
from utils.testPassBCButils import create_virtualenv_and_install
from utils.logger.logger_util import wrap_filepath_with_thread_id
from utils.config import TEST_FILE_PATH,CONDA_COMMAND,OUTPUT_INSTALL_INFO,SINGLE_PROCESS,ERROR_LOG_PATH,ENV_BASE
def run_tests_in_virtualenv(venv_dir,test_file_path,outputTestInfo=True):
    '''
    #TODO:尝试修改testPassBCB和该逻辑之间的连接关系，直接根据returncode，而非通过raise
    '''
    python_executable = os.path.join(venv_dir, 'bin', 'python') if os.name != 'nt' else os.path.join(venv_dir, 'Scripts', 'python.exe')

    if not os.path.exists(python_executable):
        print(f"未找到虚拟环境的 Python 可执行文件: {python_executable}")
        sys.exit(1)

    print("运行测试中...")
    # 使用 subprocess.run 替代 check_call，并捕获输出
    result = subprocess.run(
        [python_executable, "-m", "unittest", test_file_path],
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
def passTaskTest(task_id,dep,outputTestInfo=True,specifiedcode=None,test_file_path=TEST_FILE_PATH,DeprecateAdapt=False):
    '''
    params:
        task_id: str
        dep: dict[str,str]
        outputTestInfo: bool
        specifiedcode: tuple[str,str]
        test_file_path: str
        DeprecateAdapt: bool,如果启用会额外返回stderr信息，方便进行deprecation验证
    return:
        bool if DeprecateAdapt is False
        tuple[bool,stderr] if DeprecateAdapt is True
    '''
    if type(task_id) is int:
        task_id = str(task_id)
    if not SINGLE_PROCESS:
        test_file_path = wrap_filepath_with_thread_id(test_file_path)
    # function_file_path = 'utils/function.py'

    # 格式为dict[task_id,dict[frozenset(dep),bool]]  True需要adapt，False不需要adapt
    function_code = specifiedcode[0]
    test_code = specifiedcode[1]
    version_combination = dep
    venv_dir = os.path.join(
        ENV_BASE,
        '_'.join([f"{pkg}{ver}" for pkg, ver in sorted(version_combination.items())])
        
    )
    matplotlib_Agg ='import matplotlib.pyplot as plt\nplt.switch_backend("Agg")\n' if 'matplotlib' in version_combination else ''
    # unit_test_code = 'if __name__ == "__main__":\n    unittest.main()'
    with open(test_file_path, 'w') as f_tests:
        f_tests.write(matplotlib_Agg+function_code+"\n"+test_code+"\n")
    try:
        create_virtualenv_and_install(venv_dir, version_combination,conda_command=CONDA_COMMAND,show_output=OUTPUT_INSTALL_INFO)
    except Exception as e:
        print(f"虚拟环境创建失败,环境有误，跳过: {e}")
        return True
    
    
    # if DeprecateAdapt:
    # # 进行deprecate测试，判断是否通过测试且需要removeDeprecation
    #     try:
    #         result = run_tests_in_virtualenv(venv_dir, test_file_path,outputTestInfo=outputTestInfo)
    #         return True,result.stderr
    #     except Exception as e:
    #         print(f"deprecate测试失败，{task_id}的依赖{dep}需要适配\n")
    #         return False,None
    # else:
    try:
        result = run_tests_in_virtualenv(venv_dir, test_file_path,outputTestInfo=outputTestInfo)
        print(f"refcode测试通过，{task_id}的依赖{dep}无需适配")
        return True if not DeprecateAdapt else (True,result.stderr)

    except subprocess.CalledProcessError as e:
        with open(f"{ERROR_LOG_PATH}/task_{task_id.replace('/', '_')}_error.log","a") as f:
            f.write(f"refcode测试失败，{task_id}的依赖{dep}需要适配\n")
            f.write(f"错误信息:\n{e}\n")
            f.write(f"stderr:\n{e.stderr}\n")
            f.write(f"stdout:\n{e.stdout}\n")
            f.write(f"Traceback:\n {traceback.format_exc()}\n")
        print(f"refcode测试失败，{task_id}的依赖{dep}需要适配")
        result = False if not DeprecateAdapt else (False,None)
        return result
    except Exception as e:
        with open(f"logs/error_log/task_{task_id.replace('/', '_')}_error.log", "a") as f:
            # 处理其他异常
            f.write(f"发生未知错误: {str(e)}\n")
            f.write("主进程的 Traceback:\n")
            traceback.print_exc(file=f)
        return False if not DeprecateAdapt else (False,None)