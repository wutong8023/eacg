from config.createTarcode_config import ENV_BASE,OUTPUT_TEST_INFO,OUTPUT_INSTALL_INFO,CONDA_COMMAND,TEST_FILE_PATH,ERROR_LOG_PATH,SINGLE_PROCESS,PARALLEL_ROAD_SEARCH,PARALLEL_ROAD_SEARCH_THREAD
import os
import traceback
import subprocess

def passTaskTest1(task_id,dep,outputTestInfo=True,specifiedcode=None,test_file_path=TEST_FILE_PATH,DeprecateAdapt=False,single_thread=None,return_test_stats=False):
    '''
        deprecateAdapt: bool,如果启用会额外返回stderr信息，方便进行deprecation验证
    '''
    import tempfile
    import shutil
    
    #使用single-thread override SINGLE_PROCESS
    if single_thread!=None:
        if single_thread==False:
            test_file_path = wrap_filepath_with_thread_id(test_file_path)
    elif not SINGLE_PROCESS:
        test_file_path = wrap_filepath_with_thread_id(test_file_path)
    
    BCB_dataset = load_bcb_dataset()
    if specifiedcode is None:
        function_code = get_refcode_by_taskid(BCB_dataset,task_id)
        test_code = get_testcases_by_taskid(BCB_dataset,task_id)
    else:
        function_code = specifiedcode[0]
        test_code = specifiedcode[1]
    
    version_combination = dep
    venv_dir = os.path.join(
        ENV_BASE,
        '_'.join([f"{pkg}{ver}" for pkg, ver in sorted(version_combination.items())])
    )
    
    matplotlib_Agg ='import matplotlib.pyplot as plt\nplt.switch_backend("Agg")\n' if 'matplotlib' in version_combination else ''
    
    # 创建临时目录确保父目录存在
    temp_dir = os.path.dirname(test_file_path)
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
    
    # 写入测试文件
    with open(test_file_path, 'w') as f_tests:
        f_tests.write(matplotlib_Agg+function_code+"\n"+test_code+"\n")
    
    try:
        create_virtualenv_and_install(venv_dir, version_combination,conda_command=CONDA_COMMAND,show_output=OUTPUT_INSTALL_INFO)
    except Exception as e:
        print(f"虚拟环境创建失败,环境有误，跳过: {e}")
        return True
    
    try:
        result = run_tests_in_virtualenv(venv_dir, test_file_path,outputTestInfo=outputTestInfo, return_test_stats=return_test_stats)
        test_stats = getattr(result, 'test_stats', None)
        
        if test_stats:
            print(f"refcode测试通过，{task_id}的依赖{dep}无需适配 - 测试统计: {test_stats['passed_tests']}/{test_stats['total_tests']} 通过 (成功率: {test_stats['success_rate']:.2%})")
        else:
            print(f"refcode测试通过，{task_id}的依赖{dep}无需适配")
            
        if not DeprecateAdapt:
            return True
        else:
            return (True, result.stderr, test_stats)

    except subprocess.CalledProcessError as e:
        error_details = {
            'stderr': e.stderr,
            'stdout': e.stdout,
            'returncode': e.returncode,
            'cmd': str(e.cmd)
        }
        
        # 尝试解析失败测试的统计信息
        failed_test_stats = parse_unittest_output(e.stdout if e.stdout else "", e.stderr if e.stderr else "")
        
        with open(f"{ERROR_LOG_PATH}/task_{str(task_id).replace('/', '_')}_error.log","a") as f:
            f.write(f"refcode测试失败，{task_id}的依赖{dep}需要适配\n")
            f.write(f"错误信息:\n{e}\n")
            f.write(f"stderr:\n{e.stderr}\n")
            f.write(f"stdout:\n{e.stdout}\n")
            if failed_test_stats:
                f.write(f"测试统计: 总数={failed_test_stats['total_tests']}, 通过={failed_test_stats['passed_tests']}, "
                       f"失败={failed_test_stats['failed_tests']}, 错误={failed_test_stats['error_tests']}\n")
            f.write(f"Traceback:\n {traceback.format_exc()}\n")
        
        if failed_test_stats:
            print(f"refcode测试失败，{task_id}的依赖{dep}需要适配 - 测试统计: {failed_test_stats['passed_tests']}/{failed_test_stats['total_tests']} 通过")
        else:
            print(f"refcode测试失败，{task_id}的依赖{dep}需要适配")
        
        if DeprecateAdapt:
            return (False, error_details, failed_test_stats)
        else:
            return False
    
    except Exception as e:
        error_details = {
            'exception': str(e),
            'traceback': traceback.format_exc()
        }
        
        with open(f"{ERROR_LOG_PATH}/task_{str(task_id).replace('/', '_')}_error.log","a") as f:
            f.write(f"测试过程中发生异常，{task_id}的依赖{dep}\n")
            f.write(f"异常信息:\n{e}\n")
            f.write(f"Traceback:\n {traceback.format_exc()}\n")
        print(f"测试过程中发生异常，{task_id}的依赖{dep}: {e}")
        
        # 对于一般异常，没有测试统计信息
        general_exception_stats = None
        
        if DeprecateAdapt:
            return (False, error_details, general_exception_stats)
        else:
            return False