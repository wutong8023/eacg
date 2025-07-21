import os
import subprocess
import sys
import json
from utils.pack2download import pack2download
from utils.config import ENV_BASE
def create_virtualenv_and_install(venv_dir, version_combination, conda_command="mamba", 
                                piponly_filepath="cache_library_version/piponly_maxpatchPacks.json",
                                show_output=True,piponly=False,conda_only=False):
    '''
    描述
        创建指定依赖的虚拟环境（对于能通过 conda 安装的包，使用 conda 安装；对于只能通过 pip 安装的包，使用 pip 安装）。
    参数
        venv_dir: 虚拟环境目录路径。
        version_combination: 包及其版本的字典，例如 {"numpy": "1.21.0", "pandas": "1.3.0"}。
        conda_command: conda 命令（默认为 "mamba"）。
        piponly_maxpatchPacks: 只能通过 pip 安装的包及其版本范围（字典形式），例如 {"some_package": {"1.2.3", "1.2.4"}}。
        piponly_filepath: 只能通过 pip 安装的包的配置文件路径。
        show_output: 是否显示安装过程的输出（默认为False）。
    '''
    assert os.path.normpath(venv_dir)!=os.path.normpath(ENV_BASE), "venv_dir不能为ENV_BASE"

    try:
        with open(piponly_filepath,"r") as f:
            piponly_maxpatchPacks = json.load(f)
    except:
        piponly_maxpatchPacks = {}
    conda_packs = []
    pip_packs = []
    if piponly:
        for pack,ver in version_combination.items():
            pip_packs.append(pack)
        pip_packs.remove("python")
        conda_packs.append("python")
    elif conda_only:
        for pack,ver in version_combination.items():
            conda_packs.append(pack)
    else:
        for pack,ver in version_combination.items():
            if pack in piponly_maxpatchPacks:
                if ver in piponly_maxpatchPacks[pack]:
                    pip_packs.append(pack)
                else:
                    conda_packs.append(pack)
            else:
                conda_packs.append(pack)
    
    package_versions = [f"{pack2download(pkg,type='conda')}={ver}" for pkg, ver in version_combination.items() if pkg not in pip_packs]
    create_env_cmd = [conda_command, 'create', '-p', venv_dir] + package_versions + ['--yes']
    
    if os.path.exists(venv_dir):
        if show_output:
            print("虚拟环境目录已存在: {}".format(venv_dir))

        # Verify the existence of the Python executable
        python_executable = os.path.join(venv_dir, 'bin', 'python') if os.name != 'nt' else os.path.join(venv_dir, 'Scripts', 'python.exe')
        if not os.path.exists(python_executable):
            if show_output:
                print("未找到虚拟环境的 Python 可执行文件: {}，将重新创建环境。".format(python_executable))
            try:
                subprocess.check_call(['rm', '-rf', venv_dir])
                if show_output:
                    subprocess.check_call(create_env_cmd)
                else:
                    subprocess.check_call(create_env_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                if show_output:
                    print("安装依赖失败: {}".format(e))
                raise Exception("安装依赖失败: {}".format(e))
            if show_output:
                print("虚拟环境 {} 创建成功，依赖安装完成。".format(venv_dir))

    else:
        if show_output:
            print("使用 mamba 创建虚拟环境并安装依赖: {}".format(package_versions))
        try:
            if show_output:
                subprocess.check_call(create_env_cmd)
            else:
                subprocess.check_call(create_env_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            if show_output:
                print("未找到 mamba，请确保其已安装并在 PATH 中。")
            raise Exception("未找到 mamba，请确保其已安装并在 PATH 中。")
        except subprocess.CalledProcessError as e:
            if show_output:
                print("安装依赖失败: {}".format(e))
            raise Exception("安装依赖失败: {}".format(e))
        
        if show_output:
            print("虚拟环境 {} 创建成功，依赖安装完成。".format(venv_dir))

    # 继续在指定mamba虚拟环境中通过pip安装piponly_packs
    if pip_packs:
        if show_output:
            print("开始通过pip安装包: {}".format(pip_packs))
        python_executable = os.path.join(venv_dir, 'bin', 'python') if os.name != 'nt' else os.path.join(venv_dir, 'Scripts', 'python.exe')
        pip_executable = os.path.join(venv_dir, 'bin', 'pip') if os.name != 'nt' else os.path.join(venv_dir, 'Scripts', 'pip.exe')
        
        for pkg in pip_packs:
            ver = version_combination[pkg]
            pip_install_cmd = [python_executable, "-m", "pip", "install", f"{pack2download(pkg)}=={ver}"]
            try:
                if show_output:
                    subprocess.check_call(pip_install_cmd)
                else:
                    subprocess.check_call(pip_install_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if show_output:
                    print("成功安装 {}=={}".format(pkg, ver))
            except subprocess.CalledProcessError as e:
                if show_output:
                    print("pip安装 {}=={} 失败: {}".format(pkg, ver, e))
                # 清理环境并退出
                subprocess.check_call(['rm', '-rf', venv_dir])
                raise Exception(f"pip安装 {pkg}=={ver} 失败: {e}")
