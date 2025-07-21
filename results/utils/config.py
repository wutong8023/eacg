from config.paths import CONDA_ENV_PATH

# Test environment location
ENV_BASE = CONDA_ENV_PATH
TEST_FILE_PATH = "data/temp/test_function.py" # 临时测试文件位置
CONDA_COMMAND = "mamba" # "mamba" or "conda", 创建测试环境的conda命令
SINGLE_PROCESS = False# 是否single thread运行
OUTPUT_INSTALL_INFO = False# 是否输出安装信息,创建测试环境用
ERROR_LOG_PATH = "logs/error_log/" # 错误日志位置
