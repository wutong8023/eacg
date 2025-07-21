# Configuration settings for the application

from config.paths import CONDA_ENV_PATH

BACKUP_ENV_BASE = CONDA_ENV_PATH
ENV_BASE = CONDA_ENV_PATH
JUMP_INSTALL = True # 是否跳过需要新建的环境
# Other configuration settings can be added here
CONDA_COMMAND = "mamba" # "mamba" or "conda", 创建测试环境的conda命令
OUTPUT_TEST_INFO = False # 是否输出测试信息
OUTPUT_INSTALL_INFO = True# 是否输出安装信息
ANNOTATE = True# 是否标注，不标注就只找bound

HUMAN_EDIT = False # 是否启用人工编辑
HUMAN_CHOICE = False#是否save的一些选项(saveTarcode,saveEvolveinfo)
AI_EVOLVEINFO = True # 是否启用AI生成evolveinfo

SINGLE_PROCESS = False# 是否single thread运行

START_TASK_ID =1# 开始注释的task_id
END_TASK_ID=1145
TEST_FILE_PATH = "data/temp/test_function.py"
ERROR_LOG_PATH = "logs/error_log"
DEPRECATION_KEYWORDS =["DeprecationWarning","FutureWarning","PendingDeprecationWarning","DeprecationWarning",'deprecate']
DEPRECATION_LOG_PATH = "logs/deprecate_log"
DEPRECATE_ADAPT = True # 是否启用deprecate adapt

PARALLEL_ROAD_SEARCH = False # 是否启用并行搜索
PARALLEL_ROAD_SEARCH_THREAD = 1 # 并行搜索的线程数

UPDATE_PROCESS_BOUND = False # 是否更新process bound
TO_FILTER_TASK_ID = ["BigCodeBench/21","BigCodeBench/51","BigCodeBench/82","BigCodeBench/101","BigCodeBench/119","BigCodeBench/40","BigCodeBench/108","BigCodeBench/215","BigCodeBench/303","BigCodeBench/383","BigCodeBench/361","BigCodeBench/362","BigCodeBench/235","BigCodeBench/227","BigCodeBench/352","BigCodeBench/409","BigCodeBench/410"]