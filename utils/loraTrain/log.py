import logging
import os
from datetime import datetime

def setup_logging(args, rank=None):
    """
    配置日志记录，支持分布式训练，为每个rank创建单独的日志文件
    """
    if args.log_dir:
        log_dir = args.log_dir
    else:
        # 使用固定的基础目录
        base_log_dir = 'logs'
        # 根据数据集类型和时间戳创建唯一的日志目录
        dataset_name = args.dataset_type
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(base_log_dir, f'train_lora_{dataset_name}_{current_time}')

    os.makedirs(log_dir, exist_ok=True)
    
    # 根据rank确定日志文件名
    if rank is not None:
        log_filename_complete = f"train_lora_{args.dataset_type}_rank{rank}_complete.log"
        log_filename_errors = f"train_lora_{args.dataset_type}_rank{rank}_errors.log"
    else:
        # 兼容单进程模式
        log_filename_complete = f"train_lora_{args.dataset_type}_complete.log"
        log_filename_errors = f"train_lora_{args.dataset_type}_errors.log"

    log_file_path_complete = os.path.join(log_dir, log_filename_complete)
    log_file_path_errors = os.path.join(log_dir, log_filename_errors)

    # 清除现有的handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 配置根logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path_complete, mode='a'),
            logging.StreamHandler()
        ]
    )
    
    # 配置只记录错误的handler
    error_handler = logging.FileHandler(log_file_path_errors, mode='a')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(error_handler)

    # 记录日志目录
    logging.info(f"日志文件将保存在: {log_dir}")
    logging.info(f"完整日志: {log_file_path_complete}")
    logging.info(f"错误日志: {log_file_path_errors}")
    
    return log_dir