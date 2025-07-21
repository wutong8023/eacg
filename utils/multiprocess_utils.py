import json
import os
import time
import fcntl
import tempfile
import shutil
from pathlib import Path
import logging

class MultiprocessSafeFileHandler:
    """多进程安全的文件操作类"""
    
    def __init__(self, max_retries=5, retry_delay=0.1):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def safe_write_json(self, data, filepath, use_temp_file=True):
        """
        多进程安全的JSON文件写入
        
        Args:
            data: 要写入的数据
            filepath: 目标文件路径
            use_temp_file: 是否使用临时文件原子写入
        """
        if use_temp_file:
            return self._atomic_write_json(data, filepath)
        else:
            return self._locked_write_json(data, filepath)
    
    def safe_read_json(self, filepath, default=None):
        """
        多进程安全的JSON文件读取
        
        Args:
            filepath: 文件路径
            default: 读取失败时的默认值
            
        Returns:
            读取的数据或默认值
        """
        for attempt in range(self.max_retries):
            try:
                if not os.path.exists(filepath):
                    logging.warning(f"File {filepath} does not exist")
                    return default
                
                # 检查文件是否为空或正在写入
                if os.path.getsize(filepath) == 0:
                    logging.warning(f"File {filepath} is empty, retrying...")
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    # 使用文件锁
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # 共享锁
                    try:
                        content = f.read().strip()
                        if not content:
                            logging.warning(f"File {filepath} content is empty")
                            return default
                        
                        data = json.loads(content)
                        logging.info(f"Successfully read JSON from {filepath}")
                        return data
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # 释放锁
                        
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    logging.error(f"Failed to parse JSON from {filepath} after {self.max_retries} attempts")
                    return default
                    
            except (IOError, OSError) as e:
                logging.error(f"File I/O error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    logging.error(f"Failed to read file {filepath} after {self.max_retries} attempts")
                    return default
                    
            except Exception as e:
                logging.error(f"Unexpected error reading {filepath}: {e}")
                return default
        
        return default
    
    def _atomic_write_json(self, data, filepath):
        """使用临时文件的原子写入操作"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 创建临时文件
            temp_fd, temp_path = tempfile.mkstemp(
                suffix='.json.tmp',
                dir=os.path.dirname(filepath)
            )
            
            try:
                with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())  # 强制写入磁盘
                
                # 原子性移动文件
                shutil.move(temp_path, filepath)
                logging.info(f"Successfully wrote JSON to {filepath} using atomic operation")
                return True
                
            except Exception as e:
                # 清理临时文件
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise e
                
        except Exception as e:
            logging.error(f"Failed to write JSON to {filepath}: {e}")
            return False
    
    def _locked_write_json(self, data, filepath):
        """使用文件锁的写入操作"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                # 使用排他锁
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())  # 强制写入磁盘
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # 释放锁
            
            logging.info(f"Successfully wrote JSON to {filepath} using file lock")
            return True
            
        except Exception as e:
            logging.error(f"Failed to write JSON to {filepath}: {e}")
            return False
    
    def wait_for_file(self, filepath, timeout=30):
        """等待文件存在且可读"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                # 尝试读取以验证文件完整性
                try:
                    with open(filepath, 'r') as f:
                        json.load(f)
                    return True
                except json.JSONDecodeError:
                    pass  # 继续等待
            time.sleep(0.1)
        return False
    
    def backup_and_recover(self, filepath, backup_suffix='.backup'):
        """备份和恢复文件"""
        backup_path = filepath + backup_suffix
        
        # 创建备份
        if os.path.exists(filepath):
            try:
                shutil.copy2(filepath, backup_path)
                logging.info(f"Created backup: {backup_path}")
            except Exception as e:
                logging.error(f"Failed to create backup: {e}")
        
        # 恢复函数
        def recover():
            if os.path.exists(backup_path):
                try:
                    shutil.copy2(backup_path, filepath)
                    logging.info(f"Recovered from backup: {backup_path}")
                    return True
                except Exception as e:
                    logging.error(f"Failed to recover from backup: {e}")
            return False
        
        return recover

# 全局实例
safe_file_handler = MultiprocessSafeFileHandler()

def safe_write_coordination_data(data, filepath):
    """安全写入coordination数据"""
    return safe_file_handler.safe_write_json(data, filepath, use_temp_file=True)

def safe_read_coordination_data(filepath):
    """安全读取coordination数据"""
    return safe_file_handler.safe_read_json(filepath, default={"unprocessed_data": []}) 