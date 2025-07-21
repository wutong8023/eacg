import threading
from utils.config import SINGLE_PROCESS
# 创建全局线程ID映射字典和对应的锁
thread_id2id = {}
thread_id2id_lock = threading.Lock()
next_id = 0

def get_thread_sequential_id():
    """
    为线程获取顺序ID，如果已存在则返回已分配的ID，
    如果不存在则分配新的ID
    """
    global next_id
    thread_id = threading.get_ident()
    
    with thread_id2id_lock:
        if thread_id not in thread_id2id:
            thread_id2id[thread_id] = next_id
            next_id += 1
        return thread_id2id[thread_id]

def wrap_filepath_with_thread_id(filepath):
    '''
    将文件名加上顺序线程id
    '''
    if SINGLE_PROCESS:
        return filepath
    sequential_id = get_thread_sequential_id()
    filepath_no_suffix = filepath.split(".")[0]
    suffix = filepath.split(".")[1]
    filepath_thread_id = f"{filepath_no_suffix}_{sequential_id}.{suffix}"
    return filepath_thread_id

def clear_thread_id_mapping():
    """
    清理线程ID映射
    在需要重置映射关系时调用
    """
    global next_id
    with thread_id2id_lock:
        thread_id2id.clear()
        next_id = 0

class Tee:
    def __init__(self, *files):
        self.files = files
        self._error_reported = False

    def write(self, obj):
        for f in self.files:
            try:
                # Check if f has the necessary file-like attributes
                if hasattr(f, 'write') and hasattr(f, 'closed') and not f.closed:
                    f.write(obj)
                    f.flush()
            except Exception as e:
                if not self._error_reported:
                    self._error_reported = True
                    import sys
                    sys.__stderr__.write(f"Write error: {type(e).__name__}: {str(e)}\n")
                    sys.__stderr__.write(f"File object type: {type(f)}\n")
                    sys.__stderr__.write(f"File status: {'closed' if hasattr(f, 'closed') and f.closed else 'N/A'}\n")
                    self._error_reported = False
    
    def flush(self):
        for f in self.files:
            try:
                # Check if f has the necessary file-like attributes
                if hasattr(f, 'flush') and hasattr(f, 'closed') and not f.closed:
                    f.flush()
            except Exception as e:
                if not self._error_reported:
                    self._error_reported = True
                    import sys
                    sys.__stderr__.write(f"Flush error: {type(e).__name__}: {str(e)}\n")
                    sys.__stderr__.write(f"File object type: {type(f)}\n")
                    sys.__stderr__.write(f"File status: {'closed' if hasattr(f, 'closed') and f.closed else 'N/A'}\n")
                    self._error_reported = False