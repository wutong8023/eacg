import torch
import time
import logging
import os
from collections import defaultdict
from datetime import datetime

class GPUMemoryProfiler:
    def __init__(self, log_dir="logs/memory_profiler", enable_file_logging=True):
        self.records = defaultdict(list)
        self.start_time = time.time()
        self.enable_file_logging = enable_file_logging
        
        # 设置日志记录
        if enable_file_logging:
            self._setup_logging(log_dir)
        else:
            self.logger = None
    
    def _setup_logging(self, log_dir):
        """设置日志记录器"""
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"memory_profiler_{timestamp}.log")
        
        # 配置日志记录器
        self.logger = logging.getLogger(f"MemoryProfiler_{timestamp}")
        self.logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 记录初始化信息
        self.logger.info("=" * 80)
        self.logger.info("GPU内存分析器初始化")
        self.logger.info("=" * 80)
        self.logger.info(f"日志文件: {log_file}")
        self.logger.info(f"CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            self.logger.info(f"GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                self.logger.info(f"GPU {i}: {gpu_name} ({total_memory:.1f}GB)")
    
    def _get_all_gpu_memory_info(self):
        """获取所有GPU的内存信息"""
        gpu_memory_info = []
        
        if not torch.cuda.is_available():
            return gpu_memory_info
        
        for i in range(torch.cuda.device_count()):
            try:
                # 设置当前设备以获取准确信息
                torch.cuda.set_device(i)
                
                allocated = torch.cuda.memory_allocated(i) / 1024**2
                reserved = torch.cuda.memory_reserved(i) / 1024**2
                max_allocated = torch.cuda.max_memory_allocated(i) / 1024**2
                max_reserved = torch.cuda.max_memory_reserved(i) / 1024**2
                total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**2
                
                gpu_info = {
                    'device_id': i,
                    'allocated': allocated,
                    'reserved': reserved,
                    'max_allocated': max_allocated,
                    'max_reserved': max_reserved,
                    'total_memory': total_memory,
                    'free_memory': total_memory - reserved,
                    'utilization_percent': (allocated / total_memory) * 100
                }
                gpu_memory_info.append(gpu_info)
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"获取GPU {i} 内存信息失败: {e}")
                continue
        
        return gpu_memory_info
    
    def record(self, event_name, additional_info=None):
        """记录所有GPU的内存使用情况"""
        if not torch.cuda.is_available():
            if self.logger:
                self.logger.warning("CUDA不可用，无法记录GPU内存")
            return
        
        # 获取所有GPU的内存信息
        gpu_memory_info = self._get_all_gpu_memory_info()
        
        if not gpu_memory_info:
            if self.logger:
                self.logger.warning("无法获取任何GPU的内存信息")
            return
        
        # 计算总体统计
        total_allocated = sum(gpu['allocated'] for gpu in gpu_memory_info)
        total_reserved = sum(gpu['reserved'] for gpu in gpu_memory_info)
        total_max_allocated = sum(gpu['max_allocated'] for gpu in gpu_memory_info)
        total_max_reserved = sum(gpu['max_reserved'] for gpu in gpu_memory_info)
        
        # 获取当前时间
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # 记录内存信息
        memory_info = {
            "time": elapsed_time,
            "total_allocated": total_allocated,
            "total_reserved": total_reserved,
            "total_max_allocated": total_max_allocated,
            "total_max_reserved": total_max_reserved,
            "gpu_details": gpu_memory_info,
            "timestamp": datetime.fromtimestamp(current_time).isoformat()
        }
        
        self.records[event_name].append(memory_info)
        
        # 记录到日志
        if self.logger:
            # 总体信息
            log_message = f"事件: {event_name} | 总分配: {total_allocated:7.1f}MB | 总保留: {total_reserved:7.1f}MB | 总最大分配: {total_max_allocated:7.1f}MB | 总最大保留: {total_max_reserved:7.1f}MB | 耗时: {elapsed_time:.2f}s"
            
            if additional_info:
                log_message += f" | 额外信息: {additional_info}"
            
            self.logger.info(log_message)
            
            # 各GPU详细信息
            for gpu in gpu_memory_info:
                gpu_log = f"  GPU {gpu['device_id']}: 分配={gpu['allocated']:7.1f}MB, 保留={gpu['reserved']:7.1f}MB, 利用率={gpu['utilization_percent']:5.1f}%, 可用={gpu['free_memory']:7.1f}MB"
                self.logger.info(gpu_log)
    
    def record_detailed(self, event_name, tensor_info=None, model_info=None):
        """记录详细的内存信息"""
        if not torch.cuda.is_available():
            if self.logger:
                self.logger.warning("CUDA不可用，无法记录详细GPU内存")
            return
        
        # 记录基础信息
        self.record(event_name)
        
        # 记录张量信息
        if tensor_info and self.logger:
            self.logger.info(f"张量信息 - {event_name}:")
            for name, tensor in tensor_info.items():
                if hasattr(tensor, 'shape') and hasattr(tensor, 'dtype'):
                    size_mb = tensor.numel() * tensor.element_size() / (1024**2)
                    self.logger.info(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}, size={size_mb:.2f}MB, device={tensor.device}")
        
        # 记录模型信息
        if model_info and self.logger:
            self.logger.info(f"模型信息 - {event_name}:")
            if 'total_params' in model_info:
                self.logger.info(f"  总参数: {model_info['total_params']:,}")
            if 'trainable_params' in model_info:
                self.logger.info(f"  可训练参数: {model_info['trainable_params']:,}")
            if 'model_size_mb' in model_info:
                self.logger.info(f"  模型大小: {model_info['model_size_mb']:.2f}MB")
    
    def print_report(self):
        """打印内存使用报告"""
        if not self.records:
            print("没有记录的内存数据")
            return
        
        print("\n==== GPU Memory Timeline ====")
        for event, data in self.records.items():
            last = data[-1]
            print(f"{event:<20} | 总分配: {last['total_allocated']:7.1f}MB | 总保留: {last['total_reserved']:7.1f}MB")
        
        # 记录到日志
        if self.logger:
            self.logger.info("=" * 80)
            self.logger.info("内存使用时间线报告")
            self.logger.info("=" * 80)
            
            for event, data in self.records.items():
                last = data[-1]
                self.logger.info(f"{event:<20} | 总分配: {last['total_allocated']:7.1f}MB | 总保留: {last['total_reserved']:7.1f}MB | 总最大分配: {last['total_max_allocated']:7.1f}MB | 总最大保留: {last['total_max_reserved']:7.1f}MB")
    
    def generate_summary_report(self):
        """生成详细的内存使用摘要报告"""
        if not self.records:
            return "没有记录的内存数据"
        
        # 找到峰值内存使用
        peak_total_allocated = 0
        peak_total_reserved = 0
        peak_event = ""
        peak_gpu_details = []
        
        for event, data in self.records.items():
            for record in data:
                if record['total_allocated'] > peak_total_allocated:
                    peak_total_allocated = record['total_allocated']
                    peak_total_reserved = record['total_reserved']
                    peak_event = event
                    peak_gpu_details = record['gpu_details']
        
        # 计算内存增长
        initial_record = next(iter(self.records.values()))[0]
        final_record = next(iter(self.records.values()))[-1]
        memory_growth = final_record['total_allocated'] - initial_record['total_allocated']
        
        # 生成报告
        report = f"""
内存使用摘要报告
================

峰值内存使用:
  事件: {peak_event}
  总分配内存: {peak_total_allocated:.2f} MB
  总保留内存: {peak_total_reserved:.2f} MB

内存增长:
  初始: {initial_record['total_allocated']:.2f} MB
  最终: {final_record['total_allocated']:.2f} MB
  增长: {memory_growth:.2f} MB

峰值时各GPU使用情况:
"""
        
        for gpu in peak_gpu_details:
            report += f"  GPU {gpu['device_id']}: 分配={gpu['allocated']:.2f}MB, 保留={gpu['reserved']:.2f}MB, 利用率={gpu['utilization_percent']:.1f}%\n"
        
        report += "\n事件统计:\n"
        
        for event, data in self.records.items():
            avg_total_allocated = sum(r['total_allocated'] for r in data) / len(data)
            avg_total_reserved = sum(r['total_reserved'] for r in data) / len(data)
            report += f"  {event}: 平均总分配={avg_total_allocated:.2f}MB, 平均总保留={avg_total_reserved:.2f}MB\n"
        
        # 记录到日志
        if self.logger:
            self.logger.info("=" * 80)
            self.logger.info("内存使用摘要报告")
            self.logger.info("=" * 80)
            self.logger.info(report)
        
        return report
    
    def save_to_json(self, filepath=None):
        """将记录保存为JSON文件"""
        import json
        
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"logs/memory_profiler/memory_data_{timestamp}.json"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 转换数据为可序列化格式
        serializable_records = {}
        for event, data in self.records.items():
            serializable_records[event] = []
            for record in data:
                serializable_record = {
                    'time': record['time'],
                    'total_allocated': record['total_allocated'],
                    'total_reserved': record['total_reserved'],
                    'total_max_allocated': record['total_max_allocated'],
                    'total_max_reserved': record['total_max_reserved'],
                    'timestamp': record['timestamp'],
                    'gpu_details': record['gpu_details']
                }
                serializable_records[event].append(serializable_record)
        
        # 保存到文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_records, f, indent=2, ensure_ascii=False)
        
        if self.logger:
            self.logger.info(f"内存数据已保存到: {filepath}")
        
        return filepath
    
    def cleanup(self):
        """清理资源"""
        if self.logger:
            # 移除所有处理器
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
                handler.close()
            
            self.logger.info("内存分析器清理完成")
            self.logger = None