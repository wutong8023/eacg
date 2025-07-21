#!/usr/bin/env python3
"""
快速内存调试工具
提供简单的接口来快速监控和调试训练过程中的内存使用情况
"""

import os
import sys
import torch
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from memoryDebugger import MemoryDebugger

class QuickMemoryDebugger:
    """快速内存调试器"""
    
    def __init__(self, log_dir: str = None, enable_logging: bool = True):
        if log_dir is None:
            log_dir = f"logs/quick_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.log_dir = log_dir
        self.enable_logging = enable_logging
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 初始化调试器
        self.debugger = MemoryDebugger(log_dir=log_dir, enable_real_time=False)
        
        # 设置简单的日志记录
        if enable_logging:
            self.logger = self._setup_simple_logger()
        else:
            self.logger = None
    
    def _setup_simple_logger(self):
        """设置简单的日志记录器"""
        logger = logging.getLogger("QuickMemoryDebug")
        logger.setLevel(logging.INFO)
        
        # 清除现有处理器
        logger.handlers.clear()
        
        # 创建文件处理器
        log_file = os.path.join(self.log_dir, "quick_debug.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        
        return logger
    
    def log_info(self, message: str):
        """记录信息"""
        if self.logger:
            self.logger.info(message)
        print(f"[INFO] {message}")
    
    def log_warning(self, message: str):
        """记录警告"""
        if self.logger:
            self.logger.warning(message)
        print(f"[WARNING] {message}")
    
    def log_error(self, message: str):
        """记录错误"""
        if self.logger:
            self.logger.error(message)
        print(f"[ERROR] {message}")
    
    def get_memory_snapshot(self, name: str = "snapshot") -> Dict:
        """获取内存快照"""
        self.log_info(f"获取内存快照: {name}")
        
        memory_info = self.debugger.get_current_memory_usage()
        
        # 简化的内存信息
        snapshot = {
            'name': name,
            'timestamp': memory_info['timestamp'],
            'gpu_summary': []
        }
        
        for gpu_mem in memory_info['gpu_memory']:
            gpu_summary = {
                'gpu_id': gpu_mem['device_id'],
                'allocated_mb': gpu_mem['allocated_mb'],
                'utilization_percent': gpu_mem['utilization_percent'],
                'free_mb': gpu_mem['free_mb']
            }
            snapshot['gpu_summary'].append(gpu_summary)
            
            self.log_info(f"  GPU {gpu_mem['device_id']}: "
                         f"{gpu_mem['allocated_mb']:.1f}MB / "
                         f"{gpu_mem['total_mb']:.1f}MB "
                         f"({gpu_mem['utilization_percent']:.1f}%)")
        
        return snapshot
    
    def analyze_model_quick(self, model, name: str = "model") -> Dict:
        """快速分析模型参数"""
        self.log_info(f"快速分析模型: {name}")
        
        analysis = self.debugger.analyze_model_parameters(model, name)
        
        # 提取关键信息
        summary = analysis['summary']
        quick_summary = {
            'model_name': name,
            'total_params': summary['total_parameters'],
            'trainable_params': summary['trainable_parameters'],
            'memory_mb': summary['total_memory_mb'],
            'param_efficiency': summary['trainable_parameters'] / summary['total_parameters'] * 100
        }
        
        self.log_info(f"  总参数: {quick_summary['total_params']:,}")
        self.log_info(f"  可训练参数: {quick_summary['trainable_params']:,}")
        self.log_info(f"  内存占用: {quick_summary['memory_mb']:.2f} MB")
        self.log_info(f"  参数效率: {quick_summary['param_efficiency']:.2f}%")
        
        # 检查是否有异常
        self._check_model_anomalies(analysis)
        
        return quick_summary
    
    def _check_model_anomalies(self, analysis: Dict):
        """检查模型异常"""
        summary = analysis['summary']
        
        # 检查异常大的参数
        for group_name, group_info in analysis['parameter_groups'].items():
            for param_info in group_info['parameters']:
                if param_info['memory_mb'] > 100:  # 超过100MB的参数
                    self.log_warning(f"大参数检测: {param_info['name']} "
                                   f"({param_info['memory_mb']:.2f} MB)")
                
                # 检查数据异常
                if 'data_stats' in param_info:
                    stats = param_info['data_stats']
                    if isinstance(stats, dict) and 'has_nan' in stats:
                        if stats['has_nan']:
                            self.log_error(f"参数包含NaN: {param_info['name']}")
                        if stats['has_inf']:
                            self.log_error(f"参数包含Inf: {param_info['name']}")
    
    def compare_snapshots(self, snapshot1: Dict, snapshot2: Dict) -> Dict:
        """比较两个内存快照"""
        self.log_info(f"比较内存快照: {snapshot1['name']} vs {snapshot2['name']}")
        
        comparison = {
            'from': snapshot1['name'],
            'to': snapshot2['name'],
            'gpu_changes': []
        }
        
        for gpu1, gpu2 in zip(snapshot1['gpu_summary'], snapshot2['gpu_summary']):
            change = {
                'gpu_id': gpu1['gpu_id'],
                'memory_change_mb': gpu2['allocated_mb'] - gpu1['allocated_mb'],
                'utilization_change': gpu2['utilization_percent'] - gpu1['utilization_percent']
            }
            comparison['gpu_changes'].append(change)
            
            self.log_info(f"  GPU {gpu1['gpu_id']}: "
                         f"内存变化 {change['memory_change_mb']:+.1f}MB, "
                         f"利用率变化 {change['utilization_change']:+.1f}%")
        
        return comparison
    
    def monitor_function(self, func, *args, **kwargs):
        """监控函数执行的内存变化"""
        func_name = func.__name__ if hasattr(func, '__name__') else str(func)
        self.log_info(f"开始监控函数: {func_name}")
        
        # 执行前快照
        before_snapshot = self.get_memory_snapshot(f"{func_name}_before")
        
        try:
            # 执行函数
            result = func(*args, **kwargs)
            
            # 执行后快照
            after_snapshot = self.get_memory_snapshot(f"{func_name}_after")
            
            # 比较快照
            comparison = self.compare_snapshots(before_snapshot, after_snapshot)
            
            self.log_info(f"函数 {func_name} 执行完成")
            
            return result, comparison
            
        except Exception as e:
            error_snapshot = self.get_memory_snapshot(f"{func_name}_error")
            self.log_error(f"函数 {func_name} 执行失败: {e}")
            
            error_comparison = self.compare_snapshots(before_snapshot, error_snapshot)
            
            raise e
    
    def check_memory_growth(self, threshold_mb: float = 100.0):
        """检查内存增长"""
        memory_info = self.debugger.get_current_memory_usage()
        
        for gpu_mem in memory_info['gpu_memory']:
            if gpu_mem['allocated_mb'] > threshold_mb:
                self.log_warning(f"GPU {gpu_mem['device_id']} 内存使用超过阈值: "
                               f"{gpu_mem['allocated_mb']:.1f}MB > {threshold_mb}MB")
            
            if gpu_mem['utilization_percent'] > 90:
                self.log_warning(f"GPU {gpu_mem['device_id']} 利用率过高: "
                               f"{gpu_mem['utilization_percent']:.1f}%")
    
    def cleanup_memory(self):
        """清理内存"""
        self.log_info("清理GPU内存...")
        
        before_snapshot = self.get_memory_snapshot("before_cleanup")
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        after_snapshot = self.get_memory_snapshot("after_cleanup")
        
        # 比较清理效果
        comparison = self.compare_snapshots(before_snapshot, after_snapshot)
        
        self.log_info("内存清理完成")
        
        return comparison
    
    def generate_quick_report(self) -> str:
        """生成快速报告"""
        self.log_info("生成快速报告...")
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("快速内存调试报告")
        report_lines.append("=" * 60)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"日志目录: {self.log_dir}")
        report_lines.append("")
        
        # 当前内存状态
        current_memory = self.debugger.get_current_memory_usage()
        report_lines.append("当前内存状态:")
        
        for gpu_mem in current_memory['gpu_memory']:
            report_lines.append(f"  GPU {gpu_mem['device_id']}: "
                              f"{gpu_mem['allocated_mb']:.1f}MB / "
                              f"{gpu_mem['total_mb']:.1f}MB "
                              f"({gpu_mem['utilization_percent']:.1f}%)")
        
        # 系统内存
        sys_mem = current_memory['system_memory']
        report_lines.append(f"系统内存: {sys_mem['used_gb']:.1f}GB / "
                          f"{sys_mem['total_gb']:.1f}GB ({sys_mem['percent']:.1f}%)")
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        report_file = os.path.join(self.log_dir, "quick_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.log_info(f"快速报告已保存: {report_file}")
        
        return report_content

# 全局快速调试器实例
_quick_debugger = None

def get_quick_debugger(log_dir: str = None) -> QuickMemoryDebugger:
    """获取全局快速调试器实例"""
    global _quick_debugger
    if _quick_debugger is None:
        _quick_debugger = QuickMemoryDebugger(log_dir=log_dir)
    return _quick_debugger

# 便捷函数
def memory_snapshot(name: str = "snapshot"):
    """快速内存快照"""
    debugger = get_quick_debugger()
    return debugger.get_memory_snapshot(name)

def analyze_model(model, name: str = "model"):
    """快速分析模型"""
    debugger = get_quick_debugger()
    return debugger.analyze_model_quick(model, name)

def monitor_function(func, *args, **kwargs):
    """监控函数内存使用"""
    debugger = get_quick_debugger()
    return debugger.monitor_function(func, *args, **kwargs)

def check_memory_status(threshold_mb: float = 100.0):
    """检查内存状态"""
    debugger = get_quick_debugger()
    return debugger.check_memory_growth(threshold_mb)

def cleanup_memory():
    """清理内存"""
    debugger = get_quick_debugger()
    return debugger.cleanup_memory()

def generate_report():
    """生成报告"""
    debugger = get_quick_debugger()
    return debugger.generate_quick_report()

# 使用示例
if __name__ == "__main__":
    # 示例：快速内存调试
    print("🚀 快速内存调试示例")
    
    # 获取初始内存快照
    initial_snapshot = memory_snapshot("initial")
    
    # 模拟一些内存操作
    def dummy_operation():
        import time
        if torch.cuda.is_available():
            dummy_tensor = torch.randn(1000, 1000).cuda()
            time.sleep(1)
            return dummy_tensor
        else:
            dummy_tensor = torch.randn(1000, 1000)
            time.sleep(1)
            return dummy_tensor
    
    # 监控函数执行
    result, comparison = monitor_function(dummy_operation)
    print(f"内存变化: {comparison}")
    
    # 检查内存状态
    check_memory_status(threshold_mb=50.0)
    
    # 清理内存
    cleanup_comparison = cleanup_memory()
    
    # 生成报告
    report = generate_report()
    print("\n报告预览:")
    print(report[:500] + "...")
    
    print("\n✅ 快速内存调试完成!") 