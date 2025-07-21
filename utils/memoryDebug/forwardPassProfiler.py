#!/usr/bin/env python3
"""
前向传播内存分析器
用于监控模型前向传播过程中每个模块的显存占用情况
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict, OrderedDict
import json
import os
from datetime import datetime
import traceback

class TensorInfo:
    """Tensor信息类，用于存储tensor的详细信息"""
    
    def __init__(self, tensor: torch.Tensor, name: str = ""):
        self.name = name
        self.shape = list(tensor.shape)
        self.dtype = str(tensor.dtype)
        self.device = str(tensor.device)
        self.element_size = tensor.element_size()
        self.numel = tensor.numel()
        self.memory_mb = (self.numel * self.element_size) / (1024**2)
        self.requires_grad = tensor.requires_grad
        
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "shape": self.shape,
            "dtype": self.dtype,
            "device": self.device,
            "element_size_bytes": self.element_size,
            "num_elements": self.numel,
            "memory_mb": self.memory_mb,
            "requires_grad": self.requires_grad
        }
        
    def __str__(self) -> str:
        return f"{self.name}: {self.shape} {self.dtype} on {self.device} ({self.memory_mb:.2f}MB)"

class ModuleProfiler:
    """单个模块的性能分析器"""
    
    def __init__(self, module_name: str, module: nn.Module):
        self.module_name = module_name
        self.module = module
        self.call_count = 0
        self.forward_records = []
        
    def profile_forward(self, inputs, outputs, gpu_memory_before, gpu_memory_after):
        """记录一次forward调用的详细信息"""
        record = {
            "call_id": self.call_count,
            "module_name": self.module_name,
            "timestamp": datetime.now().isoformat(),
            "gpu_memory_before_mb": gpu_memory_before,
            "gpu_memory_after_mb": gpu_memory_after,
            "memory_delta_mb": gpu_memory_after - gpu_memory_before,
            "inputs": [],
            "outputs": [],
            "module_parameters": self._analyze_module_parameters()
        }
        
        # 分析输入tensors
        if isinstance(inputs, (tuple, list)):
            for i, inp in enumerate(inputs):
                if isinstance(inp, torch.Tensor):
                    tensor_info = TensorInfo(inp, f"input_{i}")
                    record["inputs"].append(tensor_info.to_dict())
        elif isinstance(inputs, torch.Tensor):
            tensor_info = TensorInfo(inputs, "input_0")
            record["inputs"].append(tensor_info.to_dict())
        elif isinstance(inputs, dict):
            # 处理字典形式的输入（如训练时的batch数据）
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    tensor_info = TensorInfo(value, key)
                    record["inputs"].append(tensor_info.to_dict())
                    
                    # 打印详细的tensor信息
                    self.logger.info(f"    📊 {key}: {value.shape} (dtype: {value.dtype}, device: {value.device})")
            
        # 分析输出tensors
        if isinstance(outputs, (tuple, list)):
            for i, out in enumerate(outputs):
                if isinstance(out, torch.Tensor):
                    tensor_info = TensorInfo(out, f"output_{i}")
                    record["outputs"].append(tensor_info.to_dict())
        elif isinstance(outputs, torch.Tensor):
            tensor_info = TensorInfo(outputs, "output_0")
            record["outputs"].append(tensor_info.to_dict())
            
        self.forward_records.append(record)
        self.call_count += 1
        
        return record
        
    def _analyze_module_parameters(self) -> Dict:
        """分析模块参数"""
        param_info = {
            "total_params": 0,
            "trainable_params": 0,
            "total_memory_mb": 0,
            "parameters": []
        }
        
        for name, param in self.module.named_parameters():
            if param is not None:
                tensor_info = TensorInfo(param, name)
                param_info["parameters"].append(tensor_info.to_dict())
                param_info["total_params"] += param.numel()
                param_info["total_memory_mb"] += tensor_info.memory_mb
                if param.requires_grad:
                    param_info["trainable_params"] += param.numel()
                    
        return param_info

class ForwardPassProfiler:
    """前向传播内存分析器主类"""
    
    def __init__(self, log_dir: str = "logs/forward_pass_profiler", enable_detailed_logging: bool = True):
        self.log_dir = log_dir
        self.enable_detailed_logging = enable_detailed_logging
        self.module_profilers = {}
        self.wrapped_modules = {}
        self.original_forwards = {}
        self.global_call_order = []
        self.is_profiling = False
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = logging.getLogger(f"ForwardPassProfiler_{timestamp}")
        logger.setLevel(logging.INFO)
        
        # 文件处理器
        log_file = os.path.join(self.log_dir, f"forward_pass_profile_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # 格式器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
        
    def _get_gpu_memory(self) -> float:
        """获取当前GPU内存使用量（MB）"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**2)
        return 0.0
        
    def wrap_module(self, module: nn.Module, module_name: str):
        """包装一个模块以监控其前向传播"""
        if module_name in self.wrapped_modules:
            return  # 已经包装过了
            
        # 创建模块分析器
        profiler = ModuleProfiler(module_name, module)
        self.module_profilers[module_name] = profiler
        
        # 保存原始forward方法
        original_forward = module.forward
        self.original_forwards[module_name] = original_forward
        
        def profiled_forward(*args, **kwargs):
            if not self.is_profiling:
                return original_forward(*args, **kwargs)
                
            # 记录调用顺序
            call_id = len(self.global_call_order)
            self.global_call_order.append(module_name)
            
            # 记录前向传播前的GPU内存
            gpu_memory_before = self._get_gpu_memory()
            
            try:
                # 执行原始forward
                outputs = original_forward(*args, **kwargs)
                
                # 记录前向传播后的GPU内存
                gpu_memory_after = self._get_gpu_memory()
                
                # 分析这次调用
                record = profiler.profile_forward(
                    inputs=args,
                    outputs=outputs,
                    gpu_memory_before=gpu_memory_before,
                    gpu_memory_after=gpu_memory_after
                )
                
                # 详细日志
                if self.enable_detailed_logging:
                    memory_delta = gpu_memory_after - gpu_memory_before
                    self.logger.info(f"模块 {module_name} [调用#{call_id}]: 内存变化 {memory_delta:+.2f}MB "
                                   f"(前: {gpu_memory_before:.2f}MB, 后: {gpu_memory_after:.2f}MB)")
                    
                    # 记录输入输出tensor信息
                    if record["inputs"]:
                        input_memory = sum(inp["memory_mb"] for inp in record["inputs"])
                        self.logger.info(f"  输入tensors: {len(record['inputs'])}个, 总内存: {input_memory:.2f}MB")
                        
                        # 详细打印输入tensor信息
                        for inp in record["inputs"]:
                            self.logger.info(f"    📊 {inp['name']}: {inp['shape']} "
                                           f"(dtype: {inp['dtype']}, device: {inp['device']}, "
                                           f"memory: {inp['memory_mb']:.2f}MB)")
                        
                    if record["outputs"]:
                        output_memory = sum(out["memory_mb"] for out in record["outputs"])
                        self.logger.info(f"  输出tensors: {len(record['outputs'])}个, 总内存: {output_memory:.2f}MB")
                        
                        # 详细打印输出tensor信息
                        for out in record["outputs"]:
                            self.logger.info(f"    📊 {out['name']}: {out['shape']} "
                                           f"(dtype: {out['dtype']}, device: {out['device']}, "
                                           f"memory: {out['memory_mb']:.2f}MB)")
                
                return outputs
                
            except Exception as e:
                self.logger.error(f"模块 {module_name} 前向传播出错: {e}")
                traceback.print_exc()
                raise e
        
        # 替换forward方法
        module.forward = profiled_forward
        self.wrapped_modules[module_name] = module
        
    def wrap_model(self, model: nn.Module, prefix: str = ""):
        """递归包装模型的所有子模块"""
        for name, child in model.named_children():
            module_name = f"{prefix}.{name}" if prefix else name
            
            # 包装这个子模块
            self.wrap_module(child, module_name)
            
            # 递归包装子模块的子模块
            if len(list(child.children())) > 0:
                self.wrap_model(child, module_name)
                
        # 也包装根模块
        if not prefix:  # 只在最顶层包装根模块
            self.wrap_module(model, "root_model")
            
    def start_profiling(self):
        """开始性能分析"""
        self.is_profiling = True
        self.global_call_order = []
        self.logger.info("🚀 开始前向传播内存分析")
        
    def stop_profiling(self):
        """停止性能分析"""
        self.is_profiling = False
        self.logger.info("🛑 停止前向传播内存分析")
        
    def unwrap_all(self):
        """恢复所有模块的原始forward方法"""
        for module_name, module in self.wrapped_modules.items():
            if module_name in self.original_forwards:
                module.forward = self.original_forwards[module_name]
                
        self.wrapped_modules.clear()
        self.original_forwards.clear()
        self.logger.info("✅ 已恢复所有模块的原始forward方法")
        
    def generate_summary_report(self) -> Dict:
        """生成汇总报告"""
        report = {
            "profiling_summary": {
                "total_modules": len(self.module_profilers),
                "total_forward_calls": sum(p.call_count for p in self.module_profilers.values()),
                "call_order": self.global_call_order
            },
            "module_statistics": {},
            "memory_analysis": {
                "peak_memory_usage": 0,
                "total_memory_allocated": 0,
                "most_memory_intensive_modules": []
            }
        }
        
        # 统计每个模块
        module_stats = []
        for module_name, profiler in self.module_profilers.items():
            if profiler.call_count == 0:
                continue
                
            # 计算该模块的统计信息
            memory_deltas = [record["memory_delta_mb"] for record in profiler.forward_records]
            avg_memory_delta = sum(memory_deltas) / len(memory_deltas)
            max_memory_delta = max(memory_deltas)
            min_memory_delta = min(memory_deltas)
            
            # 计算输入输出内存统计
            total_input_memory = 0
            total_output_memory = 0
            for record in profiler.forward_records:
                total_input_memory += sum(inp["memory_mb"] for inp in record["inputs"])
                total_output_memory += sum(out["memory_mb"] for out in record["outputs"])
                
            module_stat = {
                "module_name": module_name,
                "call_count": profiler.call_count,
                "avg_memory_delta_mb": avg_memory_delta,
                "max_memory_delta_mb": max_memory_delta,
                "min_memory_delta_mb": min_memory_delta,
                "total_input_memory_mb": total_input_memory,
                "total_output_memory_mb": total_output_memory,
                "parameter_memory_mb": profiler.forward_records[0]["module_parameters"]["total_memory_mb"] if profiler.forward_records else 0
            }
            
            module_stats.append(module_stat)
            report["module_statistics"][module_name] = module_stat
            
        # 按内存使用量排序
        module_stats.sort(key=lambda x: x["max_memory_delta_mb"], reverse=True)
        report["memory_analysis"]["most_memory_intensive_modules"] = module_stats[:10]
        
        # 计算峰值内存
        all_memory_values = []
        for profiler in self.module_profilers.values():
            for record in profiler.forward_records:
                all_memory_values.extend([record["gpu_memory_before_mb"], record["gpu_memory_after_mb"]])
                
        if all_memory_values:
            report["memory_analysis"]["peak_memory_usage"] = max(all_memory_values)
            report["memory_analysis"]["total_memory_allocated"] = sum(all_memory_values) / len(all_memory_values)
            
        return report
        
    def save_detailed_report(self, filename: str = None):
        """保存详细报告到JSON文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"forward_pass_detailed_report_{timestamp}.json"
            
        filepath = os.path.join(self.log_dir, filename)
        
        # 构建详细报告
        detailed_report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_modules": len(self.module_profilers),
                "profiling_enabled": self.is_profiling
            },
            "module_details": {}
        }
        
        for module_name, profiler in self.module_profilers.items():
            detailed_report["module_details"][module_name] = {
                "call_count": profiler.call_count,
                "forward_records": profiler.forward_records
            }
            
        # 添加汇总信息
        detailed_report["summary"] = self.generate_summary_report()
        
        # 保存到文件
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(detailed_report, f, ensure_ascii=False, indent=2)
            self.logger.info(f"📄 详细报告已保存到: {filepath}")
        except Exception as e:
            self.logger.error(f"保存详细报告失败: {e}")
            
        return filepath
        
    def print_summary(self):
        """打印汇总信息"""
        report = self.generate_summary_report()
        
        print("\n" + "="*80)
        print("🔍 前向传播内存分析汇总报告")
        print("="*80)
        
        print(f"📊 总模块数: {report['profiling_summary']['total_modules']}")
        print(f"📞 总调用次数: {report['profiling_summary']['total_forward_calls']}")
        print(f"🏔️  峰值内存使用: {report['memory_analysis']['peak_memory_usage']:.2f}MB")
        
        print(f"\n🔥 内存使用最多的前5个模块:")
        for i, module_stat in enumerate(report['memory_analysis']['most_memory_intensive_modules'][:5]):
            print(f"  {i+1}. {module_stat['module_name']}: "
                  f"最大内存增量 {module_stat['max_memory_delta_mb']:.2f}MB, "
                  f"平均内存增量 {module_stat['avg_memory_delta_mb']:.2f}MB, "
                  f"调用次数 {module_stat['call_count']}")
                  
        print("="*80)

def profile_model_forward(model: nn.Module, inputs, log_dir: str = "logs/forward_pass_profiler") -> ForwardPassProfiler:
    """
    便捷函数：对模型的一次前向传播进行完整的内存分析
    
    Args:
        model: 要分析的模型
        inputs: 模型输入（可以是tensor、tuple或dict）
        log_dir: 日志保存目录
        
    Returns:
        ForwardPassProfiler实例，可用于进一步分析
    """
    profiler = ForwardPassProfiler(log_dir=log_dir)
    
    try:
        # 包装模型
        profiler.wrap_model(model)
        
        # 开始分析
        profiler.start_profiling()
        
        # 执行前向传播
        with torch.no_grad():
            if isinstance(inputs, dict):
                outputs = model(**inputs)
            elif isinstance(inputs, (tuple, list)):
                outputs = model(*inputs)
            else:
                outputs = model(inputs)
                
        # 停止分析
        profiler.stop_profiling()
        
        # 生成报告
        profiler.print_summary()
        profiler.save_detailed_report()
        
        return profiler
        
    except Exception as e:
        profiler.logger.error(f"前向传播分析过程中出错: {e}")
        raise e
    finally:
        # 确保恢复原始状态
        profiler.unwrap_all()

# 上下文管理器支持
class ProfiledModel:
    """上下文管理器，用于临时分析模型"""
    
    def __init__(self, model: nn.Module, log_dir: str = "logs/forward_pass_profiler"):
        self.model = model
        self.profiler = ForwardPassProfiler(log_dir=log_dir)
        
    def __enter__(self):
        self.profiler.wrap_model(self.model)
        self.profiler.start_profiling()
        return self.profiler
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.stop_profiling()
        self.profiler.print_summary()
        self.profiler.save_detailed_report()
        self.profiler.unwrap_all() 