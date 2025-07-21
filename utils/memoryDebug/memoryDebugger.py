import torch
import logging
import time
import json
import os
from datetime import datetime
from collections import defaultdict
import traceback
import threading
import psutil
from typing import Dict, List, Optional, Tuple, Any

class MemoryDebugger:
    """
    显存监控和调试工具，用于分析模型训练时各个参数矩阵的显存占用情况
    """
    
    def __init__(self, log_dir: str = "logs/memory_debug", enable_real_time: bool = True):
        """
        初始化显存调试器
        
        Args:
            log_dir: 日志保存目录
            enable_real_time: 是否启用实时监控
        """
        self.log_dir = log_dir
        self.enable_real_time = enable_real_time
        self.monitoring_active = False
        self.memory_history = []
        self.parameter_registry = {}
        self.gradient_registry = {}
        self.checkpoint_history = []
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志记录
        self.logger = self._setup_logger()
        
        # 初始化GPU信息
        self.gpu_info = self._get_gpu_info()
        
        self.logger.info("=" * 80)
        self.logger.info("内存调试器初始化完成")
        self.logger.info("=" * 80)
        self.logger.info(f"日志目录: {log_dir}")
        self.logger.info(f"实时监控: {enable_real_time}")
        self.logger.info(f"检测到GPU数量: {len(self.gpu_info)}")
        
        for i, gpu in enumerate(self.gpu_info):
            self.logger.info(f"GPU {i}: {gpu['name']} ({gpu['total_memory_gb']:.1f}GB)")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("MemoryDebugger")
        logger.setLevel(logging.INFO)
        
        # 清除现有的处理器
        logger.handlers.clear()
        
        # 创建文件处理器
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"memory_debug_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _get_gpu_info(self) -> List[Dict]:
        """获取GPU信息"""
        gpu_info = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    'device_id': i,
                    'name': props.name,
                    'total_memory_gb': props.total_memory / (1024**3),
                    'total_memory_mb': props.total_memory / (1024**2)
                })
        return gpu_info
    
    def get_current_memory_usage(self) -> Dict:
        """获取当前显存使用情况"""
        memory_info = {
            'timestamp': datetime.now().isoformat(),
            'gpu_memory': [],
            'system_memory': self._get_system_memory()
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    torch.cuda.set_device(i)
                    allocated = torch.cuda.memory_allocated(i)
                    reserved = torch.cuda.memory_reserved(i)
                    max_allocated = torch.cuda.max_memory_allocated(i)
                    max_reserved = torch.cuda.max_memory_reserved(i)
                    
                    total_memory = torch.cuda.get_device_properties(i).total_memory
                    
                    gpu_memory = {
                        'device_id': i,
                        'allocated_mb': allocated / (1024**2),
                        'reserved_mb': reserved / (1024**2),
                        'max_allocated_mb': max_allocated / (1024**2),
                        'max_reserved_mb': max_reserved / (1024**2),
                        'total_mb': total_memory / (1024**2),
                        'free_mb': (total_memory - reserved) / (1024**2),
                        'utilization_percent': (allocated / total_memory) * 100
                    }
                    
                    memory_info['gpu_memory'].append(gpu_memory)
                    
                except Exception as e:
                    self.logger.error(f"获取GPU {i} 内存信息失败: {e}")
        
        return memory_info
    
    def _get_system_memory(self) -> Dict:
        """获取系统内存使用情况"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent
        }
    
    def analyze_model_parameters(self, model, model_name: str = "model") -> Dict:
        """
        分析模型参数的显存占用情况
        
        Args:
            model: 待分析的模型
            model_name: 模型名称
            
        Returns:
            参数分析结果
        """
        self.logger.info(f"开始分析模型参数: {model_name}")
        
        analysis_result = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'parameter_groups': {},
            'device_distribution': defaultdict(list),
            'memory_usage': defaultdict(float),
            'parameter_counts': defaultdict(int),
            'gradient_info': {}
        }
        
        total_params = 0
        total_trainable_params = 0
        total_memory_mb = 0
        
        # 分析每个参数
        for name, param in model.named_parameters():
            param_info = self._analyze_parameter(name, param)
            
            # 按层级分组
            layer_group = self._get_layer_group(name)
            if layer_group not in analysis_result['parameter_groups']:
                analysis_result['parameter_groups'][layer_group] = {
                    'parameters': [],
                    'total_params': 0,
                    'trainable_params': 0,
                    'memory_mb': 0
                }
            
            analysis_result['parameter_groups'][layer_group]['parameters'].append(param_info)
            analysis_result['parameter_groups'][layer_group]['total_params'] += param_info['num_params']
            analysis_result['parameter_groups'][layer_group]['memory_mb'] += param_info['memory_mb']
            
            if param_info['requires_grad']:
                analysis_result['parameter_groups'][layer_group]['trainable_params'] += param_info['num_params']
            
            # 设备分布统计
            device_str = str(param_info['device'])
            analysis_result['device_distribution'][device_str].append({
                'name': name,
                'memory_mb': param_info['memory_mb'],
                'params': param_info['num_params']
            })
            
            analysis_result['memory_usage'][device_str] += param_info['memory_mb']
            analysis_result['parameter_counts'][device_str] += param_info['num_params']
            
            total_params += param_info['num_params']
            total_memory_mb += param_info['memory_mb']
            
            if param_info['requires_grad']:
                total_trainable_params += param_info['num_params']
        
        # 分析梯度信息
        analysis_result['gradient_info'] = self._analyze_gradients(model)
        
        # 汇总统计
        analysis_result['summary'] = {
            'total_parameters': total_params,
            'trainable_parameters': total_trainable_params,
            'total_memory_mb': total_memory_mb,
            'memory_per_param_bytes': (total_memory_mb * 1024 * 1024) / total_params if total_params > 0 else 0
        }
        
        # 记录到注册表
        self.parameter_registry[model_name] = analysis_result
        
        # 详细日志输出
        self._log_parameter_analysis(analysis_result)
        
        return analysis_result
    
    def _analyze_parameter(self, name: str, param: torch.Tensor) -> Dict:
        """分析单个参数的详细信息"""
        param_info = {
            'name': name,
            'shape': list(param.shape),
            'dtype': str(param.dtype),
            'device': str(param.device),
            'requires_grad': param.requires_grad,
            'num_params': param.numel(),
            'memory_mb': param.numel() * param.element_size() / (1024**2),
            'is_lora': 'lora' in name.lower(),
            'is_embedding': 'embed' in name.lower(),
            'is_linear': any(keyword in name.lower() for keyword in ['linear', 'proj', 'fc']),
            'is_norm': any(keyword in name.lower() for keyword in ['norm', 'layer_norm', 'batch_norm']),
            'is_attention': any(keyword in name.lower() for keyword in ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj']),
        }
        
        # 分析参数数据范围
        if param.is_cuda:
            try:
                param_info['data_stats'] = {
                    'min': param.min().item(),
                    'max': param.max().item(),
                    'mean': param.mean().item(),
                    'std': param.std().item(),
                    'has_nan': torch.isnan(param).any().item(),
                    'has_inf': torch.isinf(param).any().item()
                }
            except:
                param_info['data_stats'] = {'error': 'Cannot compute stats'}
        
        return param_info
    
    def _get_layer_group(self, param_name: str) -> str:
        """根据参数名确定层级分组"""
        name_lower = param_name.lower()
        
        if 'lora' in name_lower:
            return 'lora_adapters'
        elif 'embed' in name_lower:
            return 'embeddings'
        elif 'lm_head' in name_lower or 'classifier' in name_lower:
            return 'output_layers'
        elif 'norm' in name_lower:
            return 'normalization'
        elif any(keyword in name_lower for keyword in ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj']):
            return 'attention_layers'
        elif any(keyword in name_lower for keyword in ['mlp', 'feed_forward', 'ffn', 'gate_proj', 'up_proj', 'down_proj']):
            return 'feedforward_layers'
        elif 'layers' in name_lower:
            return 'transformer_layers'
        else:
            return 'other'
    
    def _analyze_gradients(self, model) -> Dict:
        """分析梯度信息"""
        gradient_info = {
            'total_gradient_memory_mb': 0,
            'gradient_count': 0,
            'gradient_distribution': defaultdict(list),
            'gradient_stats': {}
        }
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_memory_mb = param.grad.numel() * param.grad.element_size() / (1024**2)
                gradient_info['total_gradient_memory_mb'] += grad_memory_mb
                gradient_info['gradient_count'] += 1
                
                device_str = str(param.device)
                gradient_info['gradient_distribution'][device_str].append({
                    'name': name,
                    'memory_mb': grad_memory_mb,
                    'shape': list(param.grad.shape)
                })
                
                # 梯度统计
                try:
                    gradient_info['gradient_stats'][name] = {
                        'norm': param.grad.norm().item(),
                        'min': param.grad.min().item(),
                        'max': param.grad.max().item(),
                        'mean': param.grad.mean().item(),
                        'std': param.grad.std().item()
                    }
                except:
                    gradient_info['gradient_stats'][name] = {'error': 'Cannot compute gradient stats'}
        
        return gradient_info
    
    def _log_parameter_analysis(self, analysis_result: Dict):
        """记录参数分析结果到日志"""
        self.logger.info("=" * 80)
        self.logger.info(f"模型参数分析结果: {analysis_result['model_name']}")
        self.logger.info("=" * 80)
        
        summary = analysis_result['summary']
        self.logger.info(f"总参数数量: {summary['total_parameters']:,}")
        self.logger.info(f"可训练参数数量: {summary['trainable_parameters']:,}")
        self.logger.info(f"总内存占用: {summary['total_memory_mb']:.2f} MB")
        self.logger.info(f"平均每参数内存: {summary['memory_per_param_bytes']:.2f} bytes")
        
        # 按层级分组统计
        self.logger.info("\n按层级分组统计:")
        for group_name, group_info in analysis_result['parameter_groups'].items():
            self.logger.info(f"  {group_name}:")
            self.logger.info(f"    参数数量: {group_info['total_params']:,}")
            self.logger.info(f"    可训练参数: {group_info['trainable_params']:,}")
            self.logger.info(f"    内存占用: {group_info['memory_mb']:.2f} MB")
            
            # 列出该组中内存占用最大的参数
            if group_info['parameters']:
                top_params = sorted(group_info['parameters'], key=lambda x: x['memory_mb'], reverse=True)[:3]
                self.logger.info(f"    内存占用最大的参数:")
                for param in top_params:
                    self.logger.info(f"      {param['name']}: {param['memory_mb']:.2f} MB ({param['shape']})")
        
        # 设备分布统计
        self.logger.info("\n设备分布统计:")
        for device, params in analysis_result['device_distribution'].items():
            total_memory = sum(p['memory_mb'] for p in params)
            total_params = sum(p['params'] for p in params)
            self.logger.info(f"  {device}: {len(params)} 个参数, {total_params:,} 个元素, {total_memory:.2f} MB")
        
        # 梯度信息
        gradient_info = analysis_result['gradient_info']
        if gradient_info['gradient_count'] > 0:
            self.logger.info(f"\n梯度信息:")
            self.logger.info(f"  有梯度的参数数量: {gradient_info['gradient_count']}")
            self.logger.info(f"  梯度总内存占用: {gradient_info['total_gradient_memory_mb']:.2f} MB")
    
    def create_memory_checkpoint(self, checkpoint_name: str, model=None, additional_info: Dict = None):
        """
        创建内存检查点
        
        Args:
            checkpoint_name: 检查点名称
            model: 模型对象（可选）
            additional_info: 额外信息
        """
        self.logger.info(f"创建内存检查点: {checkpoint_name}")
        
        checkpoint = {
            'name': checkpoint_name,
            'timestamp': datetime.now().isoformat(),
            'memory_usage': self.get_current_memory_usage(),
            'additional_info': additional_info or {}
        }
        
        if model is not None:
            checkpoint['model_analysis'] = self.analyze_model_parameters(model, checkpoint_name)
        
        self.checkpoint_history.append(checkpoint)
        
        # 保存检查点到文件
        checkpoint_file = os.path.join(self.log_dir, f"checkpoint_{checkpoint_name}_{int(time.time())}.json")
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        self.logger.info(f"检查点已保存: {checkpoint_file}")
        
        return checkpoint
    
    def start_real_time_monitoring(self, interval: float = 1.0):
        """
        启动实时内存监控
        
        Args:
            interval: 监控间隔（秒）
        """
        if not self.enable_real_time:
            self.logger.warning("实时监控未启用")
            return
        
        if self.monitoring_active:
            self.logger.warning("实时监控已在运行")
            return
        
        self.monitoring_active = True
        self.logger.info(f"启动实时内存监控，间隔: {interval}秒")
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    memory_info = self.get_current_memory_usage()
                    self.memory_history.append(memory_info)
                    
                    # 检查内存异常
                    self._check_memory_anomalies(memory_info)
                    
                    # 限制历史记录长度
                    if len(self.memory_history) > 1000:
                        self.memory_history = self.memory_history[-1000:]
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    self.logger.error(f"实时监控出错: {e}")
                    time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_real_time_monitoring(self):
        """停止实时内存监控"""
        if self.monitoring_active:
            self.monitoring_active = False
            self.logger.info("停止实时内存监控")
            
            # 保存监控历史
            self.save_monitoring_history()
    
    def _check_memory_anomalies(self, memory_info: Dict):
        """检查内存异常"""
        for gpu_mem in memory_info['gpu_memory']:
            # 检查内存使用率是否过高
            if gpu_mem['utilization_percent'] > 90:
                self.logger.warning(f"GPU {gpu_mem['device_id']} 内存使用率过高: {gpu_mem['utilization_percent']:.1f}%")
            
            # 检查内存分配是否异常增长
            if len(self.memory_history) > 5:
                recent_usage = [h['gpu_memory'][gpu_mem['device_id']]['allocated_mb'] 
                              for h in self.memory_history[-5:]]
                if len(recent_usage) > 1:
                    growth_rate = (recent_usage[-1] - recent_usage[0]) / len(recent_usage)
                    if growth_rate > 100:  # 每次增长超过100MB
                        self.logger.warning(f"GPU {gpu_mem['device_id']} 内存增长异常: {growth_rate:.2f} MB/step")
    
    def save_monitoring_history(self):
        """保存监控历史到文件"""
        if not self.memory_history:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = os.path.join(self.log_dir, f"memory_history_{timestamp}.json")
        
        with open(history_file, 'w') as f:
            json.dump(self.memory_history, f, indent=2, default=str)
        
        self.logger.info(f"监控历史已保存: {history_file}")
    
    def compare_checkpoints(self, checkpoint1_name: str, checkpoint2_name: str) -> Dict:
        """
        比较两个检查点的内存使用情况
        
        Args:
            checkpoint1_name: 第一个检查点名称
            checkpoint2_name: 第二个检查点名称
            
        Returns:
            比较结果
        """
        checkpoint1 = None
        checkpoint2 = None
        
        for checkpoint in self.checkpoint_history:
            if checkpoint['name'] == checkpoint1_name:
                checkpoint1 = checkpoint
            elif checkpoint['name'] == checkpoint2_name:
                checkpoint2 = checkpoint
        
        if not checkpoint1 or not checkpoint2:
            self.logger.error(f"找不到检查点: {checkpoint1_name} 或 {checkpoint2_name}")
            return {}
        
        comparison = {
            'checkpoint1': checkpoint1_name,
            'checkpoint2': checkpoint2_name,
            'gpu_memory_diff': [],
            'model_parameter_diff': {}
        }
        
        # 比较GPU内存
        for i, (gpu1, gpu2) in enumerate(zip(checkpoint1['memory_usage']['gpu_memory'], 
                                           checkpoint2['memory_usage']['gpu_memory'])):
            diff = {
                'device_id': i,
                'allocated_diff_mb': gpu2['allocated_mb'] - gpu1['allocated_mb'],
                'reserved_diff_mb': gpu2['reserved_mb'] - gpu1['reserved_mb'],
                'utilization_diff_percent': gpu2['utilization_percent'] - gpu1['utilization_percent']
            }
            comparison['gpu_memory_diff'].append(diff)
        
        # 比较模型参数（如果有）
        if 'model_analysis' in checkpoint1 and 'model_analysis' in checkpoint2:
            model1 = checkpoint1['model_analysis']
            model2 = checkpoint2['model_analysis']
            
            comparison['model_parameter_diff'] = {
                'total_params_diff': model2['summary']['total_parameters'] - model1['summary']['total_parameters'],
                'trainable_params_diff': model2['summary']['trainable_parameters'] - model1['summary']['trainable_parameters'],
                'memory_diff_mb': model2['summary']['total_memory_mb'] - model1['summary']['total_memory_mb']
            }
        
        # 记录比较结果
        self.logger.info(f"检查点比较结果: {checkpoint1_name} vs {checkpoint2_name}")
        for diff in comparison['gpu_memory_diff']:
            self.logger.info(f"  GPU {diff['device_id']}: 内存分配差异 {diff['allocated_diff_mb']:.2f} MB, "
                           f"利用率差异 {diff['utilization_diff_percent']:.1f}%")
        
        return comparison
    
    def generate_memory_report(self, output_file: str = None) -> str:
        """
        生成内存使用报告
        
        Args:
            output_file: 输出文件路径（可选）
            
        Returns:
            报告内容
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("内存使用报告")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {datetime.now().isoformat()}")
        report_lines.append(f"监控历史记录数: {len(self.memory_history)}")
        report_lines.append(f"检查点数量: {len(self.checkpoint_history)}")
        report_lines.append("")
        
        # 当前内存状态
        current_memory = self.get_current_memory_usage()
        report_lines.append("当前内存状态:")
        for gpu_mem in current_memory['gpu_memory']:
            report_lines.append(f"  GPU {gpu_mem['device_id']}: "
                              f"{gpu_mem['allocated_mb']:.2f} MB / {gpu_mem['total_mb']:.2f} MB "
                              f"({gpu_mem['utilization_percent']:.1f}%)")
        
        # 系统内存
        sys_mem = current_memory['system_memory']
        report_lines.append(f"系统内存: {sys_mem['used_gb']:.2f} GB / {sys_mem['total_gb']:.2f} GB "
                          f"({sys_mem['percent']:.1f}%)")
        report_lines.append("")
        
        # 模型参数统计
        if self.parameter_registry:
            report_lines.append("模型参数统计:")
            for model_name, analysis in self.parameter_registry.items():
                summary = analysis['summary']
                report_lines.append(f"  {model_name}:")
                report_lines.append(f"    总参数: {summary['total_parameters']:,}")
                report_lines.append(f"    可训练参数: {summary['trainable_parameters']:,}")
                report_lines.append(f"    内存占用: {summary['total_memory_mb']:.2f} MB")
            report_lines.append("")
        
        # 检查点历史
        if self.checkpoint_history:
            report_lines.append("检查点历史:")
            for checkpoint in self.checkpoint_history[-5:]:  # 最近5个检查点
                report_lines.append(f"  {checkpoint['name']} ({checkpoint['timestamp']})")
                for gpu_mem in checkpoint['memory_usage']['gpu_memory']:
                    report_lines.append(f"    GPU {gpu_mem['device_id']}: {gpu_mem['allocated_mb']:.2f} MB")
            report_lines.append("")
        
        # 内存使用趋势
        if len(self.memory_history) > 1:
            report_lines.append("内存使用趋势:")
            first_record = self.memory_history[0]
            last_record = self.memory_history[-1]
            
            for i, (first_gpu, last_gpu) in enumerate(zip(first_record['gpu_memory'], 
                                                         last_record['gpu_memory'])):
                trend = last_gpu['allocated_mb'] - first_gpu['allocated_mb']
                report_lines.append(f"  GPU {i}: 变化 {trend:+.2f} MB")
        
        report_content = "\n".join(report_lines)
        
        # 保存到文件
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.log_dir, f"memory_report_{timestamp}.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"内存报告已保存: {output_file}")
        
        return report_content
    
    def generate_detailed_memory_json_report(self, model, model_name: str = "model", 
                                          optimizer=None, stage: str = "training") -> Dict:
        """
        生成详细的JSON格式显存占用统计表
        
        Args:
            model: 模型对象
            model_name: 模型名称
            optimizer: 优化器对象（可选）
            stage: 训练阶段（model_creation, training, validation等）
            
        Returns:
            详细的JSON格式统计表
        """
        self.logger.info(f"🔍 生成详细显存占用统计表 - 阶段: {stage}")
        
        # 获取当前GPU内存状态
        gpu_memory = self.get_current_memory_usage()
        
        # 基础统计结构
        memory_report = {
            "stage": stage,
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "gpu_memory_status": gpu_memory,
            "detailed_memory_breakdown": {},
            "memory_summary": {},
            "occupancy_percentages": {}
        }
        
        # 1. 分析模型参数
        self.logger.info("📊 分析模型参数...")
        parameters_analysis = self._analyze_model_parameters_detailed(model)
        memory_report["detailed_memory_breakdown"]["parameters"] = parameters_analysis
        
        # 2. 分析梯度
        self.logger.info("📊 分析梯度...")
        gradients_analysis = self._analyze_gradients_detailed(model)
        memory_report["detailed_memory_breakdown"]["gradients"] = gradients_analysis
        
        # 3. 分析优化器状态
        if optimizer is not None:
            self.logger.info("📊 分析优化器状态...")
            optimizer_analysis = self._analyze_optimizer_detailed(optimizer)
            memory_report["detailed_memory_breakdown"]["optimizer"] = optimizer_analysis
        else:
            memory_report["detailed_memory_breakdown"]["optimizer"] = {
                "total_memory_mb": 0,
                "states": {},
                "message": "No optimizer provided"
            }
        
        # 4. 分析基础模型组件
        self.logger.info("📊 分析基础模型组件...")
        base_model_analysis = self._analyze_base_model_components(model)
        memory_report["detailed_memory_breakdown"]["base_model"] = base_model_analysis
        
        # 5. 估算激活值内存（近似）
        self.logger.info("📊 估算激活值内存...")
        activation_analysis = self._estimate_activation_memory(model)
        memory_report["detailed_memory_breakdown"]["activations"] = activation_analysis
        
        # 6. 计算总体统计和占用率
        self.logger.info("📊 计算总体统计...")
        memory_report["memory_summary"] = self._calculate_memory_summary(memory_report["detailed_memory_breakdown"])
        memory_report["occupancy_percentages"] = self._calculate_occupancy_percentages(
            memory_report["memory_summary"], gpu_memory
        )
        
        # 7. 保存JSON报告
        self.logger.info("💾 保存JSON报告...")
        self._save_json_report(memory_report, stage)
        
        return memory_report
    
    def _analyze_model_parameters_detailed(self, model) -> Dict:
        """详细分析模型参数"""
        parameters_info = {
            "total_memory_mb": 0,
            "total_parameters": 0,
            "trainable_parameters": 0,
            "frozen_parameters": 0,
            "parameter_groups": {},
            "device_distribution": {},
            "dtype_distribution": {},
            "detailed_parameters": []
        }
        
        # 按参数组分类
        param_groups = {
            "base_model": [],
            "lora_adapters": [],
            "embeddings": [],
            "attention_layers": [],
            "feedforward_layers": [],
            "normalization": [],
            "output_layers": [],
            "other": []
        }
        
        for name, param in model.named_parameters():
            param_info = {
                "name": name,
                "shape": list(param.shape),
                "dtype": str(param.dtype),
                "device": str(param.device),
                "requires_grad": param.requires_grad,
                "num_params": param.numel(),
                "memory_mb": param.numel() * param.element_size() / (1024**2),
                "memory_bytes": param.numel() * param.element_size(),
                "element_size": param.element_size()
            }
            
            # 分类到对应的组
            group = self._classify_parameter_group(name)
            param_groups[group].append(param_info)
            
            # 累计统计
            parameters_info["total_memory_mb"] += param_info["memory_mb"]
            parameters_info["total_parameters"] += param_info["num_params"]
            
            if param.requires_grad:
                parameters_info["trainable_parameters"] += param_info["num_params"]
            else:
                parameters_info["frozen_parameters"] += param_info["num_params"]
            
            # 设备分布
            device_str = str(param.device)
            if device_str not in parameters_info["device_distribution"]:
                parameters_info["device_distribution"][device_str] = {"memory_mb": 0, "count": 0}
            parameters_info["device_distribution"][device_str]["memory_mb"] += param_info["memory_mb"]
            parameters_info["device_distribution"][device_str]["count"] += 1
            
            # 数据类型分布
            dtype_str = str(param.dtype)
            if dtype_str not in parameters_info["dtype_distribution"]:
                parameters_info["dtype_distribution"][dtype_str] = {"memory_mb": 0, "count": 0}
            parameters_info["dtype_distribution"][dtype_str]["memory_mb"] += param_info["memory_mb"]
            parameters_info["dtype_distribution"][dtype_str]["count"] += 1
            
            parameters_info["detailed_parameters"].append(param_info)
        
        # 汇总参数组信息
        for group_name, group_params in param_groups.items():
            parameters_info["parameter_groups"][group_name] = {
                "count": len(group_params),
                "total_memory_mb": sum(p["memory_mb"] for p in group_params),
                "total_parameters": sum(p["num_params"] for p in group_params),
                "trainable_parameters": sum(p["num_params"] for p in group_params if p["requires_grad"]),
                "parameters": group_params
            }
        
        return parameters_info
    
    def _analyze_gradients_detailed(self, model) -> Dict:
        """详细分析梯度"""
        gradients_info = {
            "total_memory_mb": 0,
            "gradient_count": 0,
            "gradient_groups": {},
            "device_distribution": {},
            "detailed_gradients": []
        }
        
        # 按梯度组分类
        grad_groups = {
            "base_model": [],
            "lora_adapters": [],
            "embeddings": [],
            "attention_layers": [],
            "feedforward_layers": [],
            "normalization": [],
            "output_layers": [],
            "other": []
        }
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_info = {
                    "name": name,
                    "shape": list(param.grad.shape),
                    "dtype": str(param.grad.dtype),
                    "device": str(param.grad.device),
                    "memory_mb": param.grad.numel() * param.grad.element_size() / (1024**2),
                    "memory_bytes": param.grad.numel() * param.grad.element_size(),
                    "element_size": param.grad.element_size()
                }
                
                # 分类到对应的组
                group = self._classify_parameter_group(name)
                grad_groups[group].append(grad_info)
                
                # 累计统计
                gradients_info["total_memory_mb"] += grad_info["memory_mb"]
                gradients_info["gradient_count"] += 1
                
                # 设备分布
                device_str = str(param.grad.device)
                if device_str not in gradients_info["device_distribution"]:
                    gradients_info["device_distribution"][device_str] = {"memory_mb": 0, "count": 0}
                gradients_info["device_distribution"][device_str]["memory_mb"] += grad_info["memory_mb"]
                gradients_info["device_distribution"][device_str]["count"] += 1
                
                gradients_info["detailed_gradients"].append(grad_info)
        
        # 汇总梯度组信息
        for group_name, group_grads in grad_groups.items():
            gradients_info["gradient_groups"][group_name] = {
                "count": len(group_grads),
                "total_memory_mb": sum(g["memory_mb"] for g in group_grads),
                "gradients": group_grads
            }
        
        return gradients_info
    
    def _analyze_optimizer_detailed(self, optimizer) -> Dict:
        """详细分析优化器状态"""
        optimizer_info = {
            "total_memory_mb": 0,
            "optimizer_type": type(optimizer).__name__,
            "states": {},
            "param_groups": [],
            "device_distribution": {}
        }
        
        try:
            # 分析优化器状态
            for param_id, param in enumerate(optimizer.param_groups):
                group_info = {
                    "group_id": param_id,
                    "lr": param.get('lr', 0),
                    "weight_decay": param.get('weight_decay', 0),
                    "params_count": len(param['params']),
                    "params_memory_mb": 0
                }
                
                for param_tensor in param['params']:
                    if param_tensor in optimizer.state:
                        state = optimizer.state[param_tensor]
                        param_memory = 0
                        
                        # 计算状态占用的内存
                        for state_key, state_value in state.items():
                            if hasattr(state_value, 'numel') and hasattr(state_value, 'element_size'):
                                state_memory_mb = state_value.numel() * state_value.element_size() / (1024**2)
                                param_memory += state_memory_mb
                                
                                # 设备分布
                                device_str = str(state_value.device)
                                if device_str not in optimizer_info["device_distribution"]:
                                    optimizer_info["device_distribution"][device_str] = {"memory_mb": 0, "states": 0}
                                optimizer_info["device_distribution"][device_str]["memory_mb"] += state_memory_mb
                                optimizer_info["device_distribution"][device_str]["states"] += 1
                        
                        group_info["params_memory_mb"] += param_memory
                
                optimizer_info["param_groups"].append(group_info)
                optimizer_info["total_memory_mb"] += group_info["params_memory_mb"]
        
        except Exception as e:
            optimizer_info["error"] = f"Failed to analyze optimizer: {str(e)}"
        
        return optimizer_info
    
    def _analyze_base_model_components(self, model) -> Dict:
        """分析基础模型组件"""
        base_model_info = {
            "model_type": type(model).__name__,
            "total_memory_mb": 0,
            "components": {},
            "layer_analysis": {}
        }
        
        # 分析模型结构
        try:
            for name, module in model.named_modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    weight_memory = module.weight.numel() * module.weight.element_size() / (1024**2)
                    
                    if hasattr(module, 'bias') and module.bias is not None:
                        bias_memory = module.bias.numel() * module.bias.element_size() / (1024**2)
                    else:
                        bias_memory = 0
                    
                    component_info = {
                        "module_type": type(module).__name__,
                        "weight_memory_mb": weight_memory,
                        "bias_memory_mb": bias_memory,
                        "total_memory_mb": weight_memory + bias_memory
                    }
                    
                    base_model_info["components"][name] = component_info
                    base_model_info["total_memory_mb"] += component_info["total_memory_mb"]
        
        except Exception as e:
            base_model_info["error"] = f"Failed to analyze base model: {str(e)}"
        
        return base_model_info
    
    def _estimate_activation_memory(self, model) -> Dict:
        """分析激活值内存"""
        try:
            # 使用详细方法估算激活值内存
            activation_details = self._estimate_activation_memory_detailed(model)
            
            # 构造与原有格式兼容的结果
            activation_info = {
                "estimated_memory_mb": activation_details["total_memory_mb"],
                "estimation_method": activation_details["estimation_method"],
                "notes": activation_details["notes"],
                "model_config": activation_details["model_config"],
                "breakdown": activation_details["breakdown"]
            }
            
            return activation_info
            
        except Exception as e:
            return {
                "estimated_memory_mb": 0,
                "estimation_method": "failed",
                "error": f"Failed to estimate activation memory: {str(e)}"
            }
    
    def _classify_parameter_group(self, param_name: str) -> str:
        """分类参数组"""
        name_lower = param_name.lower()
        
        if 'lora' in name_lower:
            return 'lora_adapters'
        elif 'embed' in name_lower:
            return 'embeddings'
        elif 'lm_head' in name_lower or 'classifier' in name_lower:
            return 'output_layers'
        elif 'norm' in name_lower:
            return 'normalization'
        elif any(keyword in name_lower for keyword in ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj']):
            return 'attention_layers'
        elif any(keyword in name_lower for keyword in ['mlp', 'feed_forward', 'ffn', 'gate_proj', 'up_proj', 'down_proj']):
            return 'feedforward_layers'
        elif 'base_model' in name_lower:
            return 'base_model'
        else:
            return 'other'
    
    def _calculate_memory_summary(self, memory_breakdown: Dict) -> Dict:
        """计算内存总结"""
        summary = {
            "total_memory_mb": 0,
            "component_breakdown": {}
        }
        
        for component, info in memory_breakdown.items():
            component_memory = 0
            
            if isinstance(info, dict):
                # 尝试获取 total_memory_mb 字段
                if 'total_memory_mb' in info:
                    component_memory = info['total_memory_mb']
                elif 'estimated_memory_mb' in info:  # 处理激活值估算
                    component_memory = info['estimated_memory_mb']
                elif isinstance(info, dict) and 'memory_mb' in info:
                    component_memory = info['memory_mb']
                else:
                    # 如果没有直接的内存字段，尝试从子组件累加
                    if 'parameter_groups' in info:
                        for group_name, group_info in info['parameter_groups'].items():
                            if isinstance(group_info, dict) and 'total_memory_mb' in group_info:
                                component_memory += group_info['total_memory_mb']
                    elif 'gradient_groups' in info:
                        for group_name, group_info in info['gradient_groups'].items():
                            if isinstance(group_info, dict) and 'total_memory_mb' in group_info:
                                component_memory += group_info['total_memory_mb']
            elif isinstance(info, (int, float)):
                # 如果直接是数值
                component_memory = info
            else:
                # 其他类型，跳过
                self.logger.warning(f"跳过组件 {component}，类型为 {type(info)}")
                continue
            
            summary["component_breakdown"][component] = component_memory
            summary["total_memory_mb"] += component_memory
        
        return summary
    
    def _calculate_occupancy_percentages(self, memory_summary: Dict, gpu_memory: Dict) -> Dict:
        """计算占用率百分比"""
        occupancy = {
            "total_used_percentage": 0,
            "component_percentages": {}
        }
        
        try:
            if gpu_memory and 'gpu_memory' in gpu_memory:
                gpu_memory_list = gpu_memory['gpu_memory']
                
                # 处理gpu_memory是列表的情况
                if isinstance(gpu_memory_list, list) and len(gpu_memory_list) > 0:
                    # 使用第一个GPU的信息计算占用率
                    device_info = gpu_memory_list[0]
                    total_gpu_memory = device_info.get('total_mb', 0)
                    
                    if total_gpu_memory > 0:
                        # 计算总占用率
                        occupancy["total_used_percentage"] = (memory_summary["total_memory_mb"] / total_gpu_memory) * 100
                        
                        # 计算各组件占用率
                        for component, component_memory in memory_summary["component_breakdown"].items():
                            occupancy["component_percentages"][component] = (component_memory / total_gpu_memory) * 100
                
                # 处理gpu_memory是字典的情况（向后兼容）
                elif isinstance(gpu_memory_list, dict):
                    for device_id, device_info in gpu_memory_list.items():
                        if 'total_memory_mb' in device_info:
                            total_gpu_memory = device_info['total_memory_mb']
                        elif 'total_mb' in device_info:
                            total_gpu_memory = device_info['total_mb']
                        else:
                            continue
                        
                        if total_gpu_memory > 0:
                            # 计算总占用率
                            occupancy["total_used_percentage"] = (memory_summary["total_memory_mb"] / total_gpu_memory) * 100
                            
                            # 计算各组件占用率
                            for component, component_memory in memory_summary["component_breakdown"].items():
                                occupancy["component_percentages"][component] = (component_memory / total_gpu_memory) * 100
                            
                            break  # 只计算第一个GPU的占用率
                else:
                    self.logger.warning(f"无法识别的gpu_memory类型: {type(gpu_memory_list)}")
            else:
                self.logger.warning("没有找到GPU内存信息，无法计算占用率")
                
        except Exception as e:
            self.logger.error(f"计算占用率时出错: {e}")
            # 返回默认值，避免程序崩溃
            occupancy = {
                "total_used_percentage": 0,
                "component_percentages": {component: 0 for component in memory_summary.get("component_breakdown", {})}
            }
        
        return occupancy
    
    def _save_json_report(self, memory_report: Dict, stage: str):
        """保存JSON报告到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"memory_report_{stage}_{timestamp}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(memory_report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"💾 JSON报告已保存: {filepath}")
            
            # 同时保存一个简化版本用于快速查看
            simplified_report = {
                "stage": memory_report["stage"],
                "timestamp": memory_report["timestamp"],
                "memory_summary": memory_report["memory_summary"],
                "occupancy_percentages": memory_report["occupancy_percentages"],
                "gpu_memory_status": memory_report["gpu_memory_status"]
            }
            
            simplified_filename = f"memory_summary_{stage}_{timestamp}.json"
            simplified_filepath = os.path.join(self.log_dir, simplified_filename)
            
            with open(simplified_filepath, 'w', encoding='utf-8') as f:
                json.dump(simplified_report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"💾 简化报告已保存: {simplified_filepath}")
            
        except Exception as e:
            self.logger.error(f"保存JSON报告失败: {e}")
    
    def generate_brief_memory_report(self, model, model_name: str = "model", 
                                   optimizer=None, stage: str = "training") -> Dict:
        """
        生成简洁的内存占用报告
        
        Args:
            model: 模型对象
            model_name: 模型名称
            optimizer: 优化器对象（可选）
            stage: 训练阶段
            
        Returns:
            简洁的内存占用报告
        """
        self.logger.info(f"📊 生成简洁内存报告 - 阶段: {stage}")
        
        # 获取当前GPU内存状态
        gpu_memory = self.get_current_memory_usage()
        
        # 初始化简洁报告结构
        brief_report = {
            "stage": stage,
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "gpu_memory_status": gpu_memory,
            "memory_breakdown": {
                "base_model": {"memory_mb": 0, "percentage": 0},
                "lora_parameters": {"memory_mb": 0, "percentage": 0},
                "gradients": {"memory_mb": 0, "percentage": 0},
                "optimizer_states": {"memory_mb": 0, "percentage": 0},
                "activations": {"memory_mb": 0, "percentage": 0}
            },
            "total_memory_mb": 0,
            "gpu_utilization_percentage": 0
        }
        
        # 1. 分析基础模型内存
        base_model_memory = self._calculate_base_model_memory(model)
        brief_report["memory_breakdown"]["base_model"]["memory_mb"] = base_model_memory
        
        # 2. 分析LoRA参数内存
        lora_memory = self._calculate_lora_memory(model)
        brief_report["memory_breakdown"]["lora_parameters"]["memory_mb"] = lora_memory
        
        # 3. 分析梯度内存
        gradient_memory = self._calculate_gradient_memory(model)
        brief_report["memory_breakdown"]["gradients"]["memory_mb"] = gradient_memory
        
        # 4. 分析优化器状态内存
        optimizer_memory = 0
        if optimizer is not None:
            optimizer_memory = self._calculate_optimizer_memory(optimizer)
        brief_report["memory_breakdown"]["optimizer_states"]["memory_mb"] = optimizer_memory
        
        # 5. 估算激活值内存（使用详细方法）
        activation_details = self._estimate_activation_memory_detailed(model)
        activation_memory = activation_details["total_memory_mb"]
        brief_report["memory_breakdown"]["activations"]["memory_mb"] = activation_memory
        brief_report["memory_breakdown"]["activations"]["details"] = activation_details
        
        # 6. 计算总内存和百分比
        total_memory = (base_model_memory + lora_memory + gradient_memory + 
                       optimizer_memory + activation_memory)
        brief_report["total_memory_mb"] = total_memory
        
        # 计算各组件占比
        if total_memory > 0:
            for component in brief_report["memory_breakdown"]:
                component_memory = brief_report["memory_breakdown"][component]["memory_mb"]
                percentage = (component_memory / total_memory) * 100
                brief_report["memory_breakdown"][component]["percentage"] = percentage
        
        # 计算GPU利用率
        try:
            if gpu_memory and 'gpu_memory' in gpu_memory:
                gpu_memory_list = gpu_memory['gpu_memory']
                
                # 处理gpu_memory是列表的情况
                if isinstance(gpu_memory_list, list) and len(gpu_memory_list) > 0:
                    # 使用第一个GPU的信息计算利用率
                    device_info = gpu_memory_list[0]
                    if 'total_mb' in device_info:
                        gpu_total = device_info['total_mb']
                    elif 'total_memory_mb' in device_info:
                        gpu_total = device_info['total_memory_mb']
                    else:
                        gpu_total = 0
                    
                    if gpu_total > 0:
                        brief_report["gpu_utilization_percentage"] = (total_memory / gpu_total) * 100
                
                # 处理gpu_memory是字典的情况（向后兼容）
                elif isinstance(gpu_memory_list, dict):
                    for device_info in gpu_memory_list.values():
                        if 'total_memory_mb' in device_info:
                            gpu_total = device_info['total_memory_mb']
                        elif 'total_mb' in device_info:
                            gpu_total = device_info['total_mb']
                        else:
                            continue
                        
                        if gpu_total > 0:
                            brief_report["gpu_utilization_percentage"] = (total_memory / gpu_total) * 100
                        break
                else:
                    self.logger.warning(f"无法识别的gpu_memory类型: {type(gpu_memory_list)}")
            else:
                self.logger.warning("没有找到GPU内存信息，无法计算GPU利用率")
        except Exception as e:
            self.logger.error(f"计算GPU利用率时出错: {e}")
            brief_report["gpu_utilization_percentage"] = 0
        
        # 保存简洁报告
        self._save_brief_report(brief_report, stage)
        
        # 打印到日志
        self._log_brief_report(brief_report)
        
        return brief_report
    
    def _calculate_base_model_memory(self, model) -> float:
        """计算基础模型内存（不包括LoRA）"""
        base_memory = 0
        for name, param in model.named_parameters():
            if 'lora' not in name.lower():
                base_memory += param.numel() * param.element_size() / (1024**2)
        return base_memory
    
    def _calculate_lora_memory(self, model) -> float:
        """计算LoRA参数内存"""
        lora_memory = 0
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                lora_memory += param.numel() * param.element_size() / (1024**2)
        return lora_memory
    
    def _calculate_gradient_memory(self, model) -> float:
        """计算梯度内存"""
        gradient_memory = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gradient_memory += param.grad.numel() * param.grad.element_size() / (1024**2)
        return gradient_memory
    
    def _calculate_optimizer_memory(self, optimizer) -> float:
        """计算优化器状态内存"""
        optimizer_memory = 0
        try:
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    if param in optimizer.state:
                        state = optimizer.state[param]
                        for state_key, state_value in state.items():
                            if hasattr(state_value, 'numel') and hasattr(state_value, 'element_size'):
                                optimizer_memory += state_value.numel() * state_value.element_size() / (1024**2)
        except Exception as e:
            self.logger.warning(f"计算优化器内存时出错: {e}")
        return optimizer_memory
    
    def _estimate_activation_memory_simple(self, model) -> float:
        """
        简单估算激活值内存（仅估算前向传播中的中间激活值）
        不包括参数本身的内存，避免重复计算
        """
        try:
            # 基于模型架构估算激活值内存
            # 这里使用更合理的估算方法，不基于参数总量
            
            # 估算隐藏层大小（假设典型的transformer架构）
            hidden_size = 4096  # 默认隐藏层大小
            seq_length = 512    # 默认序列长度
            batch_size = 1      # 默认批次大小
            
            # 尝试从模型配置中获取实际值
            try:
                if hasattr(model, 'config'):
                    config = model.config
                    hidden_size = getattr(config, 'hidden_size', hidden_size)
                    if hasattr(config, 'max_position_embeddings'):
                        seq_length = min(seq_length, config.max_position_embeddings)
                elif hasattr(model, 'model') and hasattr(model.model, 'config'):
                    config = model.model.config
                    hidden_size = getattr(config, 'hidden_size', hidden_size)
                    if hasattr(config, 'max_position_embeddings'):
                        seq_length = min(seq_length, config.max_position_embeddings)
            except:
                pass
            
            # 估算激活值内存（单位：MB）
            # 包括：attention激活值、MLP激活值、层归一化激活值等
            # 使用float16精度计算（2字节每个数值）
            bytes_per_element = 2
            
            # 基本激活值估算
            attention_activations = batch_size * seq_length * hidden_size * bytes_per_element
            mlp_activations = batch_size * seq_length * hidden_size * 4 * bytes_per_element  # MLP通常4倍扩展
            
            # 估算总激活值内存
            total_activation_bytes = attention_activations + mlp_activations
            activation_memory_mb = total_activation_bytes / (1024**2)
            
            self.logger.debug(f"估算激活值内存: {activation_memory_mb:.2f} MB (hidden_size={hidden_size}, seq_length={seq_length})")
            
            return activation_memory_mb
            
        except Exception as e:
            self.logger.warning(f"估算激活值内存时出错: {e}")
            return 0

    def _estimate_activation_memory_detailed(self, model) -> Dict:
        """
        详细估算激活值内存并提供breakdown
        """
        try:
            # 基础配置参数
            hidden_size = 4096
            seq_length = 512
            batch_size = 1
            num_layers = 32
            
            # 尝试从模型配置中获取实际值
            try:
                if hasattr(model, 'config'):
                    config = model.config
                    hidden_size = getattr(config, 'hidden_size', hidden_size)
                    num_layers = getattr(config, 'num_hidden_layers', num_layers)
                    if hasattr(config, 'max_position_embeddings'):
                        seq_length = min(seq_length, config.max_position_embeddings)
                elif hasattr(model, 'model') and hasattr(model.model, 'config'):
                    config = model.model.config
                    hidden_size = getattr(config, 'hidden_size', hidden_size)
                    num_layers = getattr(config, 'num_hidden_layers', num_layers)
                    if hasattr(config, 'max_position_embeddings'):
                        seq_length = min(seq_length, config.max_position_embeddings)
            except:
                pass
            
            bytes_per_element = 2  # float16
            
            # 详细的激活值分类
            activation_breakdown = {
                "embedding_activations": {
                    "memory_mb": 0,
                    "description": "词嵌入层激活值"
                },
                "attention_activations": {
                    "memory_mb": 0,
                    "description": "注意力层激活值（Q、K、V矩阵运算结果）"
                },
                "mlp_activations": {
                    "memory_mb": 0,
                    "description": "MLP层激活值（前馈网络中间结果）"
                },
                "layer_norm_activations": {
                    "memory_mb": 0,
                    "description": "层归一化激活值"
                },
                "output_activations": {
                    "memory_mb": 0,
                    "description": "输出层激活值"
                }
            }
            
            # 1. 词嵌入激活值
            embedding_memory = batch_size * seq_length * hidden_size * bytes_per_element
            activation_breakdown["embedding_activations"]["memory_mb"] = embedding_memory / (1024**2)
            
            # 2. 注意力激活值（每层）
            # Q、K、V矩阵 + 注意力分数 + 输出
            attention_per_layer = batch_size * seq_length * hidden_size * 4 * bytes_per_element
            attention_total = attention_per_layer * num_layers
            activation_breakdown["attention_activations"]["memory_mb"] = attention_total / (1024**2)
            
            # 3. MLP激活值（每层）
            # 假设MLP有4倍隐藏层大小的中间层
            mlp_per_layer = batch_size * seq_length * hidden_size * 4 * bytes_per_element
            mlp_total = mlp_per_layer * num_layers
            activation_breakdown["mlp_activations"]["memory_mb"] = mlp_total / (1024**2)
            
            # 4. 层归一化激活值
            layer_norm_per_layer = batch_size * seq_length * hidden_size * bytes_per_element
            layer_norm_total = layer_norm_per_layer * num_layers * 2  # 每层通常有2个LayerNorm
            activation_breakdown["layer_norm_activations"]["memory_mb"] = layer_norm_total / (1024**2)
            
            # 5. 输出层激活值
            output_memory = batch_size * seq_length * hidden_size * bytes_per_element
            activation_breakdown["output_activations"]["memory_mb"] = output_memory / (1024**2)
            
            # 计算总内存
            total_activation_memory = sum(
                component["memory_mb"] for component in activation_breakdown.values()
            )
            
            # 计算百分比
            if total_activation_memory > 0:
                for component in activation_breakdown.values():
                    component["percentage"] = (component["memory_mb"] / total_activation_memory) * 100
            
            return {
                "total_memory_mb": total_activation_memory,
                "estimation_method": "detailed_layer_based",
                "model_config": {
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "seq_length": seq_length,
                    "batch_size": batch_size
                },
                "breakdown": activation_breakdown,
                "notes": "基于模型架构的详细激活值内存估算，不包括参数本身"
            }
            
        except Exception as e:
            self.logger.warning(f"详细估算激活值内存时出错: {e}")
            return {
                "total_memory_mb": 0,
                "estimation_method": "failed",
                "error": str(e),
                "breakdown": {}
            }
    
    def _save_brief_report(self, brief_report: Dict, stage: str):
        """保存简洁报告到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"brief_memory_report_{stage}_{timestamp}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(brief_report, f, ensure_ascii=False, indent=2)
            self.logger.info(f"💾 简洁报告已保存: {filepath}")
        except Exception as e:
            self.logger.error(f"保存简洁报告失败: {e}")
    
    def _log_brief_report(self, brief_report: Dict):
        """将简洁报告记录到日志"""
        self.logger.info("=" * 60)
        self.logger.info(f"📊 简洁内存报告 - {brief_report['stage']}")
        self.logger.info("=" * 60)
        
        breakdown = brief_report["memory_breakdown"]
        total_memory = brief_report["total_memory_mb"]
        gpu_utilization = brief_report["gpu_utilization_percentage"]
        
        self.logger.info(f"🎯 总内存占用: {total_memory:.2f} MB")
        self.logger.info(f"🖥️  GPU利用率: {gpu_utilization:.2f}%")
        self.logger.info("")
        self.logger.info("📋 各组件内存占用:")
        
        # 按内存占用排序
        sorted_components = sorted(breakdown.items(), key=lambda x: x[1]["memory_mb"], reverse=True)
        
        for i, (component, info) in enumerate(sorted_components):
            memory_mb = info["memory_mb"]
            percentage = info["percentage"]
            
            # 中文组件名映射
            component_names = {
                "base_model": "基础模型",
                "lora_parameters": "LoRA参数",
                "gradients": "梯度",
                "optimizer_states": "优化器状态",
                "activations": "激活值"
            }
            
            display_name = component_names.get(component, component)
            self.logger.info(f"  {i+1}. {display_name}: {memory_mb:.2f} MB ({percentage:.1f}%)")
            
            # 如果是激活值，显示详细breakdown
            if component == "activations" and "details" in info:
                details = info["details"]
                if "breakdown" in details and details["breakdown"]:
                    self.logger.info(f"     📊 激活值详细分解:")
                    for act_type, act_info in details["breakdown"].items():
                        act_memory = act_info["memory_mb"]
                        act_percentage = act_info.get("percentage", 0)
                        act_desc = act_info.get("description", "")
                        
                        # 中文激活值类型映射
                        act_type_names = {
                            "embedding_activations": "词嵌入激活",
                            "attention_activations": "注意力激活",
                            "mlp_activations": "MLP激活",
                            "layer_norm_activations": "层归一化激活",
                            "output_activations": "输出层激活"
                        }
                        
                        act_display_name = act_type_names.get(act_type, act_type)
                        self.logger.info(f"       - {act_display_name}: {act_memory:.2f} MB ({act_percentage:.1f}%)")
                        if act_desc:
                            self.logger.info(f"         {act_desc}")
                    
                    # 显示模型配置信息
                    if "model_config" in details:
                        config = details["model_config"]
                        self.logger.info(f"     🔧 模型配置: hidden_size={config.get('hidden_size', 'N/A')}, "
                                       f"num_layers={config.get('num_layers', 'N/A')}, "
                                       f"seq_length={config.get('seq_length', 'N/A')}")
        
        self.logger.info("=" * 60)
    
    def __enter__(self):
        """上下文管理器入口"""
        if self.enable_real_time:
            self.start_real_time_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        if self.monitoring_active:
            self.stop_real_time_monitoring()
        
        # 生成最终报告
        try:
            self.generate_memory_report()
        except Exception as e:
            self.logger.error(f"生成最终报告失败: {e}")

# 便捷函数
def create_memory_debugger(log_dir: str = "logs/memory_debug", enable_real_time: bool = True) -> MemoryDebugger:
    """创建内存调试器实例"""
    return MemoryDebugger(log_dir=log_dir, enable_real_time=enable_real_time)

def debug_model_memory(model, model_name: str = "model", log_dir: str = "logs/memory_debug") -> Dict:
    """
    快速调试模型内存使用情况
    
    Args:
        model: 模型对象
        model_name: 模型名称
        log_dir: 日志目录
        
    Returns:
        分析结果
    """
    debugger = MemoryDebugger(log_dir=log_dir, enable_real_time=False)
    return debugger.analyze_model_parameters(model, model_name)

def monitor_training_memory(training_function, *args, **kwargs):
    """
    监控训练过程中的内存使用
    
    Args:
        training_function: 训练函数
        *args: 训练函数参数
        **kwargs: 训练函数关键字参数
        
    Returns:
        训练函数的返回值
    """
    with create_memory_debugger() as debugger:
        debugger.create_memory_checkpoint("training_start")
        
        try:
            result = training_function(*args, **kwargs)
            debugger.create_memory_checkpoint("training_end")
            return result
        except Exception as e:
            debugger.create_memory_checkpoint("training_error")
            raise e

# 使用示例
if __name__ == "__main__":
    # 示例：使用内存调试器
    with create_memory_debugger() as debugger:
        # 创建检查点
        debugger.create_memory_checkpoint("start")
        
        # 模拟一些内存操作
        import time
        dummy_tensor = torch.randn(1000, 1000).cuda() if torch.cuda.is_available() else torch.randn(1000, 1000)
        time.sleep(2)
        
        debugger.create_memory_checkpoint("after_tensor_creation")
        
        # 清理张量
        del dummy_tensor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        debugger.create_memory_checkpoint("after_cleanup")
        
        # 比较检查点
        comparison = debugger.compare_checkpoints("start", "after_tensor_creation")
        print("检查点比较结果:", comparison) 