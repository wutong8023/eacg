"""
集成内存调试器的训练工具
提供便捷的接口在训练过程中监控和调试内存使用情况
"""

import os
import sys
import torch
import logging
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from datetime import datetime
import json

# 添加当前路径到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from memoryDebugger import MemoryDebugger, create_memory_debugger
from utils.loraTrain.loraTrainUtils import buildandTrainLoraModel, getEquipAdaptorModel

class TrainingMemoryProfiler:
    """训练内存性能分析器"""
    
    def __init__(self, log_dir: str = "logs/training_memory_debug", enable_real_time: bool = True):
        self.log_dir = log_dir
        self.enable_real_time = enable_real_time
        self.debugger = create_memory_debugger(log_dir=log_dir, enable_real_time=enable_real_time)
        self.training_stages = []
        self.json_reports = {}  # 存储各阶段的JSON报告
        
    def profile_model_creation(self, config: Dict, model_name: str = "lora_model"):
        """
        监控模型创建过程的内存使用
        
        Args:
            config: 模型配置
            model_name: 模型名称
            
        Returns:
            创建的模型和分析结果
        """
        print(f"🔍 开始监控模型创建过程: {model_name}")
        
        # 创建起始检查点
        self.debugger.create_memory_checkpoint(f"{model_name}_creation_start")
        
        try:
            # 创建模型
            model = getEquipAdaptorModel(config)
            
            # 创建模型创建完成检查点
            self.debugger.create_memory_checkpoint(f"{model_name}_creation_complete", model=model)
            
            # 🆕 生成详细的JSON报告 - 模型创建阶段
            print("📊 生成模型创建阶段的详细JSON报告...")
            json_report = self.debugger.generate_detailed_memory_json_report(
                model=model,
                model_name=model_name,
                optimizer=None,
                stage="model_creation"
            )
            self.json_reports["model_creation"] = json_report
            
            # 🆕 生成简洁报告 - 模型创建阶段
            print("📄 生成模型创建阶段的简洁报告...")
            brief_report = self.debugger.generate_brief_memory_report(
                model=model,
                model_name=model_name,
                optimizer=None,
                stage="model_creation"
            )
            self.json_reports["model_creation_brief"] = brief_report
            
            # 分析模型创建过程的内存变化
            comparison = self.debugger.compare_checkpoints(
                f"{model_name}_creation_start",
                f"{model_name}_creation_complete"
            )
            
            # 记录到训练阶段
            self.training_stages.append({
                'stage': 'model_creation',
                'timestamp': datetime.now().isoformat(),
                'memory_comparison': comparison,
                'json_report': json_report,
                'brief_report': brief_report
            })
            
            return model, comparison
            
        except Exception as e:
            print(f"❌ 模型创建过程出错: {e}")
            raise
    
    def profile_training_process(self, config: Dict, dataloader, precision: str = 'fp16', 
                               pkg: str = None, version: str = None, knowledge_type: str = None):
        """
        监控训练过程的内存使用
        
        Args:
            config: 训练配置
            dataloader: 数据加载器
            precision: 精度设置
            pkg: 包名
            version: 版本
            knowledge_type: 知识类型
            
        Returns:
            训练好的模型和分析结果
        """
        print(f"🚀 开始监控训练过程: {pkg}-{version}")
        
        # 获取模型（从之前的checkpoint或重新创建）
        model_name = f"{pkg}_{version}" if pkg and version else "lora_model"
        
        # 创建训练开始检查点
        self.debugger.create_memory_checkpoint(f"{model_name}_training_start")
        
        try:
            # 执行训练
            from utils.loraTrain.loraTrainUtils import buildandTrainLoraModel
            
            # 🆕 在训练开始前生成报告
            if hasattr(self, '_current_model'):
                print("📊 生成训练开始阶段的详细JSON报告...")
                json_report_start = self.debugger.generate_detailed_memory_json_report(
                    model=self._current_model,
                    model_name=model_name,
                    optimizer=None,
                    stage="training_start"
                )
                self.json_reports["training_start"] = json_report_start
                
                # 🆕 生成简洁报告 - 训练开始阶段
                print("📄 生成训练开始阶段的简洁报告...")
                brief_report_start = self.debugger.generate_brief_memory_report(
                    model=self._current_model,
                    model_name=model_name,
                    optimizer=None,
                    stage="training_start"
                )
                self.json_reports["training_start_brief"] = brief_report_start
            
            # 执行实际的训练
            trained_model = buildandTrainLoraModel(config, dataloader, precision, pkg, version, knowledge_type=knowledge_type)
            
            # 创建训练完成检查点
            self.debugger.create_memory_checkpoint(f"{model_name}_training_complete", model=trained_model)
            
            # 🆕 生成详细的JSON报告 - 训练完成阶段
            print("📊 生成训练完成阶段的详细JSON报告...")
            json_report_complete = self.debugger.generate_detailed_memory_json_report(
                model=trained_model,
                model_name=model_name,
                optimizer=None,  # 注意：这里可能需要传入实际的optimizer
                stage="training_complete"
            )
            self.json_reports["training_complete"] = json_report_complete
            
            # 🆕 生成简洁报告 - 训练完成阶段
            print("📄 生成训练完成阶段的简洁报告...")
            brief_report_complete = self.debugger.generate_brief_memory_report(
                model=trained_model,
                model_name=model_name,
                optimizer=None,
                stage="training_complete"
            )
            self.json_reports["training_complete_brief"] = brief_report_complete
            
            # 比较训练前后的内存变化
            comparison = self.debugger.compare_checkpoints(
                f"{model_name}_training_start",
                f"{model_name}_training_complete"
            )
            
            # 记录到训练阶段
            stage_info = {
                'stage': 'training_process',
                'timestamp': datetime.now().isoformat(),
                'memory_comparison': comparison,
                'json_report_start': json_report_start if 'json_report_start' in locals() else None,
                'json_report_complete': json_report_complete,
                'brief_report_start': brief_report_start if 'brief_report_start' in locals() else None,
                'brief_report_complete': brief_report_complete
            }
            
            self.training_stages.append(stage_info)
            
            return trained_model, comparison
            
        except Exception as e:
            print(f"❌ 训练过程出错: {e}")
            raise
    
    def profile_stage_with_json_report(self, stage_name: str, stage_function, model=None, optimizer=None, *args, **kwargs):
        """
        监控特定阶段并生成JSON报告
        
        Args:
            stage_name: 阶段名称
            stage_function: 阶段函数
            model: 模型对象（如果可用）
            optimizer: 优化器对象（如果可用）
            *args, **kwargs: 传递给阶段函数的参数
            
        Returns:
            阶段函数的结果和分析结果
        """
        print(f"🔍 开始监控阶段: {stage_name}")
        
        # 创建阶段开始检查点
        self.debugger.create_memory_checkpoint(f"{stage_name}_start")
        
        try:
            # 🆕 在阶段开始前生成JSON报告（如果有模型）
            if model is not None:
                print(f"📊 生成{stage_name}阶段开始的详细JSON报告...")
                json_report_start = self.debugger.generate_detailed_memory_json_report(
                    model=model,
                    model_name=f"{stage_name}_model",
                    optimizer=optimizer,
                    stage=f"{stage_name}_start"
                )
                self.json_reports[f"{stage_name}_start"] = json_report_start
            
            # 执行阶段函数
            result = stage_function(*args, **kwargs)
            
            # 创建阶段完成检查点
            self.debugger.create_memory_checkpoint(f"{stage_name}_complete", model=model)
            
            # 🆕 在阶段完成后生成JSON报告（如果有模型）
            if model is not None:
                print(f"📊 生成{stage_name}阶段完成的详细JSON报告...")
                json_report_complete = self.debugger.generate_detailed_memory_json_report(
                    model=model,
                    model_name=f"{stage_name}_model",
                    optimizer=optimizer,
                    stage=f"{stage_name}_complete"
                )
                self.json_reports[f"{stage_name}_complete"] = json_report_complete
            
            # 比较阶段前后的内存变化
            comparison = self.debugger.compare_checkpoints(
                f"{stage_name}_start",
                f"{stage_name}_complete"
            )
            
            # 记录到训练阶段
            stage_info = {
                'stage': stage_name,
                'timestamp': datetime.now().isoformat(),
                'memory_comparison': comparison
            }
            
            if model is not None:
                stage_info['json_report_start'] = json_report_start if 'json_report_start' in locals() else None
                stage_info['json_report_complete'] = json_report_complete if 'json_report_complete' in locals() else None
            
            self.training_stages.append(stage_info)
            
            return result, comparison
            
        except Exception as e:
            print(f"❌ 阶段 {stage_name} 出错: {e}")
            raise

    def profile_stage(self, stage_name: str, stage_function, *args, **kwargs):
        """
        监控特定阶段的内存使用
        
        Args:
            stage_name: 阶段名称
            stage_function: 阶段函数
            *args, **kwargs: 传递给阶段函数的参数
            
        Returns:
            阶段函数的结果和分析结果
        """
        print(f"🔍 开始监控阶段: {stage_name}")
        
        # 创建阶段开始检查点
        self.debugger.create_memory_checkpoint(f"{stage_name}_start")
        
        try:
            # 执行阶段函数
            result = stage_function(*args, **kwargs)
            
            # 创建阶段完成检查点
            self.debugger.create_memory_checkpoint(f"{stage_name}_complete")
            
            # 比较阶段前后的内存变化
            comparison = self.debugger.compare_checkpoints(
                f"{stage_name}_start",
                f"{stage_name}_complete"
            )
            
            # 记录到训练阶段
            self.training_stages.append({
                'stage': stage_name,
                'timestamp': datetime.now().isoformat(),
                'memory_comparison': comparison
            })
            
            return result, comparison
            
        except Exception as e:
            print(f"❌ 阶段 {stage_name} 出错: {e}")
            raise
    
    def get_json_reports(self) -> Dict:
        """获取所有JSON报告"""
        return self.json_reports
    
    def save_all_json_reports(self):
        """保存所有JSON报告到文件"""
        print("💾 保存所有JSON报告...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_reports_file = os.path.join(self.log_dir, f"all_memory_reports_{timestamp}.json")
        
        try:
            with open(all_reports_file, 'w', encoding='utf-8') as f:
                json.dump(self.json_reports, f, ensure_ascii=False, indent=2)
            
            print(f"💾 所有JSON报告已保存: {all_reports_file}")
            
            # 生成汇总报告
            summary_report = self._generate_summary_report()
            summary_file = os.path.join(self.log_dir, f"memory_summary_all_stages_{timestamp}.json")
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_report, f, ensure_ascii=False, indent=2)
            
            print(f"💾 汇总报告已保存: {summary_file}")
            
        except Exception as e:
            print(f"❌ 保存JSON报告失败: {e}")
    
    def _generate_summary_report(self) -> Dict:
        """生成汇总报告"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "stages_analyzed": list(self.json_reports.keys()),
            "memory_progression": [],
            "peak_memory_usage": {"stage": None, "memory_mb": 0},
            "memory_efficiency": {}
        }
        
        for stage, report in self.json_reports.items():
            if 'memory_summary' in report:
                memory_info = {
                    "stage": stage,
                    "total_memory_mb": report['memory_summary']['total_memory_mb'],
                    "component_breakdown": report['memory_summary']['component_breakdown'],
                    "occupancy_percentage": report.get('occupancy_percentages', {}).get('total_used_percentage', 0)
                }
                summary["memory_progression"].append(memory_info)
                
                # 找出峰值内存使用
                if memory_info["total_memory_mb"] > summary["peak_memory_usage"]["memory_mb"]:
                    summary["peak_memory_usage"] = {
                        "stage": stage,
                        "memory_mb": memory_info["total_memory_mb"]
                    }
        
        return summary
    
    def compare_training_stages(self):
        """比较训练各阶段的内存使用"""
        print("📊 比较训练各阶段的内存使用...")
        
        # 找到所有相关的检查点
        checkpoints = [cp['name'] for cp in self.debugger.checkpoint_history]
        
        comparisons = []
        
        # 比较训练开始和完成
        if "training_start" in checkpoints and "training_complete" in checkpoints:
            comparison = self.debugger.compare_checkpoints("training_start", "training_complete")
            comparisons.append(("训练开始 vs 训练完成", comparison))
        
        # 比较模型创建前后
        creation_start = [cp for cp in checkpoints if cp.endswith("_creation_start")]
        creation_complete = [cp for cp in checkpoints if cp.endswith("_creation_complete")]
        
        for start_cp, complete_cp in zip(creation_start, creation_complete):
            comparison = self.debugger.compare_checkpoints(start_cp, complete_cp)
            comparisons.append((f"模型创建: {start_cp} vs {complete_cp}", comparison))
        
        return comparisons
    
    def detect_memory_issues(self) -> Dict:
        """检测内存使用问题"""
        print("🔍 检测内存使用问题...")
        
        issues = {
            'memory_leaks': [],
            'excessive_usage': [],
            'abnormal_growth': [],
            'fragmentation': []
        }
        
        if len(self.debugger.checkpoint_history) < 2:
            print("⚠️  检查点数量不足，无法检测内存问题")
            return issues
        
        # 检查内存泄漏
        for i in range(len(self.debugger.checkpoint_history) - 1):
            current = self.debugger.checkpoint_history[i]
            next_cp = self.debugger.checkpoint_history[i + 1]
            
            for gpu_idx in range(len(current['memory_usage']['gpu_memory'])):
                current_allocated = current['memory_usage']['gpu_memory'][gpu_idx]['allocated_mb']
                next_allocated = next_cp['memory_usage']['gpu_memory'][gpu_idx]['allocated_mb']
                
                growth = next_allocated - current_allocated
                
                # 检查异常增长
                if growth > 500:  # 500MB增长认为异常
                    issues['abnormal_growth'].append({
                        'from_checkpoint': current['name'],
                        'to_checkpoint': next_cp['name'],
                        'gpu_id': gpu_idx,
                        'growth_mb': growth
                    })
        
        # 检查过度使用
        for checkpoint in self.debugger.checkpoint_history:
            for gpu_mem in checkpoint['memory_usage']['gpu_memory']:
                if gpu_mem['utilization_percent'] > 95:
                    issues['excessive_usage'].append({
                        'checkpoint': checkpoint['name'],
                        'gpu_id': gpu_mem['device_id'],
                        'utilization_percent': gpu_mem['utilization_percent']
                    })
        
        # 检查内存碎片化
        for checkpoint in self.debugger.checkpoint_history:
            for gpu_mem in checkpoint['memory_usage']['gpu_memory']:
                reserved = gpu_mem['reserved_mb']
                allocated = gpu_mem['allocated_mb']
                
                if reserved > 0 and allocated > 0:
                    fragmentation_ratio = (reserved - allocated) / reserved
                    if fragmentation_ratio > 0.3:  # 30%以上的碎片化
                        issues['fragmentation'].append({
                            'checkpoint': checkpoint['name'],
                            'gpu_id': gpu_mem['device_id'],
                            'fragmentation_ratio': fragmentation_ratio,
                            'wasted_mb': reserved - allocated
                        })
        
        # 输出问题摘要
        print("\n📋 内存问题检测结果:")
        print(f"  异常增长: {len(issues['abnormal_growth'])} 个")
        print(f"  过度使用: {len(issues['excessive_usage'])} 个")
        print(f"  内存碎片化: {len(issues['fragmentation'])} 个")
        
        return issues
    
    def generate_training_report(self, output_file: str = None) -> str:
        """生成训练内存报告"""
        print("📄 生成训练内存报告...")
        
        # 生成基础报告
        base_report = self.debugger.generate_memory_report(output_file)
        
        # 添加训练特定信息
        training_info = []
        training_info.append("\n" + "=" * 80)
        training_info.append("训练特定信息")
        training_info.append("=" * 80)
        
        # 添加阶段比较
        comparisons = self.compare_training_stages()
        training_info.append("\n训练阶段比较:")
        for name, comparison in comparisons:
            training_info.append(f"\n{name}:")
            for diff in comparison['gpu_memory_diff']:
                training_info.append(f"  GPU {diff['device_id']}: "
                                   f"内存变化 {diff['allocated_diff_mb']:+.2f} MB")
        
        # 添加问题检测结果
        issues = self.detect_memory_issues()
        training_info.append("\n内存问题检测:")
        
        if issues['abnormal_growth']:
            training_info.append("  异常增长:")
            for issue in issues['abnormal_growth']:
                training_info.append(f"    {issue['from_checkpoint']} -> {issue['to_checkpoint']}: "
                                   f"GPU {issue['gpu_id']} 增长 {issue['growth_mb']:.2f} MB")
        
        if issues['excessive_usage']:
            training_info.append("  过度使用:")
            for issue in issues['excessive_usage']:
                training_info.append(f"    {issue['checkpoint']}: "
                                   f"GPU {issue['gpu_id']} 使用率 {issue['utilization_percent']:.1f}%")
        
        if issues['fragmentation']:
            training_info.append("  内存碎片化:")
            for issue in issues['fragmentation']:
                training_info.append(f"    {issue['checkpoint']}: "
                                   f"GPU {issue['gpu_id']} 碎片化 {issue['fragmentation_ratio']:.1%}, "
                                   f"浪费 {issue['wasted_mb']:.2f} MB")
        
        # 合并报告
        full_report = base_report + "\n".join(training_info)
        
        # 保存完整报告
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(full_report)
            print(f"📄 训练内存报告已保存: {output_file}")
        
        return full_report
    
    def cleanup(self):
        """清理资源"""
        if self.debugger.monitoring_active:
            self.debugger.stop_real_time_monitoring()
        
        # 生成最终报告
        self.generate_training_report()

# 上下文管理器
@contextmanager
def memory_profiled_training(log_dir: str = "logs/training_memory_debug", 
                           enable_real_time: bool = True):
    """
    上下文管理器，用于在训练过程中监控内存使用
    
    Args:
        log_dir: 日志目录
        enable_real_time: 是否启用实时监控
        
    Yields:
        TrainingMemoryProfiler 实例
    """
    profiler = TrainingMemoryProfiler(log_dir=log_dir, enable_real_time=enable_real_time)
    
    try:
        yield profiler
    finally:
        # 在退出时保存所有JSON报告
        profiler.save_all_json_reports()
        
        # 生成最终的训练报告
        final_report = profiler.generate_training_report()
        print("✅ 内存分析完成")
        print(f"📊 详细报告已保存到: {log_dir}")


def monitor_memory(log_dir: str = "logs/training_memory_debug", 
                  enable_real_time: bool = True):
    """
    装饰器，用于监控函数执行过程中的内存使用
    
    Args:
        log_dir: 日志目录
        enable_real_time: 是否启用实时监控
        
    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with memory_profiled_training(log_dir=log_dir, enable_real_time=enable_real_time) as profiler:
                return func(profiler, *args, **kwargs)
        return wrapper
    return decorator


def debug_lora_training(config: Dict, dataloader, precision: str = 'fp16', 
                       pkg: str = None, version: str = None, knowledge_type: str = None,
                       log_dir: str = "logs/lora_training_debug"):
    """
    调试LoRA训练过程的内存使用，生成详细的JSON报告
    
    Args:
        config: 训练配置
        dataloader: 数据加载器
        precision: 精度设置
        pkg: 包名
        version: 版本
        knowledge_type: 知识类型
        log_dir: 日志目录
        
    Returns:
        训练好的模型和分析结果（包括详细的JSON报告）
    """
    print(f"🔍 开始调试LoRA训练: {pkg}-{version}")
    
    with memory_profiled_training(log_dir=log_dir, enable_real_time=True) as profiler:
        # 🆕 监控模型创建过程，生成JSON报告
        print("📊 第一阶段：模型创建")
        model, creation_analysis = profiler.profile_model_creation(config, f"{pkg}_{version}")
        
        # 保存当前模型引用供后续使用
        profiler._current_model = model
        
        # 🆕 监控训练过程，生成JSON报告
        print("📊 第二阶段：模型训练")
        trained_model, training_analysis = profiler.profile_training_process(
            config, dataloader, precision, pkg, version, knowledge_type
        )
        
        # 🆕 获取所有JSON报告
        json_reports = profiler.get_json_reports()
        
        # 🆕 生成内存问题检测报告
        print("📊 第三阶段：内存问题检测")
        memory_issues = profiler.detect_memory_issues()
        
        # 🆕 生成训练完整性报告
        print("📊 第四阶段：生成完整性报告")
        stage_comparisons = profiler.compare_training_stages()
        
        # 构建综合分析结果
        comprehensive_analysis = {
            'creation_analysis': creation_analysis,
            'training_analysis': training_analysis,
            'memory_issues': memory_issues,
            'stage_comparisons': stage_comparisons,
            'json_reports': json_reports,  # 🆕 包含所有阶段的详细JSON报告
            'summary': {
                'total_stages_analyzed': len(json_reports),
                'stages': list(json_reports.keys()),
                'peak_memory_stage': None,
                'memory_efficiency_score': None
            }
        }
        
        # 🆕 计算峰值内存使用阶段
        peak_memory = 0
        peak_stage = None
        for stage, report in json_reports.items():
            if 'memory_summary' in report:
                memory_usage = report['memory_summary']['total_memory_mb']
                if memory_usage > peak_memory:
                    peak_memory = memory_usage
                    peak_stage = stage
        
        comprehensive_analysis['summary']['peak_memory_stage'] = peak_stage
        comprehensive_analysis['summary']['peak_memory_mb'] = peak_memory
        
        # 🆕 打印详细的JSON报告摘要到日志
        print("="*80)
        print("📊 详细内存分析报告摘要")
        print("="*80)
        
        for stage, report in json_reports.items():
            if 'memory_summary' in report:
                memory_summary = report['memory_summary']
                occupancy = report.get('occupancy_percentages', {})
                
                print(f"\n🔍 阶段: {stage}")
                print(f"  总内存使用: {memory_summary['total_memory_mb']:.2f} MB")
                print(f"  GPU占用率: {occupancy.get('total_used_percentage', 0):.2f}%")
                print(f"  组件分解:")
                
                for component, memory_mb in memory_summary['component_breakdown'].items():
                    component_percentage = occupancy.get('component_percentages', {}).get(component, 0)
                    print(f"    {component}: {memory_mb:.2f} MB ({component_percentage:.2f}%)")
        
        print(f"\n🏆 峰值内存使用阶段: {peak_stage} ({peak_memory:.2f} MB)")
        print(f"📂 详细JSON报告已保存到: {log_dir}")
        print("="*80)
        
        return trained_model, comprehensive_analysis


def analyze_model_memory_only(model, model_name: str = "model", 
                            log_dir: str = "logs/model_memory_analysis"):
    """
    仅分析模型的内存使用情况，生成详细的JSON报告
    
    Args:
        model: 模型对象
        model_name: 模型名称
        log_dir: 日志目录
        
    Returns:
        内存分析结果
    """
    print(f"🔍 开始分析模型内存: {model_name}")
    
    # 创建内存调试器
    debugger = create_memory_debugger(log_dir=log_dir, enable_real_time=False)
    
    try:
        # 🆕 生成详细的JSON报告
        print("📊 生成详细的JSON报告...")
        json_report = debugger.generate_detailed_memory_json_report(
            model=model,
            model_name=model_name,
            optimizer=None,
            stage="model_analysis"
        )
        
        # 🆕 传统的参数分析（向后兼容）
        parameter_analysis = debugger.analyze_model_parameters(model, model_name)
        
        # 🆕 生成内存报告
        memory_report = debugger.generate_memory_report()
        
        # 构建综合分析结果
        analysis_result = {
            'json_report': json_report,
            'parameter_analysis': parameter_analysis,
            'memory_report': memory_report,
            'summary': {
                'total_parameters': parameter_analysis['summary']['total_parameters'],
                'total_memory_mb': json_report['memory_summary']['total_memory_mb'],
                'gpu_occupancy_percentage': json_report.get('occupancy_percentages', {}).get('total_used_percentage', 0)
            }
        }
        
        print("✅ 模型内存分析完成")
        print(f"📊 详细报告已保存到: {log_dir}")
        
        return analysis_result
        
    except Exception as e:
        print(f"❌ 模型内存分析失败: {e}")
        raise


# 🆕 新增函数：生成内存优化建议
def generate_memory_optimization_suggestions(json_reports: Dict) -> Dict:
    """
    基于JSON报告生成内存优化建议
    
    Args:
        json_reports: 各阶段的JSON报告字典
        
    Returns:
        优化建议字典
    """
    suggestions = {
        "general_suggestions": [],
        "parameter_optimization": [],
        "gradient_optimization": [],
        "optimizer_optimization": [],
        "training_optimization": []
    }
    
    # 分析所有阶段的报告
    for stage, report in json_reports.items():
        if 'detailed_memory_breakdown' not in report:
            continue
            
        breakdown = report['detailed_memory_breakdown']
        occupancy = report.get('occupancy_percentages', {})
        
        # 检查GPU占用率
        total_occupancy = occupancy.get('total_used_percentage', 0)
        if total_occupancy > 90:
            suggestions["general_suggestions"].append(
                f"阶段{stage}: GPU占用率过高({total_occupancy:.1f}%)，建议降低batch_size或使用梯度累积"
            )
        elif total_occupancy < 50:
            suggestions["general_suggestions"].append(
                f"阶段{stage}: GPU利用率较低({total_occupancy:.1f}%)，可以考虑增加batch_size提高训练效率"
            )
        
        # 检查参数分布
        if 'parameters' in breakdown:
            param_info = breakdown['parameters']
            lora_memory = param_info['parameter_groups'].get('lora_adapters', {}).get('total_memory_mb', 0)
            base_memory = param_info['parameter_groups'].get('base_model', {}).get('total_memory_mb', 0)
            
            if lora_memory > base_memory * 0.1:  # LoRA参数占比过高
                suggestions["parameter_optimization"].append(
                    f"阶段{stage}: LoRA参数占用过多内存({lora_memory:.1f}MB)，建议降低rank(r)值"
                )
        
        # 检查梯度内存
        if 'gradients' in breakdown:
            grad_memory = breakdown['gradients']['total_memory_mb']
            if grad_memory > 1000:  # 梯度内存超过1GB
                suggestions["gradient_optimization"].append(
                    f"阶段{stage}: 梯度内存占用过高({grad_memory:.1f}MB)，建议使用梯度检查点或混合精度训练"
                )
        
        # 检查优化器内存
        if 'optimizer' in breakdown:
            optimizer_memory = breakdown['optimizer']['total_memory_mb']
            if optimizer_memory > 2000:  # 优化器内存超过2GB
                suggestions["optimizer_optimization"].append(
                    f"阶段{stage}: 优化器内存占用过高({optimizer_memory:.1f}MB)，建议使用AdamW或其他内存友好的优化器"
                )
    
    return suggestions


# 使用示例
if __name__ == "__main__":
    # 示例1：使用装饰器监控函数
    @monitor_memory(log_dir="logs/decorated_training")
    def example_training_function(profiler, config):
        # 在这个函数中，profiler已经可用
        model, analysis = profiler.profile_model_creation(config, "example_model")
        return model
    
    # 示例2：使用上下文管理器
    # with memory_profiled_training(log_dir="logs/context_training") as profiler:
    #     model, analysis = profiler.profile_model_creation(config, "context_model")
    #     
    #     # 获取所有JSON报告
    #     json_reports = profiler.get_json_reports()
    #     
    #     # 生成优化建议
    #     suggestions = generate_memory_optimization_suggestions(json_reports)
    
    pass 