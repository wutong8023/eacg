#!/usr/bin/env python3
"""
将前向传播分析器集成到现有训练代码中
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Callable
import os
from datetime import datetime
import traceback

try:
    from .forwardPassProfiler import ForwardPassProfiler, ProfiledModel, profile_model_forward
    from .trainWithForwardProfiler import TrainingForwardProfiler
except ImportError:
    print("Warning: 无法导入前向传播分析器模块，分析功能将被禁用")
    ForwardPassProfiler = None
    ProfiledModel = None
    TrainingForwardProfiler = None

def enhance_training_with_forward_profiling(
    original_training_function: Callable,
    enable_profiling: bool = True,
    profile_first_batch: bool = True,
    profile_memory_pattern: bool = True,
    continuous_profiling: bool = False,
    log_dir: str = "logs/enhanced_forward_profiling"
):
    """
    装饰器：为现有训练函数添加前向传播分析功能
    
    Args:
        original_training_function: 原始训练函数
        enable_profiling: 是否启用分析
        profile_first_batch: 是否分析第一个batch
        profile_memory_pattern: 是否分析内存使用模式
        continuous_profiling: 是否启用持续分析
        log_dir: 日志目录
        
    Returns:
        增强后的训练函数
    """
    
    def enhanced_training(*args, **kwargs):
        if not enable_profiling or ForwardPassProfiler is None:
            # 如果未启用分析或模块不可用，直接执行原函数
            return original_training_function(*args, **kwargs)
            
        print("\n" + "🔬" + "="*70)
        print("🔬 启动增强型前向传播分析训练")
        print("🔬" + "="*70)
        
        # 从参数中提取模型和数据加载器（需要根据实际函数签名调整）
        model = None
        dataloader = None
        
        # 尝试从args和kwargs中找到model和dataloader
        for arg in args:
            if isinstance(arg, nn.Module):
                model = arg
                break
                
        # 常见的参数名
        for key in ['model', 'lora_model', 'dataloader', 'train_dataloader']:
            if key in kwargs and kwargs[key] is not None:
                if isinstance(kwargs[key], nn.Module):
                    model = kwargs[key]
                elif hasattr(kwargs[key], '__iter__'):  # 可能是dataloader
                    dataloader = kwargs[key]
                    
        if model is None:
            print("⚠️ 警告：未找到模型对象，跳过前向传播分析")
            return original_training_function(*args, **kwargs)
            
        # 创建分析器
        profiler_mgr = TrainingForwardProfiler(log_dir=log_dir)
        analysis_results = {}
        
        try:
            # 阶段1：分析第一个batch（如果可用）
            if profile_first_batch and dataloader is not None:
                try:
                    print("\n🔍 阶段1：第一个batch前向传播分析")
                    first_batch = next(iter(dataloader))
                    analysis_results["first_batch"] = profiler_mgr.profile_single_batch(
                        model, first_batch, stage="first_batch_analysis"
                    )
                except Exception as e:
                    print(f"⚠️ 第一个batch分析失败: {e}")
                    
            # 阶段2：内存使用模式分析（如果可用）
            if profile_memory_pattern and dataloader is not None:
                try:
                    print("\n📊 阶段2：内存使用模式分析")
                    analysis_results["memory_pattern"] = profiler_mgr.analyze_memory_pattern(
                        model, dataloader, num_batches=2
                    )
                except Exception as e:
                    print(f"⚠️ 内存模式分析失败: {e}")
                    
            # 阶段3：持续分析（如果启用）
            if continuous_profiling:
                print("\n📈 阶段3：设置持续前向传播分析")
                try:
                    profiler_mgr.setup_continuous_profiling(model)
                    profiler_mgr.start_continuous_profiling()
                except Exception as e:
                    print(f"⚠️ 持续分析设置失败: {e}")
                    continuous_profiling = False
                    
            # 执行原始训练函数
            print("\n🚂 开始执行训练...")
            training_result = original_training_function(*args, **kwargs)
            
            # 停止持续分析
            if continuous_profiling:
                try:
                    analysis_results["continuous"] = profiler_mgr.stop_continuous_profiling()
                except Exception as e:
                    print(f"⚠️ 停止持续分析失败: {e}")
                    
            # 生成最终报告
            _generate_enhanced_training_report(analysis_results, log_dir)
            
            return training_result
            
        except Exception as e:
            print(f"❌ 增强训练过程中出错: {e}")
            traceback.print_exc()
            
            # 确保清理资源
            try:
                if profiler_mgr.profiler is not None:
                    profiler_mgr.stop_continuous_profiling()
            except:
                pass
                
            # 仍然执行原始训练函数
            print("🔄 回退到原始训练函数...")
            return original_training_function(*args, **kwargs)
    
    return enhanced_training

def _generate_enhanced_training_report(analysis_results: Dict, log_dir: str):
    """生成增强训练的分析报告"""
    try:
        import json
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "enhanced_training_summary": {
                "first_batch_analyzed": "first_batch" in analysis_results,
                "memory_pattern_analyzed": "memory_pattern" in analysis_results,
                "continuous_profiling_completed": "continuous" in analysis_results
            },
            "analysis_results": analysis_results,
            "recommendations": []
        }
        
        # 生成建议
        if "memory_pattern" in analysis_results:
            pattern = analysis_results["memory_pattern"].get("memory_pattern", {})
            if pattern.get("potential_issues"):
                report["recommendations"].extend([
                    "检测到潜在的内存使用问题",
                    "建议查看详细的模块分析报告",
                    "考虑优化内存使用最高的模块"
                ])
                
        # 保存报告
        report_file = os.path.join(log_dir, "enhanced_training_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        print(f"📋 增强训练分析报告已保存: {report_file}")
        
    except Exception as e:
        print(f"⚠️ 生成增强训练报告失败: {e}")

# 简化的集成函数，专门针对LoRA训练
def add_forward_profiling_to_lora_training(
    model,
    dataloader, 
    enable_first_batch_analysis: bool = True,
    enable_memory_pattern_analysis: bool = False,
    enable_continuous_profiling: bool = False,
    analysis_log_dir: str = "logs/lora_forward_profiling"
) -> Dict:
    """
    为LoRA训练添加前向传播分析
    
    Args:
        model: LoRA模型
        dataloader: 数据加载器
        enable_first_batch_analysis: 是否分析第一个batch
        enable_memory_pattern_analysis: 是否分析内存模式
        enable_continuous_profiling: 是否启用持续分析
        analysis_log_dir: 分析日志目录
        
    Returns:
        分析结果字典
    """
    if ForwardPassProfiler is None:
        print("⚠️ 前向传播分析器不可用，跳过分析")
        return {}
        
    print("\n" + "🔬" + "="*60)
    print("🔬 LoRA训练前向传播分析")
    print("🔬" + "="*60)
    
    profiler_mgr = TrainingForwardProfiler(log_dir=analysis_log_dir)
    results = {}
    
    try:
        # 第一个batch分析
        if enable_first_batch_analysis:
            print("\n🔍 分析第一个batch的前向传播...")
            try:
                first_batch = next(iter(dataloader))
                results["first_batch_analysis"] = profiler_mgr.profile_single_batch(
                    model, first_batch, stage="lora_first_batch"
                )
                print("✅ 第一个batch分析完成")
            except Exception as e:
                print(f"❌ 第一个batch分析失败: {e}")
                
        # 内存模式分析
        if enable_memory_pattern_analysis:
            print("\n📊 分析内存使用模式...")
            try:
                results["memory_pattern_analysis"] = profiler_mgr.analyze_memory_pattern(
                    model, dataloader, num_batches=2
                )
                print("✅ 内存模式分析完成")
            except Exception as e:
                print(f"❌ 内存模式分析失败: {e}")
                
        # 持续分析设置
        if enable_continuous_profiling:
            print("\n📈 设置持续前向传播分析...")
            try:
                profiler_mgr.setup_continuous_profiling(model)
                profiler_mgr.start_continuous_profiling()
                results["continuous_profiler"] = profiler_mgr
                print("✅ 持续分析已启动")
            except Exception as e:
                print(f"❌ 持续分析设置失败: {e}")
                
        return results
        
    except Exception as e:
        print(f"❌ LoRA训练前向传播分析出错: {e}")
        return {}

def finalize_lora_forward_profiling(analysis_results: Dict, log_dir: str = "logs/lora_forward_profiling"):
    """
    完成LoRA训练的前向传播分析
    
    Args:
        analysis_results: 从add_forward_profiling_to_lora_training返回的结果
        log_dir: 日志目录
    """
    if "continuous_profiler" in analysis_results:
        try:
            profiler_mgr = analysis_results["continuous_profiler"]
            continuous_results = profiler_mgr.stop_continuous_profiling()
            analysis_results["continuous_analysis"] = continuous_results
            print("✅ 持续前向传播分析已完成")
        except Exception as e:
            print(f"❌ 完成持续分析失败: {e}")
            
    # 生成最终报告
    try:
        _generate_lora_profiling_summary(analysis_results, log_dir)
    except Exception as e:
        print(f"⚠️ 生成LoRA分析总结失败: {e}")

def _generate_lora_profiling_summary(analysis_results: Dict, log_dir: str):
    """生成LoRA前向传播分析总结"""
    import json
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "lora_forward_profiling_summary": {
            "analyses_completed": list(analysis_results.keys()),
            "total_analyses": len(analysis_results)
        },
        "key_findings": [],
        "performance_insights": [],
        "detailed_results": analysis_results
    }
    
    # 分析关键发现
    if "first_batch_analysis" in analysis_results:
        batch_analysis = analysis_results["first_batch_analysis"]
        if "summary" in batch_analysis:
            memory_analysis = batch_analysis["summary"].get("memory_analysis", {})
            peak_memory = memory_analysis.get("peak_memory_usage", 0)
            if peak_memory > 0:
                summary["key_findings"].append(f"第一个batch峰值内存使用: {peak_memory:.2f}MB")
                
    if "memory_pattern_analysis" in analysis_results:
        pattern_analysis = analysis_results["memory_pattern_analysis"]
        pattern = pattern_analysis.get("memory_pattern", {})
        if pattern.get("potential_issues"):
            summary["key_findings"].extend(pattern["potential_issues"])
            
    # 生成性能洞察
    summary["performance_insights"].append("建议查看详细的JSON报告获取更多信息")
    if summary["key_findings"]:
        summary["performance_insights"].append("检测到潜在的内存优化机会")
        
    # 保存总结
    summary_file = os.path.join(log_dir, "lora_forward_profiling_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        
    print(f"📋 LoRA前向传播分析总结已保存: {summary_file}")
    
    # 打印简要总结
    print("\n" + "📊" + "="*60)
    print("📊 LoRA前向传播分析总结")
    print("📊" + "="*60)
    print(f"✅ 完成的分析: {len(analysis_results)}")
    if summary["key_findings"]:
        print("🔍 关键发现:")
        for finding in summary["key_findings"]:
            print(f"  - {finding}")
    print("📊" + "="*60)

# 使用示例
if __name__ == "__main__":
    print("""
LoRA训练前向传播分析器集成示例:

# 方式1：装饰器方式（推荐用于新代码）
@enhance_training_with_forward_profiling(
    enable_profiling=True,
    profile_first_batch=True,
    continuous_profiling=False
)
def train_lora_model(model, dataloader, ...):
    # 原有的训练代码
    pass

# 方式2：手动集成方式（推荐用于现有代码）
def train_lora_with_profiling(model, dataloader, ...):
    # 在训练开始前进行分析
    analysis_results = add_forward_profiling_to_lora_training(
        model=model,
        dataloader=dataloader,
        enable_first_batch_analysis=True,
        enable_memory_pattern_analysis=True,
        enable_continuous_profiling=True
    )
    
    try:
        # 执行实际训练
        training_result = train_lora_model(model, dataloader, ...)
        
    finally:
        # 完成分析
        finalize_lora_forward_profiling(analysis_results)
    
    return training_result

# 方式3：最简单的方式（仅分析第一个batch）
analysis = add_forward_profiling_to_lora_training(
    model=my_model,
    dataloader=my_dataloader,
    enable_first_batch_analysis=True
)
    """) 