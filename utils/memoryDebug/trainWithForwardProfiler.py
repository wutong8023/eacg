#!/usr/bin/env python3
"""
集成前向传播分析器到训练过程中
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any
import os
from datetime import datetime
from .forwardPassProfiler import ForwardPassProfiler, ProfiledModel, profile_model_forward

class TrainingForwardProfiler:
    """训练过程中的前向传播分析器"""
    
    def __init__(self, log_dir: str = "logs/training_forward_profiler"):
        self.log_dir = log_dir
        self.profiler = None
        self.analysis_results = []
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
    def profile_single_batch(self, model: nn.Module, batch_data: Dict, stage: str = "training") -> Dict:
        """
        分析单个batch的前向传播
        
        Args:
            model: 模型
            batch_data: batch数据，包含input_ids, labels, attention_mask等
            stage: 阶段名称
            
        Returns:
            分析结果字典
        """
        print(f"\n🔍 开始分析 {stage} 阶段的前向传播...")
        
        # 准备输入数据
        inputs = {
            'input_ids': batch_data['input_ids'],
            'attention_mask': batch_data['attention_mask'],
            'labels': batch_data['labels']
        }
        
        # 使用便捷函数进行分析
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stage_log_dir = os.path.join(self.log_dir, f"{stage}_{timestamp}")
        
        profiler = profile_model_forward(model, inputs, log_dir=stage_log_dir)
        
        # 生成汇总报告
        summary = profiler.generate_summary_report()
        
        # 保存结果
        result = {
            "stage": stage,
            "timestamp": timestamp,
            "batch_shape": {
                "input_ids": list(batch_data['input_ids'].shape),
                "attention_mask": list(batch_data['attention_mask'].shape),
                "labels": list(batch_data['labels'].shape)
            },
            "summary": summary,
            "log_directory": stage_log_dir
        }
        
        self.analysis_results.append(result)
        
        print(f"✅ {stage} 阶段分析完成，详细报告保存在: {stage_log_dir}")
        
        return result
        
    def setup_continuous_profiling(self, model: nn.Module) -> ForwardPassProfiler:
        """
        设置持续的前向传播分析
        
        Args:
            model: 要分析的模型
            
        Returns:
            ForwardPassProfiler实例
        """
        if self.profiler is not None:
            print("⚠️ 警告：已存在活跃的分析器，将先停止现有分析器")
            self.stop_continuous_profiling()
            
        self.profiler = ForwardPassProfiler(log_dir=self.log_dir, enable_detailed_logging=False)
        self.profiler.wrap_model(model)
        
        print("🚀 持续前向传播分析已设置完成")
        return self.profiler
        
    def start_continuous_profiling(self):
        """开始持续分析"""
        if self.profiler is None:
            raise RuntimeError("请先调用 setup_continuous_profiling() 设置分析器")
        self.profiler.start_profiling()
        print("▶️ 开始持续前向传播分析")
        
    def stop_continuous_profiling(self) -> Optional[Dict]:
        """停止持续分析并生成报告"""
        if self.profiler is None:
            return None
            
        self.profiler.stop_profiling()
        
        # 生成报告
        summary = self.profiler.generate_summary_report()
        report_file = self.profiler.save_detailed_report()
        
        # 清理
        self.profiler.unwrap_all()
        self.profiler = None
        
        print("⏹️ 持续前向传播分析已停止")
        
        return {
            "summary": summary,
            "detailed_report_file": report_file
        }
        
    def analyze_memory_pattern(self, model: nn.Module, dataloader, num_batches: int = 3) -> Dict:
        """
        分析多个batch的内存使用模式
        
        Args:
            model: 模型
            dataloader: 数据加载器
            num_batches: 分析的batch数量
            
        Returns:
            内存模式分析结果
        """
        print(f"\n📊 开始分析 {num_batches} 个batch的内存使用模式...")
        
        batch_results = []
        
        with ProfiledModel(model, log_dir=self.log_dir) as profiler:
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                    
                print(f"\n分析第 {i+1}/{num_batches} 个batch...")
                
                # 执行前向传播
                with torch.no_grad():
                    inputs = {
                        'input_ids': batch['input_ids'],
                        'attention_mask': batch['attention_mask'],
                        'labels': batch['labels']
                    }
                    outputs = model(**inputs)
                
                # 记录这个batch的结果
                batch_result = {
                    "batch_id": i,
                    "batch_shape": {
                        "input_ids": list(batch['input_ids'].shape),
                        "attention_mask": list(batch['attention_mask'].shape),
                        "labels": list(batch['labels'].shape)
                    },
                    "output_shape": list(outputs.logits.shape) if hasattr(outputs, 'logits') else "unknown"
                }
                batch_results.append(batch_result)
                
        # 生成模式分析
        summary = profiler.generate_summary_report()
        
        # 分析内存使用模式
        memory_pattern = self._analyze_batch_memory_pattern(batch_results, summary)
        
        result = {
            "num_batches_analyzed": num_batches,
            "batch_results": batch_results,
            "memory_pattern": memory_pattern,
            "overall_summary": summary
        }
        
        print(f"✅ 内存使用模式分析完成")
        
        return result
        
    def _analyze_batch_memory_pattern(self, batch_results: list, summary: Dict) -> Dict:
        """分析batch间的内存使用模式"""
        pattern = {
            "memory_consistency": "unknown",
            "peak_modules": [],
            "potential_issues": []
        }
        
        if "memory_analysis" in summary:
            memory_analysis = summary["memory_analysis"]
            
            # 获取最耗内存的模块
            if "most_memory_intensive_modules" in memory_analysis:
                pattern["peak_modules"] = memory_analysis["most_memory_intensive_modules"][:5]
                
            # 分析潜在问题
            if memory_analysis.get("peak_memory_usage", 0) > 10000:  # 超过10GB
                pattern["potential_issues"].append("峰值内存使用过高")
                
            # 检查模块调用一致性
            total_calls = summary.get("profiling_summary", {}).get("total_forward_calls", 0)
            if total_calls > len(batch_results) * 100:  # 假设每个batch不应该超过100次模块调用
                pattern["potential_issues"].append("模块调用次数异常高")
                
            pattern["memory_consistency"] = "normal" if len(pattern["potential_issues"]) == 0 else "有问题"
            
        return pattern

def integrate_forward_profiler_into_training(
    model: nn.Module, 
    dataloader,
    training_function,
    enable_batch_analysis: bool = True,
    enable_continuous_profiling: bool = False,
    num_analysis_batches: int = 2
) -> Dict:
    """
    将前向传播分析器集成到训练函数中
    
    Args:
        model: 模型
        dataloader: 数据加载器
        training_function: 训练函数
        enable_batch_analysis: 是否启用单batch分析
        enable_continuous_profiling: 是否启用持续分析
        num_analysis_batches: 分析的batch数量
        
    Returns:
        分析结果汇总
    """
    profiler = TrainingForwardProfiler()
    results = {
        "batch_analysis": None,
        "continuous_profiling": None,
        "memory_pattern": None
    }
    
    try:
        # 1. 单batch分析（在训练前）
        if enable_batch_analysis and len(dataloader) > 0:
            print("\n" + "="*60)
            print("🔍 第一阶段：训练前单batch前向传播分析")
            print("="*60)
            
            # 获取第一个batch进行分析
            first_batch = next(iter(dataloader))
            results["batch_analysis"] = profiler.profile_single_batch(
                model, first_batch, stage="pre_training"
            )
            
        # 2. 内存使用模式分析
        if num_analysis_batches > 0:
            print("\n" + "="*60)
            print("🔍 第二阶段：多batch内存使用模式分析")
            print("="*60)
            
            results["memory_pattern"] = profiler.analyze_memory_pattern(
                model, dataloader, num_batches=num_analysis_batches
            )
            
        # 3. 持续分析（在训练过程中）
        if enable_continuous_profiling:
            print("\n" + "="*60)
            print("🔍 第三阶段：训练过程持续分析")
            print("="*60)
            
            # 设置持续分析
            profiler.setup_continuous_profiling(model)
            profiler.start_continuous_profiling()
            
            try:
                # 执行训练（这里只是示例，实际使用时传入真正的训练函数）
                print("🚂 开始训练过程...")
                training_result = training_function()
                
            finally:
                # 停止分析
                results["continuous_profiling"] = profiler.stop_continuous_profiling()
                
        # 生成最终报告
        print("\n" + "="*60)
        print("📋 生成最终分析报告")
        print("="*60)
        
        final_report = _generate_final_report(results)
        
        # 保存最终报告
        report_file = os.path.join(profiler.log_dir, "final_forward_profiling_report.json")
        import json
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)
            
        print(f"📄 最终报告已保存到: {report_file}")
        
        return final_report
        
    except Exception as e:
        print(f"❌ 前向传播分析过程中出错: {e}")
        raise e

def _generate_final_report(results: Dict) -> Dict:
    """生成最终的分析报告"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "analysis_summary": {
            "batch_analysis_completed": results["batch_analysis"] is not None,
            "continuous_profiling_completed": results["continuous_profiling"] is not None,
            "memory_pattern_analysis_completed": results["memory_pattern"] is not None
        },
        "key_findings": [],
        "recommendations": [],
        "detailed_results": results
    }
    
    # 分析关键发现
    if results["memory_pattern"]:
        pattern = results["memory_pattern"]["memory_pattern"]
        if pattern["memory_consistency"] != "normal":
            report["key_findings"].append(f"内存使用一致性: {pattern['memory_consistency']}")
            report["key_findings"].extend(pattern["potential_issues"])
            
        if pattern["peak_modules"]:
            top_module = pattern["peak_modules"][0]
            report["key_findings"].append(
                f"最耗内存模块: {top_module['module_name']} "
                f"(最大内存增量: {top_module['max_memory_delta_mb']:.2f}MB)"
            )
            
    # 生成建议
    if results["memory_pattern"] and results["memory_pattern"]["memory_pattern"]["potential_issues"]:
        report["recommendations"].append("建议优化内存使用最高的模块")
        report["recommendations"].append("考虑使用梯度累积减少单次前向传播的batch size")
        
    return report

# 使用示例函数
def example_usage():
    """使用示例"""
    print("前向传播分析器使用示例:")
    print("="*50)
    
    print("""
# 方式1：简单的单次分析
from utils.memoryDebug.forwardPassProfiler import profile_model_forward

# 准备输入数据
inputs = {
    'input_ids': torch.randint(0, 1000, (2, 512)),
    'attention_mask': torch.ones(2, 512),
    'labels': torch.randint(0, 1000, (2, 512))
}

# 分析前向传播
profiler = profile_model_forward(model, inputs)

# 方式2：上下文管理器方式
from utils.memoryDebug.forwardPassProfiler import ProfiledModel

with ProfiledModel(model) as profiler:
    for batch in dataloader:
        outputs = model(**batch)
        
# 方式3：集成到训练中
from utils.memoryDebug.trainWithForwardProfiler import integrate_forward_profiler_into_training

def my_training_function():
    # 你的训练代码
    return training_result

results = integrate_forward_profiler_into_training(
    model=model,
    dataloader=dataloader,
    training_function=my_training_function,
    enable_batch_analysis=True,
    enable_continuous_profiling=True,
    num_analysis_batches=3
)
    """)

if __name__ == "__main__":
    example_usage() 