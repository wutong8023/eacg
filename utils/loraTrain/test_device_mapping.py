#!/usr/bin/env python3
"""
测试LoRA训练设备映射功能
"""
import sys
import os
import torch
import argparse
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.loraTrain.loraTrainUtils import (
    create_balanced_device_map,
    create_dynamic_device_map,
    analyze_device_balance,
    create_layer_balanced_mapping
)

def test_balanced_device_map():
    """测试均衡设备映射"""
    print("🧪 测试均衡设备映射...")
    
    # 使用一个已知的模型路径进行测试
    model_path = "/datanfs2/chenrongyi/models/Llama-3.1-8B"
    
    try:
        device_map = create_balanced_device_map(
            model_path,
            force_balance=True,
            exclude_cpu=True
        )
        
        if device_map == "auto":
            print("✅ 回退到auto策略")
        elif device_map == "cpu":
            print("✅ 使用CPU策略")
        elif isinstance(device_map, dict):
            print(f"✅ 创建均衡设备映射成功: {len(device_map)} 个组件")
            
            # 分析均衡性
            balance_info = analyze_device_balance(device_map)
            print(f"   不均衡系数: {balance_info['imbalance_ratio']:.3f}")
            print(f"   设备分布: {balance_info['device_distribution']}")
        else:
            print("❌ 未知的设备映射类型")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_dynamic_device_map():
    """测试动态设备映射"""
    print("\n🧪 测试动态设备映射...")
    
    # 使用一个已知的模型路径进行测试
    model_path = "/datanfs2/chenrongyi/models/Llama-3.1-8B"
    
    try:
        model, tokenizer, device_map_info = create_dynamic_device_map(
            model_path,
            balance_threshold=0.3,
            force_balance=True,
            exclude_cpu=True
        )
        
        if model is None:
            print("❌ 动态设备映射失败")
            print(f"   策略: {device_map_info['strategy']}")
            print(f"   原因: {device_map_info['reason']}")
        else:
            print(f"✅ 动态设备映射成功")
            print(f"   策略: {device_map_info['strategy']}")
            print(f"   原因: {device_map_info['reason']}")
            
            # 检查模型设备映射
            if hasattr(model, 'hf_device_map'):
                balance_info = analyze_device_balance(model.hf_device_map)
                print(f"   不均衡系数: {balance_info['imbalance_ratio']:.3f}")
                print(f"   设备分布: {balance_info['device_distribution']}")
            
            # 清理内存
            del model, tokenizer
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_analyze_device_balance():
    """测试设备均衡性分析"""
    print("\n🧪 测试设备均衡性分析...")
    
    # 模拟auto策略的不均衡分配
    auto_device_map = {
        "model.embed_tokens": 0,
        "model.layers.0": 0, "model.layers.1": 0, "model.layers.2": 0,
        "model.layers.3": 0, "model.layers.4": 0, "model.layers.5": 0,
        "model.layers.6": 0, "model.layers.7": 0,
        "model.layers.8": 1, "model.layers.9": 1, "model.layers.10": 1,
        "model.layers.11": 1, "model.layers.12": 1, "model.layers.13": 1,
        "model.layers.14": 1, "model.layers.15": 1, "model.layers.16": 1,
        "model.layers.17": 1, "model.layers.18": 1, "model.layers.19": 1,
        "model.layers.20": 1,
        "model.layers.21": 2, "model.layers.22": 2, "model.layers.23": 2,
        "model.layers.24": 2, "model.layers.25": 2, "model.layers.26": 2,
        "model.layers.27": 2, "model.layers.28": 2, "model.layers.29": 2,
        "model.layers.30": 2, "model.layers.31": 2,
        "model.norm": 2, "lm_head": 2,
    }
    
    print("📊 分析Auto策略分配:")
    balance_info = analyze_device_balance(auto_device_map)
    print(f"   不均衡系数: {balance_info['imbalance_ratio']:.3f}")
    print(f"   设备分布: {balance_info['device_distribution']}")
    print(f"   层分布: {balance_info['layer_distribution']}")
    
    # 模拟balanced策略的均衡分配
    balanced_device_map = {
        "model.embed_tokens": 0,
        "model.layers.0": 0, "model.layers.1": 0, "model.layers.2": 0,
        "model.layers.3": 0, "model.layers.4": 0, "model.layers.5": 0,
        "model.layers.6": 0, "model.layers.7": 0, "model.layers.8": 0,
        "model.layers.9": 0, "model.layers.10": 0,
        "model.layers.11": 1, "model.layers.12": 1, "model.layers.13": 1,
        "model.layers.14": 1, "model.layers.15": 1, "model.layers.16": 1,
        "model.layers.17": 1, "model.layers.18": 1, "model.layers.19": 1,
        "model.layers.20": 1, "model.layers.21": 1,
        "model.layers.22": 2, "model.layers.23": 2, "model.layers.24": 2,
        "model.layers.25": 2, "model.layers.26": 2, "model.layers.27": 2,
        "model.layers.28": 2, "model.layers.29": 2, "model.layers.30": 2,
        "model.layers.31": 2,
        "model.norm": 2, "lm_head": 2,
    }
    
    print("\n📊 分析Balanced策略分配:")
    balance_info = analyze_device_balance(balanced_device_map)
    print(f"   不均衡系数: {balance_info['imbalance_ratio']:.3f}")
    print(f"   设备分布: {balance_info['device_distribution']}")
    print(f"   层分布: {balance_info['layer_distribution']}")

def test_layer_balanced_mapping():
    """测试层级均衡映射"""
    print("\n🧪 测试层级均衡映射...")
    
    # 模拟GPU信息
    available_gpus = [
        {"device_id": 0, "free_memory_gb": 20.0},
        {"device_id": 1, "free_memory_gb": 20.0},
        {"device_id": 2, "free_memory_gb": 20.0},
    ]
    
    # 模拟模型配置
    class MockConfig:
        def __init__(self):
            self.architectures = ["LlamaForCausalLM"]
    
    model_config = MockConfig()
    
    try:
        device_map = create_layer_balanced_mapping(32, available_gpus, model_config)
        
        if device_map:
            print(f"✅ 层级均衡映射创建成功: {len(device_map)} 个组件")
            
            # 检查是否包含预期的组件
            expected_components = ["model.embed_tokens", "lm_head", "model.norm"]
            for component in expected_components:
                if component in device_map:
                    print(f"   ✓ 找到组件: {component} -> GPU {device_map[component]}")
                else:
                    print(f"   ✗ 缺少组件: {component}")
            
            # 分析均衡性
            balance_info = analyze_device_balance(device_map)
            print(f"   不均衡系数: {balance_info['imbalance_ratio']:.3f}")
            print(f"   设备分布: {balance_info['device_distribution']}")
        else:
            print("❌ 层级均衡映射创建失败")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_gpu_environment():
    """测试GPU环境"""
    print("\n🧪 测试GPU环境...")
    
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"GPU数量: {num_gpus}")
        
        for i in range(num_gpus):
            try:
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / (1024**3)
                allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)
                free_memory = total_memory - allocated_memory
                
                print(f"GPU {i} ({props.name}):")
                print(f"   总内存: {total_memory:.1f}GB")
                print(f"   已用内存: {allocated_memory:.1f}GB")
                print(f"   可用内存: {free_memory:.1f}GB")
            except Exception as e:
                print(f"GPU {i}: 无法获取信息 - {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试LoRA训练设备映射功能")
    parser.add_argument("--test", type=str, default="all", 
                       choices=["all", "balanced", "dynamic", "analyze", "layer", "gpu"],
                       help="选择要运行的测试")
    parser.add_argument("--model_path", type=str, 
                       default="/datanfs2/chenrongyi/models/Llama-3.1-8B",
                       help="测试用的模型路径")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"LoRA训练设备映射功能测试 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        if args.test in ["all", "gpu"]:
            test_gpu_environment()
        
        if args.test in ["all", "analyze"]:
            test_analyze_device_balance()
        
        if args.test in ["all", "layer"]:
            test_layer_balanced_mapping()
        
        if args.test in ["all", "balanced"]:
            test_balanced_device_map()
        
        if args.test in ["all", "dynamic"]:
            test_dynamic_device_map()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试完成!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断测试")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 