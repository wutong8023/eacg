#!/usr/bin/env python3
"""
LoRA训练工具使用示例
演示如何使用新增的r值检查和均衡设备映射功能
"""

import os
import json
import sys
import torch
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.loraTrain.loraTrainUtils import (
    check_lora_r_consistency,
    create_balanced_device_map,
    apply_balanced_device_map,
    getEquipAdaptorModel,
    buildandTrainLoraModel,
    get_dataloader,
    loadTokenizer
)

def demo_r_consistency_check():
    """演示LoRA r值一致性检查功能"""
    print("=" * 60)
    print("演示 1: LoRA r值一致性检查")
    print("=" * 60)
    
    # 配置示例
    config = {
        "model_name": "microsoft/DialoGPT-medium",
        "r": 16,
        "alpha": 32,
        "target_modules": ["c_attn", "c_proj"],
        "target_layers": [0, 1, 2, 3],
        "check_r_consistency": True,
        "strict_r_check": False,  # 设置为True会在不一致时抛出异常
        "precision": "fp16",
        "device_map": "auto"
    }
    
    try:
        # 创建LoRA模型
        print("创建LoRA模型...")
        lora_model = getEquipAdaptorModel(config)
        
        # 手动检查r值一致性
        print("\n手动检查r值一致性...")
        result = check_lora_r_consistency(lora_model, config)
        
        print(f"\n检查结果:")
        print(f"- 是否一致: {result['is_consistent']}")
        print(f"- 期望r值: {result['expected_r']}")
        print(f"- 实际r值: {result['actual_r_values']}")
        
        if result['mismatched_layers']:
            print(f"- 不匹配的层: {result['mismatched_layers']}")
        
        # 清理资源
        del lora_model
        torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        print(f"演示失败: {e}")
        return None

def demo_balanced_device_map():
    """演示均衡设备映射功能"""
    print("\n" + "=" * 60)
    print("演示 2: 均衡设备映射")
    print("=" * 60)
    
    model_name = "microsoft/DialoGPT-medium"
    
    try:
        # 创建均衡设备映射
        print("创建均衡设备映射...")
        device_map = create_balanced_device_map(
            model_name_or_path=model_name,
            force_balance=False,
            exclude_cpu=True
        )
        
        print(f"\n设备映射类型: {type(device_map)}")
        if isinstance(device_map, dict):
            print(f"设备映射包含 {len(device_map)} 个组件")
            # 显示部分映射
            sample_keys = list(device_map.keys())[:10]
            print("设备映射样本:")
            for key in sample_keys:
                print(f"  {key}: cuda:{device_map[key]}")
        else:
            print(f"设备映射: {device_map}")
        
        return device_map
        
    except Exception as e:
        print(f"演示失败: {e}")
        return None

def demo_balanced_model_loading():
    """演示使用均衡设备映射加载模型"""
    print("\n" + "=" * 60)
    print("演示 3: 使用均衡设备映射加载模型")
    print("=" * 60)
    
    model_name = "microsoft/DialoGPT-medium"
    
    try:
        # 配置
        device_map_config = {
            'force_balance': False,
            'exclude_cpu': True,
            'precision': 'fp16'
        }
        
        # 使用均衡设备映射加载模型
        print("使用均衡设备映射加载模型...")
        model, tokenizer, actual_device_map = apply_balanced_device_map(
            model_name_or_path=model_name,
            device_map_config=device_map_config
        )
        
        print(f"模型类型: {type(model)}")
        print(f"分词器类型: {type(tokenizer)}")
        print(f"实际设备映射: {type(actual_device_map)}")
        
        if hasattr(model, 'hf_device_map'):
            print(f"模型设备映射: {model.hf_device_map}")
        
        # 清理资源
        del model, tokenizer
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"演示失败: {e}")
        return False

def demo_integrated_training():
    """演示集成了新功能的训练流程"""
    print("\n" + "=" * 60)
    print("演示 4: 集成训练流程")
    print("=" * 60)
    
    # 增强的配置
    config = {
        "model_name": "microsoft/DialoGPT-medium",
        "r": 16,
        "alpha": 32,
        "target_modules": ["c_attn", "c_proj"],
        "target_layers": [0, 1, 2, 3],
        "num_epochs": 1,  # 演示用少量epoch
        "learning_rate": 1e-4,
        "batch_size": 2,
        "traindata_percentage": 0.1,  # 使用少量数据进行演示
        "precision": "fp16",
        
        # 新增的均衡设备映射配置
        "use_balanced_device_map": True,
        "force_balance": False,
        "exclude_cpu": True,
        
        # 新增的r值检查配置
        "check_r_consistency": True,
        "strict_r_check": False,
        
        # 训练配置
        "target_batch_size": 4,
        "device_map": "auto"  # 如果不使用均衡映射，将使用此设置
    }
    
    try:
        # 创建虚拟数据加载器进行演示
        print("创建数据加载器...")
        
        # 加载tokenizer
        tokenizer = loadTokenizer(config["model_name"])
        
        # 创建简单的演示数据
        sample_texts = [
            "这是一个LoRA训练的示例。",
            "我们正在测试新的功能。",
            "包括r值检查和均衡设备映射。",
            "希望这些功能能够提高训练效率。"
        ]
        
        # 创建简单的数据加载器（这里只是演示，实际应用中请使用您的数据）
        from torch.utils.data import DataLoader
        from utils.loraTrain.buildandloadData import TextDataset, collate_fn
        
        dataset = TextDataset(sample_texts, tokenizer, block_size=128)
        dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
        
        print(f"数据加载器创建完成，包含 {len(dataset)} 个样本")
        
        # 使用增强的训练功能
        print("\n开始训练（演示模式）...")
        
        # 注意：这里使用虚拟的pkg和version，实际使用时请提供真实值
        lora_model = buildandTrainLoraModel(
            config=config,
            dataloader=dataloader,
            precision=config["precision"],
            pkg="demo_pkg",
            version="1.0.0",
            knowledge_type="demo"
        )
        
        print("训练完成！")
        
        # 清理资源
        del lora_model, tokenizer, dataset, dataloader
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_configuration_examples():
    """演示各种配置选项的示例"""
    print("\n" + "=" * 60)
    print("演示 5: 配置选项示例")
    print("=" * 60)
    
    configs = {
        "基础配置": {
            "model_name": "microsoft/DialoGPT-medium",
            "r": 16,
            "alpha": 32,
            "target_modules": ["c_attn", "c_proj"],
            "target_layers": [0, 1, 2, 3],
            "num_epochs": 5,
            "learning_rate": 1e-4,
            "batch_size": 4,
            "precision": "fp16"
        },
        
        "启用均衡设备映射": {
            "model_name": "microsoft/DialoGPT-medium",
            "r": 16,
            "alpha": 32,
            "target_modules": ["c_attn", "c_proj"],
            "target_layers": [0, 1, 2, 3],
            "use_balanced_device_map": True,
            "force_balance": False,
            "exclude_cpu": True
        },
        
        "启用r值检查": {
            "model_name": "microsoft/DialoGPT-medium",
            "r": 16,
            "alpha": 32,
            "target_modules": ["c_attn", "c_proj"],
            "target_layers": [0, 1, 2, 3],
            "check_r_consistency": True,
            "strict_r_check": False
        },
        
        "完整增强配置": {
            "model_name": "microsoft/DialoGPT-medium",
            "r": 16,
            "alpha": 32,
            "target_modules": ["c_attn", "c_proj"],
            "target_layers": [0, 1, 2, 3],
            "num_epochs": 5,
            "learning_rate": 1e-4,
            "batch_size": 4,
            "precision": "fp16",
            "use_balanced_device_map": True,
            "force_balance": False,
            "exclude_cpu": True,
            "check_r_consistency": True,
            "strict_r_check": False,
            "target_batch_size": 8
        }
    }
    
    for name, config in configs.items():
        print(f"\n{name}:")
        print(json.dumps(config, indent=2, ensure_ascii=False))
    
    return configs

def main():
    """主演示函数"""
    print("LoRA训练工具新功能演示")
    print("=" * 60)
    print("本演示将展示以下新功能：")
    print("1. LoRA r值一致性检查")
    print("2. 均衡设备映射")
    print("3. 使用均衡设备映射加载模型")
    print("4. 集成训练流程")
    print("5. 配置选项示例")
    print("=" * 60)
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，某些功能可能无法正常工作")
    else:
        print(f"✅ CUDA可用，检测到 {torch.cuda.device_count()} 个GPU")
    
    try:
        # 演示1: r值一致性检查
        r_result = demo_r_consistency_check()
        
        # 演示2: 均衡设备映射
        device_map_result = demo_balanced_device_map()
        
        # 演示3: 使用均衡设备映射加载模型
        loading_result = demo_balanced_model_loading()
        
        # 演示4: 集成训练流程（需要用户确认）
        if input("\n是否运行集成训练演示？(y/n): ").lower() == 'y':
            training_result = demo_integrated_training()
        else:
            training_result = None
            print("跳过集成训练演示")
        
        # 演示5: 配置选项示例
        config_examples = demo_configuration_examples()
        
        # 总结
        print("\n" + "=" * 60)
        print("演示总结")
        print("=" * 60)
        print(f"r值一致性检查: {'✅ 成功' if r_result else '❌ 失败'}")
        print(f"均衡设备映射: {'✅ 成功' if device_map_result else '❌ 失败'}")
        print(f"模型加载: {'✅ 成功' if loading_result else '❌ 失败'}")
        print(f"集成训练: {'✅ 成功' if training_result else '跳过' if training_result is None else '❌ 失败'}")
        print(f"配置示例: {'✅ 完成' if config_examples else '❌ 失败'}")
        
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        torch.cuda.empty_cache()
        print("\n演示完成！")

if __name__ == "__main__":
    main() 