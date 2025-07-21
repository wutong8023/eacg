#!/usr/bin/env python3
"""
测试不同精度加载的示例代码
"""

import torch
from loraTrainUtils import load_base_model, get_precision_info, check_lora_params

def test_precision_loading():
    """测试不同精度的模型加载"""
    
    print("=" * 60)
    print("精度支持测试")
    print("=" * 60)
    
    # 显示支持的精度信息
    get_precision_info()
    
    # 示例模型路径 (请替换为您的实际模型路径)
    model_path = "/path/to/your/model"  # 替换为实际路径
    
    # 测试不同精度
    precisions = ['fp16', 'fp32', 'bf16']
    
    for precision in precisions:
        try:
            print(f"\n{'='*20} 测试 {precision} 精度 {'='*20}")
            
            # 加载模型
            model, tokenizer = load_base_model(
                model_path, 
                device_map="auto", 
                precision=precision
            )
            
            print(f"✓ {precision} 精度模型加载成功")
            
            # 检查模型参数类型
            first_param = next(model.parameters())
            print(f"模型参数数据类型: {first_param.dtype}")
            print(f"模型所在设备: {first_param.device}")
            
            # 检查是否有LoRA参数
            has_lora, lora_params = check_lora_params(model)
            print(f"LoRA参数数量: {len(lora_params)}")
            
            # 清理内存
            del model, tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"✗ {precision} 精度加载失败: {e}")
            continue

def show_config_example():
    """展示配置文件示例"""
    
    config_example = {
        "model_name": "/path/to/your/model",
        "precision": "fp16",  # 可选: fp16, fp32, bf16
        "device_map": "auto",
        "batch_size": 4,
        "num_epochs": 3,
        "learning_rate": 1e-4,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "target_layers": [0, 1, 2],
        "r": 16,
        "alpha": 32
    }
    
    print("\n" + "="*60)
    print("配置文件示例")
    print("="*60)
    
    import json
    print(json.dumps(config_example, indent=2, ensure_ascii=False))
    
    print("\n注意事项:")
    print("1. precision 参数控制模型的数值精度")
    print("2. fp16 是默认选项，平衡性能和精度")
    print("3. fp32 精度最高但占用内存最多")
    print("4. bf16 在某些新GPU上性能更好")

if __name__ == "__main__":
    # 显示配置示例
    show_config_example()
    
    # 如果要测试实际加载，请取消注释以下行并设置正确的模型路径
    # test_precision_loading() 