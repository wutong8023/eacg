#!/usr/bin/env python3
"""
快速LoRA适配器文件修复工具 (简化版)
专注于快速检查和修复缺失的adapter_model.safetensors和adapter_config.json
"""

import os
import shutil
import re
import sys

def find_and_fix_lora_models(base_path="/datanfs2/chenrongyi/models/loraadaptors/"):
    """
    快速查找并修复LoRA模型文件
    
    Args:
        base_path: LoRA模型基础路径
        
    Returns:
        int: 成功修复的模型数量
    """
    print(f"🔍 扫描LoRA模型目录: {base_path}")
    
    fixed_count = 0
    total_count = 0
    complete_count = 0
    
    # 遍历所有模型目录
    for root, dirs, files in os.walk(base_path):
        # 检查是否包含checkpoints目录，确认是LoRA模型目录
        if "checkpoints" in dirs:
            total_count += 1
            model_name = os.path.basename(root)
            
            print(f"\n📦 检查模型: {model_name}")
            
            # 检查必需文件
            adapter_model_exists = os.path.exists(os.path.join(root, "adapter_model.safetensors"))
            adapter_config_exists = os.path.exists(os.path.join(root, "adapter_config.json"))
            
            if adapter_model_exists and adapter_config_exists:
                print(f"  ✅ 文件完整")
                complete_count += 1
                continue
            
            # 找出缺失的文件
            missing_files = []
            if not adapter_model_exists:
                missing_files.append("adapter_model.safetensors")
            if not adapter_config_exists:
                missing_files.append("adapter_config.json")
            
            print(f"  ⚠️  缺失文件: {', '.join(missing_files)}")
            
            # 查找最佳checkpoint
            checkpoints_dir = os.path.join(root, "checkpoints")
            best_checkpoint = None
            
            # 优先查找epoch4
            for checkpoint_dir in os.listdir(checkpoints_dir):
                if checkpoint_dir.endswith("epoch4"):
                    best_checkpoint = os.path.join(checkpoints_dir, checkpoint_dir)
                    break
            
            # 如果没有epoch4，找最高epoch
            if not best_checkpoint:
                max_epoch = -1
                best_checkpoint_dir = None
                for checkpoint_dir in os.listdir(checkpoints_dir):
                    match = re.search(r'epoch(\d+)$', checkpoint_dir)
                    if match:
                        epoch_num = int(match.group(1))
                        if epoch_num > max_epoch:
                            max_epoch = epoch_num
                            best_checkpoint_dir = checkpoint_dir
                
                if best_checkpoint_dir:
                    best_checkpoint = os.path.join(checkpoints_dir, best_checkpoint_dir)
            
            # 如果还是没找到，使用第一个checkpoint
            if not best_checkpoint:
                checkpoint_dirs = [d for d in os.listdir(checkpoints_dir) 
                                 if os.path.isdir(os.path.join(checkpoints_dir, d))]
                if checkpoint_dirs:
                    best_checkpoint = os.path.join(checkpoints_dir, checkpoint_dirs[0])
            
            if not best_checkpoint:
                print(f"  ❌ 没有可用的checkpoint")
                continue
            
            checkpoint_name = os.path.basename(best_checkpoint)
            print(f"  🔧 使用checkpoint: {checkpoint_name}")
            
            # 复制缺失的文件
            success = True
            copied_files = []
            
            for missing_file in missing_files:
                src_file = os.path.join(best_checkpoint, missing_file)
                dst_file = os.path.join(root, missing_file)
                
                if os.path.exists(src_file):
                    try:
                        shutil.copy2(src_file, dst_file)
                        copied_files.append(missing_file)
                        print(f"    ✅ 复制: {missing_file}")
                    except Exception as e:
                        print(f"    ❌ 复制失败: {missing_file} - {e}")
                        success = False
                else:
                    print(f"    ❌ checkpoint中缺少: {missing_file}")
                    success = False
            
            if success and copied_files:
                print(f"  🎉 修复成功! 恢复了 {len(copied_files)} 个文件")
                fixed_count += 1
            elif not copied_files:
                print(f"  ⚠️  没有需要复制的文件")
            else:
                print(f"  ❌ 修复失败")
    
    return fixed_count, total_count, complete_count

def main():
    """主函数"""
    print("=" * 60)
    print("🚀 快速LoRA适配器文件修复工具")
    print("=" * 60)
    
    # 允许通过命令行参数指定路径
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = "/datanfs2/chenrongyi/models/loraadaptors/"
    
    if not os.path.exists(base_path):
        print(f"❌ 路径不存在: {base_path}")
        return
    
    try:
        fixed_count, total_count, complete_count = find_and_fix_lora_models(base_path)
        
        print("\n" + "=" * 60)
        print("📊 修复完成报告")
        print("=" * 60)
        print(f"总模型数: {total_count}")
        print(f"完整模型: {complete_count}")
        print(f"成功修复: {fixed_count}")
        print(f"需要修复: {total_count - complete_count}")
        
        if fixed_count > 0:
            print(f"\n🎉 成功获取并恢复了 {fixed_count} 个模型的文件!")
        
        if total_count == 0:
            print("\n⚠️  没有找到任何LoRA模型目录")
        elif fixed_count == 0 and complete_count < total_count:
            print(f"\n⚠️  有 {total_count - complete_count} 个模型无法自动修复，需要手动检查")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")

if __name__ == "__main__":
    main() 