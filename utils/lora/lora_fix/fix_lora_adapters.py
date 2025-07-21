#!/usr/bin/env python3
"""
LoRA适配器模型文件修复工具
用于检查和修复缺失的adapter_model.safetensors和配置文件
"""

import os
import shutil
import glob
import argparse
import logging
import re
from datetime import datetime
from pathlib import Path

def setup_logging(verbose=False):
    """设置日志配置"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def find_lora_model_directories(base_path):
    """
    查找所有LoRA模型目录
    
    Args:
        base_path: 基础路径，如 /datanfs2/chenrongyi/models/loraadaptors/
        
    Returns:
        list: LoRA模型目录列表
    """
    lora_dirs = []
    
    # 遍历所有模型目录
    if not os.path.exists(base_path):
        logging.error(f"基础路径不存在: {base_path}")
        return lora_dirs
    
    # 查找模式: base_path/model_name/package_name/package_version_knowledge_type_...
    for model_dir in os.listdir(base_path):
        model_path = os.path.join(base_path, model_dir)
        if not os.path.isdir(model_path):
            continue
            
        for pkg_dir in os.listdir(model_path):
            pkg_path = os.path.join(model_path, pkg_dir)
            if not os.path.isdir(pkg_path):
                continue
                
            for version_dir in os.listdir(pkg_path):
                version_path = os.path.join(pkg_path, version_dir)
                if os.path.isdir(version_path):
                    # 检查是否包含checkpoints目录，确认是LoRA模型目录
                    checkpoints_path = os.path.join(version_path, "checkpoints")
                    if os.path.exists(checkpoints_path):
                        lora_dirs.append(version_path)
                        logging.debug(f"找到LoRA模型目录: {version_path}")
    
    logging.info(f"总共找到 {len(lora_dirs)} 个LoRA模型目录")
    return lora_dirs

def check_required_files(model_dir):
    """
    检查主目录是否包含必需的文件
    
    Args:
        model_dir: 模型目录路径
        
    Returns:
        dict: 缺失文件的状态
    """
    required_files = {
        'adapter_model.safetensors': False,
        'adapter_config.json': False,
    }
    
    # 可选文件（如果存在会一起复制）
    optional_files = [
        'README.md',
        'training_args.bin',
        'trainer_state.json',
        'pytorch_model.bin'  # 可能有些模型使用这个格式
    ]
    
    for filename in required_files.keys():
        file_path = os.path.join(model_dir, filename)
        if os.path.exists(file_path):
            required_files[filename] = True
            logging.debug(f"找到文件: {file_path}")
    
    missing_count = sum(1 for exists in required_files.values() if not exists)
    
    return {
        'required_files': required_files,
        'missing_count': missing_count,
        'all_present': missing_count == 0
    }

def find_best_checkpoint(checkpoints_dir):
    """
    找到最佳的checkpoint目录（优先epoch4，然后是最高epoch）
    
    Args:
        checkpoints_dir: checkpoints目录路径
        
    Returns:
        str or None: 最佳checkpoint目录路径
    """
    if not os.path.exists(checkpoints_dir):
        return None
    
    checkpoint_dirs = []
    for item in os.listdir(checkpoints_dir):
        item_path = os.path.join(checkpoints_dir, item)
        if os.path.isdir(item_path):
            checkpoint_dirs.append(item)
    
    if not checkpoint_dirs:
        return None
    
    # 优先查找epoch4
    epoch4_pattern = re.compile(r'.*epoch4$')
    for checkpoint_dir in checkpoint_dirs:
        if epoch4_pattern.match(checkpoint_dir):
            best_checkpoint = os.path.join(checkpoints_dir, checkpoint_dir)
            logging.debug(f"找到epoch4 checkpoint: {best_checkpoint}")
            return best_checkpoint
    
    # 如果没有epoch4，找最高的epoch
    epoch_pattern = re.compile(r'.*epoch(\d+)$')
    max_epoch = -1
    best_checkpoint_dir = None
    
    for checkpoint_dir in checkpoint_dirs:
        match = epoch_pattern.match(checkpoint_dir)
        if match:
            epoch_num = int(match.group(1))
            if epoch_num > max_epoch:
                max_epoch = epoch_num
                best_checkpoint_dir = checkpoint_dir
    
    if best_checkpoint_dir:
        best_checkpoint = os.path.join(checkpoints_dir, best_checkpoint_dir)
        logging.debug(f"找到最高epoch checkpoint: {best_checkpoint} (epoch{max_epoch})")
        return best_checkpoint
    
    # 如果没有符合模式的，返回第一个
    fallback_checkpoint = os.path.join(checkpoints_dir, checkpoint_dirs[0])
    logging.debug(f"使用备用checkpoint: {fallback_checkpoint}")
    return fallback_checkpoint

def copy_files_from_checkpoint(checkpoint_dir, target_dir, required_files):
    """
    从checkpoint目录复制文件到目标目录
    
    Args:
        checkpoint_dir: checkpoint目录路径
        target_dir: 目标目录路径
        required_files: 需要复制的文件状态字典
        
    Returns:
        dict: 复制结果
    """
    copied_files = []
    failed_files = []
    
    # 首先复制必需文件
    for filename, exists in required_files.items():
        if not exists:  # 只复制缺失的文件
            src_file = os.path.join(checkpoint_dir, filename)
            dst_file = os.path.join(target_dir, filename)
            
            if os.path.exists(src_file):
                try:
                    shutil.copy2(src_file, dst_file)
                    copied_files.append(filename)
                    logging.debug(f"复制文件: {src_file} -> {dst_file}")
                except Exception as e:
                    failed_files.append((filename, str(e)))
                    logging.error(f"复制文件失败: {filename}, 错误: {e}")
            else:
                failed_files.append((filename, "源文件不存在"))
                logging.warning(f"checkpoint中缺少文件: {src_file}")
    
    # 复制可选文件
    optional_files = ['README.md', 'training_args.bin', 'trainer_state.json', 'pytorch_model.bin']
    for filename in optional_files:
        src_file = os.path.join(checkpoint_dir, filename)
        dst_file = os.path.join(target_dir, filename)
        
        if os.path.exists(src_file) and not os.path.exists(dst_file):
            try:
                shutil.copy2(src_file, dst_file)
                copied_files.append(filename)
                logging.debug(f"复制可选文件: {src_file} -> {dst_file}")
            except Exception as e:
                logging.warning(f"复制可选文件失败: {filename}, 错误: {e}")
    
    return {
        'copied_files': copied_files,
        'failed_files': failed_files,
        'success': len(failed_files) == 0
    }

def fix_single_model(model_dir, dry_run=False):
    """
    修复单个模型目录
    
    Args:
        model_dir: 模型目录路径
        dry_run: 是否为干运行模式
        
    Returns:
        dict: 修复结果
    """
    model_name = os.path.basename(model_dir)
    logging.info(f"检查模型: {model_name}")
    
    # 检查必需文件
    file_status = check_required_files(model_dir)
    
    if file_status['all_present']:
        logging.info(f"✓ 模型 {model_name} 文件完整，无需修复")
        return {
            'model_dir': model_dir,
            'status': 'complete',
            'message': '文件完整',
            'fixed': False
        }
    
    logging.info(f"⚠ 模型 {model_name} 缺少 {file_status['missing_count']} 个必需文件")
    
    # 查找最佳checkpoint
    checkpoints_dir = os.path.join(model_dir, "checkpoints")
    best_checkpoint = find_best_checkpoint(checkpoints_dir)
    
    if not best_checkpoint:
        logging.error(f"✗ 模型 {model_name} 没有可用的checkpoint")
        return {
            'model_dir': model_dir,
            'status': 'error',
            'message': '没有可用的checkpoint',
            'fixed': False
        }
    
    checkpoint_name = os.path.basename(best_checkpoint)
    logging.info(f"使用checkpoint: {checkpoint_name}")
    
    if dry_run:
        logging.info(f"[DRY RUN] 将从 {checkpoint_name} 恢复文件到 {model_name}")
        return {
            'model_dir': model_dir,
            'status': 'dry_run',
            'message': f'将从 {checkpoint_name} 恢复文件',
            'fixed': False
        }
    
    # 执行文件复制
    copy_result = copy_files_from_checkpoint(
        best_checkpoint, 
        model_dir, 
        file_status['required_files']
    )
    
    if copy_result['success']:
        logging.info(f"✓ 模型 {model_name} 修复成功，复制了 {len(copy_result['copied_files'])} 个文件")
        return {
            'model_dir': model_dir,
            'status': 'fixed',
            'message': f'从 {checkpoint_name} 恢复了 {len(copy_result["copied_files"])} 个文件',
            'copied_files': copy_result['copied_files'],
            'fixed': True
        }
    else:
        logging.error(f"✗ 模型 {model_name} 修复失败")
        return {
            'model_dir': model_dir,
            'status': 'failed',
            'message': f'复制失败: {copy_result["failed_files"]}',
            'fixed': False
        }

def generate_report(results, output_file=None):
    """
    生成修复报告
    
    Args:
        results: 修复结果列表
        output_file: 输出文件路径
    """
    total_models = len(results)
    complete_models = sum(1 for r in results if r['status'] == 'complete')
    fixed_models = sum(1 for r in results if r['status'] == 'fixed')
    failed_models = sum(1 for r in results if r['status'] == 'failed')
    error_models = sum(1 for r in results if r['status'] == 'error')
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_models': total_models,
            'complete_models': complete_models,
            'fixed_models': fixed_models,
            'failed_models': failed_models,
            'error_models': error_models,
            'success_rate': f"{(complete_models + fixed_models) / total_models * 100:.1f}%" if total_models > 0 else "0.0%"
        },
        'detailed_results': results
    }
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("LoRA适配器模型修复报告")
    print("=" * 60)
    print(f"总模型数: {total_models}")
    print(f"完整模型: {complete_models}")
    print(f"成功修复: {fixed_models}")
    print(f"修复失败: {failed_models}")
    print(f"错误模型: {error_models}")
    print(f"成功率: {report['summary']['success_rate']}")
    
    if fixed_models > 0:
        print(f"\n🎉 成功获取并恢复了 {fixed_models} 个模型的文件！")
    
    if failed_models > 0 or error_models > 0:
        print(f"\n⚠️ 有 {failed_models + error_models} 个模型需要手动处理")
        print("\n失败的模型:")
        for result in results:
            if result['status'] in ['failed', 'error']:
                print(f"  - {os.path.basename(result['model_dir'])}: {result['message']}")
    
    # 保存详细报告
    if output_file:
        try:
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n详细报告已保存到: {output_file}")
        except Exception as e:
            logging.error(f"保存报告失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="修复LoRA适配器模型文件")
    parser.add_argument("--base_path", type=str,
                       default="/datanfs2/chenrongyi/models/loraadaptors/",
                       help="LoRA模型基础路径")
    parser.add_argument("--model_name", type=str,
                       default="Llama-3.1-8B",
                       help="指定模型名称，只处理该模型下的适配器")
    parser.add_argument("--package_filter", type=str,
                       default=None,
                       help="包名过滤器（可选），只处理匹配的包")
    parser.add_argument("--dry_run", action="store_true",
                       help="干运行模式，只检查不执行修复")
    parser.add_argument("--output_report", type=str,
                       default=None,
                       help="输出报告文件路径")
    parser.add_argument("--verbose", action="store_true",
                       help="显示详细日志")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    if args.dry_run:
        logging.info("运行在干运行模式，将只检查不执行修复")
    
    # 构建搜索路径
    if args.model_name:
        search_path = os.path.join(args.base_path, args.model_name)
    else:
        search_path = args.base_path
    
    logging.info(f"开始扫描LoRA模型目录: {search_path}")
    
    # 查找所有模型目录
    model_dirs = find_lora_model_directories(args.base_path)
    
    # 过滤指定模型
    if args.model_name:
        model_dirs = [d for d in model_dirs if args.model_name in d]
    
    # 过滤指定包
    if args.package_filter:
        model_dirs = [d for d in model_dirs if args.package_filter in d]
    
    if not model_dirs:
        logging.error("没有找到符合条件的LoRA模型目录")
        return
    
    logging.info(f"将处理 {len(model_dirs)} 个模型目录")
    
    # 处理每个模型
    results = []
    for i, model_dir in enumerate(model_dirs, 1):
        logging.info(f"\n[{i}/{len(model_dirs)}] 处理: {os.path.basename(model_dir)}")
        
        try:
            result = fix_single_model(model_dir, args.dry_run)
            results.append(result)
        except Exception as e:
            logging.error(f"处理模型时发生错误: {e}")
            results.append({
                'model_dir': model_dir,
                'status': 'error',
                'message': f'处理异常: {str(e)}',
                'fixed': False
            })
    
    # 生成报告
    if args.output_report:
        report_file = args.output_report
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"lora_fix_report_{timestamp}.json"
    
    generate_report(results, report_file)

if __name__ == "__main__":
    main() 