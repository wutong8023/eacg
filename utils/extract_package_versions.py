#!/usr/bin/env python3
"""
辅助脚本：从benchmark文件中提取包版本信息
用于生成train_lora_ift.py脚本所需的package_versions.json文件
"""

import json
import argparse
import os
from collections import defaultdict
from utils.getDatasetPacks import getPackVersions


def extract_package_versions_from_benchmark(benchmark_paths, output_file):
    """
    从多个benchmark文件中提取包版本信息
    
    Args:
        benchmark_paths: list, benchmark文件路径列表
        output_file: str, 输出文件路径
    """
    aggregated_pack_versions = defaultdict(list)
    
    print(f"正在处理 {len(benchmark_paths)} 个benchmark文件...")
    
    for benchmark_path in benchmark_paths:
        print(f"正在处理: {benchmark_path}")
        
        if not os.path.exists(benchmark_path):
            print(f"警告: 文件不存在 {benchmark_path}")
            continue
            
        try:
            with open(benchmark_path, "r", encoding="utf-8") as f:
                datas = json.load(f)
            
            # 使用现有的getPackVersions函数
            pack_versions = getPackVersions(datas)
            print(f"从 {benchmark_path} 提取了 {len(pack_versions)} 个包")
            
            # 汇总包版本信息
            for pkg, versions in pack_versions.items():
                aggregated_pack_versions[pkg].extend(versions)
                
        except Exception as e:
            print(f"处理文件时出错 {benchmark_path}: {e}")
            continue
    
    # 去重并排序
    final_pack_versions = {}
    for pkg, versions in aggregated_pack_versions.items():
        # 去重并排序
        unique_versions = sorted(list(set(versions)))
        final_pack_versions[pkg] = unique_versions
    
    # 保存结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_pack_versions, f, indent=2, ensure_ascii=False)
    
    # 打印统计信息
    total_packages = len(final_pack_versions)
    total_combinations = sum(len(versions) for versions in final_pack_versions.values())
    
    print(f"\n=== 提取完成 ===")
    print(f"总计包数: {total_packages}")
    print(f"总计包版本组合: {total_combinations}")
    print(f"结果保存到: {output_file}")
    
    # 打印详细信息
    print(f"\n=== 包版本详情 ===")
    for pkg, versions in sorted(final_pack_versions.items()):
        print(f"{pkg}: {len(versions)} 个版本 -> {versions}")
    
    return final_pack_versions


def extract_from_single_benchmark(benchmark_path, output_file):
    """
    从单个benchmark文件中提取包版本信息（向后兼容）
    """
    return extract_package_versions_from_benchmark([benchmark_path], output_file)


def main():
    parser = argparse.ArgumentParser(description="从benchmark文件中提取包版本信息")
    
    # 支持两种模式：单文件和多文件
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--benchmark_file", type=str,
                      help="单个benchmark文件路径（向后兼容）")
    group.add_argument("--benchmark_files", type=str, nargs='+',
                      help="多个benchmark文件路径")
    
    parser.add_argument("--output", type=str, required=True,
                       help="输出的package_versions.json文件路径")
    
    args = parser.parse_args()
    
    # 确定输入文件列表
    if args.benchmark_file:
        benchmark_paths = [args.benchmark_file]
        print(f"使用单文件模式: {args.benchmark_file}")
    else:
        benchmark_paths = args.benchmark_files
        print(f"使用多文件模式: {len(benchmark_paths)} 个文件")
    
    # 提取包版本信息
    try:
        result = extract_package_versions_from_benchmark(benchmark_paths, args.output)
        print(f"\n✅ 成功提取包版本信息到: {args.output}")
        
    except Exception as e:
        print(f"❌ 提取过程中发生错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


# =============================================================================
# 使用示例 (Usage Examples)
# =============================================================================
#
# 1. 从单个benchmark文件提取:
# python benchmark/extract_package_versions.py \
#     --benchmark_file benchmark/data/VersiBCB_Benchmark/vace_datas.json \
#     --output package_versions.json
#
# 2. 从多个benchmark文件提取:
# python benchmark/extract_package_versions.py \
#     --benchmark_files \
#         benchmark/data/VersiBCB_Benchmark/vace_datas.json \
#         benchmark/data/VersiBCB_Benchmark/vscc_datas.json \
#         benchmark/data/VersiBCB_Benchmark/vace_datas_for_warning.json \
#     --output package_versions_combined.json
#
# 3. 然后使用生成的文件进行IFT训练:
# python benchmark/train_lora_ift.py \
#     --package_versions_file package_versions_combined.json \
#     --output_base_dir /datanfs2/chenrongyi/models/ift_models \
#     --use_benchmark_data \
#     --knowledge_type docstring
# ============================================================================= 