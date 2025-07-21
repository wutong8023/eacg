#!/usr/bin/env python3
"""
生成包版本文件的辅助脚本
从benchmark JSON文件中提取包版本信息，生成适用于批量训练脚本的包版本文件
"""

import json
import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from utils.getDatasetPacks import getPackVersions


def load_benchmark_data(benchmark_path):
    """加载benchmark数据"""
    try:
        with open(benchmark_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到benchmark文件: {benchmark_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"错误: JSON解析失败: {e}")
        return None


def generate_package_versions_file(benchmark_paths, output_file, filter_existing=False, corpus_path=None):
    """
    生成包版本文件
    
    Args:
        benchmark_paths: benchmark文件路径列表
        output_file: 输出文件路径
        filter_existing: 是否过滤掉不存在数据文件的包版本
        corpus_path: 语料库路径（用于检查数据文件是否存在）
    """
    all_pack_versions = {}
    
    # 处理多个benchmark文件
    for benchmark_path in benchmark_paths:
        print(f"处理benchmark文件: {benchmark_path}")
        
        benchmark_data = load_benchmark_data(benchmark_path)
        if benchmark_data is None:
            continue
            
        # 提取包版本信息
        pack_versions = getPackVersions(benchmark_data)
        
        # 合并到总的包版本字典中
        for pkg, versions in pack_versions.items():
            if pkg not in all_pack_versions:
                all_pack_versions[pkg] = set()
            all_pack_versions[pkg].update(versions)
        
        print(f"  - 提取到 {len(pack_versions)} 个包")
    
    # 转换为列表并排序
    pkg_version_list = []
    for pkg, versions in all_pack_versions.items():
        for version in sorted(versions):
            pkg_version_list.append(f"{pkg}:{version}")
    
    pkg_version_list.sort()
    
    print(f"总共提取到 {len(pkg_version_list)} 个包版本组合")
    
    # 如果需要过滤，检查数据文件是否存在
    if filter_existing and corpus_path:
        print("检查数据文件存在性...")
        existing_pkg_versions = []
        missing_count = 0
        
        for pkg_version in pkg_version_list:
            pkg, version = pkg_version.split(':', 1)
            data_file = os.path.join(corpus_path, pkg, f"{version}.jsonl")
            
            if os.path.exists(data_file):
                existing_pkg_versions.append(pkg_version)
            else:
                missing_count += 1
                print(f"  - 数据文件不存在: {data_file}")
        
        print(f"过滤结果: {len(existing_pkg_versions)} 个存在, {missing_count} 个不存在")
        pkg_version_list = existing_pkg_versions
    
    # 写入输出文件
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入文件头
            f.write(f"# 包版本文件 - 生成时间: {__import__('datetime').datetime.now()}\n")
            f.write(f"# 格式: pkg:version\n")
            f.write(f"# 数据来源: {', '.join(benchmark_paths)}\n")
            if filter_existing:
                f.write(f"# 已过滤不存在的数据文件 (语料库路径: {corpus_path})\n")
            f.write(f"# 总数量: {len(pkg_version_list)}\n")
            f.write("\n")
            
            # 写入包版本列表
            for pkg_version in pkg_version_list:
                f.write(f"{pkg_version}\n")
        
        print(f"✅ 包版本文件已生成: {output_file}")
        print(f"   包含 {len(pkg_version_list)} 个包版本")
        
    except Exception as e:
        print(f"❌ 写入文件失败: {e}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="从benchmark文件生成包版本文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从单个benchmark文件生成
  python utils/generate_package_versions.py \\
    --benchmark data/VersiBCB_Benchmark/vace_datas.json \\
    --output packages.txt
  
  # 从多个benchmark文件生成
  python utils/generate_package_versions.py \\
    --benchmark data/benchmark1.json data/benchmark2.json \\
    --output packages.txt
  
  # 生成时过滤不存在的数据文件
  python utils/generate_package_versions.py \\
    --benchmark data/VersiBCB_Benchmark/vace_datas.json \\
    --output packages.txt \\
    --filter-existing \\
    --corpus-path /datanfs4/chenrongyi/data/docs
        """
    )
    
    parser.add_argument(
        "--benchmark", 
        nargs='+', 
        required=True,
        help="benchmark JSON文件路径（可以指定多个）"
    )
    
    parser.add_argument(
        "--output", 
        required=True,
        help="输出的包版本文件路径"
    )
    
    parser.add_argument(
        "--filter-existing",
        action="store_true",
        help="过滤掉不存在数据文件的包版本"
    )
    
    parser.add_argument(
        "--corpus-path",
        default="/datanfs4/chenrongyi/data/docs",
        help="语料库路径（用于检查数据文件是否存在）"
    )
    
    args = parser.parse_args()
    
    # 验证benchmark文件存在
    for benchmark_path in args.benchmark:
        if not os.path.exists(benchmark_path):
            print(f"错误: benchmark文件不存在: {benchmark_path}")
            sys.exit(1)
    
    # 生成包版本文件
    success = generate_package_versions_file(
        args.benchmark,
        args.output,
        args.filter_existing,
        args.corpus_path
    )
    
    if success:
        print("\n✅ 包版本文件生成完成！")
        print(f"可以使用以下命令进行批量训练:")
        print(f"./scripts/train_lora_batch.sh --pkg-version-file {args.output}")
    else:
        print("\n❌ 包版本文件生成失败")
        sys.exit(1)


if __name__ == "__main__":
    main() 