#!/usr/bin/env python3
"""
未训练包检测工具
用于检测哪些包还没有对应的LoRA模型，帮助规划训练任务
"""

import os
import json
import argparse
import logging
from datetime import datetime
from collections import defaultdict

# 导入必要的模块
from utils.getDatasetPacks import getPackVersions
from utils.loraTrain.loraTrainUtils import loraModelExists
from benchmark.config.code.config_lora import LORA_CONFIG_PATH, load_config
from utils.loraPathConfigure import pathConfigurator

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def load_package_versions(benchmark_data_path):
    """
    从benchmark数据中加载包版本信息
    
    Args:
        benchmark_data_path: benchmark数据文件路径
        
    Returns:
        dict: 包版本信息 {pkg: [version1, version2, ...]}
    """
    try:
        with open(benchmark_data_path, "r") as f:
            datas = json.load(f)
        
        pack_versions = getPackVersions(datas)
        logging.info(f"从 {benchmark_data_path} 加载了 {len(pack_versions)} 个包")
        
        total_versions = sum(len(versions) for versions in pack_versions.values())
        logging.info(f"总计 {total_versions} 个包版本组合")
        
        return pack_versions
        
    except Exception as e:
        logging.error(f"加载包版本信息失败: {e}")
        return {}

def check_package_models(pack_versions, model_name, config, knowledge_types=['docstring', 'srccodes']):
    """
    检查包模型的存在情况
    
    Args:
        pack_versions: 包版本信息
        model_name: 模型名称
        config: 配置信息
        knowledge_types: 要检查的知识类型列表
        
    Returns:
        dict: 检查结果
    """
    results = {}
    
    for knowledge_type in knowledge_types:
        logging.info(f"\n=== 检查 {knowledge_type} 模型 ===")
        
        existing_packages = []
        missing_packages = []
        error_packages = []
        
        for pkg, versions in pack_versions.items():
            for version in versions:
                try:
                    if loraModelExists(pkg, version,model_name,config,knowledge_type): # , model_name, config, knowledge_type
                        existing_packages.append((pkg, version))
                        logging.debug(f"✓ {pkg}-{version} ({knowledge_type}) - 模型存在")
                    else:
                        missing_packages.append((pkg, version))
                        logging.debug(f"✗ {pkg}-{version} ({knowledge_type}) - 模型缺失")
                        
                except Exception as e:
                    error_packages.append((pkg, version, str(e)))
                    logging.warning(f"⚠ {pkg}-{version} ({knowledge_type}) - 检查出错: {e}")
        
        results[knowledge_type] = {
            'existing': existing_packages,
            'missing': missing_packages,
            'errors': error_packages,
            'total': len(existing_packages) + len(missing_packages) + len(error_packages)
        }
        
        logging.info(f"{knowledge_type} 统计:")
        logging.info(f"  已存在: {len(existing_packages)}")
        logging.info(f"  缺失: {len(missing_packages)}")
        logging.info(f"  错误: {len(error_packages)}")
        logging.info(f"  总计: {results[knowledge_type]['total']}")
    
    return results

def generate_training_plan(results, knowledge_types):
    """
    生成训练计划
    
    Args:
        results: 检查结果
        knowledge_types: 知识类型列表
        
    Returns:
        dict: 训练计划
    """
    plan = {}
    
    for knowledge_type in knowledge_types:
        if knowledge_type not in results:
            continue
            
        missing = results[knowledge_type]['missing']
        
        # 按包分组
        packages_to_train = defaultdict(list)
        for pkg, version in missing:
            packages_to_train[pkg].append(version)
        
        plan[knowledge_type] = {
            'packages_count': len(packages_to_train),
            'versions_count': len(missing),
            'packages': dict(packages_to_train)
        }
    
    return plan

def save_results_to_file(results, plan, output_file):
    """
    将结果保存到文件
    
    Args:
        results: 检查结果
        plan: 训练计划
        output_file: 输出文件路径
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {},
        'detailed_results': results,
        'training_plan': plan
    }
    
    # 生成摘要
    for knowledge_type, result in results.items():
        report['summary'][knowledge_type] = {
            'total_packages': result['total'],
            'existing_packages': len(result['existing']),
            'missing_packages': len(result['missing']),
            'error_packages': len(result['errors']),
            'completion_rate': f"{len(result['existing']) / result['total'] * 100:.1f}%" if result['total'] > 0 else "0.0%"
        }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logging.info(f"结果已保存到: {output_file}")
    except Exception as e:
        logging.error(f"保存结果文件失败: {e}")

def print_summary(results, plan):
    """
    打印摘要信息
    """
    print("\n" + "=" * 60)
    print("未训练包检测结果摘要")
    print("=" * 60)
    
    for knowledge_type, result in results.items():
        total = result['total']
        existing = len(result['existing'])
        missing = len(result['missing'])
        errors = len(result['errors'])
        
        completion_rate = (existing / total * 100) if total > 0 else 0
        
        print(f"\n📊 {knowledge_type.upper()} 模型统计:")
        print(f"  总包数: {total}")
        print(f"  已训练: {existing} ({completion_rate:.1f}%)")
        print(f"  未训练: {missing}")
        print(f"  检查错误: {errors}")
        
        if missing > 0:
            print(f"\n🎯 {knowledge_type.upper()} 训练计划:")
            pkg_plan = plan.get(knowledge_type, {})
            print(f"  需要训练的包: {pkg_plan.get('packages_count', 0)} 个")
            print(f"  需要训练的版本: {pkg_plan.get('versions_count', 0)} 个")
            
            if args.show_missing and 'packages' in pkg_plan:
                print(f"\n📝 缺失的包列表 ({knowledge_type}):")
                for pkg, versions in list(pkg_plan['packages'].items())[:10]:  # 只显示前10个
                    versions_str = ', '.join(versions[:3])  # 只显示前3个版本
                    if len(versions) > 3:
                        versions_str += f" ... (共{len(versions)}个版本)"
                    print(f"    {pkg}: {versions_str}")
                
                if len(pkg_plan['packages']) > 10:
                    print(f"    ... 还有 {len(pkg_plan['packages']) - 10} 个包 (使用 --show-missing 查看完整列表)")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="检测未训练的LoRA模型包")
    parser.add_argument("--benchmark_data_path", type=str, 
                       default="benchmark/data/VersiBCB_Benchmark/vace_datas.json",
                       help="benchmark数据文件路径")
    parser.add_argument("--model_name", type=str,
                       default="/datanfs2/chenrongyi/models/Llama-3.1-8B",
                       help="模型名称")
    parser.add_argument("--knowledge_types", nargs='+', 
                       default=['docstring', 'srccodes'],
                       choices=['docstring', 'srccodes'],
                       help="要检查的知识类型")
    parser.add_argument("--output_file", type=str,
                       default=None,
                       help="输出结果文件路径 (JSON格式)")
    parser.add_argument("--show_missing", action="store_true",
                       help="显示所有缺失的包列表")
    parser.add_argument("--verbose", action="store_true",
                       help="显示详细信息")
    
    global args
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup_logging()
    
    logging.info("开始检测未训练的LoRA模型包...")
    logging.info(f"Benchmark数据: {args.benchmark_data_path}")
    logging.info(f"模型名称: {args.model_name}")
    logging.info(f"知识类型: {args.knowledge_types}")
    
    # 加载配置
    try:
        config = load_config(LORA_CONFIG_PATH)
        config["model_name"] = args.model_name
        logging.info("配置加载成功")
    except Exception as e:
        logging.error(f"配置加载失败: {e}")
        return
    
    # 获取模型名称（用于路径检查）
    model_name = args.model_name.split("/")[-1]
    
    # 加载包版本信息
    pack_versions = load_package_versions(args.benchmark_data_path)
    if not pack_versions:
        logging.error("无法获取包版本信息，退出")
        return
    
    # 检查模型存在情况
    results = check_package_models(pack_versions, model_name, config, args.knowledge_types)
    
    # 生成训练计划
    plan = generate_training_plan(results, args.knowledge_types)
    
    # 打印摘要
    print_summary(results, plan)
    
    # 保存结果到文件
    if args.output_file:
        save_results_to_file(results, plan, args.output_file)
    else:
        # 自动生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"untrained_packages_report_{timestamp}.json"
        save_results_to_file(results, plan, output_file)
    
    print(f"\n✅ 检测完成！")

if __name__ == "__main__":
    main()