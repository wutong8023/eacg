#!/usr/bin/env python3
"""
Query Dependency Filter
独立的查询依赖过滤工具，用于在query层面过滤生成的queries，
确保只保留属于target_dependency的查询，从源头杜绝错误的dependency。
"""

import json
import logging
import os
import re
from typing import Dict, List, Union, Set, Tuple, Optional
from utils.getDependencyUtils import dict_to_pkg_ver_tuples

# 设置日志
logger = logging.getLogger(__name__)

class QueryDependencyFilter:
    """
    查询依赖过滤器
    
    功能：
    1. 分析queries中的API路径
    2. 根据target_dependency过滤queries
    3. 支持多种包名识别策略
    4. 提供详细的过滤统计信息
    """
    
    def __init__(self, strict_mode: bool = True, enable_alias_matching: bool = True):
        """
        初始化查询依赖过滤器
        
        Args:
            strict_mode: 严格模式，只保留明确匹配target_dependency的queries
            enable_alias_matching: 启用别名匹配（如numpy的别名np）
        """
        self.strict_mode = strict_mode
        self.enable_alias_matching = enable_alias_matching
        
        # 常见的包别名映射
        self.package_aliases = {
            'np': 'numpy',
            'pd': 'pandas', 
            'plt': 'matplotlib',
            'mpl': 'matplotlib',
            'sk': 'sklearn',
            'sklearn': 'scikit-learn',
            'tf': 'tensorflow',
            'torch': 'torch',
            'cv2': 'opencv-python',
            'scipy.stats': 'scipy',
            'scipy.optimize': 'scipy',
            'scipy.linalg': 'scipy',
            'matplotlib.pyplot': 'matplotlib',
            'matplotlib.axes': 'matplotlib',
            'seaborn': 'sns'
        }
        
        logger.info(f"QueryDependencyFilter initialized: strict_mode={strict_mode}, alias_matching={enable_alias_matching}")
    
    def extract_package_from_path(self, api_path: str) -> Optional[str]:
        """
        从API路径中提取包名
        
        Args:
            api_path: API路径，如 "numpy.array.sum", "matplotlib.axes.Axes.hist"
            
        Returns:
            提取的包名，如 "numpy", "matplotlib"
        """
        if not api_path or not isinstance(api_path, str):
            return None
        
        # 清理路径
        api_path = api_path.strip()
        
        # 分割路径
        path_parts = api_path.split('.')
        if not path_parts:
            return None
        
        # 策略1: 直接使用第一部分
        first_part = path_parts[0].lower()
        
        # 策略2: 处理常见的子模块情况
        if len(path_parts) >= 2:
            # 对于 "scipy.stats", "matplotlib.pyplot" 等情况
            two_part = f"{path_parts[0]}.{path_parts[1]}".lower()
            if two_part in self.package_aliases:
                return self.package_aliases[two_part]
            
            # 对于 "matplotlib.axes.Axes" 等情况，可能需要取前两部分
            if path_parts[0].lower() in ['matplotlib', 'scipy']:
                return path_parts[0].lower()
        
        # 策略3: 别名映射
        if self.enable_alias_matching and first_part in self.package_aliases:
            return self.package_aliases[first_part]
        
        # 策略4: 直接返回第一部分
        return first_part
    
    def normalize_package_name(self, package_name: str) -> str:
        """
        标准化包名
        
        Args:
            package_name: 原始包名
            
        Returns:
            标准化后的包名
        """
        if not package_name:
            return package_name
            
        # 转换为小写
        normalized = package_name.lower().strip()
        
        # 处理常见的包名变体
        package_mappings = {
            'scikit-learn': 'sklearn',
            'opencv-python': 'cv2',
            'pillow': 'pil',
            'beautifulsoup4': 'bs4',
            'python-dateutil': 'dateutil',
            'pytorch': 'torch'  # pytorch 包实际上是torch
        }
        
        return package_mappings.get(normalized, normalized)
    
    def get_target_packages(self, target_dependencies: Dict[str, Union[str, List[str]]]) -> Set[str]:
        """
        从target_dependencies中提取所有包名
        
        Args:
            target_dependencies: 目标依赖字典
            
        Returns:
            标准化的包名集合
        """
        target_packages = set()
        
        # 使用dict_to_pkg_ver_tuples处理不同格式
        pkg_ver_tuples = dict_to_pkg_ver_tuples(target_dependencies)
        
        for pkg, ver in pkg_ver_tuples:
            if pkg and ver is not None:  # 过滤掉None版本
                normalized_pkg = self.normalize_package_name(pkg)
                target_packages.add(normalized_pkg)
        
        logger.debug(f"Target packages: {target_packages}")
        return target_packages
    
    def is_query_relevant(self, query: Dict, target_packages: Set[str]) -> Tuple[bool, str]:
        """
        判断单个query是否与target_dependencies相关
        
        Args:
            query: 查询字典，包含path、description等字段
            target_packages: 目标包名集合
            
        Returns:
            (是否相关, 匹配理由)
        """
        if not isinstance(query, dict):
            return False, "Invalid query format"
        
        # 策略1: 检查path字段
        api_path = query.get('path', '')
        if api_path:
            extracted_package = self.extract_package_from_path(api_path)
            if extracted_package:
                normalized_package = self.normalize_package_name(extracted_package)
                if normalized_package in target_packages:
                    return True, f"Path match: {api_path} -> {normalized_package}"
        
        # 策略2: 检查description字段中的包名（如果非严格模式）
        if not self.strict_mode:
            description = query.get('description', '')
            if description:
                for target_pkg in target_packages:
                    # 检查描述中是否包含包名
                    if target_pkg in description.lower():
                        return True, f"Description match: '{target_pkg}' in description"
        
        # 策略3: 检查其他可能的字段
        for field_name in ['module', 'package', 'library']:
            if field_name in query:
                field_value = query[field_name]
                if isinstance(field_value, str):
                    normalized_value = self.normalize_package_name(field_value)
                    if normalized_value in target_packages:
                        return True, f"Field {field_name} match: {field_value} -> {normalized_value}"
        
        return False, "No match found"
    
    def filter_queries(self, queries: List[Dict], target_dependencies: Dict[str, Union[str, List[str]]]) -> Tuple[List[Dict], Dict]:
        """
        过滤queries列表，只保留与target_dependencies相关的queries
        
        Args:
            queries: 查询列表,其中每个query应该要是一个字典，包含path、description等字段(虽然兼容str类型)
            target_dependencies: 目标依赖字典
            
        Returns:
            (过滤后的queries, 统计信息)
        """
        if not queries:
            return [], {"total": 0, "kept": 0, "filtered": 0, "filter_ratio": 0.0}
        
        target_packages = self.get_target_packages(target_dependencies)
        
        if not target_packages:
            logger.warning("No valid target packages found in target_dependencies")
            return queries, {"total": len(queries), "kept": len(queries), "filtered": 0, "filter_ratio": 0.0}
        
        filtered_queries = []
        filtering_details = []
        
        for i, query in enumerate(queries):
            is_relevant, reason = self.is_query_relevant(query, target_packages)
            
            if is_relevant:
                filtered_queries.append(query)
                filtering_details.append({
                    "index": i,
                    "action": "kept",
                    "reason": reason,
                    "path": query.get('path', 'N/A') if isinstance(query, dict) else str(query)
                })
            else:
                filtering_details.append({
                    "index": i,
                    "action": "filtered",
                    "reason": reason,
                    "path": query.get('path', 'N/A') if isinstance(query, dict) else str(query)
                })
        
        # 统计信息
        total_count = len(queries)
        kept_count = len(filtered_queries)
        filtered_count = total_count - kept_count
        filter_ratio = filtered_count / total_count if total_count > 0 else 0.0
        
        stats = {
            "total": total_count,
            "kept": kept_count,
            "filtered": filtered_count,
            "filter_ratio": filter_ratio,
            "target_packages": list(target_packages),
            "details": filtering_details
        }
        
        logger.info(f"Query filtering completed: {kept_count}/{total_count} queries kept ({(1-filter_ratio)*100:.1f}%)")
        
        return filtered_queries, stats
    
    def filter_queries_from_file(self, queries_file: str, target_dependencies: Dict[str, Union[str, List[str]]], 
                                 output_file: Optional[str] = None, sample_ids: Optional[List] = None) -> Tuple[Dict, Dict]:
        """
        从文件中加载queries并进行过滤
        
        Args:
            queries_file: 生成的queries文件路径（JSON格式）
            target_dependencies: 目标依赖字典
            output_file: 输出文件路径（可选）
            sample_ids: 要处理的样本ID列表（可选，None表示处理所有）
            
        Returns:
            (过滤后的queries字典, 统计信息)
        """
        logger.info(f"Loading queries from {queries_file}")
        
        # 加载queries文件
        try:
            with open(queries_file, 'r', encoding='utf-8') as f:
                all_queries_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load queries file {queries_file}: {e}")
            raise
        
        if not isinstance(all_queries_data, dict):
            raise ValueError(f"Expected dict format in queries file, got {type(all_queries_data)}")
        
        # 过滤样本（如果指定了sample_ids）
        if sample_ids is not None:
            sample_ids_str = {str(sid) for sid in sample_ids}
            queries_data = {k: v for k, v in all_queries_data.items() if k in sample_ids_str}
            logger.info(f"Filtered to {len(queries_data)} samples based on sample_ids")
        else:
            queries_data = all_queries_data
        
        filtered_queries_data = {}
        total_stats = {
            "total_samples": len(queries_data),
            "processed_samples": 0,
            "total_queries": 0,
            "total_kept": 0,
            "total_filtered": 0,
            "sample_stats": {}
        }
        
        # 处理每个样本的queries
        for sample_id, sample_data in queries_data.items():
            if not isinstance(sample_data, dict) or 'queries' not in sample_data:
                logger.warning(f"Skipping sample {sample_id}: invalid format")
                continue
            
            queries = sample_data['queries']
            if not isinstance(queries, list):
                logger.warning(f"Skipping sample {sample_id}: queries is not a list")
                continue
            
            # 过滤queries
            filtered_queries, stats = self.filter_queries(queries, target_dependencies)
            
            # 保存过滤后的数据
            filtered_sample_data = sample_data.copy()
            filtered_sample_data['queries'] = filtered_queries
            
            # 添加过滤统计信息
            filtered_sample_data['filtering_stats'] = stats
            
            filtered_queries_data[sample_id] = filtered_sample_data
            
            # 更新总统计
            total_stats["processed_samples"] += 1
            total_stats["total_queries"] += stats["total"]
            total_stats["total_kept"] += stats["kept"]
            total_stats["total_filtered"] += stats["filtered"]
            total_stats["sample_stats"][sample_id] = stats
            
            logger.debug(f"Sample {sample_id}: {stats['kept']}/{stats['total']} queries kept")
        
        # 计算总体过滤比例
        if total_stats["total_queries"] > 0:
            total_stats["overall_filter_ratio"] = total_stats["total_filtered"] / total_stats["total_queries"]
        else:
            total_stats["overall_filter_ratio"] = 0.0
        
        logger.info(f"Overall filtering stats: {total_stats['total_kept']}/{total_stats['total_queries']} "
                   f"queries kept ({(1-total_stats['overall_filter_ratio'])*100:.1f}%)")
        
        # 保存到输出文件
        if output_file:
            try:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(filtered_queries_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Filtered queries saved to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save filtered queries to {output_file}: {e}")
                raise
        
        return filtered_queries_data, total_stats


def filter_queries_cli():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter generated queries based on target dependencies")
    parser.add_argument("--queries_file", type=str, required=True,
                       help="Path to the generated queries JSON file")
    parser.add_argument("--target_dependencies", type=str, required=True,
                       help="Target dependencies as JSON string or file path")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file path for filtered queries")
    parser.add_argument("--sample_ids_file", type=str, default=None,
                       help="File containing sample IDs to process (JSON format)")
    parser.add_argument("--strict_mode", action="store_true", default=True,
                       help="Use strict mode (only path-based matching)")
    parser.add_argument("--disable_alias_matching", action="store_true",
                       help="Disable package alias matching")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.basicConfig(level=getattr(logging, args.log_level), 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 解析target_dependencies
    try:
        if os.path.isfile(args.target_dependencies):
            with open(args.target_dependencies, 'r') as f:
                target_dependencies = json.load(f)
        else:
            target_dependencies = json.loads(args.target_dependencies)
    except Exception as e:
        logger.error(f"Failed to parse target_dependencies: {e}")
        return
    
    # 加载sample_ids（如果指定）
    sample_ids = None
    if args.sample_ids_file:
        try:
            with open(args.sample_ids_file, 'r') as f:
                sample_ids = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load sample_ids from {args.sample_ids_file}: {e}")
            return
    
    # 创建过滤器
    filter_obj = QueryDependencyFilter(
        strict_mode=args.strict_mode,
        enable_alias_matching=not args.disable_alias_matching
    )
    
    # 执行过滤
    try:
        filtered_queries, stats = filter_obj.filter_queries_from_file(
            args.queries_file, target_dependencies, args.output_file, sample_ids
        )
        
        # 输出统计信息
        print(f"\nFiltering completed successfully!")
        print(f"Processed samples: {stats['processed_samples']}")
        print(f"Total queries: {stats['total_queries']}")
        print(f"Kept queries: {stats['total_kept']}")
        print(f"Filtered queries: {stats['total_filtered']}")
        print(f"Keep ratio: {(1-stats['overall_filter_ratio'])*100:.1f}%")
        
        if args.output_file:
            print(f"Filtered queries saved to: {args.output_file}")
        
    except Exception as e:
        logger.error(f"Filtering failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    filter_queries_cli() 