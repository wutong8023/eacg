import json
import logging

def version_similarity(v1, v2):
    """
    计算两个版本的相似性，返回负的欧几里得距离用于排序
    
    Args:
        v1: str, 版本1
        v2: str, 版本2
        
    Returns:
        float: 相似性分数（越大越相似）
    """
    try:
        v1_parts = [int(x) for x in v1.split('.')]
        v2_parts = [int(x) for x in v2.split('.')]
        
        # 补齐到相同长度
        max_len = max(len(v1_parts), len(v2_parts))
        v1_parts.extend([0] * (max_len - len(v1_parts)))
        v2_parts.extend([0] * (max_len - len(v2_parts)))
        
        # 计算欧几里得距离
        distance = sum((a - b) ** 2 for a, b in zip(v1_parts, v2_parts)) ** 0.5
        return -distance  # 负距离用于排序
    except:
        return 0


def get_available_package_versions(data, pkg):
    """
    获取数据中指定包的所有可用版本
    
    Args:
        data: list, 数据列表
        pkg: str, 包名
        
    Returns:
        list: 可用版本列表
    """
    versions = set()
    for item in data:
        if 'pkg' in item and 'version' in item:
            if item['pkg'] == pkg:
                versions.add(item['version'])
    return sorted(list(versions))


def GetQAPairsFromFlatIFTData_BCB(filepath, sample_num=None, pkg=None, version=None, 
                                 data_strategy='same_minor_version', n_versions=3, 
                                 all_package_versions=None):
    '''
    从flat_IFTdata.json中获取QA对，支持多种数据加载策略
    
    Args:
        filepath: str, 文件路径
        sample_num: int, 需要采样的数量
        pkg: str, 包名
        version: str, 目标版本（格式为major.minor或major.minor.patch）
        data_strategy: str, 数据加载策略
            - 'same_minor_version': 仅匹配major.minor版本（忽略patch版本）
            - 'all_versions': 为包加载所有版本的数据
            - 'closest_n': 加载最接近的n个版本数据
        n_versions: int, closest_n策略中的版本数量
        all_package_versions: dict, 所有包版本信息（可选，用于策略优化）
        
    Returns:
        list: QA对列表
    '''
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        logging.info(f"Loaded {len(data)} examples from {filepath}")
        
        if not data:
            logging.warning("No data loaded from file")
            return []
        
        # Debug: Check the structure of the first few items
        logging.debug(f"Sample data structure (first item): {list(data[0].keys()) if data else 'No data'}")
        
        # 获取可用版本信息
        available_versions = get_available_package_versions(data, pkg) if pkg else []
        
        if pkg and not available_versions:
            logging.warning(f"Package '{pkg}' not found in data")
            return []
        
        # 根据策略确定要加载的版本
        target_versions = []
        
        if data_strategy == 'same_minor_version':
            # 原有逻辑：仅匹配major.minor版本
            if version:
                version_parts = version.split('.')
                target_major_minor = '.'.join(version_parts[:2])
                target_versions = [target_major_minor]
                logging.info(f"Using same_minor_version strategy: {target_major_minor}")
        
        elif data_strategy == 'all_versions':
            # 加载包的所有版本
            if pkg and available_versions:
                target_versions = available_versions
                logging.info(f"Using all_versions strategy: {len(target_versions)} versions for {pkg}")
            elif version:
                target_versions = [version]
        
        elif data_strategy == 'closest_n':
            # 加载最接近的n个版本
            if pkg and available_versions and version:
                # 按相似性排序
                sorted_versions = sorted(available_versions, 
                                       key=lambda v: version_similarity(v, version), 
                                       reverse=True)
                target_versions = sorted_versions[:n_versions]
                logging.info(f"Using closest_n strategy: {target_versions} (closest to {version})")
            elif version:
                target_versions = [version]
        
        else:
            logging.warning(f"Unknown data strategy: {data_strategy}, using default")
            target_versions = [version] if version else []
        
        if not target_versions:
            logging.warning(f"No target versions determined for pkg='{pkg}', version='{version}', strategy='{data_strategy}'")
            return []
        
        # 处理数据
        qa_pairs = []
        total_items = 0
        matched_items = 0
        missing_pkg_count = 0
        missing_version_count = 0
        version_match_stats = {}
        
        for i, item in enumerate(data):
            total_items += 1
            
            # Check if 'pkg' field exists
            if 'pkg' not in item:
                missing_pkg_count += 1
                if missing_pkg_count <= 3:
                    logging.warning(f"Item {i} missing 'pkg' field. Available keys: {list(item.keys())}")
                continue
            
            # Check if 'version' field exists
            if 'version' not in item:
                missing_version_count += 1
                if missing_version_count <= 3:
                    logging.warning(f"Item {i} missing 'version' field. Available keys: {list(item.keys())}")
                continue
            
            item_pkg = item.get("pkg", "")
            item_version = item.get("version", "")
            
            # Check if pkg matches
            if pkg and item_pkg != pkg:
                continue
            
            # Check if version matches according to strategy
            version_matched = False
            
            if data_strategy == 'same_minor_version':
                # Extract major.minor from the item's version
                if item_version:
                    item_version_parts = item_version.split('.')
                    if len(item_version_parts) >= 2:
                        item_major_minor = '.'.join(item_version_parts[:2])
                        
                        # Match if major.minor versions are in target_versions
                        if item_major_minor in target_versions:
                            version_matched = True
                    else:
                        logging.warning(f"Invalid version format in item {i}: {item_version}")
                        
            else:  # all_versions or closest_n
                # Direct version match
                if item_version in target_versions:
                    version_matched = True
            
            if not version_matched:
                continue
            
            # Item matches criteria
            matched_items += 1
            
            # 统计版本匹配情况
            match_version = item_version
            if data_strategy == 'same_minor_version':
                item_version_parts = item_version.split('.')
                if len(item_version_parts) >= 2:
                    match_version = '.'.join(item_version_parts[:2])
            
            version_match_stats[match_version] = version_match_stats.get(match_version, 0) + 1
            
            # Format: (query, answer)
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            output = item.get('output', '')
            
            # Combine instruction and input as the query
            if input_text:
                query = f"{instruction}\n\n{input_text}"
            else:
                query = instruction
                
            qa_pairs.append((query, output))
            
            # Apply sample limit if specified
            if sample_num and len(qa_pairs) >= sample_num:
                logging.info(f"Reached sample limit of {sample_num} items")
                break
        
        # Print statistics
        logging.info(f"Data processing statistics for {data_strategy} strategy:")
        logging.info(f"  Package: {pkg}")
        logging.info(f"  Target version: {version}")
        logging.info(f"  Target versions for loading: {target_versions}")
        logging.info(f"  Total items processed: {total_items}")
        logging.info(f"  Items missing 'pkg' field: {missing_pkg_count}")
        logging.info(f"  Items missing 'version' field: {missing_version_count}")
        logging.info(f"  Items matching criteria: {matched_items}")
        logging.info(f"  Final QA pairs returned: {len(qa_pairs)}")
        
        if version_match_stats:
            logging.info(f"  Version match breakdown:")
            for v, count in sorted(version_match_stats.items()):
                logging.info(f"    {v}: {count} items")
        
        if len(qa_pairs) == 0:
            logging.warning(f"No QA pairs found for pkg='{pkg}', version='{version}', strategy='{data_strategy}'")
            if pkg and available_versions:
                logging.info(f"Available versions for {pkg}: {available_versions[:10]}{'...' if len(available_versions) > 10 else ''}")
        
        return qa_pairs
        
    except Exception as e:
        logging.error(f"Error loading flat IFT data: {e}")
        import traceback
        traceback.print_exc()
        return []


# 向后兼容的函数别名
def GetQAPairsFromFlatIFTData_BCB_original(filepath, sample_num=None, pkg=None, version=None):
    """向后兼容的函数，使用same_minor_version策略"""
    return GetQAPairsFromFlatIFTData_BCB(
        filepath=filepath, 
        sample_num=sample_num, 
        pkg=pkg, 
        version=version, 
        data_strategy='same_minor_version'
    )