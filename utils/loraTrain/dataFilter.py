'''
数据预过滤，检查是否需要训练
'''
import json
import os
import logging
from utils.loraTrain.loraTrainUtils import loraModelExists


def getTrainDataItems(corpus_path, pkg, version, model_config):
    """
    获取训练数据项
    
    Args:
        corpus_path: 语料库路径
        pkg: 包名
        version: 版本号
        model_config: 模型配置
        
    Returns:
        tuple: (original_count, files_info) - (原始数据量, 过滤后的数据)
    """
    try:
        files_info = []
        data_file_path = f"{corpus_path}/{pkg}/{version}.jsonl"
        with open(data_file_path, "r", encoding="utf-8") as f:
            for line in f:
                files_info.append(json.loads(line))
    except FileNotFoundError:
        logging.error(f"数据文件不存在: {data_file_path}")
        return None, None
        
    original_count = len(files_info)
    if model_config["override_data_percentage"] is not None:
        files_info = files_info[:int(len(files_info)*float(model_config["override_data_percentage"]))]
    else:
        files_info = files_info[:int(len(files_info)*model_config["traindata_percentage"])]
    return original_count, files_info


def prefilter_package_version(pkg, version, config, corpus_path, knowledge_type='docstring'):
    """
    预过滤包版本，检查是否需要训练
    
    Args:
        pkg: 包名
        version: 版本号  
        config: 配置信息
        corpus_path: 语料库路径
        knowledge_type: 知识类型
        
    Returns:
        tuple: (should_train, reason, data_count)
        - should_train: 是否需要训练
        - reason: 跳过训练的原因（如果不需要训练）
        - data_count: 可用训练数据条数
    """
    try:
        model_name = config.get("model_name").split("/")[-1]
        
        # 1. 检查LoRA模型是否已存在
        if loraModelExists(pkg, version, model_name, config, knowledge_type=knowledge_type):
            return False, "lora_model_exists", 0
        
        # 2. 获取并检查训练数据
        original_count, files_info = getTrainDataItems(corpus_path, pkg, version, config)
        
        if original_count is None or files_info is None:
            return False, "data_file_not_found", 0
            
        if len(files_info) == 0:
            return False, "no_training_data_after_percentage", original_count
        
        return True, "needs_training", len(files_info)
        
    except Exception as e:
        logging.error(f"预过滤包版本时出错 {pkg}-{version}: {e}")
        return False, f"error: {str(e)}", 0


def apply_prefilter_to_package_versions(pack_versions, config, corpus_path, knowledge_type='docstring', log_details=True):
    """
    对所有包版本应用预过滤
    
    Args:
        pack_versions: 包版本字典
        config: 配置信息
        corpus_path: 语料库路径  
        knowledge_type: 知识类型
        log_details: 是否输出详细日志
        
    Returns:
        tuple: (filtered_pack_versions, stats)
        - filtered_pack_versions: 过滤后的包版本字典
        - stats: 统计信息字典
    """
    filtered_pack_versions = {}
    stats = {
        'total': 0,
        'needs_training': 0,
        'lora_exists': 0,
        'no_data': 0,
        'errors': 0,
        'details': []
    }
    
    if log_details:
        logging.info("🔍 开始预过滤包版本...")
    
    for pkg, versions in pack_versions.items():
        filtered_versions = []
        
        for version in versions:
            stats['total'] += 1
            
            should_train, reason, data_count = prefilter_package_version(
                pkg, version, config, corpus_path, knowledge_type
            )
            
            if should_train:
                filtered_versions.append(version)
                stats['needs_training'] += 1
                if log_details:
                    logging.debug(f"✅ 需要训练: {pkg}-{version} (数据量: {data_count})")
            else:
                if reason == "lora_model_exists":
                    stats['lora_exists'] += 1
                elif "no_training_data" in reason or "data_file_not_found" in reason:
                    stats['no_data'] += 1
                elif "error:" in reason:
                    stats['errors'] += 1
                
                if log_details:
                    logging.debug(f"⏭️ 跳过: {pkg}-{version} ({reason})")
                    
                stats['details'].append({
                    'pkg': pkg,
                    'version': version,
                    'reason': reason,
                    'data_count': data_count
                })
        
        if filtered_versions:
            filtered_pack_versions[pkg] = filtered_versions
    
    if log_details:
        logging.info(f"📊 预过滤统计:")
        logging.info(f"  总包版本: {stats['total']}")
        logging.info(f"  需要训练: {stats['needs_training']} ({stats['needs_training']/stats['total']*100:.1f}%)")
        logging.info(f"  模型已存在: {stats['lora_exists']} ({stats['lora_exists']/stats['total']*100:.1f}%)")  
        logging.info(f"  无训练数据: {stats['no_data']} ({stats['no_data']/stats['total']*100:.1f}%)")
        logging.info(f"  检查错误: {stats['errors']}")
    
    return filtered_pack_versions, stats