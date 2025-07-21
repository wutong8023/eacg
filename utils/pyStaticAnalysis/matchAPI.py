import json
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from difflib import SequenceMatcher

def load_target_dependencies() -> Dict[str, Dict[str, str]]:
    """
    加载所有任务的target_dependency信息
    
    Returns:
        Dict[str, Dict[str, str]]: task_id到target_dependency的映射
    """
    with open("data/VersiBCB_Benchmark/vace_datas.json", 'r') as f:
        vace_data = json.load(f)
    return {item['id']: item['target_dependency'] for item in vace_data}

def calculate_similarity(s1: str, s2: str) -> float:
    """
    计算两个字符串的相似度
    
    Args:
        s1: 第一个字符串
        s2: 第二个字符串
        
    Returns:
        float: 相似度分数 (0-1)
    """
    return SequenceMatcher(None, s1, s2).ratio()

def match_api_in_corpus(api_path: str, package: str, version: str, corpus_base: str = "/datanfs2/chenrongyi/data/docs/") -> Tuple[Optional[str], float]:
    """
    在corpus中匹配API路径，返回相似度最高的匹配
    
    Args:
        api_path: API的完整路径
        package: 包名
        version: 版本号
        corpus_base: corpus基础目录
        
    Returns:
        Tuple[Optional[str], float]: (匹配到的API路径, 相似度分数) 或 (None, 0.0)
    """
    corpus_file = os.path.join(corpus_base, package, f"{version}.jsonl")
    if not os.path.exists(corpus_file):
        return None, 0.0
    
    best_match = None
    best_similarity = 0.0
    
    with open(corpus_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            corpus_path = item['path']
            similarity = calculate_similarity(api_path, corpus_path)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = corpus_path
    
    return best_match, best_similarity

def get_corpus_item(package: str, version: str, api_path: str, corpus_base: str = "/datanfs2/chenrongyi/data/docs/") -> Optional[Dict]:
    """
    从corpus中获取指定API路径的完整数据项
    
    Args:
        package: 包名
        version: 版本号
        api_path: API路径
        corpus_base: corpus基础目录
        
    Returns:
        Optional[Dict]: 包含path、doc和signature的数据项，如果未找到则返回None
    """
    corpus_file = os.path.join(corpus_base, package, f"{version}.jsonl")
    if not os.path.exists(corpus_file):
        return None
    
    with open(corpus_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            if item['path'] == api_path:
                return item
    return None

def process_errors_with_corpus(processed_errors_file: str, corpus_base: str = "/datanfs2/chenrongyi/data/docs/") -> List[Dict]:
    """
    处理错误信息并在corpus中匹配API路径
    
    Args:
        processed_errors_file: 处理后的错误信息文件路径
        corpus_base: corpus基础目录
        
    Returns:
        List[Dict]: 添加了匹配结果的错误信息列表
    """
    # 加载target_dependency信息
    dependency_map = load_target_dependencies()
    
    # 读取处理后的错误信息
    with open(processed_errors_file, 'r') as f:
        errors = json.load(f)
    
    # 处理每个错误
    for error_item in errors:
        task_id = error_item['id']
        target_dependency = dependency_map.get(task_id, {})
        
        for error in error_item['error_infos']:
            if 'full_api_path' in error:
                full_api_path = error['full_api_path']
                if full_api_path:
                    # 从full_api_path中提取包名
                    package = full_api_path.split('.')[0]
                    if package in target_dependency:
                        version = target_dependency[package]
                        matched_path, similarity = match_api_in_corpus(full_api_path, package, version, corpus_base)
                        error['matched_path'] = matched_path
                        error['match_similarity'] = similarity
    
    return errors

def save_retrieved_items(matched_errors_file: str, corpus_base: str = "/datanfs2/chenrongyi/data/docs/") -> None:
    """
    保存匹配路径对应的完整数据项
    
    Args:
        matched_errors_file: 匹配后的错误信息文件路径
        corpus_base: corpus基础目录
    """
    # 加载target_dependency信息
    dependency_map = load_target_dependencies()
    
    # 读取匹配后的错误信息
    with open(matched_errors_file, 'r') as f:
        errors = json.load(f)
    
    # 收集每个task_id的retrieved_items
    retrieved_data = []
    for error_item in errors:
        task_id = error_item['id']
        target_dependency = dependency_map.get(task_id, {})
        retrieved_items = []
        
        for error in error_item['error_infos']:
            if 'matched_path' in error and error['matched_path']:
                package = error['matched_path'].split('.')[0]
                if package in target_dependency:
                    version = target_dependency[package]
                    corpus_item = get_corpus_item(package, version, error['matched_path'], corpus_base)
                    if corpus_item:
                        retrieved_items.append(corpus_item)
        
        if retrieved_items:
            retrieved_data.append({
                'id': task_id,
                'retrieved_items': retrieved_items,
                'api_to_match': error['full_api_path'],
                'match_similarity': error['match_similarity']
            })
    
    # 保存结果
    output_file = "data/temp/retrieved_items.json"
    with open(output_file, 'w') as f:
        json.dump(retrieved_data, f, indent=2)
    
    print(f"检索完成，结果已保存到 {output_file}")

def main():
    # 设置文件路径
    processed_errors_file = "data/temp/processed_errors.json"
    matched_errors_file = "data/temp/matched_errors.json"
    corpus_base = "/datanfs2/chenrongyi/data/docs/"
    
    # 处理错误信息并匹配API路径
    matched_errors = process_errors_with_corpus(processed_errors_file, corpus_base)
    
    # 保存匹配结果
    with open(matched_errors_file, 'w') as f:
        json.dump(matched_errors, f, indent=2)
    
    # 保存检索到的数据项
    save_retrieved_items(matched_errors_file, corpus_base)

if __name__ == "__main__":
    main() 