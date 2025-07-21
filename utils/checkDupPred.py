#!/usr/bin/env python3
"""
检查同一ID的输出是否一致，并合并相同的结果
"""
import json
import argparse
import sys
from collections import defaultdict
import os
from typing import Dict, List, Tuple, Set, Any

def load_jsonl(file_path: str) -> List[Dict]:
    """加载JSONL文件"""
    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        results.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line: {e}")
                        continue
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        sys.exit(1)
    return results

def check_consistency(results: List[Dict]) -> Tuple[Dict[str, List[Dict]], Set[str]]:
    """
    检查同一ID的输出是否一致
    
    Returns:
        Tuple[Dict[str, List[Dict]], Set[str]]: 
            - 按ID分组的输出结果
            - 不一致的ID集合
    """
    id_groups = defaultdict(list)
    inconsistent_ids = set()
    
    # 按ID分组
    for result in results:
        if 'id' not in result:
            print(f"Warning: Found result without ID: {result}")
            continue
        id_groups[str(result['id'])].append(result)
    
    # 检查每个ID组的一致性
    for id_val, group in id_groups.items():
        if len(group) == 1:
            continue
            
        # 获取第一个结果作为参考
        reference = group[0]
        
        # 检查其他结果是否与参考结果一致
        for result in group[1:]:
            if not is_consistent(reference, result):
                inconsistent_ids.add(id_val)
                break
    
    return id_groups, inconsistent_ids

def is_consistent(result1: Dict, result2: Dict) -> bool:
    """检查两个结果是否一致"""
    # 检查必要的字段是否存在
    required_fields = {'id', 'answer'}
    if not all(field in result1 and field in result2 for field in required_fields):
        return False
    
    # 检查ID是否相同
    if str(result1['id']) != str(result2['id']):
        return False
    
    # 检查answer字段是否相同
    if result1['answer'] != result2['answer']:
        return False
    
    # 检查context字段（如果存在）
    if 'context' in result1 and 'context' in result2:
        if result1['context'] != result2['context']:
            return False
    
    return True

def merge_results(id_groups: Dict[str, List[Dict]], inconsistent_ids: Set[str]) -> List[Dict]:
    """合并相同的结果"""
    merged_results = []
    
    for id_val, group in id_groups.items():
        if id_val in inconsistent_ids:
            print(f"\n❌ Inconsistent results found for ID {id_val}:")
            for i, result in enumerate(group):
                print(f"\nResult {i+1}:")
                print(f"  Answer: {result['answer'][:100]}...")
                if 'context' in result:
                    print(f"  Context: {result['context'][:100]}...")
            continue
        
        # 使用第一个结果作为合并后的结果
        merged_results.append(group[0])
    
    return merged_results

def save_results(results: List[Dict], output_path: str):
    """保存结果到文件"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
    except Exception as e:
        print(f"Error saving results to {output_path}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Check and merge duplicate predictions")
    parser.add_argument("--input_file", default="output/test/RAG/testQA_vace.jsonl", help="Input JSONL file path")
    parser.add_argument("--output", "-o", default="output/test/RAG/testQA_vace1.jsonl", help="Output JSONL file path (default: input_file.merged.jsonl)")
    args = parser.parse_args()
    
    # 设置默认输出文件路径
    if not args.output:
        base, ext = os.path.splitext(args.input_file)
        args.output = f"{base}.merged{ext}"
    
    print(f"Processing file: {args.input_file}")
    
    # 加载结果
    results = load_jsonl(args.input_file)
    print(f"Loaded {len(results)} results")
    
    # 检查一致性
    id_groups, inconsistent_ids = check_consistency(results)
    
    # 打印统计信息
    total_ids = len(id_groups)
    duplicate_ids = sum(1 for group in id_groups.values() if len(group) > 1)
    inconsistent_count = len(inconsistent_ids)
    
    print(f"\nStatistics:")
    print(f"Total unique IDs: {total_ids}")
    print(f"IDs with duplicates: {duplicate_ids}")
    print(f"Inconsistent IDs: {inconsistent_count}")
    
    if inconsistent_count > 0:
        print(f"\nFound {inconsistent_count} IDs with inconsistent results")
        print("Inconsistent IDs:", sorted(inconsistent_ids))
    else:
        print("\n✅ All duplicate results are consistent")
    
    # 合并结果
    merged_results = merge_results(id_groups, inconsistent_ids)
    
    # 保存结果
    save_results(merged_results, args.output)
    print(f"\nSaved {len(merged_results)} merged results to {args.output}")
    
    # 如果有不一致的结果，返回非零退出码
    if inconsistent_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
