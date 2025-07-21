import json

def get_baseline_passed_ids(baseline_file: str):
    """
    从baseline文件中获取passed=true的ID列表
    
    Args:
        baseline_file: baseline文件路径
    
    Returns:
        list: passed=true的ID列表
    """
    passed_ids = []
    try:
        with open(baseline_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for result in data.get('detailed_results', []):
                if result.get('passed', False):
                    passed_ids.append(result.get('task_id'))
    except Exception as e:
        print(f"Error reading baseline file: {str(e)}")
    return passed_ids

def analyze_errors(input_file: str, baseline_file: str):
    """
    分析测试结果文件，统计错误总量和错误ID，并与baseline比较
    
    Args:
        input_file: 输入文件路径
        baseline_file: baseline文件路径
    """
    error_ids = []
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                result = json.loads(line)
                total_count += 1
                
                # 检查是否有错误
                if result.get('has_error', False):
                    error_ids.append(result.get('id'))
            
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line: {line[:100]}... Error: {str(e)}")
                continue
    
    # 获取baseline中passed=true的ID
    passed_ids = get_baseline_passed_ids(baseline_file)
    
    # 找出在baseline中passed=true但在错误列表中的ID
    conflict_ids = set(passed_ids) & set(error_ids)
    
    # 打印统计结果
    print(f"\n=== 错误统计 ===")
    print(f"总测试数量: {total_count}")
    print(f"错误数量: {len(error_ids)}")
    print(f"错误率: {(len(error_ids) / total_count * 100):.1f}%")
    
    print(f"\n=== Baseline比较 ===")
    print(f"Baseline中passed=true的数量: {len(passed_ids)}")
    print(f"冲突ID数量: {len(conflict_ids)}")
    
    if conflict_ids:
        print("\n冲突ID列表 (在baseline中passed=true但在错误列表中):")
        print(sorted(list(conflict_ids)))
    
    if error_ids:
        print("\n所有错误ID列表:")
        print(sorted(error_ids))

def main():
    input_file = "data/temp/pyright_test_results_filtered.jsonl"
    baseline_file = "data/temp/baseline.json"
    mypy_file = "data/temp/mypy_test_results.jsonl"
    
    print("Analyzing pyright test results...")
    analyze_errors(mypy_file, baseline_file)

if __name__ == "__main__":
    main()
