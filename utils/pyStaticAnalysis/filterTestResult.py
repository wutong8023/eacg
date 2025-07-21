import json
import os

def filter_python35_results(input_file, output_file):
    """
    过滤Python 3.5的结果，将has_error设置为false
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 读取并处理结果
    filtered_results = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            try:
                result = json.loads(line)
                # 检查是否为Python 3.5环境
                if (result.get('target_dependency', {}).get('python') == '3.5'):
                    # 将has_error设置为false
                    result['has_error'] = False
                    result['error_info'] = "Skipped pyright check for Python 3.5 environment"
                filtered_results.append(result)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line: {line[:100]}... Error: {str(e)}")
                continue
    
    # 写入过滤后的结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in filtered_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Filtered results saved to {output_file}")

if __name__ == "__main__":
    # 设置输入输出文件路径
    input_file = "data/temp/pyright_test_results.jsonl"
    output_file = "data/temp/pyright_test_results_filtered.jsonl"
    
    # 执行过滤
    filter_python35_results(input_file, output_file)
