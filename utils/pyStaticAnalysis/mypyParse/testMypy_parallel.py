import subprocess
import os
import sys
import json
import tempfile
import datetime
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import ast
import astor
#TODO：过滤[import-untyped]的错误信息行
#
def clean_return_annotations(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            node.returns = None  # 删除返回值注解
    return astor.to_source(tree)

from tests.pyStaticAnalysis.testmypy_utils import get_error_info_from_mypy,getTargetEnvPath,run_mypy_in_env



def clean_model_output(output):
    '''
    清理模型输出，移除markdown标记等
    '''
    if "<start>" in output:
        start_index = output.find("<start>") + len("<start>")
        if "<end>" in output:
            end_index = output.find("<end>")
            content = output[start_index:end_index].replace('```python', '').replace('```', '')
        else:
            content = output[start_index:].replace('```python', '').replace('```', '')
    else:
        # 目前不要求严格start和end
        content = output.replace('```python', '').replace('```', '')
    
    return content

def read_benchmark_file(benchmark_file_path):
    '''
    读取benchmark文件，返回id到target_dependency的映射
    '''
    try:
        with open(benchmark_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        id_to_dependency = {}
        for item in data:
            if 'id' in item and ('target_dependency' in item or 'dependency' in item):
                id_to_dependency[item['id']] = item['target_dependency'] if 'target_dependency' in item else item['dependency']
        
        return id_to_dependency
    except Exception as e:
        print(f"Error reading benchmark file {benchmark_file_path}: {str(e)}")
        return {}

def get_existing_results(output_file):
    '''
    获取已经处理过的结果
    
    Args:
        output_file: 输出文件路径
    
    Returns:
        set: 已处理的id集合
    '''
    existing_ids = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'id' in data:
                            existing_ids.add(data['id'])
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: Error reading existing results: {str(e)}")
    return existing_ids


def process_single_item(item_data, id_to_dependency, output_file, lock):
    '''
    处理单个测试项
    
    Args:
        item_data: 包含id和answer的数据
        id_to_dependency: id到target_dependency的映射
        output_file: 输出文件路径
        lock: 文件写入锁
    
    Returns:
        bool: 是否成功处理
    '''
    try:
        item_id = item_data['id']
        answer = item_data['answer']
        
        # 获取对应的依赖信息
        if item_id not in id_to_dependency:
            print(f"Warning: No dependency found for id {item_id}")
            return False
        
        target_dependency = id_to_dependency[item_id]
        venv_dir = getTargetEnvPath(target_dependency)
        
        # 清理模型输出
        clean_code = clean_model_output(answer)
        
        # 清理返回值注解
        try:
            clean_code = clean_return_annotations(clean_code)
        except Exception as e:
            print(f"Warning: Failed to clean return annotations for id {item_id}: {str(e)}")
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(clean_code)
            temp_file_path = temp_file.name
        
        try:
            # 测试代码
            has_error, error_info = run_mypy_in_env(venv_dir, temp_file_path)
            
            # 准备输出数据
            result = {
                "id": item_id,
                "code": clean_code,
                "target_dependency": target_dependency,
                "has_error": has_error,
                "error_info": error_info,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # 使用锁保护文件写入
            with lock:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            return True
            
        finally:
            # 清理临时文件
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
    except Exception as e:
        print(f"Error processing item {item_data.get('id', 'unknown')}: {str(e)}")
        return False

def process_jsonl_with_benchmark(jsonl_file_path, benchmark_file_path, output_file, max_workers=4):
    '''
    从JSONL文件读取代码，从benchmark文件读取依赖，使用多线程测试代码并保存结果
    
    Args:
        jsonl_file_path: JSONL文件路径，包含id和answer字段
        benchmark_file_path: benchmark文件路径，包含target_dependency
        output_file: 输出JSONL文件路径
        max_workers: 最大线程数
    
    Returns:
        tuple: (成功测试数量, 总数量)
    '''
    # 读取benchmark文件获取依赖映射
    id_to_dependency = read_benchmark_file(benchmark_file_path)
    
    # 获取已处理的结果
    existing_ids = get_existing_results(output_file)
    print(f"Found {len(existing_ids)} existing results")
    
    # 读取所有测试项
    test_items = []
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    if 'id' not in data or 'answer' not in data:
                        continue
                    # 跳过已处理的项目
                    if data['id'] in existing_ids:
                        continue
                    test_items.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line: {line[:100]}... Error: {str(e)}")
                    continue
    except Exception as e:
        print(f"Error reading JSONL file {jsonl_file_path}: {str(e)}")
        return 0, 0
    
    total_count = len(test_items)
    if total_count == 0:
        print("No new items to process")
        return 0, 0
    
    print(f"Processing {total_count} new items...")
    
    # 创建文件写入锁
    file_lock = threading.Lock()
    
    # 使用线程池处理测试项
    success_count = 0
    start_time = time.time()
    
    print(f"Starting parallel processing with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_item = {
            executor.submit(
                process_single_item, 
                item, 
                id_to_dependency, 
                output_file, 
                file_lock
            ): item for item in test_items
        }
        
        # 处理完成的任务
        for i, future in enumerate(as_completed(future_to_item), 1):
            item = future_to_item[future]
            try:
                if future.result():
                    success_count += 1
                # 打印进度
                if i % 10 == 0 or i == total_count:
                    elapsed = time.time() - start_time
                    print(f"Progress: {i}/{total_count} ({(i/total_count*100):.1f}%) - "
                          f"Success: {success_count}/{i} - "
                          f"Time elapsed: {elapsed:.1f}s")
            except Exception as e:
                print(f"Error processing item {item.get('id', 'unknown')}: {str(e)}")
    
    total_time = time.time() - start_time
    print(f"\nProcessing completed in {total_time:.1f} seconds")
    print(f"Successfully tested {success_count} out of {total_count} code snippets")
    
    return success_count, total_count

if __name__ == "__main__":
    # Example: Process JSONL file with benchmark
    jsonl_file = "output/approach_eval/BASELINE/Llama-3.1-8B-Instruct/versibcb_vscc_Llama-3.1-8B-Instruct_maxdep10.jsonl"
    benchmark_file = "data/VersiBCB_Benchmark/vscc_datas.json"
    output_file = "data/temp/mypy_test_results_vscc.jsonl"
    
    print("Processing JSONL file with benchmark data...")
    success_count, total_count = process_jsonl_with_benchmark(
        jsonl_file, 
        benchmark_file, 
        output_file,
        max_workers=8  # 可以根据CPU核心数调整
    )
    print(f"Results saved to {output_file}") 