from utils.testPassBCB import passTaskTest
import json
from utils.config import DEPRECATION_KEYWORDS
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import argparse
import os
from datetime import datetime

def deprecateCheck(deprecate_string):
    deprecate_keyword = DEPRECATION_KEYWORDS
    if deprecate_string and any(keyword.lower() in deprecate_string.lower() for keyword in deprecate_keyword):
        deprecate=True
    else:
        deprecate=False
    return deprecate

def clean_model_output(output):
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

def test_single_prediction(code, data, ban_deprecation=False, process_id=None):
    """
    测试单个预测结果，带10秒超时限制
    
    Args:
        code: 预测的代码
        data: 包含测试信息的数据字典
        ban_deprecation: 是否禁用废弃函数
        process_id: 进程ID，用于区分测试文件
    """
    code = clean_model_output(code)
    testcode = data['testcode'] if TASK_TYPE == 'vscc' else data['target_testcode']
    specifiedcode = (code, testcode)
    test_file = f"test_{process_id}.py" if process_id is not None else None
    
    # 获取task id，统一使用id或taskid
    task_id = data.get('id')
    BCB_id = data.get('taskid')
    # 创建单个进程的池
    with mp.Pool(1) as pool:
        # 异步提交任务
        async_result = pool.apply_async(
            passTaskTest,
            kwds={
                'task_id': BCB_id,
                'dep': data['dependency'] if TASK_TYPE == 'vscc' else data['target_dependency'],
                'specifiedcode': specifiedcode,
                'DeprecateAdapt': ban_deprecation,
                'test_file_path': test_file
            }
        )
        
        try:
            # 等待结果，10秒超时
            passTest = async_result.get(timeout=10)
        except mp.TimeoutError:
            print(f"Test timeout for task {task_id}")
            return False
        except Exception as e:
            print(f"Test error for task {task_id}: {str(e)}")
            return False
        finally:
            # 确保进程池被正确关闭
            pool.terminate()
            pool.join()
    
    if ban_deprecation:
        return passTest[0] and not deprecateCheck(passTest[1])
    return passTest

def test_batch_predictions(batch_data, ban_deprecation=False, process_id=None):
    """
    测试一批数据的预测结果
    
    Args:
        batch_data: 待测试的数据批次
        ban_deprecation: 是否禁用废弃函数
        process_id: 进程ID
    """
    results = []
    for data in batch_data:
        predictions = data['model_output']
        # 获取task id，统一使用id或taskid
        task_id = data.get('id', data.get('taskid'))
        
        if not isinstance(predictions, list):
            print(f"Warning: model_output is not a list for task {task_id}")
            results.append((False, False))
            continue
            
        # 测试第一个预测，传入process_id
        pass_1 = test_single_prediction(
            predictions[0], 
            data, 
            ban_deprecation, 
            process_id=process_id
        )
        
        # 如果第一个通过，pass@3也通过
        if pass_1:
            results.append((True, True))
            continue
            
        # 测试剩余预测
        pass_3 = any(
            test_single_prediction(pred, data, ban_deprecation, process_id=process_id) 
            for pred in predictions[1:]
        )
        results.append((pass_1, pass_3))
        
    return results

def parallel_evaluate(data_list, ban_deprecation=False, num_processes=None):
    """
    并行评估pass@1和pass@3
    
    Returns:
        dict: 包含总体结果和每个task_id的详细结果
    """
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)
    
    data_count = len(data_list)
    batch_size = max(1, data_count // num_processes)
    batches = [data_list[i:i + batch_size] for i in range(0, data_count, batch_size)]
    
    pbar = tqdm(total=data_count, desc="Evaluating predictions")
    
    pass_at_1_count = 0
    pass_at_3_count = 0
    task_results = {}  # 存储每个task_id的结果
    
    # 使用进程池并行处理，并传入进程ID
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # 为每个批次创建带有进程ID的任务
        futures = []
        for i, batch in enumerate(batches):
            test_func = partial(
                test_batch_predictions, 
                ban_deprecation=ban_deprecation, 
                process_id=i  # 使用批次索引作为进程ID
            )
            futures.append((batch, executor.submit(test_func, batch)))
        
        # 处理结果
        for batch, future in futures:
            batch_results = future.result()
            for i, (pass_1, pass_3) in enumerate(batch_results):
                # 获取当前数据项
                data_item = batch[i]
                # 获取task id，统一使用id或taskid
                task_id = data_item.get('id', data_item.get('taskid'))
                
                # 记录结果
                task_results[task_id] = {
                    'pass@1': pass_1,
                    'pass@3': pass_3
                }
                
                if pass_1:
                    pass_at_1_count += 1
                if pass_3:
                    pass_at_3_count += 1
                pbar.update(1)
    
    pbar.close()
    
    results = {
        'pass@1': pass_at_1_count / data_count,
        'pass@3': pass_at_3_count / data_count,
        'total_samples': data_count,
        'pass@1_count': pass_at_1_count,
        'pass@3_count': pass_at_3_count,
        'task_results': task_results  # 添加每个task_id的详细结果
    }
    
    return results

if __name__=='__main__':
    #TODO: results中加入pass@n对应的instance id，从而定位到其对应的taskid，从而方便统计task通过率；
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='Evaluate predictions')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the prediction data JSON file')
    parser.add_argument('--ban_deprecation', type=bool, default=False,
                       help='Whether to ban deprecated functions')
    parser.add_argument('--output_dir', type=str, default='logs/evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--num_processes', type=int, default=4,
                       help='Number of processes for parallel evaluation')
    parser.add_argument('--task_type',type=str,default='vscc',
                       help='task type')
    args = parser.parse_args()
    TASK_TYPE = args.task_type
    # 从路径中提取模型名和数据集名
    path_parts = args.data_path.split('/')
    model_name = path_parts[-2]  # 例如 'deepseek-chat'
    dataset_name = path_parts[-1].split('.')[0]  # 例如 'vscc_datas'，移除.json后缀
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取数据
    with open(args.data_path, "r") as f:
        data_list = json.load(f)
    
    # 使用并行版本进行评估
    results = parallel_evaluate(
        data_list, 
        ban_deprecation=args.ban_deprecation,
        num_processes=args.num_processes
    )
    
    # 准备输出结果
    output = {
        "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "dataset_name": dataset_name,
        "ban_deprecation": args.ban_deprecation,
        "num_processes": args.num_processes,
        "results": results
    }
    
    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        args.output_dir, 
        f"{model_name}_{dataset_name}_eval_{timestamp}.json"
    )
    
    # 保存结果到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4)
    
    # 同时在控制台打印简要结果
    print(f"\nEvaluation completed. Results saved to: {output_file}")
    print(f"Summary for {model_name} on {dataset_name}:")
    print(f"Total samples: {results['total_samples']}")
    print(f"Pass@1: {results['pass@1']:.4f} ({results['pass@1_count']}/{results['total_samples']})")
    print(f"Pass@3: {results['pass@3']:.4f} ({results['pass@3_count']}/{results['total_samples']})")
    
    # 添加关于每个任务的简要统计
    print("\nTask-level results saved in output file.")
    print(f"Number of unique tasks: {len(results['task_results'])}")
    
    # 计算通过的任务数量
    pass1_tasks = sum(1 for res in results['task_results'].values() if res['pass@1'])
    pass3_tasks = sum(1 for res in results['task_results'].values() if res['pass@3'])
    
    print(f"Tasks passed@1: {pass1_tasks}/{len(results['task_results'])} ({pass1_tasks/len(results['task_results']):.4f})")
    print(f"Tasks passed@3: {pass3_tasks}/{len(results['task_results'])} ({pass3_tasks/len(results['task_results']):.4f})")
    
    # 打印一些示例任务结果
    print("\nSample task results (first 5):")
    for i, (task_id, result) in enumerate(list(results['task_results'].items())[:5]):
        print(f"  Task {task_id}: pass@1={result['pass@1']}, pass@3={result['pass@3']}")