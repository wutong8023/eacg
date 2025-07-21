#!/usr/bin/env python3
"""
单文件评估脚本 - 使用testPassBCB_new获取详细测试统计
用于对指定的预测文件进行快速评估，避免处理大量无关文件
并返回详细的测试用例统计信息

使用方法:
python utils/evaluate/single_file_evaluation_withnew.py --prediction_file path/to/prediction.json
"""

import json
import os
import argparse
import logging
import sys
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading

# Import existing evaluation functions
from utils.evaluate.evaluation import test_single_prediction, clean_model_output, deprecateCheck,clean_model_output1,clean_model_output_lora,clean_model_output_review,clean_model_output_errorfix,clean_model_output_py_wrapper_first
from utils.evaluate.config.createTarcode_config import DEPRECATION_KEYWORDS
from utils.evaluate.testPassBCB import passTaskTest1

# Global logger and original streams
logger = None
original_stdout = None
original_stderr = None

class DualStream:
    """Custom stream class that writes to both console and log file"""
    def __init__(self, console_stream, log_handler):
        self.console_stream = console_stream
        self.log_handler = log_handler
        
    def write(self, text):
        # Write to console
        self.console_stream.write(text)
        self.console_stream.flush()
        
        # Write to log file (remove extra newlines that might be added)
        if text.strip():  # Only log non-empty content
            # Remove trailing newlines since logger will add its own
            clean_text = text.rstrip('\n\r')
            if clean_text:
                self.log_handler.info(clean_text)
    
    def flush(self):
        self.console_stream.flush()
        
    def fileno(self):
        return self.console_stream.fileno()

def setup_logging_with_redirect(log_file_path):
    """Setup logging with stdout/stderr redirection to capture all output"""
    global logger, original_stdout, original_stderr
    
    # Store original streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Create logger
    logger = logging.getLogger('evaluation')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # File handler
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Create dual streams that write to both console and log
    dual_stdout = DualStream(original_stdout, logger)
    dual_stderr = DualStream(original_stderr, logger)
    
    # Redirect stdout and stderr
    sys.stdout = dual_stdout
    sys.stderr = dual_stderr
    
    return logger

def restore_streams():
    """Restore original stdout and stderr"""
    global original_stdout, original_stderr
    if original_stdout:
        sys.stdout = original_stdout
    if original_stderr:
        sys.stderr = original_stderr

def log_print(message, end='\n'):
    """Custom print function - now just uses regular print since we redirect streams"""
    print(message, end=end)

def load_benchmark_data(benchmark_path):
    """Load benchmark data from JSON file"""
    with open(benchmark_path, 'r') as f:
        return json.load(f)

def load_prediction_data(file_path):
    """Load prediction data from JSON or JSONL file"""
    data_list = []
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r') as f:
            for line in f:
                data_list.append(json.loads(line))
    else:  # .json file
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                data_list = data
            else:
                data_list = [data]
    return data_list

def get_prediction_from_data(data):
    """Extract prediction from data item"""
    if 'pred' in data:
        return data['pred']
    elif 'answer' in data:
        return data['answer']
    elif 'model_output' in data:
        if isinstance(data['model_output'], list) and len(data['model_output']) > 0:
            return data['model_output'][0]
        else:
            return data['model_output']
    else:
        return None

def extract_info_from_filename(filename):
    """Extract all information from filename
    
    Args:
        filename: e.g., versibcb_vace_bd_docstring_emb_local_4000_...
        
    Returns:
        dict: {
            'task_type': 'VACE',
            'knowledge_type': 'Doc', 
            'ban_deprecation': True,
            'benchmark_file': 'vace_datas_for_warning.json'
        }
    """
    # Get base filename without directory
    base_filename = os.path.basename(filename)
    parts = base_filename.split('_')
    
    # Extract task type (vscc or vace)
    task_type = 'Unknown'
    if len(parts) >= 2:
        task_type = parts[1].upper()  # vace -> VACE, vscc -> VSCC
    
    # Extract knowledge type and ban_deprecation
    ban_deprecation = False  # default
    
    if len(parts) >= 3:
        # Check if bd exists (indicates ban_deprecation = True and Doc knowledge)
        if 'bd' in parts[2]:
            ban_deprecation = True
    
    if 'docstring' in base_filename or 'doc' in base_filename:
        knowledge_type = 'Doc'
    elif 'srccode' in base_filename:
        knowledge_type = 'Src_code'
    else:
        knowledge_type = 'Unknown'
    
    # Determine benchmark file
    benchmark_file = ''
    if task_type.lower() == 'vscc':
        if ban_deprecation:
            benchmark_file = 'vscc_datas_for_warning.json'
        else:
            benchmark_file = 'vscc_datas.json'
    elif task_type.lower() == 'vace':
        if ban_deprecation:
            benchmark_file = 'vace_datas_for_warning.json'
        else:
            benchmark_file = 'vace_datas.json'
    
    return {
        'task_type': task_type,
        'knowledge_type': knowledge_type,
        'ban_deprecation': ban_deprecation,
        'benchmark_file': benchmark_file
    }

def test_single_item_with_stats(args_tuple,clean_func=clean_model_output):
    """Wrapper function for multiprocessing with detailed test statistics"""
    data, benchmark_data, ban_deprecation, process_id = args_tuple
    
    prediction = get_prediction_from_data(data)
    if prediction is None:
        return {
            'task_id': data.get('id', 'unknown'),
            'passed': False,
            'error_type': 'no_prediction',
            'message': "No prediction found",
            'prediction': None,
            'cleaned_code': None,
            'test_code': None,
            'full_traceback': None,
            'test_stats': None
        }
    
    task_id = data['id']
    
    # Get test information from benchmark data
    benchmark_item = next((item for item in benchmark_data if item['id'] == task_id), None)
    if not benchmark_item:
        return {
            'task_id': task_id,
            'passed': False,
            'error_type': 'no_benchmark',
            'message': "No benchmark data found",
            'prediction': prediction,
            'cleaned_code': clean_func(prediction),
            'test_code': None,
            'full_traceback': None,
            'test_stats': None
        }
    
    # For VSCC data
    if 'testcode' in benchmark_item:
        testcode = benchmark_item['testcode']
        dependency = benchmark_item['dependency']
    # For VACE data
    elif 'target_testcode' in benchmark_item:
        testcode = benchmark_item['target_testcode']
        dependency = benchmark_item['target_dependency']
    else:
        return {
            'task_id': task_id,
            'passed': False,
            'error_type': 'no_testcode',
            'message': "No testcode found",
            'prediction': prediction,
            'cleaned_code': clean_func(prediction),
            'test_code': None,
            'full_traceback': None,
            'test_stats': None
        }
    
    code = clean_func(prediction)
    specifiedcode = (code, testcode)
    
    # 为每个进程创建独立的测试文件路径
    thread_id = threading.get_ident()
    test_file = f"test_{process_id}_{thread_id}.py"
    
    try:
        with mp.Pool(1) as pool:
            async_result = pool.apply_async(
                passTaskTest1,
                kwds={
                    'task_id': task_id,
                    'dep': dependency,
                    'specifiedcode': specifiedcode,
                    'DeprecateAdapt': True,  # Always get detailed error info
                    'test_file_path': test_file,
                    'single_thread': True,
                    'return_test_stats': True  # Request test statistics
                }
            )
            
            try:
                passTest = async_result.get(timeout=TIMEOUT)
            except mp.TimeoutError:
                return {
                    'task_id': task_id,
                    'passed': False,
                    'error_type': 'timeout',
                    'message': "Test timeout",
                    'prediction': prediction,
                    'cleaned_code': code,
                    'test_code': testcode,
                    'full_traceback': None,
                    'test_stats': None
                }
            except Exception as e:
                # Capture full traceback for exceptions
                import traceback
                full_traceback = traceback.format_exc()
                return {
                    'task_id': task_id,
                    'passed': False,
                    'error_type': 'test_error',
                    'message': f"Test error: {str(e)}",
                    'prediction': prediction,
                    'cleaned_code': code,
                    'test_code': testcode,
                    'full_traceback': full_traceback,
                    'test_stats': None
                }
            finally:
                pool.terminate()
                pool.join()
    except Exception as e:
        # Capture full traceback for pool errors
        import traceback
        full_traceback = traceback.format_exc()
        return {
            'task_id': task_id,
            'passed': False,
            'error_type': 'pool_error',
            'message': f"Pool error: {str(e)}",
            'prediction': prediction,
            'cleaned_code': code,
            'test_code': testcode,
            'full_traceback': full_traceback,
            'test_stats': None
        }
    finally:
        # Clean up test file
        if os.path.exists(test_file):
            try:
                os.remove(test_file)
            except:
                pass
    
    # Process results - passTest is a tuple: (test_passed, error_or_stderr, test_stats)
    test_stats = None
    if isinstance(passTest, tuple) and len(passTest) >= 2:
        test_passed = passTest[0]
        error_or_stderr = passTest[1]
        
        # Extract test statistics if available
        if len(passTest) >= 3:
            test_stats = passTest[2]
        
        # Check for deprecation if originally requested
        if ban_deprecation and test_passed:
            result = not deprecateCheck(error_or_stderr)
        else:
            result = test_passed
        
        deprecation_info = error_or_stderr if isinstance(error_or_stderr, str) else None
    else:
        # Fallback for unexpected return format
        result = bool(passTest)
        error_or_stderr = None
        deprecation_info = None
    
    # If test failed, extract detailed error information
    error_info = None
    full_traceback = None
    test_error_details = None
    
    if not result:
        if isinstance(error_or_stderr, dict):
            # We have detailed error information
            test_error_details = error_or_stderr
            
            if 'stderr' in error_or_stderr and error_or_stderr['stderr']:
                error_info = f"Test failed with stderr:\n{error_or_stderr['stderr']}"
                if 'stdout' in error_or_stderr and error_or_stderr['stdout']:
                    error_info += f"\n\nStdout:\n{error_or_stderr['stdout']}"
            elif 'exception' in error_or_stderr:
                error_info = f"Test failed with exception: {error_or_stderr['exception']}"
                if 'traceback' in error_or_stderr:
                    full_traceback = error_or_stderr['traceback']
            else:
                error_info = "Test failed without detailed error information"
        elif isinstance(error_or_stderr, str) and error_or_stderr:
            error_info = f"Test failed with stderr: {error_or_stderr}"
            deprecation_info = error_or_stderr
        else:
            error_info = "Test failed without detailed error information"
    
    return {
        'task_id': task_id,
        'passed': result,
        'error_type': None if result else 'test_failed',
        'message': "Success" if result else error_info,
        'prediction': prediction,
        'cleaned_code': code,
        'test_code': testcode,
        'deprecation_info': deprecation_info,
        'full_traceback': full_traceback,
        'test_error_details': test_error_details,
        'test_stats': test_stats
    }

def evaluate_single_file_with_stats(prediction_data, benchmark_data, ban_deprecation=False, num_processes=4,clean_func=None):
    """Evaluate a single prediction file with detailed test statistics"""
    data_count = len(prediction_data)
    pass_count = 0
    error_count = 0
    detailed_results = []
    
    # Initialize aggregated test statistics
    total_test_cases = 0
    total_passed_tests = 0
    total_failed_tests = 0
    total_error_tests = 0
    total_skipped_tests = 0
    tasks_with_stats = 0
    
    # Initialize proportional correctness statistics
    total_proportional_correct = 0.0  # Sum of correct ratios for each task
    
    log_print(f"Starting evaluation with {num_processes} processes...")
    
    # Prepare arguments for multiprocessing
    args_list = []
    for i, data in enumerate(prediction_data):
        args_list.append((data, benchmark_data, ban_deprecation, i))
    
    # Use ProcessPoolExecutor for better error handling
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all tasks
        future_to_args = {executor.submit(test_single_item_with_stats, args,clean_func=clean_func): args for args in args_list}
        
        # Process results with progress bar
        with tqdm(total=data_count, desc="Evaluating") as pbar:
            for future in as_completed(future_to_args):
                try:
                    result = future.result()
                    detailed_results.append(result)
                    
                    if result['passed']:
                        pass_count += 1
                    elif result['error_type'] in ['timeout', 'test_error', 'pool_error']:
                        error_count += 1
                    
                    # Aggregate test statistics
                    if result.get('test_stats'):
                        stats = result['test_stats']
                        total_test_cases += stats.get('total_tests', 0)
                        total_passed_tests += stats.get('passed_tests', 0)
                        total_failed_tests += stats.get('failed_tests', 0)
                        total_error_tests += stats.get('error_tests', 0)
                        total_skipped_tests += stats.get('skipped_tests', 0)
                        tasks_with_stats += 1
                        
                        # Calculate proportional correctness for this task
                        task_total_tests = stats.get('total_tests', 0)
                        if task_total_tests > 0 :
                            task_correct_ratio = stats.get('passed_tests', 0) / task_total_tests
                            total_proportional_correct += task_correct_ratio
                        else:
                            # If test_cases total is 0, set to 0 (compilation failed)
                            total_proportional_correct += 0.0
                    else:
                        # No test stats available, only if examined as passed, will be treated as 1 correctness
                        if result['passed']:
                            total_proportional_correct += 1.0
                        
                    pbar.update(1)
                    
                except Exception as e:
                    log_print(f"Error processing result: {e}")
                    error_count += 1
                    pbar.update(1)
    
    # Calculate pass rate
    total_count = data_count
    pass_rate = pass_count / total_count if total_count > 0 else 0
    
    # Calculate aggregated test statistics
    aggregated_stats = {
        'total_tasks': total_count,
        'tasks_with_stats': tasks_with_stats,
        'total_test_cases': total_test_cases,
        'total_passed_tests': total_passed_tests,
        'total_failed_tests': total_failed_tests,
        'total_error_tests': total_error_tests,
        'total_skipped_tests': total_skipped_tests,
        'test_case_pass_rate': total_passed_tests / total_test_cases if total_test_cases > 0 else 0,
        'total_proportional_correct': total_proportional_correct,
        'avg_proportional_correct': total_proportional_correct / total_count if total_count > 0 else 0
    }
    
    return pass_rate, pass_count, total_count, error_count, detailed_results, aggregated_stats

def print_summary_results_with_stats(file_info, pass_rate, pass_count, total_count, error_count, aggregated_stats):
    """Print summary results with test statistics to console"""
    log_print("\n" + "="*60)
    log_print("EVALUATION RESULTS WITH TEST STATISTICS")
    log_print("="*60)
    log_print(f"File Information:")
    log_print(f"  Task Type:        {file_info['task_type']}")
    log_print(f"  Knowledge Type:   {file_info['knowledge_type']}")
    log_print(f"  Ban Deprecation:  {file_info['ban_deprecation']}")
    log_print(f"  Benchmark File:   {file_info['benchmark_file']}")
    log_print(f"\nTask-Level Results:")
    log_print(f"  Total Tasks:      {total_count}")
    log_print(f"  Passed Tasks:     {pass_count}")
    log_print(f"  Failed Tasks:     {total_count - pass_count}")
    log_print(f"  Error Tasks:      {error_count}")
    log_print(f"  Task Pass Rate:   {pass_rate:.4f} ({pass_rate*100:.2f}%)")
    
    log_print(f"\nTest Case-Level Statistics:")
    log_print(f"  Tasks with Stats: {aggregated_stats['tasks_with_stats']}/{aggregated_stats['total_tasks']}")
    log_print(f"  Total Test Cases: {aggregated_stats['total_test_cases']}")
    log_print(f"  Passed Tests:     {aggregated_stats['total_passed_tests']}")
    log_print(f"  Failed Tests:     {aggregated_stats['total_failed_tests']}")
    log_print(f"  Error Tests:      {aggregated_stats['total_error_tests']}")
    log_print(f"  Skipped Tests:    {aggregated_stats['total_skipped_tests']}")
    log_print(f"  Test Case Pass Rate: {aggregated_stats['test_case_pass_rate']:.4f} ({aggregated_stats['test_case_pass_rate']*100:.2f}%)")
    
    log_print(f"\nProportional Correctness Statistics:")
    log_print(f"  Total Proportional Correct: {aggregated_stats['total_proportional_correct']:.4f}")
    log_print(f"  Avg Proportional Correct:   {aggregated_stats['avg_proportional_correct']:.4f} ({aggregated_stats['avg_proportional_correct']*100:.2f}%)")
    
    if aggregated_stats['tasks_with_stats'] > 0:
        avg_tests_per_task = aggregated_stats['total_test_cases'] / aggregated_stats['tasks_with_stats']
        log_print(f"  Avg Tests/Task:   {avg_tests_per_task:.1f}")
    
    log_print("="*60)

def print_detailed_errors_with_stats(detailed_results):
    """Print detailed error information and test statistics for failed tests"""
    failed_results = [r for r in detailed_results if not r['passed']]
    
    if failed_results:
        log_print(f"\n{'='*60}")
        log_print("DETAILED ERROR INFORMATION WITH TEST STATISTICS")
        log_print("="*60)
        
        for result in failed_results:
            log_print(f"\nTask ID: {result['task_id']}")
            log_print(f"Error Type: {result['error_type']}")
            log_print(f"Message: {result['message']}")
            
            # Print test statistics if available
            if result.get('test_stats'):
                stats = result['test_stats']
                task_total_tests = stats.get('total_tests', 0)
                task_passed_tests = stats.get('passed_tests', 0)
                
                # Calculate proportional correctness for this task
                if task_total_tests > 0:
                    proportional_correct = task_passed_tests / task_total_tests
                else:
                    proportional_correct = 0.0
                
                log_print(f"Test Statistics:")
                log_print(f"  - Total Tests: {task_total_tests}")
                log_print(f"  - Passed: {task_passed_tests}")
                log_print(f"  - Failed: {stats.get('failed_tests', 0)}")
                log_print(f"  - Errors: {stats.get('error_tests', 0)}")
                log_print(f"  - Skipped: {stats.get('skipped_tests', 0)}")
                log_print(f"  - Success Rate: {stats.get('success_rate', 0):.2%}")
                log_print(f"  - Proportional Correct: {proportional_correct:.4f}")
                log_print(f"  - Execution Time: {stats.get('execution_time', 0):.3f}s")
            else:
                log_print(f"Test Statistics: No stats available (proportional correct = 0.0)")
            
            if result.get('full_traceback'):
                log_print("Full Traceback:")
                log_print(result['full_traceback'])
            
            if result.get('deprecation_info'):
                log_print("Deprecation Info:")
                log_print(result['deprecation_info'])
            
            if result.get('test_error_details'):
                log_print("Test Error Details:")
                log_print(result['test_error_details'])
            
            log_print("-" * 40)
        
        log_print("="*60)

def save_detailed_results_with_stats(detailed_results, file_info, pass_rate, pass_count, total_count, error_count, 
                                    aggregated_stats, prediction_file, output_file=None):
    """Save detailed results with test statistics to JSON file"""
    
    if output_file is None:
        # Create output filename based on input filename
        base_name = os.path.splitext(os.path.basename(prediction_file))[0]
        output_file = f"{base_name}_detailed_results_with_stats.json"
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Prepare summary data
    summary = {
        "evaluation_info": {
            "source_file": prediction_file,
            "knowledge_type": file_info['knowledge_type'],
            "task_type": file_info['task_type'],
            "ban_deprecation": file_info['ban_deprecation'],
            "benchmark_file": file_info['benchmark_file'],
            "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "task_level_statistics": {
            "total_tasks": total_count,
            "passed_tasks": pass_count,
            "failed_tasks": total_count - pass_count,
            "error_tasks": error_count,
            "task_pass_rate": pass_rate
        },
        "test_case_level_statistics": aggregated_stats,
        "detailed_results": detailed_results
    }
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    log_print(f"\nDetailed results with test statistics saved to: {output_file}")
    return output_file

def getCleanFuncByStr(function_name):
    if function_name=='basic':
        return clean_model_output
    elif function_name=='loose':
        return clean_model_output1
    elif function_name=='lora':
        return clean_model_output_lora
    elif function_name=='review':
        return clean_model_output_review
    elif function_name=='errorfix':
        return clean_model_output_errorfix 
    elif function_name=='py_wrapper_first':
        return clean_model_output_py_wrapper_first
    else:
        raise ValueError(f"Invalid function name: {function_name}")

def main():
    global TIMEOUT
    TIMEOUT=200
    parser = argparse.ArgumentParser(description='单文件评估 (带测试统计) - 对指定的预测文件进行快速评估，并返回详细的测试用例统计')
    parser.add_argument('--prediction_file', type=str, default='/datanfs2/chenrongyi/codes/EM-LLM-model/output/approach_eval/MEMORY/Llama-3.1-8B/versibcb_vscc_gen_docstring_emb_local_4000_Llama-3.1-8B_n_init128_n_mem4096_n_local4096_repr_topk4_block_size128_chunk_size512.jsonl',
                       help='要评估的预测文件路径 (JSON or JSONL)')
    parser.add_argument('--benchmark_dir', type=str, default='data/output/VersiBCB_Benchmark',
                       help='基准数据目录')
    parser.add_argument('--benchmark_file', type=str, default='vace_datas.json',
                       help='指定基准文件,需提供文件名')
    parser.add_argument('--ban_deprecation', type=bool, default=None,
                       help='是否禁用过时API(可选，如果不指定将根据文件名自动确定)')
    parser.add_argument('--output_file','-of', type=str, default=None,
                       help='详细结果输出文件路径 (JSON格式，可选)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大评估样本数量 (用于快速测试)')
    parser.add_argument('--num_processes', type=int, default=4,
                       help='并行进程数量')
    parser.add_argument('--no_save', action='store_true',
                       help='不保存详细结果文件')
    parser.add_argument('--clean_func','-cf', type=str, default='lora',choices=['basic','loose','lora','review','errorfix','py_wrapper_first'],
                       help='清理函数，可选值: basic, loose, lora, review, errorfix, py_wrapper_first')
    parser.add_argument('--log_file', type=str, default=None,
                       help='日志文件路径 (可选，如果不指定将根据预测文件名自动生成)')
    parser.add_argument('--task_ids', type=str, default=None,
                       help='指定要检测的task ID，用逗号分隔 (例如: "1,2,3" 或 "task_1,task_2")，如果不指定则检测全部')
    parser.add_argument('--task_ids_file','-tidf', type=str, default=None,
                       help='包含task IDs的文件路径，为1个列表')
    args = parser.parse_args()
    
    # Setup log file path
    if args.log_file is None:
        base_name = os.path.splitext(os.path.basename(args.prediction_file))[0]
        args.log_file = f"{base_name}_evaluation_with_stats.log"
    
    try:
        # Setup logging with stream redirection
        setup_logging_with_redirect(args.log_file)
        log_print(f"Evaluation with test statistics started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_print(f"Log file: {args.log_file}")
        
        # Check if prediction file exists
        if not os.path.exists(args.prediction_file):
            log_print(f"错误: 预测文件不存在: {args.prediction_file}")
            return
        
        log_print(f"加载预测文件: {args.prediction_file}")
        
        # Extract file information
        file_info = extract_info_from_filename(args.prediction_file)
        
        # Override with manual settings if provided
        if args.benchmark_file:
            file_info['benchmark_file'] = args.benchmark_file
        if args.ban_deprecation is not None:
            file_info['ban_deprecation'] = args.ban_deprecation
        
        log_print(f"文件信息: {file_info}")
        
        # Load benchmark data
        benchmark_path = os.path.join(args.benchmark_dir, file_info['benchmark_file'])
        if not os.path.exists(benchmark_path):
            log_print(f"错误: 基准文件不存在: {benchmark_path}")
            return
        
        log_print(f"加载基准数据: {benchmark_path}")
        benchmark_data = load_benchmark_data(benchmark_path)
        log_print(f"基准数据包含 {len(benchmark_data)} 个任务")
        
        # Load prediction data
        prediction_data = load_prediction_data(args.prediction_file)
        log_print(f"预测数据包含 {len(prediction_data)} 个样本")
        
        # Filter by task IDs if specified
        if args.task_ids or args.task_ids_file:
            # Parse task IDs
            specified_task_ids = []
            if args.task_ids_file:
                try:
                    with open(args.task_ids_file, 'r', encoding='utf-8') as f:
                        specified_task_ids = json.load(f)
                    log_print(f"从文件 {args.task_ids_file} 读取的Task IDs: {specified_task_ids}")
                except Exception as e:
                    log_print(f"读取task_ids文件失败: {str(e)}")
                    return
            else:
                specified_task_ids = [task_id.strip() for task_id in args.task_ids.split(',')]
                log_print(f"指定的Task IDs: {specified_task_ids}")
            
            # Filter prediction data
            original_count = len(prediction_data)
            prediction_data = [data for data in prediction_data if data.get('id', '') in specified_task_ids]
            
            if len(prediction_data) == 0:
                log_print(f"警告: 没有找到指定的Task IDs，请检查输入的ID是否正确")
                log_print(f"可用的Task IDs示例: {[data.get('id', 'unknown') for data in load_prediction_data(args.prediction_file)[:5]]}")
                return
            
            log_print(f"根据指定Task IDs过滤后: {len(prediction_data)} 个样本 (从 {original_count} 个中筛选)")
        
        # Limit samples if specified (applied after task ID filtering)
        if args.max_samples and len(prediction_data) > args.max_samples:
            prediction_data = prediction_data[:args.max_samples]
            log_print(f"限制为 {args.max_samples} 个样本进行测试")
        
        # Run evaluation with test statistics
        log_print(f"\n开始评估...")
        log_print(f"任务类型: {file_info['task_type']}")
        log_print(f"知识类型: {file_info['knowledge_type']}")
        log_print(f"禁用过时API: {file_info['ban_deprecation']}")
        log_print(f"实际评估样本数: {len(prediction_data)}")
        
        pass_rate, pass_count, total_count, error_count, detailed_results, aggregated_stats = evaluate_single_file_with_stats(
            prediction_data, benchmark_data, file_info['ban_deprecation'], args.num_processes, getCleanFuncByStr(args.clean_func)
        )
        
        # Print summary with test statistics
        print_summary_results_with_stats(file_info, pass_rate, pass_count, total_count, error_count, aggregated_stats)
        
        # Print detailed errors with test statistics
        print_detailed_errors_with_stats(detailed_results)
        
        # Save detailed results with test statistics if requested
        if not args.no_save:
            save_detailed_results_with_stats(
                detailed_results, file_info, pass_rate, pass_count, total_count, error_count, aggregated_stats,
                args.prediction_file, args.output_file
            )
        
        log_print(f"\nEvaluation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always restore original streams
        restore_streams()

if __name__ == '__main__':
    main() 