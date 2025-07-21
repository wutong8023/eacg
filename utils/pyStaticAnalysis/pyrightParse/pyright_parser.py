import subprocess
import os
import sys
import json
import tempfile
import datetime
import threading
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import logging

@dataclass
class PyrightDiagnostic:
    """Pyright诊断信息数据结构"""
    file: str
    severity: str
    message: str
    range: Dict[str, Dict[str, int]]  # start/end line and character
    rule: Optional[str] = None
    code: Optional[str] = None

@dataclass
class PyrightResult:
    """Pyright分析结果"""
    has_error: bool
    diagnostics: List[PyrightDiagnostic]
    raw_json: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None

class PyrightParser:
    """Pyright代码分析器，返回原始结果"""
    
    def __init__(self, enable_logging: bool = True):
        """
        初始化Pyright解析器
        
        Args:
            enable_logging: 是否启用日志记录
        """
        if enable_logging:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def ensure_pyright_installed(self, venv_dir: str, target_dependency: Dict[str, str]) -> bool:
        """
        确保在指定conda环境中安装了pyright和必要的stubs
        
        Args:
            venv_dir: conda环境目录
            target_dependency: 目标依赖信息
        
        Returns:
            bool: 是否成功安装或已存在
        """
        # 获取conda环境中的pip路径
        pip_executable = os.path.join(venv_dir, 'bin', 'pip') if os.name != 'nt' else os.path.join(venv_dir, 'Scripts', 'pip.exe')
        pyright_executable = os.path.join(venv_dir, 'bin', 'pyright') if os.name != 'nt' else os.path.join(venv_dir, 'Scripts', 'pyright.exe')
        
        if os.path.exists(pyright_executable):
            # 检查是否需要安装matplotlib-stubs
            if 'matplotlib' in target_dependency:
                try:
                    # 使用conda run命令在指定环境中安装matplotlib-stubs
                    conda_run_cmd = ['conda', 'run', '-p', venv_dir, 'pip', 'install', 'matplotlib-stubs']
                    result = subprocess.run(conda_run_cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        self.logger.warning(f"Failed to install matplotlib-stubs: {result.stderr}")
                except Exception as e:
                    self.logger.warning(f"Error installing matplotlib-stubs: {str(e)}")
            return True
            
        try:
            # 使用conda run命令在指定环境中安装pyright
            conda_run_cmd = ['conda', 'run', '-p', venv_dir, 'pip', 'install', 'pyright']
            result = subprocess.run(conda_run_cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and 'matplotlib' in target_dependency:
                # 安装matplotlib-stubs
                conda_run_cmd = ['conda', 'run', '-p', venv_dir, 'pip', 'install', 'matplotlib-stubs']
                result = subprocess.run(conda_run_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    self.logger.warning(f"Failed to install matplotlib-stubs: {result.stderr}")
            
            return True
        except Exception as e:
            self.logger.error(f"安装pyright时出错: {str(e)}")
            return False
    
    def format_error_info(self, error_json: str) -> str:
        """
        格式化pyright的JSON输出，提取关键错误信息
        
        Args:
            error_json: pyright输出的JSON字符串
        
        Returns:
            str: 格式化后的错误信息
        """
        try:
            data = json.loads(error_json)
            if not data.get('generalDiagnostics'):
                return "No errors found"
                
            formatted_errors = []
            for diagnostic in data['generalDiagnostics']:
                file_path = diagnostic['file']
                severity = diagnostic['severity']
                message = diagnostic['message']
                start_line = diagnostic['range']['start']['line'] + 1  # 转换为1-based行号
                start_char = diagnostic['range']['start']['character']
                
                # 读取错误行内容
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if start_line <= len(lines):
                            error_line = lines[start_line - 1].rstrip()
                            formatted_errors.append(
                                f"Line {start_line}: {error_line}\n"
                                f"Error: {message}\n"
                            )
                except Exception as e:
                    formatted_errors.append(f"Error reading file {file_path}: {str(e)}")
            
            return "\n".join(formatted_errors)
        except json.JSONDecodeError:
            return error_json
        except Exception as e:
            return f"Error processing pyright output: {str(e)}"
    
    def run_pyright_analysis(self, venv_dir: str, file_path: str, target_dependency: Dict[str, str]) -> Tuple[bool, str, Optional[List[Dict]]]:
        """
        在指定conda环境中运行pyright分析指定文件
        
        Args:
            venv_dir: conda环境目录
            file_path: 要分析的文件路径
            target_dependency: 目标依赖信息
        
        Returns:
            Tuple[bool, str, Optional[List[Dict]]]: (has_error, error_info, raw_diagnostics)
        """
        # 确保pyright已安装
        if not self.ensure_pyright_installed(venv_dir, target_dependency):
            return True, "Failed to install pyright in the conda environment", None
        
        # 获取conda环境中的pyright路径
        pyright_executable = os.path.join(venv_dir, 'bin', 'pyright') if os.name != 'nt' else os.path.join(venv_dir, 'Scripts', 'pyright.exe')
        
        if not os.path.exists(pyright_executable):
            return True, f"Pyright executable not found in {venv_dir} after installation attempt", None
        
        try:
            # 使用conda run命令在指定环境中运行pyright
            conda_run_cmd = ['conda', 'run', '-p', venv_dir, 'pyright', '--outputjson', file_path]
            
            result = subprocess.run(
                conda_run_cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60秒超时
            )
            
            has_error = result.returncode != 0
            if has_error:
                try:
                    error_info = self.format_error_info(result.stdout)
                    print(error_info)
                except Exception as e:
                    error_info = result.stdout
                    print(f"Error formatting error info: {str(e)}")
            else:
                error_info = ""
            
            # 解析原始诊断信息
            raw_diagnostics = None
            if has_error:
                try:
                    data = json.loads(result.stdout)
                    if 'generalDiagnostics' in data:
                        raw_diagnostics = data['generalDiagnostics']
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse pyright output: {str(e)}")
            
            return has_error, error_info, raw_diagnostics
            
        except subprocess.TimeoutExpired:
            return True, "Pyright analysis timed out after 60 seconds", None
        except subprocess.CalledProcessError as e:
            return True, f"Error running pyright: {str(e)}", None
        except Exception as e:
            return True, f"Unexpected error: {str(e)}", None
    
    def analyze_code_string(self, code: str, venv_dir: str, target_dependency: Dict[str, str], filename: str = "temp.py") -> Tuple[bool, str, Optional[List[Dict]]]:
        """
        分析代码字符串
        
        Args:
            code: 要分析的代码字符串
            venv_dir: conda环境目录
            target_dependency: 目标依赖信息
            filename: 临时文件名
        
        Returns:
            Tuple[bool, str, Optional[List[Dict]]]: (has_error, error_info, raw_diagnostics)
        """
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name
        
        try:
            # 运行pyright分析
            return self.run_pyright_analysis(venv_dir, temp_file_path, target_dependency)
        finally:
            # 清理临时文件
            try:
                os.unlink(temp_file_path)
            except:
                pass
    
    def _format_diagnostic_to_error_info(self, diagnostic: Dict[str, Any], code_lines: List[str]) -> Dict[str, str]:
        """
        将诊断信息格式化为错误信息字典
        
        Args:
            diagnostic: 诊断信息字典
            code_lines: 代码行列表
        
        Returns:
            Dict[str, str]: 格式化的错误信息
        """
        # 获取行号和列号
        start_line = diagnostic['range']['start']['line'] + 1  # 转换为1-based
        start_char = diagnostic['range']['start']['character']
        
        # 获取错误行代码
        error_line = ""
        if start_line <= len(code_lines):
            error_line = code_lines[start_line - 1].rstrip()
        
        # 构建错误信息字符串
        error_info = f"Line {start_line}: {error_line}\nError: {diagnostic['message']}"
        
        return {
            "error_info": error_info,
            "tool": "pyright",
            "rule": diagnostic.get('rule', 'unknown')
        }
    
    def get_error_info_from_pyright(self, generated_code: str, target_dependency: Dict[str, str], return_list: bool = True) -> List[Dict[str, str]]:
        """
        从pyright获取错误信息，返回格式化的错误信息列表
        与testPyright_parallel.py中的get_error_info_from_pyright函数兼容
        
        Args:
            generated_code: 生成的代码
            target_dependency: 目标依赖信息
            return_list: 是否返回列表格式（兼容参数）
        
        Returns:
            List[Dict[str, str]]: 错误信息列表，每个元素包含error_info、tool、rule字段
        """
        from utils.pyStaticAnalysis.testmypy_utils import getTargetEnvPath
        
        venv_dir = getTargetEnvPath(target_dependency)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(generated_code)
            temp_file_path = temp_file.name
        
        try:
            # 运行pyright分析
            has_error, error_info, raw_diagnostics = self.run_pyright_analysis(venv_dir, temp_file_path, target_dependency)
            
            # 添加f-string错误检查
            from utils.pyStaticAnalysis.fstring35detect import getFStringErrorInfo
            fstring_errors = getFStringErrorInfo(generated_code, target_dependency)
            
            # 格式化错误信息
            error_infos = []
            
            # 处理pyright错误
            if has_error and raw_diagnostics:
                code_lines = generated_code.splitlines()
                for diagnostic in raw_diagnostics:
                    if diagnostic['severity'] == 'error':  # 只处理error级别的诊断
                        error_dict = self._format_diagnostic_to_error_info(diagnostic, code_lines)
                        error_infos.append(error_dict)
            
            # 添加f-string错误
            if fstring_errors:
                for fstring_error in fstring_errors:
                    error_infos.append({
                        "error_info": fstring_error,
                        "tool": "pyright",
                        "rule": "f-string"
                    })
            
            return error_infos
            
        finally:
            # 清理临时文件
            try:
                os.unlink(temp_file_path)
            except:
                pass

def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="Pyright代码分析器，返回combined_errors_vscc.json格式的结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python pyright_parser.py jsonl_file benchmark_file output_file --max-workers 8
        """
    )
    
    parser.add_argument(
        "jsonl_file",
        help="JSONL文件路径，包含id和answer字段"
    )
    
    parser.add_argument(
        "benchmark_file", 
        help="benchmark文件路径，包含target_dependency"
    )
    
    parser.add_argument(
        "output_file",
        help="输出文件路径"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="最大工作线程数 (默认: 4)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="启用详细日志输出"
    )
    
    return parser

def get_existing_results(output_file):
    """
    获取已经处理过的结果
    
    Args:
        output_file: 输出文件路径
    
    Returns:
        set: 已处理的id集合
    """
    existing_ids = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    if 'id' in item:
                        existing_ids.add(item['id'])
        except Exception as e:
            print(f"Warning: Error reading existing results: {str(e)}")
    return existing_ids

def clean_model_output(output):
    """
    清理模型输出，移除markdown标记等
    """
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
    """
    读取benchmark文件，返回id到target_dependency的映射
    """
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

def process_single_item(item_data, id_to_dependency, parser, lock):
    """
    处理单个测试项
    
    Args:
        item_data: 包含id和answer的数据
        id_to_dependency: id到target_dependency的映射
        parser: PyrightParser实例
        lock: 文件写入锁
    
    Returns:
        Optional[Dict]: 处理后的结果，如果失败则返回None
    """
    try:
        item_id = item_data['id']
        answer = item_data['answer']
        
        # 获取对应的依赖信息
        if item_id not in id_to_dependency:
            print(f"Warning: No dependency found for id {item_id}")
            return None
        
        target_dependency = id_to_dependency[item_id]
        
        # 清理模型输出
        clean_code = clean_model_output(answer)
        from utils.pyStaticAnalysis.clean_utils import clean_annotations
        clean_code = clean_annotations(clean_code)
        
        # 获取错误信息
        error_infos = parser.get_error_info_from_pyright(clean_code, target_dependency)
        
        # 只返回有错误的项
        if error_infos:
            return {
                "id": item_id,
                "error_infos": error_infos
            }
        
        return None
        
    except Exception as e:
        print(f"Error processing item {item_data.get('id', 'unknown')}: {str(e)}")
        return None

def process_jsonl_with_benchmark(jsonl_file_path, benchmark_file_path, output_file, max_workers=4):
    """
    从JSONL文件读取代码，从benchmark文件读取依赖，使用多线程测试代码并保存结果
    
    Args:
        jsonl_file_path: JSONL文件路径，包含id和answer字段
        benchmark_file_path: benchmark文件路径，包含target_dependency
        output_file: 输出文件路径
        max_workers: 最大线程数
    
    Returns:
        tuple: (成功测试数量, 总数量)
    """
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
    
    # 创建解析器和锁
    parser = PyrightParser(enable_logging=False)
    file_lock = threading.Lock()
    
    # 存储结果
    results = []
    
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
                parser,
                file_lock
            ): item for item in test_items
        }
        
        # 处理完成的任务
        for i, future in enumerate(as_completed(future_to_item), 1):
            item = future_to_item[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    success_count += 1
                # 打印进度
                if i % 10 == 0 or i == total_count:
                    elapsed = time.time() - start_time
                    print(f"Progress: {i}/{total_count} ({(i/total_count*100):.1f}%) - "
                          f"Success: {success_count}/{i} - "
                          f"Time elapsed: {elapsed:.1f}s")
            except Exception as e:
                print(f"Error processing item {item.get('id', 'unknown')}: {str(e)}")
    
    # 加载现有结果并合并
    existing_results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
        except Exception as e:
            print(f"Warning: Error reading existing results: {str(e)}")
    
    # 合并结果
    all_results = existing_results + results
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    total_time = time.time() - start_time
    print(f"\nProcessing completed in {total_time:.1f} seconds")
    print(f"Successfully processed {success_count} out of {total_count} code snippets")
    print(f"Results saved to {output_file}")
    
    return success_count, total_count

def main():
    """命令行入口点"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # 验证文件是否存在
    if not os.path.exists(args.jsonl_file):
        print(f"Error: JSONL file not found: {args.jsonl_file}")
        sys.exit(1)
    
    if not os.path.exists(args.benchmark_file):
        print(f"Error: Benchmark file not found: {args.benchmark_file}")
        sys.exit(1)
    
    print("Processing JSONL file with benchmark data...")
    success_count, total_count = process_jsonl_with_benchmark(
        args.jsonl_file,
        args.benchmark_file,
        args.output_file,
        max_workers=args.max_workers
    )
    
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main() 