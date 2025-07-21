from datasets import load_from_disk
import json
import chromadb
import hashlib
import os
from sentence_transformers import SentenceTransformer
import re
import tiktoken
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.docTokenDistribute import distribute_doc_tokens,truncate_doc
from benchmark.config.code.config import (
    VSCC_LOW_BOUND, VSCC_HIGH_BOUND, RAG_DOCUMENT_NUM,
    FC_MAX_TOKEN_LENGTH, RAG_MAX_TOKEN_LENGTH, TOGETHER_API_KEY_PATH,
    RAG_COLLECTION_BASE,
    LOCAL_EMBEDDING_MODEL, TOGETHERAI_EMBEDDING_MODEL,
    DOCSTRING_EMBEDDING_BASE_PATH, SRCCODE_EMBEDDING_BASE_PATH,
    DOCSTRING_CORPUS_PATH, SRCCODE_CORPUS_PATH,QUERY_CACHE_DIR
)
import torch
from chromadb.api.types import Documents, EmbeddingFunction
import time
import pickle
import numpy as np
from utils.RAGutils.RAGEmbedding import PrecomputedEmbeddingsManager, CustomEmbeddingFunction
from utils.RAGutils.document_utils import truncate_context, truncate_BCB_context, get_version
from utils.prompt_utils import format_prompt
from utils.RAGutils.RAGRetriever import RAGContextRetriever, getKnowledgeDocs
from utils.loraTrain.loraTrainUtils import inference
from benchmark.config.code.VersiBCB_compress import VersiBCB_VACE_RAGCompress_complete,VersiBCB_VACE_RAGCompress_instruct
# from utils.infer import inference
import traceback
from chromadb.config import Settings
import logging
import argparse # Added for argument parsing
from accelerate import Accelerator # Import Accelerator
from together import Together
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.output_manager import OutputManager
import fcntl  # 添加文件锁支持
import shutil  # 添加文件操作支持
import tempfile  # 添加临时文件支持
import threading  # 添加线程锁支持

def formatInputAndInfer(item_data, generated_code, error_info, args, error_fix_mode, retrieved_info, context):
    '''
    合并format_prompt和inference两个操作，支持本地和远程推理
    
    Args:
        item_data: 包含样本数据的元组
        generated_code: 已生成的代码（用于多轮推理）
        error_info: 错误信息（用于error_fix模式）
        args: 参数命名空间
        error_fix_mode: 是否为错误修复模式
        retrieved_info: 检索到的信息
        context: 上下文信息
    
    Returns:
        response_text: 推理结果文本
    '''
    global local_model, local_tokenizer
    
    i, data, current_dataset, current_task, current_ban_deprecation, current_model_path, current_approach, max_tokens, max_new_tokens, review_mode, generated_code_dict, error_infos_dict, retrieved_info_dict = item_data
    sample_id = data.get('id', f'index_{i}')
    rank = getattr(args, 'rank', 0)
    
    try:
        # 处理本地推理
        if args.inference_type == "local":
            if local_model is None or local_tokenizer is None:
                raise RuntimeError("Local model or tokenizer not initialized for local inference.")
            
            model_max_length = getattr(local_model.config, "max_position_embeddings", None)
            if model_max_length is None:
                logging.error(f"Worker {rank}: max_position_embeddings not found in model config for {current_model_path}. Cannot proceed for sample {sample_id}.")
                raise RuntimeError("max_position_embeddings not found in model config.")
            
            desired_max_new_tokens = max_new_tokens
            buffer_for_model_specific_tokens = 5
            max_len_for_prompt_itself = model_max_length - desired_max_new_tokens - buffer_for_model_specific_tokens
            
            if max_len_for_prompt_itself <= 0:
                logging.error(f"Worker {rank}: Model max length {model_max_length} is too short. Sample ID: {sample_id}")
                return {
                    "id": sample_id, 
                    "error": "Model max length too short", 
                    "traceback": ""
                }

            # 处理generated_code_dict的类型转换
            generated_code_dict = {int(k): v for k, v in generated_code_dict.items()}
            
            # 获取generated_target_code用于review模式
            # 在多轮推理中，优先使用传入的generated_code参数
            generated_target_code = None
            if generated_code and generated_code.strip():  # 如果传入了有效的generated_code
                generated_target_code = generated_code
            elif review_mode and sample_id in generated_code_dict:
                generated_target_code = generated_code_dict[sample_id]
            elif review_mode:
                logging.warning(f"Worker {rank}: Review mode enabled but no generated code found for sample with id: {sample_id}")
            
            # 格式化输入
            input_for_api = format_prompt(
                data, context, current_dataset, current_task, current_ban_deprecation,
                tokenizer=local_tokenizer, 
                max_prompt_token_length=max_len_for_prompt_itself,
                truncate=False,
                review_mode=review_mode,
                generated_target_code=generated_target_code,
                error_fix_mode=error_fix_mode,
                error_info=error_info,
                retrieved_info=retrieved_info
            )
            logging.debug(f"Worker {rank}: input_for_api: {input_for_api}")
            
            # 检查prompt长度
            actual_prompt_token_ids = local_tokenizer.encode(input_for_api, add_special_tokens=True)
            actual_prompt_length = len(actual_prompt_token_ids)

            if actual_prompt_length + desired_max_new_tokens > model_max_length:
                error_message = (
                    f"Actual prompt length ({actual_prompt_length}) + desired_max_new_tokens ({desired_max_new_tokens}) "
                    f"exceeds model_max_length ({model_max_length}). Sample ID: {sample_id}."
                )
                logging.error(f"Worker {rank}: {error_message}")
                return {
                    "id": sample_id, 
                    "error": "Prompt too long for desired_max_new_tokens", 
                    "traceback": f"Details: Prompt: {actual_prompt_length}, Desired new: {desired_max_new_tokens}, Model max: {model_max_length}"
                }
            
            # 执行推理
            stop_tokens = args.stop_tokens
            logging.info(f"stop tokens: {stop_tokens}")
            response_text = inference(
                local_model, local_tokenizer, input_for_api, 
                max_new_tokens=desired_max_new_tokens, temperature=args.temperature, top_p=args.top_p, 
                inference_type="local",
                stop_tokens=stop_tokens
            )
            
        else:  # 远程推理 (huggingface or togetherai)
            # 处理generated_code_dict的类型转换
            generated_code_dict = {int(k): v for k, v in generated_code_dict.items()}
            
            # 获取generated_target_code用于review模式
            # 在多轮推理中，优先使用传入的generated_code参数
            generated_target_code = None
            if generated_code and generated_code.strip():  # 如果传入了有效的generated_code
                generated_target_code = generated_code
            elif review_mode and sample_id in generated_code_dict:
                generated_target_code = generated_code_dict[sample_id]
            elif review_mode:
                logging.warning(f"Worker {rank}: Review mode enabled but no generated code found for sample {sample_id}")
            
            # 格式化输入（远程推理使用None作为tokenizer）
            input_for_api = format_prompt(
                data, context, current_dataset, current_task, current_ban_deprecation,
                tokenizer=None,  # 远程推理不需要本地tokenizer
                max_prompt_token_length=args.max_tokens,  # 使用通用的max_token限制
                truncate=True,  # 允许截断以确保安全，API也可能截断
                review_mode=review_mode,
                generated_target_code=generated_target_code,
                error_fix_mode=error_fix_mode,
                error_info=error_info,
                retrieved_info=retrieved_info
            )
            
            # 远程推理的token计算复杂且依赖API，所以依赖API限制
            desired_max_new_tokens = max_new_tokens

            # 执行远程推理
            response_text = inference(
                model=None, tokenizer=None, prompt=input_for_api,
                max_new_tokens=desired_max_new_tokens, temperature=args.temperature, top_p=args.top_p,
                inference_type=args.inference_type,
                api_key=args.api_key,
                model_name=args.api_model_name,
                api_base_url=getattr(args, 'huggingface_api_base_url', None)
            )
        
        return response_text
        
    except Exception as e:
        tb_str = traceback.format_exc()
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "sample_id": sample_id,
            "error_location": "formatInputAndInfer"
        }
        logging.error(f"Worker {rank}: Error in formatInputAndInfer for sample {sample_id}: {error_details}")
        logging.error(f"Worker {rank}: Full traceback:\n{tb_str}")
        return {
            "id": sample_id, 
            "error": str(e), 
            "traceback": tb_str, 
            "error_details": error_details
        }

def parseOutput2Code(output):
    '''
        用于finalcode的场景,找到<finalcode>和</finalcode>包裹的代码。然后再获取```python```包裹的代码进行处理
        #TODO:实际上是mvBCBbuilder的code，可以利用subfolder取消该调用
    '''
    # 获取<finalcode>和</finalcode>包裹的代码
    finalcode_start_index = output.find("<finalcode>") + len("<finalcode>")
    finalcode_end_index = output.find("</finalcode>")
    if finalcode_start_index != -1 and finalcode_end_index != -1:
        finalcode = output[finalcode_start_index:finalcode_end_index]
    else:
        finalcode = output
    # 获取```python```包裹的代码
    if "```python" in finalcode:
        start_marker = "```python"
        end_marker = "```"
        start_index = finalcode.find(start_marker) + len(start_marker)
        remaining_text = finalcode[start_index:]
        end_index = remaining_text.find(end_marker)
        if end_index != -1:
            finalcode = remaining_text[:end_index]
        else:
            finalcode = remaining_text
    return finalcode
from utils.pyStaticAnalysis.testmypy_utils import get_error_info_from_mypy
from utils.pyStaticAnalysis.testPyright_parallel import get_error_info_from_pyright
def getErrorInfoFromStaticAnalyser(static_analyser_name,generated_code,target_dependency):
    '''
    从static_analyser_name中获取error_info,使用mypy进行
    params:
        static_analyser_name: 静态分析器名称
        generated_code: 生成的代码
    return:
        error_info: 错误信息,list[str]或list[dict]
    '''
    if static_analyser_name == "mypy":
        return get_error_info_from_mypy(generated_code,target_dependency)
    elif static_analyser_name == "pyright":
        return get_error_info_from_pyright(generated_code,target_dependency,return_list=True)
    else:
        raise ValueError(f"Unsupported static analyser: {static_analyser_name}")
# Logging configurations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chromadb_processing_local.log', mode='a'), # Changed log file name
        logging.StreamHandler()
    ]
)

# Global variables for clients and model
chroma_client = None
rag_retriever = None
local_model = None
local_tokenizer = None
together_client = None # Keep for potential 'togetherai' embedding source
accelerator = None # Global accelerator object
file_handler = None # 全局文件处理器实例

class MultiprocessSafeFileHandler:
    """
    多进程安全的文件处理器，支持原子写入、重试机制和备份恢复
    """
    
    def __init__(self, file_path, max_retries=3, retry_delay=1.0, backup_enabled=True):
        """
        初始化文件处理器
        
        Args:
            file_path (str): 目标文件路径
            max_retries (int): 最大重试次数
            retry_delay (float): 重试延迟（秒）
            backup_enabled (bool): 是否启用备份功能
        """
        self.file_path = file_path
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backup_enabled = backup_enabled
        self.backup_path = f"{file_path}.backup"
        
        # 线程锁，确保同一进程内的线程安全
        self._thread_lock = threading.Lock()
        
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 如果文件不存在，创建空文件
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                pass  # 创建空文件
    
    def _create_backup(self):
        """创建备份文件"""
        if self.backup_enabled and os.path.exists(self.file_path):
            try:
                shutil.copy2(self.file_path, self.backup_path)
                return True
            except Exception as e:
                logging.warning(f"Failed to create backup: {e}")
                return False
        return True
    
    def _restore_from_backup(self):
        """从备份恢复文件"""
        if self.backup_enabled and os.path.exists(self.backup_path):
            try:
                shutil.copy2(self.backup_path, self.file_path)
                logging.info(f"Restored file from backup: {self.file_path}")
                return True
            except Exception as e:
                logging.error(f"Failed to restore from backup: {e}")
                return False
        return False
    
    def _acquire_lock(self, file_obj, lock_type=fcntl.LOCK_EX):
        """
        获取文件锁（带重试）
        
        Args:
            file_obj: 文件对象
            lock_type: 锁类型 (fcntl.LOCK_EX 或 fcntl.LOCK_SH)
        """
        for attempt in range(self.max_retries):
            try:
                fcntl.flock(file_obj.fileno(), lock_type | fcntl.LOCK_NB)
                return True
            except (OSError, IOError) as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # 指数退避
                else:
                    logging.error(f"Failed to acquire lock after {self.max_retries} attempts: {e}")
                    return False
        return False
    
    def _release_lock(self, file_obj):
        """释放文件锁"""
        try:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
            return True
        except Exception as e:
            logging.warning(f"Failed to release lock: {e}")
            return False
    
    def safe_append_jsonl(self, data, rank=0):
        """
        安全地追加JSONL数据到文件
        
        Args:
            data (dict): 要追加的数据
            rank (int): worker rank（用于日志）
            
        Returns:
            bool: 操作是否成功
        """
        with self._thread_lock:  # 线程锁
            return self._safe_append_jsonl_impl(data, rank)
    
    def _safe_append_jsonl_impl(self, data, rank=0):
        """安全追加JSONL的实际实现"""
        for attempt in range(self.max_retries):
            try:
                # 创建备份
                if not self._create_backup():
                    logging.warning(f"Worker {rank}: Failed to create backup on attempt {attempt + 1}")
                
                # 使用临时文件进行原子写入
                temp_file = None
                try:
                    # 创建临时文件
                    temp_fd, temp_path = tempfile.mkstemp(
                        dir=os.path.dirname(self.file_path),
                        prefix=f".tmp_{os.path.basename(self.file_path)}_",
                        suffix=".jsonl"
                    )
                    temp_file = os.fdopen(temp_fd, 'w', encoding='utf-8')
                    
                    # 获取原文件的共享锁进行读取
                    with open(self.file_path, 'r', encoding='utf-8') as original_file:
                        if self._acquire_lock(original_file, fcntl.LOCK_SH):
                            try:
                                # 复制原文件内容到临时文件
                                original_file.seek(0)
                                shutil.copyfileobj(original_file, temp_file)
                                
                                # 追加新数据
                                json.dump(data, temp_file, ensure_ascii=False)
                                temp_file.write('\n')
                                temp_file.flush()
                                os.fsync(temp_file.fileno())  # 强制同步到磁盘
                                
                            finally:
                                self._release_lock(original_file)
                        else:
                            raise IOError("Failed to acquire shared lock for reading")
                    
                    temp_file.close()
                    temp_file = None
                    
                    # 获取原文件的排他锁进行替换
                    with open(self.file_path, 'r+', encoding='utf-8') as original_file:
                        if self._acquire_lock(original_file, fcntl.LOCK_EX):
                            try:
                                # 原子替换文件
                                if os.name == 'nt':  # Windows
                                    # Windows需要先删除目标文件
                                    backup_temp = f"{self.file_path}.replace_backup"
                                    shutil.move(self.file_path, backup_temp)
                                    shutil.move(temp_path, self.file_path)
                                    os.remove(backup_temp)
                                else:  # Unix/Linux
                                    os.rename(temp_path, self.file_path)
                                
                                logging.debug(f"Worker {rank}: Successfully appended data using atomic write")
                                return True
                                
                            finally:
                                self._release_lock(original_file)
                        else:
                            raise IOError("Failed to acquire exclusive lock for writing")
                            
                except Exception as e:
                    # 清理临时文件
                    if temp_file:
                        temp_file.close()
                    if 'temp_path' in locals() and os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                    
                    if attempt < self.max_retries - 1:
                        logging.warning(f"Worker {rank}: Attempt {attempt + 1} failed: {e}, retrying...")
                        time.sleep(self.retry_delay * (2 ** attempt))
                    else:
                        logging.error(f"Worker {rank}: All {self.max_retries} attempts failed for JSONL append")
                        
                        # 尝试从备份恢复
                        if self._restore_from_backup():
                            logging.info(f"Worker {rank}: File restored from backup")
                        
                        raise e
                        
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logging.error(f"Worker {rank}: Failed to append JSONL data after {self.max_retries} attempts: {e}")
                    return False
        
        return False
    
    def safe_read_jsonl(self, rank=0):
        """
        安全地读取JSONL文件内容
        
        Args:
            rank (int): worker rank（用于日志）
            
        Returns:
            list: 解析的数据列表，失败时返回空列表
        """
        with self._thread_lock:
            return self._safe_read_jsonl_impl(rank)
    
    def _safe_read_jsonl_impl(self, rank=0):
        """安全读取JSONL的实际实现"""
        for attempt in range(self.max_retries):
            try:
                results = []
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    if self._acquire_lock(f, fcntl.LOCK_SH):
                        try:
                            for line_num, line in enumerate(f, 1):
                                line = line.strip()
                                if line:
                                    try:
                                        data = json.loads(line)
                                        results.append(data)
                                    except json.JSONDecodeError as e:
                                        logging.warning(f"Worker {rank}: Failed to parse line {line_num}: {e}")
                                        continue
                            return results
                        finally:
                            self._release_lock(f)
                    else:
                        raise IOError("Failed to acquire shared lock for reading")
                        
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logging.warning(f"Worker {rank}: Read attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(self.retry_delay)
                else:
                    logging.error(f"Worker {rank}: Failed to read JSONL file after {self.max_retries} attempts: {e}")
                    return []
        
        return []
    
    def safe_read_json(self, rank=0):
        """Safe read of a single JSON object from file with multiprocess protection."""
        for attempt in range(self.max_retries):
            try:
                return self._safe_read_json_impl(rank)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logging.warning(f"Worker {rank}: Read attempt {attempt + 1} failed for {self.file_path}: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Worker {rank}: All read attempts failed for {self.file_path}: {e}")
                    raise
    
    def _safe_read_json_impl(self, rank=0):
        """Implementation of safe JSON object reading with file locking."""
        try:
            with open(self.file_path, 'r') as f:
                # Acquire shared lock for reading
                self._acquire_lock(f, fcntl.LOCK_SH)
                
                try:
                    # Read the entire file content
                    content = f.read().strip()
                    
                    if not content:
                        logging.warning(f"Worker {rank}: JSON file {self.file_path} is empty")
                        return None
                    
                    # Parse as single JSON object
                    data = json.loads(content)
                    logging.debug(f"Worker {rank}: Successfully read JSON object from {self.file_path}")
                    return data
                    
                finally:
                    # Release the lock
                    self._release_lock(f)
                    
        except FileNotFoundError:
            logging.warning(f"Worker {rank}: JSON file {self.file_path} not found")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Worker {rank}: JSON decode error in {self.file_path}: {e}")
            # Try to restore from backup if available
            if self.backup_enabled and self._restore_from_backup():
                logging.info(f"Worker {rank}: Restored from backup, retrying JSON read")
                return self._safe_read_json_impl(rank)
            else:
                raise
        except Exception as e:
            logging.error(f"Worker {rank}: Failed to read JSON file {self.file_path}: {e}")
            raise
    
    def safe_write_json(self, data, rank=0):
        """Safe write of a single JSON object to file with multiprocess protection."""
        for attempt in range(self.max_retries):
            try:
                return self._safe_write_json_impl(data, rank)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logging.warning(f"Worker {rank}: Write attempt {attempt + 1} failed for {self.file_path}: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Worker {rank}: All write attempts failed for {self.file_path}: {e}")
                    raise
    
    def _safe_write_json_impl(self, data, rank=0):
        """Implementation of safe JSON object writing with atomic operations."""
        try:
            # Create backup if enabled
            if self.backup_enabled:
                self._create_backup()
            
            # Write to temporary file first (atomic operation)
            temp_file = f"{self.file_path}.tmp.{rank}.{int(time.time())}"
            
            try:
                with open(temp_file, 'w') as f:
                    # Acquire exclusive lock for writing
                    self._acquire_lock(f, fcntl.LOCK_EX)
                    
                    try:
                        # Write the JSON object
                        json.dump(data, f, indent=2)
                        f.flush()
                        os.fsync(f.fileno())  # Force write to disk
                        
                        logging.debug(f"Worker {rank}: Successfully wrote JSON object to temp file {temp_file}")
                        
                    finally:
                        # Release the lock
                        self._release_lock(f)
                
                # Atomically replace the original file
                shutil.move(temp_file, self.file_path)
                logging.debug(f"Worker {rank}: Successfully moved temp file to {self.file_path}")
                
                return True
                
            except Exception as e:
                # Clean up temp file on error
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                raise e
                
        except Exception as e:
            logging.error(f"Worker {rank}: Failed to write JSON file {self.file_path}: {e}")
            # Try to restore from backup if available
            if self.backup_enabled and self._restore_from_backup():
                logging.info(f"Worker {rank}: Restored from backup after write failure")
            raise

# ChromaDB settings
chroma_settings = chromadb.Settings(
    allow_reset=True,
    anonymized_telemetry=False,
    is_persistent=True,
)

def apply_rope_max_distance_limit(model, max_distance, rank=0):
    """
    对Llama3.1模型应用RoPE最大距离限制
    实现逻辑：对于超出local window size的token，使用相同的位置编码距离
    
    Args:
        model: 加载的模型实例
        max_distance: 最大RoPE距离（local window size）
        rank: worker rank（用于日志）
    """
    try:
        # 检查模型是否为Llama架构
        model_type = getattr(model.config, 'model_type', '').lower()
        architecture = getattr(model.config, 'architectures', [])
        
        is_llama = (model_type == 'llama' or 
                   any('llama' in arch.lower() for arch in architecture))
        
        if not is_llama:
            logging.warning(f"Worker {rank}: Model type '{model_type}' with architectures {architecture} "
                          f"is not recognized as Llama model. RoPE distance limit may not work as expected.")
        
        logging.info(f"Worker {rank}: Applying RoPE max distance limit: {max_distance}")
        logging.info(f"Worker {rank}: Model type: {model_type}, Architectures: {architecture}")
        logging.info(f"Worker {rank}: Using fixed distance strategy - tokens beyond window will use same relative encoding")
        
        # 获取模型的所有层
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'layers'):
            layers = model.layers
        else:
            logging.error(f"Worker {rank}: Cannot find model layers. RoPE limit not applied.")
            return
        
        modified_layers = 0
        
        # 遍历所有transformer层
        for layer_idx, layer in enumerate(layers):
            # 查找self-attention层
            if hasattr(layer, 'self_attn'):
                attn_layer = layer.self_attn
                
                # 查找RoPE相关的组件
                rope_component = None
                if hasattr(attn_layer, 'rotary_emb'):
                    rope_component = attn_layer.rotary_emb
                elif hasattr(attn_layer, 'rope'):
                    rope_component = attn_layer.rope
                
                if rope_component is not None:
                    # 保存原始的forward方法
                    original_forward = rope_component.forward
                    
                    def rope_forward_with_fixed_distance(self, x, position_ids=None, seq_len=None):
                        """
                        RoPE forward with fixed distance for tokens beyond window
                        相对于当前要生成的token位置，从后向前计算距离
                        """
                        # 如果没有position_ids，使用原始方法
                        if position_ids is None:
                            return original_forward(x, position_ids=position_ids, seq_len=seq_len)
                        
                        # 应用固定距离逻辑
                        if position_ids.dim() == 2:
                            # batch维度存在的情况
                            batch_size, seq_length = position_ids.shape
                            
                            # 为每个batch item应用固定距离逻辑
                            adjusted_position_ids = position_ids.clone()
                            for batch_idx in range(batch_size):
                                current_positions = position_ids[batch_idx]
                                if len(current_positions) > 1:
                                    # 当前要生成的token位置（序列中的最后一个位置）
                                    current_pos = current_positions[-1]
                                    
                                    # 计算每个历史token与当前token的距离（从后向前）
                                    deltas = current_pos - current_positions
                                    
                                    # 超出max_distance的距离固定为max_distance
                                    adjusted_deltas = torch.where(
                                        deltas > max_distance,
                                        torch.tensor(max_distance, device=deltas.device, dtype=deltas.dtype),
                                        deltas
                                    )
                                    
                                    # 重新计算调整后的位置
                                    adjusted_position_ids[batch_idx] = current_pos - adjusted_deltas
                        else:
                            # 1维情况
                            if len(position_ids) > 1:
                                # 当前要生成的token位置（序列中的最后一个位置）
                                current_pos = position_ids[-1]
                                
                                # 计算每个历史token与当前token的距离（从后向前）
                                deltas = current_pos - position_ids
                                
                                # 超出max_distance的距离固定为max_distance
                                adjusted_deltas = torch.where(
                                    deltas > max_distance,
                                    torch.tensor(max_distance, device=deltas.device, dtype=deltas.dtype),
                                    deltas
                                )
                                
                                # 重新计算调整后的位置
                                adjusted_position_ids = current_pos - adjusted_deltas
                            else:
                                adjusted_position_ids = position_ids
                        
                        # 使用调整后的position_ids调用原始forward
                        return original_forward(x, position_ids=adjusted_position_ids, seq_len=seq_len)
                    
                    # 绑定新的forward方法
                    import types
                    rope_component.forward = types.MethodType(rope_forward_with_fixed_distance, rope_component)
                    
                    modified_layers += 1
                    
                    if layer_idx < 3:  # 只记录前几层的详细信息
                        logging.debug(f"Worker {rank}: Layer {layer_idx} RoPE component modified with fixed distance strategy")
        
        if modified_layers > 0:
            logging.info(f"Worker {rank}: ✅ Successfully applied RoPE fixed distance limit ({max_distance}) "
                        f"to {modified_layers} layers")
            logging.info(f"Worker {rank}: Tokens beyond position {max_distance} will use the same relative position encoding")
        else:
            logging.warning(f"Worker {rank}: ⚠️ No RoPE components found in model. "
                           f"RoPE distance limit not applied.")
            
    except Exception as e:
        logging.error(f"Worker {rank}: Error applying RoPE max distance limit: {e}")
        import traceback
        logging.error(f"Worker {rank}: RoPE limit traceback: {traceback.format_exc()}")

def compress_knowledge(context, data, current_task, args_namespace, rank=0):
    """
    对检索到的context进行知识压缩
    
    Args:
        context: 原始检索到的context
        data: 数据字典，包含description等信息
        current_task: 当前任务类型
        args_namespace: 参数命名空间
        rank: worker rank（用于日志）
    
    Returns:
        compressed_context: 压缩后的context
    """
    global local_model, local_tokenizer
    
    if not context or context.strip() == "":
        logging.info(f"Worker {rank}: Empty context, skipping compression")
        return context
    
    # 只对VACE任务进行压缩（根据prompt模板）
    if current_task != "VACE":
        logging.info(f"Worker {rank}: Task {current_task} not supported for compression, returning original context")
        return context
    
    try:
        # 构建压缩prompt
        description = data.get("description", "")
        origin_code = data.get("origin_code", "")
        origin_dependency = data.get("origin_dependency", {})
        target_dependency = data.get("target_dependency", {})
        
        # 格式化依赖信息
        origin_dep_str = json.dumps(origin_dependency, indent=2) if origin_dependency else "{}"
        target_dep_str = json.dumps(target_dependency, indent=2) if target_dependency else "{}"
        
        compression_prompt = VersiBCB_VACE_RAGCompress_instruct.format(
            context=context,
            description=description,
            origin_dependency=origin_dep_str,
            origin_code=origin_code,
            target_dependency=target_dep_str
        )
        
        logging.info(f"Worker {rank}: Starting knowledge compression for sample {data.get('id', 'unknown')}")
        logging.info(f"Worker {rank}: Original context length: {len(context)} chars")
        
        # 直接调用inference函数进行压缩
        compressed_context = inference(
            model=local_model, 
            tokenizer=local_tokenizer, 
            prompt=compression_prompt,
            max_new_tokens=1024,
            temperature=0.1,
            top_p=0.9,
            inference_type=args_namespace.inference_type,
            api_key=args_namespace.api_key,
            model_name=args_namespace.api_model_name,
            api_base_url=getattr(args_namespace, 'huggingface_api_base_url', None)
        )
        
        # 清理压缩后的context
        if compressed_context:
            compressed_context = compressed_context.strip()
            logging.info(f"Worker {rank}: Compression completed. Compressed context length: {len(compressed_context)} chars")
            logging.info(f"Worker {rank}: Compression ratio: {len(compressed_context)/len(context)*100:.1f}%")
            
            # 保存压缩结果到临时文件
            try:
                temp_dir = "data/temp/knowledge_compression_results"
                os.makedirs(temp_dir, exist_ok=True)
                
                sample_id = data.get('id', 'unknown')
                temp_file_path = os.path.join(temp_dir, "compressed_contexts.jsonl")
                
                compression_record = {
                    "sample_id": sample_id,
                    "original_len": len(context),
                    "compressed_len": len(compressed_context),
                    "compressed_context": compressed_context
                }
                
                # 追加到JSONL文件
                with open(temp_file_path, 'a', encoding='utf-8') as f:
                    json.dump(compression_record, f, ensure_ascii=False)
                    f.write('\n')
                
                logging.info(f"Worker {rank}: Compression result appended to {temp_file_path}")
                
            except Exception as save_error:
                logging.warning(f"Worker {rank}: Failed to save compression result to temp file: {save_error}")
            
            return compressed_context
        else:
            logging.warning(f"Worker {rank}: Compression returned empty result, using original context")
            return context
            
    except Exception as e:
        logging.error(f"Worker {rank}: Error during knowledge compression: {e}")
        logging.error(f"Worker {rank}: Compression error traceback: {traceback.format_exc()}")
        logging.warning(f"Worker {rank}: Using original context due to compression failure")
        return context

def get_collection_hash(dep):
    """
    为dependency和version生成唯一的hash
    支持两种格式：
    1. dict[pkg, ver] - 标准格式
    2. dict[pkg, list[ver]] - appendSrcDep后的格式
    """
    # 使用dict_to_pkg_ver_tuples处理两种格式
    from utils.getDependencyUtils import dict_to_pkg_ver_tuples
    pkg_ver_tuples = dict_to_pkg_ver_tuples(dep)
    
    # 过滤掉None版本的依赖项，然后排序并生成字符串
    valid_deps = [(pkg, ver) for pkg, ver in pkg_ver_tuples if ver is not None]
    sorted_deps = sorted(valid_deps)
    dep_str = "_".join([f"{pkg}_{ver}" for pkg, ver in sorted_deps])
    return hashlib.md5(dep_str.encode()).hexdigest()[:60]

def get_collection_name(data, dataset, task):
    if dataset == "VersiCode":
        pkg, ver = data["dependency"], get_version(data["version"])
        return get_collection_hash({pkg:ver})
    elif dataset == "VersiBCB":
        dep = data["dependency"] if task == "VSCC" else data["target_dependency"]
        return get_collection_hash(dep)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

def get_query_from_data(data, task):
    query = data["description"]
    if task == "VACE":
        query += data["origin_code"]
    return query

def get_or_create_collection(current_chroma_client, collection_name, embedding_function_instance):
    collection = current_chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function_instance,
    )
    return collection

def load_data(dataset, task, Ban_Deprecation):
    if dataset == "VersiCode":
        # Placeholder for VersiCode data loading
        logging.warning("VersiCode data loading not fully implemented in this script.")
        datas = [] # Return empty list or implement loading
        # with open("benchmark/data/VersiCode_Benchmark/blockcode_completion.json", "r") as f:
        #     datas = json.load(f)
        pass
    elif dataset == "VersiBCB":
        if args.specified_bench_path is None:
            data_path = 'data/VersiBCB_Benchmark'
            data_name = f"{task.lower()}_datas{'_for_warning' if Ban_Deprecation else ''}.json"
            with open(os.path.join(data_path,data_name), "r") as f:
                datas = json.load(f)
        else:
            with open(args.specified_bench_path, "r") as f:
                datas = json.load(f)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return datas

def load_existing_results(output_path, rank=0):
    existing_results = []
    existing_ids = set()
    
    if not os.path.exists(output_path):
        logging.info(f"Worker {rank}: No existing results file found at {output_path}, starting fresh")
        return [], set()
    
    try:
        # 使用多进程安全的文件处理器读取
        temp_handler = MultiprocessSafeFileHandler(
            file_path=output_path,
            max_retries=3,
            retry_delay=1.0,
            backup_enabled=False  # 读取时不需要备份
        )
        
        results_data = temp_handler.safe_read_jsonl(rank)
        
        for result in results_data:
            existing_results.append(result)
            if "id" in result:
                existing_ids.add(result["id"])
        
        logging.info(f"Worker {rank}: Found {len(existing_ids)} existing results in {output_path}")
        return existing_results, existing_ids
        
    except Exception as e:
        logging.error(f"Worker {rank}: Error loading existing results from {output_path}: {e}")
        return [], set()

def analyze_duplicate_ids(output_path, rank=0):
    """分析输出文件中是否存在重复的ID"""
    if not os.path.exists(output_path):
        logging.info(f"Worker {rank}: No output file found at {output_path}")
        return
    
    id_counts = {}
    duplicate_ids = []
    line_count = 0
    
    try:
        with open(output_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line_count += 1
                try:
                    result = json.loads(line.strip())
                    sample_id = result.get("id")
                    if sample_id is not None:
                        if sample_id in id_counts:
                            id_counts[sample_id] += 1
                            if sample_id not in duplicate_ids:
                                duplicate_ids.append(sample_id)
                        else:
                            id_counts[sample_id] = 1
                except json.JSONDecodeError:
                    logging.warning(f"Worker {rank}: Failed to parse line {line_num} in {output_path}")
                    continue
        
        total_unique_ids = len(id_counts)
        total_duplicate_count = sum(count - 1 for count in id_counts.values() if count > 1)
        
        logging.info(f"Worker {rank}: File analysis for {output_path}:")
        logging.info(f"  Total lines: {line_count}")
        logging.info(f"  Unique IDs: {total_unique_ids}")
        logging.info(f"  Duplicate IDs: {len(duplicate_ids)}")
        logging.info(f"  Total duplicate entries: {total_duplicate_count}")
        
        if duplicate_ids:
            logging.warning(f"Worker {rank}: Found duplicate IDs: {duplicate_ids[:10]}")
            if len(duplicate_ids) > 10:
                logging.warning(f"Worker {rank}: ... and {len(duplicate_ids) - 10} more duplicate IDs")
            
            # 显示每个重复ID的出现次数
            for dup_id in duplicate_ids[:5]:  # 只显示前5个
                count = id_counts[dup_id]
                logging.warning(f"Worker {rank}: ID {dup_id} appears {count} times")
        else:
            logging.info(f"Worker {rank}: No duplicate IDs found")
            
    except Exception as e:
        logging.error(f"Worker {rank}: Error analyzing file {output_path}: {e}")

def cleanup_temp_files(output_path, rank=0):
    """清理与当前任务相关的临时文件"""
    try:
        import tempfile
        import hashlib
        
        temp_dir = tempfile.gettempdir()
        output_hash = hashlib.md5(output_path.encode()).hexdigest()[:8]
        
        # 清理coordination文件
        unprocessed_data_file = os.path.join(temp_dir, f"unprocessed_data_{output_hash}.json")
        if os.path.exists(unprocessed_data_file):
            try:
                os.remove(unprocessed_data_file)
                logging.info(f"Worker {rank}: Removed coordination file: {unprocessed_data_file}")
            except Exception as e:
                logging.warning(f"Worker {rank}: Failed to remove coordination file: {e}")
        
        # 清理allocation文件
        allocation_file = os.path.join(temp_dir, f"worker_{rank}_allocation_{output_hash}.json")
        if os.path.exists(allocation_file):
            try:
                os.remove(allocation_file)
                logging.info(f"Worker {rank}: Removed allocation file: {allocation_file}")
            except Exception as e:
                logging.warning(f"Worker {rank}: Failed to remove allocation file: {e}")
        
        # 清理备份文件（如果存在且任务完成）
        global file_handler
        if file_handler and file_handler.backup_enabled:
            backup_path = f"{output_path}.backup"
            if os.path.exists(backup_path):
                try:
                    os.remove(backup_path)
                    logging.info(f"Worker {rank}: Removed backup file: {backup_path}")
                except Exception as e:
                    logging.warning(f"Worker {rank}: Failed to remove backup file: {e}")
                    
    except Exception as e:
        logging.warning(f"Worker {rank}: Error during cleanup: {e}")

def load_target_task_ids(target_task_ids_file):
    """
    从json中加载对应target_ids
    
    Args:
        target_task_ids_file (str): 包含目标task_id的文件路径
        
    Returns:
        set: 目标task_id的集合，如果文件不存在或为None则返回None
    """
    if target_task_ids_file is None:
        return None
    with open(target_task_ids_file, 'r') as f:
        target_task_ids = json.load(f)
    return target_task_ids

def load_generated_target_code(jsonl_file_path):
    """
    Description:
        Load generated target code from jsonl file（对应数据为metadata形式）
    Args:
        jsonl_file_path: the path of the jsonl file
    Returns:
        generated_code_dict: a dictionary of generated target code, key is the id of the sample, value is the generated target code
    """
    generated_code_dict = {}
    
    if not jsonl_file_path or not os.path.exists(jsonl_file_path):
        logging.warning(f"Generated target code file not found: {jsonl_file_path}")
        return generated_code_dict
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if 'id' in data and 'answer' in data:
                        generated_code_dict[data['id']] = data['answer']
                    else:
                        logging.warning(f"Missing 'id' or 'answer' field in line {line_num}: {line}")
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse JSON line {line_num}: {line}, error: {e}")
    
    logging.info(f"Loaded {len(generated_code_dict)} generated target code items from {jsonl_file_path}")
    return generated_code_dict

def load_error_information(error_infos_filepath):
    """Load error information from JSON file"""
    error_infos_dict = {}
    
    if not error_infos_filepath or not os.path.exists(error_infos_filepath):
        logging.warning(f"Error information file not found: {error_infos_filepath}")
        return error_infos_dict
    
    try:
        with open(error_infos_filepath, 'r', encoding='utf-8') as f:
            error_data = json.load(f)
            
        for error_item in error_data:
            sample_id = error_item.get('id')
            error_list = error_item.get('error_infos', [])
            
            if error_list:  # 只保存有错误信息的样本
                # 将错误信息格式化为字符串
                formatted_errors = []
                for error in error_list:
                    error_info = error.get('error_info', '')
                    tool = error.get('tool', 'unknown')
                    rule = error.get('rule', 'unknown')
                    formatted_errors.append(f"[{tool}] {rule}: {error_info}")
                
                error_infos_dict[sample_id] = "\n".join(formatted_errors)
                
        logging.info(f"Loaded error information for {len(error_infos_dict)} samples from {error_infos_filepath}")
        return error_infos_dict
        
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse error information JSON file: {error_infos_filepath}, error: {e}")
        return error_infos_dict
    except Exception as e:
        logging.error(f"Error loading error information from {error_infos_filepath}: {e}")
        return error_infos_dict

def load_retrieved_information(retrieved_info_filepath):
    """Load retrieved information from JSON file"""
    retrieved_info_dict = {}
    
    if not retrieved_info_filepath or not os.path.exists(retrieved_info_filepath):
        logging.warning(f"Retrieved information file not found: {retrieved_info_filepath}")
        return retrieved_info_dict
    
    try:
        with open(retrieved_info_filepath, 'r', encoding='utf-8') as f:
            retrieved_data = json.load(f)
        
        for item in retrieved_data:
            sample_id = item.get('id')
            retrieved_items = item.get('retrieved_items', [])
            api_to_match = item.get('api_to_match', '')
            if retrieved_items:  # 只保存有检索信息的样本
                # 将检索信息格式化为字符串
                formatted_retrieved = []
                for retrieved_item in retrieved_items:
                    path = retrieved_item.get('path', '')
                    doc = retrieved_item.get('doc', '')
                    item_type = retrieved_item.get('type', '')
                    signature = retrieved_item.get('signature', '')
                    if path == api_to_match:
                        formatted_item = f"API: {path}"
                    else:
                        formatted_item = f"Not found {api_to_match} in the retrieved items, but API: {path} could be the API achieve same functionality as {api_to_match}.Please identify carefully."
                    if item_type:
                        formatted_item += f" (Type: {item_type})"
                    if signature:
                        formatted_item += f"\nSignature: {signature}"
                    if doc:
                        # 截断过长的文档
                        doc_truncated = doc[:100] + "..." if len(doc) > 100 else doc
                        formatted_item += f"\nDocumentation: {doc_truncated}"
                    formatted_retrieved.append(formatted_item)
                
                retrieved_info_dict[sample_id] = "\n\n".join(formatted_retrieved)
        
        logging.info(f"Loaded retrieved information for {len(retrieved_info_dict)} samples from {retrieved_info_filepath}")
        return retrieved_info_dict
        
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse retrieved information JSON file: {retrieved_info_filepath}, error: {e}")
        return retrieved_info_dict
    except Exception as e:
        logging.error(f"Error loading retrieved information from {retrieved_info_filepath}: {e}")
        return retrieved_info_dict

def empty_results(output_path, rank=0):
    """安全地清空结果文件"""
    global file_handler
    
    try:
        # 创建文件处理器
        if file_handler is None or file_handler.file_path != output_path:
            file_handler = MultiprocessSafeFileHandler(
                file_path=output_path,
                max_retries=3,
                retry_delay=1.0,
                backup_enabled=True
            )
        
        # 使用原子写入创建空文件
        with open(output_path, 'w', encoding='utf-8') as f:
            pass  # 创建空文件
        
        logging.info(f"Worker {rank}: Cleared results file: {output_path}")
        
    except Exception as e:
        logging.error(f"Worker {rank}: Failed to clear results file {output_path}: {e}")
        # 作为后备方案，使用简单方法
        with open(output_path, 'w', encoding='utf-8') as f:
            pass

def safe_append_to_jsonl(file_path, data, rank=0):
    """使用多进程安全文件处理器追加数据到JSONL文件"""
    global file_handler
    
    # 如果当前文件处理器不存在或路径不匹配，创建新的处理器
    if file_handler is None or file_handler.file_path != file_path:
        file_handler = MultiprocessSafeFileHandler(
            file_path=file_path,
            max_retries=3,
            retry_delay=1.0,
            backup_enabled=True
        )
        logging.info(f"Worker {rank}: Created new file handler for {file_path}")
    
    return file_handler.safe_append_jsonl(data, rank)

def append_to_jsonl(file_path, data):
    with open(file_path, 'a') as f:
        f.write(json.dumps(data) + '\n')

def initialize_resources(args):
    """Initializes ChromaDB, RAG retriever, and local model/tokenizer using device_map='auto'."""
    global chroma_client, rag_retriever, local_model, local_tokenizer, together_client, accelerator

    rank = getattr(args, 'rank', 0)
    world_size = getattr(args, 'world_size', 1)
    
    logging.info(f"Worker {rank}/{world_size}: Initializing resources...")

    # Initialize Accelerator - still useful for device context and other preparations if needed
    accelerator = Accelerator()
    logging.info(f"Worker {rank}: Accelerator initialized. Device: {accelerator.device}, Distributed Type: {accelerator.distributed_type}")

    # Initialize Together AI client (needed if embedding source is 'togetherai')
    if args.embedding_source == 'togetherai':
        try:
            with open(TOGETHER_API_KEY_PATH, "r") as f:
                api_key = f.read().strip()
            together_client = Together(api_key=api_key)
            logging.info(f"Worker {rank}: Together AI client initialized for embeddings.")
        except Exception as e:
            logging.error(f"Worker {rank}: Failed to initialize Together AI client: {e}")
            # Depending on strictness, you might want to exit or raise an error
            # For now, it will only affect 'togetherai' embeddings

    # Initialize ChromaDB client
    embed_model_name = args.local_embedding_model if args.embedding_source == 'local' else args.togetherai_embedding_model
    embed_model_name = embed_model_name.split('/')[-1]
    db_path = os.path.join(args.rag_collection_base, args.knowledge_type, embed_model_name)
    os.makedirs(db_path, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=db_path, settings=chroma_settings)
    logging.info(f"Worker {rank}: ChromaDB client initialized at {db_path}")

    # Configure embedding parameters
    embedding_args = {
        'source': args.embedding_source,
        'model_name': args.local_embedding_model if args.embedding_source == 'local' else args.togetherai_embedding_model,
        'together_client': together_client if args.embedding_source == 'togetherai' else None,
        'batch_size': 64
    }

    # Initialize RAG Retriever (if approach is RAG)
    if args.approach == 'RAG':
        rag_retriever = RAGContextRetriever(
            chroma_client=chroma_client, 
            embed_func_args=embedding_args, 
            corpus_type=args.knowledge_type,
            rag_collection_base=args.rag_collection_base,
            knowledge_type=args.knowledge_type,
            embedding_source=args.embedding_source,
            docstring_embedding_base_path=args.docstring_embedding_base_path,
            srccode_embedding_base_path=args.srccode_embedding_base_path,
            max_dependency_num=args.max_dependency_num,
            append_srcDep=args.append_srcDep,
            query_cache_dir=args.query_cache_dir,
            rag_document_num=args.rag_document_num,
            docstring_corpus_path=args.docstring_corpus_path,
            srccode_corpus_path=args.srccode_corpus_path,
            generated_queries_file=args.generated_queries_file,
            use_generated_queries=args.use_generated_queries,
            fixed_docs_per_query=args.fixed_docs_per_query,
            enable_dependency_filtering=args.enable_dependency_filtering,
            enable_query_dependency_filtering=args.enable_query_dependency_filtering,
            query_filter_strict_mode=args.query_filter_strict_mode,
            max_doc_tokens=args.max_doc_tokens,
            doc_truncate_tokenizer=args.doc_truncate_tokenizer,
            api_name_str_match=args.api_name_str_match
        )
        logging.info(f"Worker {rank}: RAGContextRetriever initialized.")
        if args.use_generated_queries:
            logging.info(f"Worker {rank}: Using generated queries from {args.generated_queries_file}")
        else:
            logging.info(f"Worker {rank}: Using default query generation (description + origin_code)")
        
        # Log the new parameters
        if args.fixed_docs_per_query is not None:
            logging.info(f"Worker {rank}: Using fixed docs mode: {args.fixed_docs_per_query} docs per query")
        else:
            logging.info(f"Worker {rank}: Using progressive retrieval mode")
        
        logging.info(f"Worker {rank}: Dependency filtering: {'enabled' if args.enable_dependency_filtering else 'disabled'}")
        logging.info(f"Worker {rank}: Query dependency filtering: {'enabled' if args.enable_query_dependency_filtering else 'disabled'}")
        if args.enable_query_dependency_filtering:
            logging.info(f"Worker {rank}: Query filter strict mode: {args.query_filter_strict_mode}")
        
        # Log the new document processing parameters
        if args.max_doc_tokens:
            logging.info(f"Worker {rank}: Individual document max tokens: {args.max_doc_tokens} tokens (using {args.doc_truncate_tokenizer})")
        else:
            logging.info(f"Worker {rank}: No individual document token limit")
        
        if args.api_name_str_match:
            logging.info(f"Worker {rank}: API name string matching: enabled (using edit distance)")
        else:
            logging.info(f"Worker {rank}: API name string matching: disabled (using vector similarity)")

    # Initialize local model and tokenizer

    if args.inference_type == "local":
        if args.precision == "bf16":
            precision = torch.bfloat16 
        elif args.precision == "fp16":
            precision = torch.float16
        elif args.precision == "int8":
            precision = torch.int8
        else:
            precision = torch.float32

        logging.info(f"Worker {rank}: Loading local model {args.model} with device_map='auto'...")
        try:
            # 为不同worker设置不同的CUDA设备
            if world_size > 1 and torch.cuda.is_available():
                device_id = rank % torch.cuda.device_count()
                torch.cuda.set_device(device_id)
                logging.info(f"Worker {rank}: Set CUDA device to {device_id}")
            
            # Using device_map="auto" for automatic model distribution
            # trust_remote_code=True might be necessary for some models
            local_model = AutoModelForCausalLM.from_pretrained(
                args.model, 
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=precision
            )
            local_tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            
            logging.info(f"Worker {rank}: Local model {args.model} loaded with device_map='auto'.")
            logging.info(f"Worker {rank}: Model device map: {local_model.hf_device_map}")

            # 🔧 应用RoPE distance限制（如果启用）
            if args.enable_rope_max_distance:
                apply_rope_max_distance_limit(local_model, args.rope_max_distance, rank)

        except Exception as e:
            logging.error(f"Worker {rank}: Error loading local model {args.model} with device_map='auto': {e}")
            # Attempt to clear cache and retry once if it's an OOM during initial load, though device_map should handle it better
            if "out of memory" in str(e).lower():
                logging.warning(f"Worker {rank}: Caught OOM error, attempting to clear cache and retry loading...")
                torch.cuda.empty_cache()
                try:
                    local_model = AutoModelForCausalLM.from_pretrained(
                        args.model, 
                        device_map="auto",
                        trust_remote_code=True,
                        torch_dtype=precision
                    )
                    local_tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
                    logging.info(f"Worker {rank}: Successfully loaded model {args.model} after clearing cache.")
                    logging.info(f"Worker {rank}: Model device map: {local_model.hf_device_map}")
                    
                    # 🔧 应用RoPE distance限制（如果启用）
                    if args.enable_rope_max_distance:
                        apply_rope_max_distance_limit(local_model, args.rope_max_distance, rank)
                        
                except Exception as e2:
                    logging.error(f"Worker {rank}: Still failed to load model after clearing cache: {e2}")
                    raise e2 # Re-raise the exception to stop execution
            else:
                raise # Re-raise other exceptions
    else:
        logging.info(f"Worker {rank}: Using remote inference, no local model loading required.")


def process_single_sample(item_data, args_namespace):
    """
    Description:Processes a single data sample.
    args:
    
    return:
     answer_dict:
        id:
        answer:
        context:
    """
    global rag_retriever, local_model, local_tokenizer, accelerator # Access global resources

    i, data, current_dataset, current_task, current_ban_deprecation, current_model_path, current_approach, max_tokens, max_new_tokens, review_mode, generated_code_dict, error_infos_dict, retrieved_info_dict = item_data
    sample_id = data.get('id', f'index_{i}')
    rank = getattr(args_namespace, 'rank', 0)

    try:
        context = ""
        if current_approach == 'RAG':
            if rag_retriever is None:
                raise RuntimeError("RAG retriever not initialized. Ensure approach is RAG and initialization succeeded.")
            
            try:
                if args_namespace.QACacheContext_path is not None:
                    from utils.io_utils import loadJsonl
                    if os.path.exists(args_namespace.QACacheContext_path):
                        QA_cache_context = loadJsonl(args_namespace.QACacheContext_path)
                        from utils.RAGutils.queryResultByInfer.contextLoad import loadQAContext
                        context = loadQAContext(data,max_tokens,QA_cache_context)
                    else:
                        logging.info(f"Worker {rank}: QACacheContext_path not found, retrieving context from RAG")
                else:
                    context = rag_retriever.retrieve_context(
                        data,
                        current_dataset,
                        current_task,
                        max_tokens 
                    )
                
                # 检查是否返回了None（锁冲突）
                if context is None:
                    logging.info(f"Worker {rank}: Skipping sample {sample_id} due to collection building lock conflict")
                    return {"id": sample_id, "skipped": True, "reason": "collection_lock_conflict"}
                    
            except Exception as rag_error:
                # RAG检索出现错误时，记录详细错误信息并使用空context继续处理
                error_details = {
                    "error_type": type(rag_error).__name__,
                    "error_message": str(rag_error),
                    "error_location": "RAG context retrieval"
                }
                logging.error(f"Worker {rank}: RAG retrieval failed for sample {sample_id}: {error_details}")
                logging.error(f"Worker {rank}: RAG error traceback: {traceback.format_exc()}")
                
                # 使用空context继续处理，避免因RAG失败而整个样本失败
                context = ""
                logging.warning(f"Worker {rank}: Using empty context for sample {sample_id} due to RAG failure")
                
        elif current_approach == 'BASELINE':
            context = ""
        else:
            raise ValueError(f"Unsupported approach: {current_approach}")
        
        # 知识压缩步骤 - 对RAG检索到的context进行压缩
        if current_approach == 'RAG' and context:
            # 检查是否启用知识压缩
            if args_namespace.enable_knowledge_compression:
                try:
                    original_context = context
                    context = compress_knowledge(context, data, current_task, args_namespace, rank)
                    
                    # 记录压缩结果
                    if context != original_context:
                        compression_ratio = len(context) / len(original_context) if len(original_context) > 0 else 0
                        logging.info(f"Worker {rank}: Knowledge compression applied for sample {sample_id}, "
                                   f"compression ratio: {compression_ratio:.2%}")
                    else:
                        logging.info(f"Worker {rank}: Knowledge compression skipped or failed for sample {sample_id}")
                        
                except Exception as compression_error:
                    logging.error(f"Worker {rank}: Knowledge compression failed for sample {sample_id}: {compression_error}")
                    logging.warning(f"Worker {rank}: Using original context due to compression failure")
                    # context保持原值，继续处理
        
        # 跳过生成环节
        if args_namespace.skip_generation:
            return {"id": sample_id, "answer": "", "context": context, "input": ""}
        
        # Prepare for inference based on type
        if args_namespace.inference_type == "local":
            if local_model is None or local_tokenizer is None:
                raise RuntimeError("Local model or tokenizer not initialized for local inference.")
            
            model_max_length = getattr(local_model.config, "max_position_embeddings", None)
            if model_max_length is None:
                logging.error(f"Worker {rank}: max_position_embeddings not found in model config for {current_model_path}. Cannot proceed for sample {sample_id}.")
                raise RuntimeError("max_position_embeddings not found in model config.")
            
            desired_max_new_tokens = max_new_tokens
            buffer_for_model_specific_tokens = 5
            max_len_for_prompt_itself = model_max_length - desired_max_new_tokens - buffer_for_model_specific_tokens
            
            if max_len_for_prompt_itself <= 0:
                logging.error(f"Worker {rank}: Model max length {model_max_length} is too short. Sample ID: {sample_id}")
                return {"id": sample_id, "error": "Model max length too short", "traceback": ""}

            # TODO:临时补丁，后面需要更好的修复
            generated_code_dict = {int(k): v for k, v in generated_code_dict.items()}
            # Get generated target code for review mode
            generated_target_code = None
            if review_mode and sample_id in generated_code_dict:
                generated_target_code = generated_code_dict[sample_id]
            elif review_mode:
                logging.warning(f"Worker {rank}: Review mode enabled but no generated code found for sample with id: {sample_id}")
                logging.warning(f"type of sample_id: {type(sample_id)}")
                logging.warning(f"type for first key of generated_code_dict: {type(list(generated_code_dict.keys())[0])}")
                logging.warning(f"first 10 keys of generated_code_dict: {list(generated_code_dict.keys())[:10]}")
                logging.warning(f"Worker {rank}: Generated code dict length: {len(generated_code_dict)}")
            
            # Check for error information for error_fix mode
            error_info = ""
            error_fix_mode = False
            retrieved_info = ""
            if review_mode and sample_id in error_infos_dict:
                error_info = error_infos_dict[sample_id]
                # 检查是否有对应的retrieved_info
                if sample_id in retrieved_info_dict:
                    retrieved_info = retrieved_info_dict[sample_id]
                    logging.info(f"Worker {rank}: Using error_fix mode with retrieved info for sample {sample_id} with {len(error_info)} error(s)")
                else:
                    retrieved_info = ""
                    logging.info(f"Worker {rank}: Using error_fix mode for sample {sample_id} with {len(error_info)} error(s), no retrieved info available")
            
            if args_namespace.enable_multiround_infer:
                round = args_namespace.round
                generated_code = ""  # 初始化生成的代码
                for round_idx in range(round):
                    logging.info(f"Worker {rank}: Starting round {round_idx + 1}/{round} for sample {sample_id}")
                    
                    # 使用formatInputAndInfer进行推理
                    response_text = formatInputAndInfer(
                        item_data, generated_code, error_info, args_namespace, 
                        error_fix_mode, retrieved_info, context
                    )
                    
                    # 检查推理结果
                    if isinstance(response_text, dict) and "error" in response_text:
                        logging.error(f"Worker {rank}: Error in round {round_idx + 1} for sample {sample_id}: {response_text['error']}")
                        return response_text  # 返回错误信息
                    
                    # 解析生成的代码
                    generated_code = parseOutput2Code(response_text)
                    logging.info(f"Worker {rank}: Round {round_idx + 1} completed, generated code length: {len(generated_code)}")
                    
                    # 如果启用了每轮更新错误信息
                    if args_namespace.updateErrorEachRound and round_idx < round - 1:  # 最后一轮不需要更新
                        try:
                            target_dependency = data.get('target_dependency', None) if "target_dependency" in data else data["dependency"]
                            error_info = getErrorInfoFromStaticAnalyser("pyright", generated_code, target_dependency)
                            logging.info(f"Worker {rank}: error_info: {error_info}")
                            logging.info(f"Worker {rank}: Updated error info for round {round_idx + 1}, error count: {len(error_info) if error_info else 0}")
                            if len(error_info) == 0:
                                return {"id": sample_id, "answer": response_text, "context": context, "input": "","error_info":error_info}
                        except Exception as e:
                            logging.warning(f"Worker {rank}: Failed to update error info for round {round_idx + 1}: {e}")
                            error_info = None

                return {"id": sample_id, "answer": response_text, "context": context, "input": "","error_info":error_info}
            # 使用formatInputAndInfer进行单轮推理
            response_text = formatInputAndInfer(
                item_data, "", error_info, args_namespace, 
                error_fix_mode, retrieved_info, context
            )
            
            # 检查推理结果
            if isinstance(response_text, dict) and "error" in response_text:
                return response_text  # 返回错误信息
        else: # Remote inference (huggingface or togetherai)
            # 使用formatInputAndInfer进行远程推理
            response_text = formatInputAndInfer(
                item_data, "", error_info, args_namespace, 
                error_fix_mode, retrieved_info, context
            )
            
            # 检查推理结果
            if isinstance(response_text, dict) and "error" in response_text:
                return response_text  # 返回错误信息
        # 之前这里用的是input_for_api，这里input改为空了
        return {"id": sample_id, "answer": response_text, "context": context, "input": "","error_info":error_info}

    except Exception as e:
        tb_str = traceback.format_exc()
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "sample_id": sample_id,
            "error_location": "general processing"
        }
        logging.error(f"Worker {rank}: Error processing sample {sample_id}: {error_details}")
        logging.error(f"Worker {rank}: Full traceback:\n{tb_str}")
        return {"id": sample_id, "error": str(e), "traceback": tb_str, "error_details": error_details}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-process local model prediction script with device_map='auto'.")
    parser.add_argument("--dataset", type=str, default="VersiBCB", help="Dataset name (e.g., VersiBCB, VersiCode)")
    parser.add_argument("--task", type=str, default="VACE", help="Task type (e.g., VACE, VSCC)")
    parser.add_argument("--specified_bench_path", type=str, default=None, help="指定要预测的benchmark路径")
    parser.add_argument("--Ban_Deprecation", type=lambda x: x.lower() == 'true', default=False, help="Whether to ban deprecation-related information")
    parser.add_argument("--approach", type=str, default="RAG", choices=['RAG', 'BASELINE'], help="Approach: RAG or BASELINE")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="Path or name of the local model")
    # model_type is implicitly "local" for this script
    parser.add_argument("--selfdefined_OutputPath", type=str, default=None, help="Custom output path for results.jsonl")
    parser.add_argument("--knowledge_type", type=str, default="srccodes", choices=['srccodes', 'docstring'], help="Type of knowledge for RAG (srccodes or docstring)")
    parser.add_argument("--embedding_source", type=str, default="local", choices=['local', 'togetherai'], help="Source for embeddings (local or togetherai)")
    parser.add_argument("--precision", type=str, default="fp16", choices=['fp32', 'fp16', 'bf16', 'int8'], help="Precision for model (fp32, fp16, bf16, int8)")
    parser.add_argument("--max_dependency_num", type=int, default=None,help="最大依赖数量")
    parser.add_argument("--append_srcDep", action="store_true", help="是否添加源依赖")
    parser.add_argument("--max_tokens", type=int, default=8000, help="最大retrieve的token数量")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="最大生成token数量")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for remote inference")

    parser.add_argument("--rag_collection_base", type=str, default=RAG_COLLECTION_BASE, help="RAG collection base path")
    parser.add_argument("--local_embedding_model", type=str, default=LOCAL_EMBEDDING_MODEL, help="Local embedding model")
    parser.add_argument("--togetherai_embedding_model", type=str, default=TOGETHERAI_EMBEDDING_MODEL, help="TogetherAI embedding model")
    parser.add_argument("--docstring_embedding_base_path", type=str, default=DOCSTRING_EMBEDDING_BASE_PATH, help="Docstring embedding base path")
    parser.add_argument("--srccode_embedding_base_path", type=str, default=SRCCODE_EMBEDDING_BASE_PATH, help="Source code embedding base path")
    parser.add_argument("--query_cache_dir", type=str, default=QUERY_CACHE_DIR, help="Query cache directory")
    parser.add_argument("--rag_document_num", type=int, default=RAG_DOCUMENT_NUM, help="RAG document number")
    parser.add_argument("--docstring_corpus_path", type=str, default=DOCSTRING_CORPUS_PATH, help="Docstring corpus path")
    parser.add_argument("--srccode_corpus_path", type=str, default=SRCCODE_CORPUS_PATH, help="Source code corpus path")
    parser.add_argument("--inference_type", type=str, default="local", choices=['local', 'huggingface', 'togetherai'], help="Inference type: local, huggingface, or togetherai")
    parser.add_argument("--api_key", type=str, default=None, help="API key for remote inference (HuggingFace or TogetherAI)")
    parser.add_argument("--api_model_name", type=str, default=None, help="Model name for remote API inference if API inference is used")
    parser.add_argument("--huggingface_api_base_url", type=str, default=None, help="Custom base URL for HuggingFace API (optional)")
    parser.add_argument("--max_concurrent_requests", type=int, default=5, help="Maximum concurrent requests for remote API inference")
    parser.add_argument("--overwrite_existing_results", action="store_true", help="Overwrite existing results,先删除所有结果，然后直接写")
    
    parser.add_argument("--skip_generation", action="store_true", help="Skip generation")
    parser.add_argument("--newWhenExist", action="store_true", help="New when exist,如果文件存在，则创建新的文件")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for nucleus sampling")
    parser.add_argument("--base_output_dir", type=str, default=None, help="Base output directory")
    
    # 添加并发推理参数
    parser.add_argument("--rank", type=int, default=0, help="Worker rank for distributed inference")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of workers for distributed inference")
    
    # 添加针对性生成选项
    parser.add_argument("--target_task_ids_file", type=str, default=None, 
                       help="Path to file containing target task IDs (one per line). Only these tasks will be processed. Supports load balancing across multiple workers.")
    
    # 添加RoPE distance限制选项
    parser.add_argument("--enable_rope_max_distance", action="store_true", 
                       help="Enable maximum RoPE distance limit for Llama3.1 models. When enabled, positions beyond max_distance will be clamped to max_distance.")
    parser.add_argument("--rope_max_distance", type=int, default=131072,
                       help="Maximum RoPE distance value. Positions beyond this value will be clamped to this maximum. Default: 131072 (128K tokens)")
    
    # 添加生成查询相关参数
    parser.add_argument("--generated_queries_file", type=str, default=None,
                       help="Path to the file containing generated queries (JSON format)")
    parser.add_argument("--use_generated_queries", action="store_true",
                       help="Use generated queries instead of simple concatenation for RAG retrieval")
    
    # 添加固定文档数量和dependency过滤参数
    parser.add_argument("--fixed_docs_per_query", type=int, default=None,
                       help="Fixed number of documents to retrieve per query. If not set, uses progressive retrieval mode.")
    parser.add_argument("--enable_dependency_filtering", action="store_true", default=True,
                       help="Enable filtering of documents to only include APIs from target dependencies")
    parser.add_argument("--disable_dependency_filtering", action="store_true",
                       help="Disable dependency filtering (overrides enable_dependency_filtering)")
    
    # 添加查询层面的dependency过滤参数
    parser.add_argument("--enable_query_dependency_filtering", action="store_true",
                       help="Enable query-level dependency filtering to filter out queries not belonging to target dependencies")
    parser.add_argument("--query_filter_strict_mode", action="store_true", default=True,
                       help="Use strict mode for query dependency filtering (only path-based matching)")
    parser.add_argument("--query_filter_nonstrict_mode", action="store_true",
                       help="Use non-strict mode for query dependency filtering (includes description-based matching)")
    
    # 添加知识压缩参数
    parser.add_argument("--enable_knowledge_compression", action="store_true", default=False,
                       help="Enable knowledge compression step after RAG retrieval")
    parser.add_argument("--disable_knowledge_compression", action="store_true",
                       help="Disable knowledge compression (overrides enable_knowledge_compression)")
    
    # 添加单个文档预截断参数
    parser.add_argument("--max_doc_tokens", type=int, default=None,
                       help="Maximum number of tokens for individual document's doc field. If set, each document's doc field will be truncated to this token limit.")
    parser.add_argument("--doc_truncate_tokenizer", type=str, default="/datanfs4/chenrongyi/models/Llama-3.1-8B-Instruct",
                       help="Tokenizer model for measuring doc field tokens. Default: Llama3.1-8B tokenizer")
    
    # 添加API名称字符串匹配参数
    parser.add_argument("--api_name_str_match", action="store_true", default=False,
                       help="Enable API name string matching using edit distance instead of vector similarity")
    # 基于已经生成的code做review
    parser.add_argument("--review_on_output", action="store_true", default=False,
                       help="Review on output")
    parser.add_argument("--generated_target_code_path", type=str, default=None,
                       help="Path to the generated target code,用于加入到review_prompt中进行代码生成")
    # 添加错误信息文件路径参数，用于error_fix推理(目前仅在review模式下开启)
    parser.add_argument("--errorinfos_filepath", type=str, default=None,
                       help="Path to the error information JSON file for error_fix inference in review mode")
    parser.add_argument("--skip_noerrorinfo_samples", action="store_true", default=True,
                       help="Skip samples with no error information")
    # 添加检索信息文件路径参数，用于error_fix推理中的API检索信息
    parser.add_argument("--retrieved_info_filepath", type=str, default=None,
                       help="Path to the retrieved information JSON file for error_fix inference with API retrieval")
    parser.add_argument("--QACacheContext_path", type=str, default=None,
                       help="Path to the QA cache context JSON file for RAG retrieval")
    parser.add_argument("--appendContext", action="store_true", default=False,
                       help="Append context to the output file for RAG retrieval")
    parser.add_argument("--stop_tokens",nargs="*", default=None,
                       help="Stop tokens for inference")
    # 多轮推理和应用错误信息更新
    parser.add_argument("--enable_multiround_infer", action="store_true", default=False,
                       help="Enable multiround inference")
    parser.add_argument("--round", type=int, default=2,
                       help="Number of rounds for multiround inference")
    parser.add_argument("--updateErrorEachRound", action="store_true", default=False,
                       help="Update error information each round")

    args = parser.parse_args()

    rank = args.rank
    world_size = args.world_size
    
    # 处理dependency filtering参数冲突
    if args.disable_dependency_filtering:
        args.enable_dependency_filtering = False
    
    # 处理query filtering参数冲突
    if args.query_filter_nonstrict_mode:
        args.query_filter_strict_mode = False
    
    # 处理knowledge compression参数冲突
    if args.disable_knowledge_compression:
        args.enable_knowledge_compression = False
    
    # 安全检查：local推理模式下强制使用单worker，避免重复ID问题
    if args.inference_type == "local" and args.num_workers > 1:
        logging.warning(f"Local inference mode detected with num_workers={args.num_workers}. "
                       f"Forcing num_workers=1 to prevent duplicate ID issues.")
        args.num_workers = 1
    
    logging.info(f"Worker {rank}/{world_size}: Starting local prediction with args: {args}")
    logging.info(f"Configuration: inference_type={args.inference_type}, num_workers={args.num_workers}")
    logging.info(f"📁 Multiprocess-safe file handling: ENABLED (atomic writes, file locking, backup/recovery)")

    # Initialize resources (Chroma, RAG retriever, Model with device_map="auto")
    initialize_resources(args)

    # Load data
    logging.info(f"Worker {rank}: Loading data...")
    datas = load_data(args.dataset, args.task, args.Ban_Deprecation)
    logging.info(f"Worker {rank}: Data loaded. Total samples: {len(datas)}")
    
    # Load generated target code if review mode is enabled
    generated_code_dict = {}
    if args.review_on_output:
        if args.generated_target_code_path:
            logging.info(f"Worker {rank}: Loading generated target code from {args.generated_target_code_path}")
            generated_code_dict = load_generated_target_code(args.generated_target_code_path)
            logging.info(f"Worker {rank}: Loaded {len(generated_code_dict)} generated code items")
        else:
            logging.warning(f"Worker {rank}: review_on_output is True but generated_target_code_path is not provided")
            args.review_on_output = False

    # Load error information if provided for error_fix inference
    error_infos_dict = {}
    if args.errorinfos_filepath:
        logging.info(f"Worker {rank}: Loading error information from {args.errorinfos_filepath}")
        error_infos_dict = load_error_information(args.errorinfos_filepath)
        logging.info(f"Worker {rank}: Loaded error information for {len(error_infos_dict)} samples")

    # Load retrieved information if provided for error_fix inference
    retrieved_info_dict = {}
    if args.retrieved_info_filepath:
        logging.info(f"Worker {rank}: Loading retrieved information from {args.retrieved_info_filepath}")
        retrieved_info_dict = load_retrieved_information(args.retrieved_info_filepath)
        logging.info(f"Worker {rank}: Loaded retrieved information for {len(retrieved_info_dict)} samples")

    # 创建输出管理器
    output_manager = OutputManager(args.base_output_dir)



    # Determine output path - 所有worker共享同一个输出文件
    if args.selfdefined_OutputPath:
        output_path = args.selfdefined_OutputPath
        # Generate config path based on output path
        output_dir = os.path.dirname(output_path)
        base_filename = os.path.splitext(os.path.basename(output_path))[0]
        
        # Check for existing config files and add suffix if needed
        suffix = 0
        while True:
            if suffix == 0:
                config_filename = f"{base_filename}_config.json"
            else:
                config_filename = f"{base_filename}_{suffix}_config.json"
            
            config_path = os.path.join(output_dir, config_filename)
            
            if not os.path.exists(config_path):
                break
            suffix += 1
    else:
        # 确定方法类型（RAG或FC）
        approach = "RAG" if args.approach == "RAG" else "BASELINE"
        
        # 生成基础文件名
        dataset_suffix = f"{args.dataset.lower()}_{args.task.lower()}{'_BD' if args.Ban_Deprecation else ''}{'_REVIEW' if args.review_on_output else ''}{'_ERRORFIX' if args.errorinfos_filepath else ''}"
        base_filename = output_manager.generate_base_filename(
            dataset=dataset_suffix,
            model_name=args.model,
            approach=approach,
            corpus_type=args.knowledge_type if args.approach == "RAG" else None,
            embedding_source=args.embedding_source if args.approach == "RAG" else None,
            max_tokens=args.max_tokens if args.approach == "RAG" else None,
            inference_type=args.inference_type,
            max_dependency_num=args.max_dependency_num if hasattr(args, 'max_dependency_num') else None
        )
        model_name = args.model.split('/')[-1]
        # 获取输出路径和配置路径
        output_path, config_path = output_manager.get_output_path_and_config(
            approach=approach,
            base_filename=base_filename,
            newWhenExist=args.newWhenExist,
            model_name_or_path=args.api_model_name if args.inference_type != "local" else model_name
        )
        logging.info(f"Worker {rank}: Output path: {output_path}")
        print(f"Worker {rank}: Output path: {output_path}")
        
    # 只有rank 0负责保存配置文件
    if rank == 0:
        if args.overwrite_existing_results:
            empty_results(output_path, rank)
            logging.info(f"Worker {rank}: Cleared existing results file")
        else:
            # 等待其他worker一点时间，确保文件状态稳定
            import time
            time.sleep(1)
    else:
        # 其他worker等待rank 0完成文件初始化
        import time
        time.sleep(3)
        logging.info(f"Worker {rank}: Waiting for rank 0 to complete file initialization...")
        
    # 🔧 修复异步问题：确保所有worker基于相同的数据视图
    # 只有rank 0负责数据过滤和分配决策，其他worker等待分配结果
    target_task_ids = load_target_task_ids(args.target_task_ids_file)
    if rank == 0:
        # rank 0负责完整的数据过滤
        logging.info(f"Worker {rank}: Acting as coordinator, performing data filtering...")
        
        # 加载已有结果
        existing_results, existing_ids = load_existing_results(output_path, rank)
        
        # Load target task IDs

        
        # Prepare data to process - 先过滤出范围内且未处理的数据
        unprocessed_data = []
        skipped_by_task_id_filter = 0
        
        from benchmark.pred_utils import convertTarDep2TrueDep
        for i, data_item in enumerate(datas):
            # 检查范围
            data_item = convertTarDep2TrueDep(data_item)
            if not (VSCC_LOW_BOUND <= i < VSCC_HIGH_BOUND):
                continue
                
            sample_id = data_item.get("id", None)
            
            # 检查是否已处理
            if sample_id in existing_ids:
                continue
                
            # 检查是否在目标task_id列表中（如果指定了的话）
            if target_task_ids is not None:
                if sample_id not in target_task_ids:
                    skipped_by_task_id_filter += 1
                    continue
            # 存在error_info file,需要根据error_info id进行过滤
            if args.errorinfos_filepath and args.skip_noerrorinfo_samples:
                if sample_id not in error_infos_dict:
                    continue
            unprocessed_data.append((i, data_item, args.dataset, args.task, args.Ban_Deprecation, args.model, args.approach, args.max_tokens, args.max_new_tokens, args.review_on_output, generated_code_dict, error_infos_dict, retrieved_info_dict))
        
        # 记录过滤统计
        if target_task_ids is not None:
            logging.info(f"Worker {rank}: Target task ID filtering enabled. "
                        f"Loaded {len(target_task_ids)} target task IDs, "
                        f"skipped {skipped_by_task_id_filter} samples not in target list")
        
        # 保存过滤后的数据到临时文件，供其他worker读取
        import tempfile
        import hashlib
        temp_dir = tempfile.gettempdir()
        # 🔧 修复：使用输出文件路径的hash作为唯一标识，而不是PID
        output_hash = hashlib.md5(output_path.encode()).hexdigest()[:8]
        unprocessed_data_file = os.path.join(temp_dir, f"unprocessed_data_{output_hash}.json")
        
        # 将unprocessed_data序列化保存
        serializable_data = []
        for item in unprocessed_data:
            # 将tuple转换为list以便JSON序列化
            serializable_item = list(item)
            serializable_data.append(serializable_item)
        
        coordination_data = {
            "unprocessed_data": serializable_data,
            "total_unprocessed": len(unprocessed_data),
            "existing_count": len(existing_ids),
            "world_size": world_size,
            "coordinator_rank": rank
        }
        
        try:
            # 使用multiprocess-safe file handler来写入coordination文件
            temp_file_handler = MultiprocessSafeFileHandler(unprocessed_data_file)
            # 使用safe_write_json方法写入单个JSON对象
            temp_file_handler.safe_write_json(coordination_data, rank)
            logging.info(f"Worker {rank}: Saved coordination data to {unprocessed_data_file}")
            logging.info(f"Worker {rank}: Total unprocessed samples: {len(unprocessed_data)}")
        except Exception as e:
            logging.error(f"Worker {rank}: Failed to save coordination data: {e}")
            raise
        
        # 等待一下确保文件写入完成
        time.sleep(1)
        
    else:
        # 其他worker等待并读取rank 0的过滤结果
        import tempfile
        import hashlib
        temp_dir = tempfile.gettempdir()
        # 🔧 修复：使用相同的输出文件路径hash来找到coordination文件
        output_hash = hashlib.md5(output_path.encode()).hexdigest()[:8]
        unprocessed_data_file = os.path.join(temp_dir, f"unprocessed_data_{output_hash}.json")
        
        # 等待coordination文件出现
        max_wait_time = 60  # 最多等待60秒
        wait_interval = 1
        waited_time = 0
        
        while not os.path.exists(unprocessed_data_file) and waited_time < max_wait_time:
            logging.info(f"Worker {rank}: Waiting for coordination data from rank 0... ({waited_time}s)")
            time.sleep(wait_interval)
            waited_time += wait_interval
        
        if not os.path.exists(unprocessed_data_file):
            logging.error(f"Worker {rank}: Coordination data file not found after {max_wait_time}s")
            raise FileNotFoundError(f"Coordination data file not found: {unprocessed_data_file}")
        
        # 读取coordination数据 - 使用multiprocess-safe方式
        try:
            # 使用multiprocess-safe file handler来读取coordination文件
            temp_file_handler = MultiprocessSafeFileHandler(unprocessed_data_file)
            coordination_data = temp_file_handler.safe_read_json(rank)
            
            if coordination_data is None:
                raise ValueError("Coordination file is empty or could not be read")
            
            # 重构unprocessed_data
            unprocessed_data = []
            for item_list in coordination_data["unprocessed_data"]:
                # 将list转换回tuple
                unprocessed_data.append(tuple(item_list))
            
            logging.info(f"Worker {rank}: Loaded coordination data from rank 0")
            logging.info(f"Worker {rank}: Total unprocessed samples: {len(unprocessed_data)}")
            
        except Exception as e:
            logging.error(f"Worker {rank}: Failed to load coordination data: {e}")
            # 提供更详细的错误信息和恢复建议
            logging.error(f"Worker {rank}: Coordination file: {unprocessed_data_file}")
            if os.path.exists(unprocessed_data_file):
                try:
                    with open(unprocessed_data_file, 'r') as f:
                        content = f.read()
                        logging.error(f"Worker {rank}: File content preview (first 500 chars): {content[:500]}")
                        if len(content) > 500:
                            logging.error(f"Worker {rank}: File content preview (last 500 chars): {content[-500:]}")
                except:
                    logging.error(f"Worker {rank}: Could not read file content for debugging")
            else:
                logging.error(f"Worker {rank}: Coordination file does not exist")
            raise
    
    # 🔧 现在所有worker都有相同的unprocessed_data，进行交叉分配
    total_unprocessed = len(unprocessed_data)
    
    # 🔧 修复：添加详细的调试信息来跟踪分配过程
    logging.info(f"Worker {rank}: Before allocation - total_unprocessed: {total_unprocessed}")
    logging.info(f"Worker {rank}: unprocessed_data sample IDs: {[item[1].get('id', f'index_{item[0]}') for item in unprocessed_data[:20]]}")
    if len(unprocessed_data) > 20:
        logging.info(f"Worker {rank}: ... and {len(unprocessed_data) - 20} more unprocessed samples")
    
    if world_size > 1 and total_unprocessed > 0:
        # 🔧 改进：使用交叉划分（插值划分）而不是连续划分
        # 这样可以确保每个worker都分配到不同复杂度的任务，避免负载不均衡
        worker_unprocessed_data = []
        
        # 🔧 修复：添加详细的分配过程日志
        logging.info(f"Worker {rank}: Starting interleaved allocation with rank={rank}, world_size={world_size}")
        
        # 交叉划分：Worker i 获取索引为 i, i+world_size, i+2*world_size, ... 的任务
        allocated_indices = []
        for idx in range(rank, total_unprocessed, world_size):
            worker_unprocessed_data.append(unprocessed_data[idx])
            allocated_indices.append(idx)
        
        # 🔧 新增：验证分配的索引
        logging.info(f"Worker {rank}: Allocated indices in unprocessed_data: {allocated_indices[:10]}")
        if len(allocated_indices) > 10:
            logging.info(f"Worker {rank}: ... and {len(allocated_indices) - 10} more indices")
        
        to_process = worker_unprocessed_data
        
        logging.info(f"🔄 Worker {rank}: Using interleaved allocation strategy (cross-interleaved)")
        logging.info(f"Worker {rank}: Processing {len(to_process)} samples using interleaved indices")
        logging.info(f"  Allocation pattern: indices {rank}, {rank + world_size}, {rank + 2*world_size}, ...")
        
        # 显示分配的任务示例
        example_indices = [rank + i * world_size for i in range(min(5, len(to_process)))]
        logging.info(f"  Example global indices: {example_indices}")
        if len(to_process) > 5:
            logging.info(f"  ... and {len(to_process) - 5} more samples")
        
        # 显示各worker的任务分配情况
        if rank == 0:
            logging.info(f"Task allocation across all workers:")
            for worker_id in range(world_size):
                worker_task_count = len([idx for idx in range(worker_id, total_unprocessed, world_size)])
                percentage = (worker_task_count / total_unprocessed) * 100
                logging.info(f"  Worker {worker_id}: {worker_task_count} tasks ({percentage:.1f}%)")
        
        # 🔧 新增：任务分配一致性检查
        # 检查各worker分配的task_id是否有重叠
        current_worker_ids = set()
        current_worker_sample_ids = []
        for item in to_process:
            sample_id = item[1].get('id', f'index_{item[0]}')
            current_worker_ids.add(sample_id)
            current_worker_sample_ids.append(sample_id)
        
        logging.info(f"Worker {rank}: Assigned {len(current_worker_ids)} unique task IDs")
        logging.info(f"Worker {rank}: First 10 assigned task IDs: {current_worker_sample_ids[:10]}")
        
        # 🔧 新增：验证分配的唯一性
        if len(current_worker_ids) != len(current_worker_sample_ids):
            logging.error(f"Worker {rank}: CRITICAL - Found duplicate task IDs in worker allocation!")
            logging.error(f"Worker {rank}: Unique IDs: {len(current_worker_ids)}, Total IDs: {len(current_worker_sample_ids)}")
        
        # 保存当前worker的任务分配到临时文件用于调试
        import tempfile
        temp_dir = tempfile.gettempdir()
        # 🔧 修复：使用输出文件hash确保同一任务的worker分配文件在同一组
        output_hash = hashlib.md5(output_path.encode()).hexdigest()[:8]
        allocation_file = os.path.join(temp_dir, f"worker_{rank}_allocation_{output_hash}.json")
        try:
            with open(allocation_file, 'w') as f:
                allocation_data = {
                    "worker_rank": rank,
                    "world_size": world_size,
                    "assigned_task_ids": current_worker_sample_ids,  # 保留顺序
                    "unique_task_ids": list(current_worker_ids),
                    "total_assigned": len(current_worker_sample_ids),
                    "unique_assigned": len(current_worker_ids),
                    "allocated_indices": allocated_indices,
                    "total_unprocessed": total_unprocessed
                }
                json.dump(allocation_data, f, indent=2)
            logging.info(f"Worker {rank}: Task allocation saved to {allocation_file}")
        except Exception as e:
            logging.warning(f"Worker {rank}: Failed to save allocation file: {e}")
            
    else:
        # 单worker模式或没有未处理数据
        to_process = unprocessed_data
        if world_size == 1:
            logging.info(f"Single worker processing {len(to_process)} samples")
        else:
            logging.info(f"Worker {rank}: Processing {len(to_process)} samples")

    samples_to_process_count = len(to_process)
    total_data_count = len(datas)
    if rank == 0:
        existing_count = len(existing_ids)
    else:
        existing_count = coordination_data["existing_count"]
    
    logging.info(f"Worker {rank}: Total samples in data: {total_data_count}")
    logging.info(f"Worker {rank}: Found {existing_count} existing results.")
    if target_task_ids is not None:
        logging.info(f"Worker {rank}: Target task ID filtering: {len(target_task_ids)} target IDs, {skipped_by_task_id_filter} samples filtered out")
    logging.info(f"Worker {rank}: Samples to process: {samples_to_process_count}")

    if samples_to_process_count == 0:
        logging.info(f"Worker {rank}: No new samples to process. Exiting.")
        exit()

    processed_count = 0
    error_count = 0
    skipped_count = 0

    # Main processing loop - modified for concurrency with remote APIs
    logging.info(f"Worker {rank}: Starting processing of {samples_to_process_count} samples...")
    
    # 添加调试信息：显示当前worker要处理的样本ID
    sample_ids_to_process = [item[1].get('id', f'index_{item[0]}') for item in to_process]
    logging.info(f"Worker {rank}: Sample IDs to process: {sample_ids_to_process[:10]}")
    if len(sample_ids_to_process) > 10:
        logging.info(f"Worker {rank}: ... and {len(sample_ids_to_process) - 10} more samples")
    
    if args.inference_type == "local" or args.num_workers == 1: # Sequential for local or single worker remote
        logging.info(f"Worker {rank}: Using sequential processing (inference_type: {args.inference_type}, num_workers: {args.num_workers})")
        for loop_idx, item_tuple in enumerate(tqdm(to_process, desc=f"Worker {rank} Processing {args.approach} predictions")):
            sample_id = item_tuple[1].get('id', f'index_{item_tuple[0]}')
            
            # 二次检查：处理前再次确认该样本是否已经被其他worker处理
            if world_size > 1:
                current_existing_results, current_existing_ids = load_existing_results(output_path, rank)
                if sample_id in current_existing_ids:
                    logging.info(f"Worker {rank}: Sample {sample_id} already processed by another worker, skipping")
                    continue
            
            # Pass args to process_single_sample
            result = process_single_sample(item_tuple, args)
            
            if args.skip_generation:
                # 确保context字段存在，即使可能为空或None
                context_value = result.get("context", "")
                if context_value is None:
                    context_value = ""
                safe_append_to_jsonl(output_path, {"id": result["id"], "answer": "", "context": context_value}, rank)
                continue
            
            if "answer" in result:
                result_dict =  {"id": result["id"], "answer": result["answer"]}
                if args.appendContext:
                    result_dict["context"] = result.get("context", "")
                result_dict["error_info"] = result.get("error_info",[])
                if safe_append_to_jsonl(output_path, result_dict, rank):
                    processed_count += 1
                    logging.debug(f"Worker {rank}: Successfully processed and wrote sample {result['id']}")
                else:
                    error_count += 1
                    logging.error(f"Worker {rank}: Failed to write result for sample {result['id']}")
            elif "skipped" in result and result["skipped"]:
                skipped_count += 1
                logging.info(f"Worker {rank}: Skipped sample {result['id']} due to {result.get('reason', 'unknown')}")
            elif "error" in result:
                error_count += 1
                error_info = result.get('error_details', {})
                if error_info:
                    logging.warning(f"Worker {rank}: Sample {result['id']} failed - Type: {error_info.get('error_type', 'Unknown')}, "
                                  f"Location: {error_info.get('error_location', 'Unknown')}, "
                                  f"Message: {error_info.get('error_message', 'Unknown error')}")
                else:
                    logging.warning(f"Worker {rank}: Sample {result['id']} failed. Error: {result.get('error', 'Unknown error')}")

            if (loop_idx + 1) % 5 == 0: 
                logging.info(f"Worker {rank}: Processed {loop_idx + 1} samples, clearing CUDA cache (if local).")
                if args.inference_type == "local":
                    torch.cuda.empty_cache()
                
            if (processed_count + error_count + skipped_count) % 10 == 0 and (processed_count + error_count + skipped_count) > 0:
                logging.info(f"Worker {rank}: Progress: {processed_count} succeeded, {error_count} failed, {skipped_count} skipped out of {samples_to_process_count}")
    else: # Concurrent for remote APIs with multiple workers/requests
        logging.info(f"Worker {rank}: Using concurrent processing for {args.inference_type} with max_concurrent_requests: {args.max_concurrent_requests}")
        
        # 为并发处理添加二次检查机制
        filtered_to_process = []
        for item_tuple in to_process:
            sample_id = item_tuple[1].get("id", f'index_{item_tuple[0]}')
            if world_size > 1:
                current_existing_results, current_existing_ids = load_existing_results(output_path, rank)
                if sample_id in current_existing_ids:
                    logging.info(f"Worker {rank}: Sample {sample_id} already processed by another worker, skipping from concurrent queue")
                    continue
            filtered_to_process.append(item_tuple)
        
        logging.info(f"Worker {rank}: After filtering, {len(filtered_to_process)} samples remain for concurrent processing")
        
        with ThreadPoolExecutor(max_workers=args.max_concurrent_requests) as executor:
            future_to_sample = {executor.submit(process_single_sample, item_tuple, args): item_tuple for item_tuple in filtered_to_process}
            
            for future in tqdm(as_completed(future_to_sample), total=len(filtered_to_process), desc=f"Worker {rank} Processing remote {args.approach} predictions"):
                item_tuple = future_to_sample[future]
                sample_id = item_tuple[1].get("id", f'index_{item_tuple[0]}') # Get sample_id from item_tuple
                try:
                    result = future.result()
                    if args.skip_generation:
                        # 确保context字段存在，即使可能为空或None
                        context_value = result.get("context", "")
                        if context_value is None:
                            context_value = ""
                        safe_append_to_jsonl(output_path, {"id": result["id"], "answer": "", "context": context_value}, rank)
                        continue
                    
                    if "answer" in result:
                        if safe_append_to_jsonl(output_path, {"id": result["id"], "answer": result["answer"]}, rank):
                            processed_count += 1
                            logging.debug(f"Worker {rank}: Successfully processed and wrote sample {result['id']}")
                        else:
                            error_count += 1
                            logging.error(f"Worker {rank}: Failed to write result for sample {result['id']}")
                    elif "skipped" in result and result["skipped"]:
                        skipped_count += 1
                        logging.info(f"Worker {rank}: Skipped sample {result['id']} due to {result.get('reason', 'unknown')}")
                    elif "error" in result:
                        error_count += 1
                        error_info = result.get('error_details', {})
                        if error_info:
                            logging.warning(f"Worker {rank}: Sample {result.get('id', sample_id)} failed - Type: {error_info.get('error_type', 'Unknown')}, "
                                          f"Location: {error_info.get('error_location', 'Unknown')}, "
                                          f"Message: {error_info.get('error_message', 'Unknown error')}")
                        else:
                            logging.warning(f"Worker {rank}: Sample {result.get('id', sample_id)} failed. Error: {result.get('error', 'Unknown error')}")
                except Exception as exc:
                    error_count += 1
                    tb_str = traceback.format_exc()
                    logging.error(f"Worker {rank}: Sample {sample_id} generated an exception: {exc}\n{tb_str}")
                
                if (processed_count + error_count + skipped_count) % 10 == 0 and (processed_count + error_count + skipped_count) > 0:
                    logging.info(f"Worker {rank}: Progress: {processed_count} succeeded, {error_count} failed, {skipped_count} skipped out of {samples_to_process_count}")

    logging.info(f"\nWorker {rank}: Finished processing! Successfully processed: {processed_count}, Failed: {error_count}, Skipped: {skipped_count}")

    # 只有rank 0负责保存配置文件
    if rank == 0:
        logging.info(f"Worker {rank}: Generating and saving configuration file...")
        try:
            # Generate and save configuration file
            approach = "RAG" if args.approach == "RAG" else "BASELINE"
            config_data = output_manager.generate_config(
                approach=approach,
                args=args,
                rag_config=None  # pred_rag doesn't use rag_config object
            )
            
            # 添加分布式推理信息
            if world_size > 1:
                config_data["experiment_info"]["distributed_inference"] = {
                    "world_size": world_size,
                    "data_parallel": True,
                    "file_locking": True
                }
                
            output_manager.save_config(config_path, config_data)
            logging.info(f"Worker {rank}: Config saved to: {config_path}")
        except Exception as e:
            logging.error(f"Worker {rank}: Error saving config: {e}")
    else:
        logging.info(f"Worker {rank}: Skipping config save (handled by rank 0)")

    logging.info(f"Worker {rank}: Results appended to: {output_path}")
    logging.info(f"Worker {rank}: Task completed")

    # 分析重复的ID
    analyze_duplicate_ids(output_path, rank)

    # 清理临时文件
    cleanup_temp_files(output_path, rank)


