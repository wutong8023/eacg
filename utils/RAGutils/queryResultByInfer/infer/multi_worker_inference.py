#!/usr/bin/env python3
"""
Multi-Worker Inference Script (Fixed GPU Allocation via CUDA_VISIBLE_DEVICES)

This script supports multi-worker inference with fixed GPU allocation per worker.
Each worker runs as an independent process with dedicated GPU(s) via CUDA_VISIBLE_DEVICES.
Uses HuggingFace's device_map="auto" for automatic model placement within each worker.

NEW FEATURES:
- All workers write directly to the same JSONL output file (thread-safe with file locking)
- Support for resuming from existing results using (original_item_id, query_index) pairs to avoid duplicate inference
- Enhanced API call support with better error handling for different inference types

IMPORTANT: Resume functionality now uses (original_item_id, query_index) pairs for more precise 
duplicate detection instead of just using result IDs. This ensures better accuracy when 
filtering already processed items.

Usage:
    # Single worker
    python multi_worker_inference.py --contexts_file contexts.jsonl --output_file results.jsonl --rank 0 --world_size 1
    
    # Multi-worker (launched by external script with CUDA_VISIBLE_DEVICES)
    CUDA_VISIBLE_DEVICES=0 python multi_worker_inference.py --contexts_file contexts.jsonl --output_file results.jsonl --rank 0 --world_size 4
    CUDA_VISIBLE_DEVICES=1 python multi_worker_inference.py --contexts_file contexts.jsonl --output_file results.jsonl --rank 1 --world_size 4
    # ... etc
    
    # Resume from existing results (skip already processed items using original_item_id + query_index)
    python multi_worker_inference.py --contexts_file contexts.jsonl --output_file results.jsonl --resume_from_existing
    
    # API-based inference (no GPU required)
    python multi_worker_inference.py --contexts_file contexts.jsonl --output_file results.jsonl --inference_type togetherai --api_key YOUR_API_KEY --api_model_name meta-llama/Llama-2-7b-chat-hf
"""

import argparse
import json
import os
import logging
import time
import sys
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import traceback
import fcntl

# Import inference system
from utils.RAGutils.queryResultByInfer.query_based_retrieval_inference import QueryBasedInference

# Import conservative configuration
try:
    from utils.RAGutils.queryResultByInfer.infer.conservative_inference_config import (
        setup_conservative_cuda_environment,
        configure_conservative_model_loading,
        monitor_gpu_memory,
        cleanup_gpu_memory,
        set_conservative_torch_settings,
        validate_worker_gpu_isolation,
        CONSERVATIVE_CONFIGS
    )
    CONSERVATIVE_CONFIG_AVAILABLE = True
except ImportError:
    CONSERVATIVE_CONFIG_AVAILABLE = False
    print("Conservative inference config not available, using basic settings")

# Torch imports (optional since we might not use distributed)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Configure logging
def setup_logging(rank: int = 0, verbose: bool = False):
    """Setup logging with file handlers for each worker"""
    
    # Create logs directory if it doesn't exist
    log_dir = "logs/multi_worker_inference"
    os.makedirs(log_dir, exist_ok=True)
    
    # Set log level
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [Worker %(worker_rank)s] - %(message)s'
)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler for this specific worker
    worker_log_file = os.path.join(log_dir, f"worker_{rank}.log")
    file_handler = logging.FileHandler(worker_log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Always debug level for file
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Error-only file handler for quick error checking
    error_log_file = os.path.join(log_dir, f"worker_{rank}_errors.log")
    error_handler = logging.FileHandler(error_log_file, mode='w', encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    # Add custom context to all log records
    class WorkerContextFilter(logging.Filter):
        def __init__(self, rank):
            super().__init__()
            self.rank = rank
            
        def filter(self, record):
            record.worker_rank = self.rank
            return True
    
    worker_filter = WorkerContextFilter(rank)
    for handler in logger.handlers:
        handler.addFilter(worker_filter)
    
    logger.info(f"Worker {rank}: Logging setup complete")
    logger.info(f"Worker {rank}: Log file: {worker_log_file}")
    logger.info(f"Worker {rank}: Error log file: {error_log_file}")
    
    return logger

# Global logger will be set up in main()
logger = None

class MultiWorkerInference:
    """
    Multi-worker inference system with fixed GPU allocation via CUDA_VISIBLE_DEVICES
    """
    
    def __init__(self, 
                 inference_system: QueryBasedInference,
                 output_file: str,
                 rank: int = 0,
                 world_size: int = 1,
                 resume_from_existing: bool = False):
        """
        Initialize multi-worker inference system
        
        Args:
            inference_system: QueryBasedInference instance
            output_file: Output file path for results
            rank: Worker rank (0-based)
            world_size: Total number of workers
            resume_from_existing: Whether to load existing results and skip processed items
        """
        self.inference_system = inference_system
        self.output_file = output_file  # All workers use same file
        self.rank = rank
        self.world_size = world_size
        self.resume_from_existing = resume_from_existing
        
        # No longer modify output file name - all workers use the same file
        
        # Statistics
        self.stats = {
            'total_contexts': 0,
            'successful_inferences': 0,
            'errors': 0,
            'total_time': 0,
            'avg_inference_time': 0
        }
        
        # Load existing results if resuming
        self.existing_pairs = set()
        if self.resume_from_existing:
            self.existing_pairs = self.load_existing_results()
            logger.info(f"Worker {rank}: Loaded {len(self.existing_pairs)} existing (original_item_id, query_index) pairs")
        
        # Log GPU environment
        if TORCH_AVAILABLE and torch.cuda.is_available():
            visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'all')
            gpu_count = torch.cuda.device_count()
            logger.info(f"Worker {rank}/{world_size} initialized")
            logger.info(f"CUDA_VISIBLE_DEVICES: {visible_devices}")
            logger.info(f"Available GPUs for this worker: {gpu_count}")
            logger.info(f"Output file: {self.output_file}")
        else:
            logger.info(f"Worker {rank}/{world_size} initialized (CPU mode)")
            logger.info(f"Output file: {self.output_file}")
    
    def load_existing_results(self) -> set:
        """
        Load existing results from output file to avoid duplicate processing
        Uses (original_item_id, query_index) pairs for more precise identification
        
        Returns:
            Set of existing (original_item_id, query_index) tuples
        """
        existing_pairs = set()
        
        if not os.path.exists(self.output_file):
            logger.info(f"Worker {self.rank}: No existing output file found")
            return existing_pairs
        
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                line_count = 0
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            result = json.loads(line)
                            original_item_id = result.get('original_item_id', '')
                            query_index = result.get('query_index', 0)
                            
                            # Only add if both fields are present and valid
                            if original_item_id and query_index is not None:
                                existing_pairs.add((original_item_id, query_index))
                                line_count += 1
                            else:
                                # Fallback to id-based filtering for backward compatibility
                                result_id = result.get('id', '')
                                if result_id:
                                    # Use id as both original_item_id and query_index 0 for fallback
                                    existing_pairs.add((result_id, 0))
                                    line_count += 1
                                    
                        except json.JSONDecodeError as e:
                            logger.warning(f"Worker {self.rank}: Invalid JSON in existing results: {e}")
                            continue
                
                logger.info(f"Worker {self.rank}: Found {len(existing_pairs)} existing (original_item_id, query_index) pairs from {line_count} lines")
                
                # Log a few examples for debugging
                if len(existing_pairs) > 0:
                    examples = list(existing_pairs)[:5]
                    logger.debug(f"Worker {self.rank}: Example existing pairs: {examples}")
                
        except Exception as e:
            logger.error(f"Worker {self.rank}: Error loading existing results: {e}")
            logger.error(f"Worker {self.rank}: Will proceed without resuming")
        
        return existing_pairs
    
    def filter_existing_contexts(self, contexts: List[Dict]) -> List[Dict]:
        """
        Filter out contexts that have already been processed based on (original_item_id, query_index) pairs
        This should be called BEFORE distributing contexts to workers
        
        Args:
            contexts: Full list of contexts to filter
            
        Returns:
            Filtered list of contexts that haven't been processed yet
        """
        if not self.resume_from_existing or not self.existing_pairs:
            logger.info(f"Worker {self.rank}: No filtering needed - resume_from_existing={self.resume_from_existing}, existing_pairs={len(self.existing_pairs)}")
            return contexts
        
        original_count = len(contexts)
        filtered_contexts = []
        
        for ctx in contexts:
            original_item_id = ctx.get('original_item_id', '')
            query_index = ctx.get('query_index', 0)
            
            # Create pair for comparison
            context_pair = (original_item_id, query_index)
            
            # If no original_item_id, fallback to id-based filtering
            if not original_item_id:
                fallback_pair = (ctx.get('id', ''), 0)
                if fallback_pair not in self.existing_pairs:
                    filtered_contexts.append(ctx)
            else:
                if context_pair not in self.existing_pairs:
                    filtered_contexts.append(ctx)
        
        filtered_count = original_count - len(filtered_contexts)
        
        if filtered_count > 0:
            logger.info(f"Worker {self.rank}: Filtered out {filtered_count} already processed contexts globally")
            logger.info(f"Worker {self.rank}: Remaining contexts after filtering: {len(filtered_contexts)}")
        else:
            logger.info(f"Worker {self.rank}: No contexts filtered (all are new)")
        
        return filtered_contexts

    def get_data_slice(self, contexts: List[Dict]) -> List[Dict]:
        """
        Get data slice for this worker based on rank and world size using interleaved allocation
        NOTE: This now expects contexts to be already filtered for existing results
        
        Args:
            contexts: Filtered list of contexts (already processed items removed)
            
        Returns:
            Slice of contexts for this worker
        """
        # Use interleaved allocation: Worker i gets indices i, i+world_size, i+2*world_size, ...
        worker_contexts = []
        for i in range(self.rank, len(contexts), self.world_size):
            worker_contexts.append(contexts[i])
        
        logger.info(f"Worker {self.rank}: processing {len(worker_contexts)} contexts "
                   f"(interleaved allocation from {len(contexts)} total filtered contexts)")
        logger.info(f"Worker {self.rank}: allocation pattern - indices: {self.rank}, "
                   f"{self.rank + self.world_size}, {self.rank + 2*self.world_size}, ...")
        
        return worker_contexts
    
    def process_context_item(self, context_item: Dict, item_index: int) -> Dict:
        """
        Process a single context item and generate inference
        
        Args:
            context_item: Context item dictionary
            item_index: Index of the item in worker's data slice
            
        Returns:
            Result dictionary with inference
        """
        start_time = time.time()
        item_id = f'item_{item_index}'  # Default ID
        
        try:
            # Step 1: Log raw context item structure for debugging
            logger.debug(f"Worker {self.rank} - Processing item {item_index}")
            logger.debug(f"Worker {self.rank} - Raw context item keys: {list(context_item.keys())}")
            logger.debug(f"Worker {self.rank} - Raw context item (first 500 chars): {str(context_item)[:500]}...")
            
            # Step 2: Extract basic information
            item_id = context_item.get('id', f'item_{item_index}')
            logger.debug(f"Worker {self.rank} - Item {item_id}: Starting processing")
            
            # Step 3: Check if this is a successful context item
            success_flag = context_item.get('success', True)
            logger.debug(f"Worker {self.rank} - Item {item_id}: Success flag = {success_flag}")
            
            if not success_flag:
                error_msg = context_item.get('error', 'Unknown error from context generation')
                logger.error(f"Worker {self.rank} - Item {item_id}: Context generation failed: {error_msg}")
                logger.error(f"Worker {self.rank} - Item {item_id}: Full context item: {context_item}")
                return {
                    'id': item_id,
                    'error': f'Context generation failed: {error_msg}',
                    'processing_time': time.time() - start_time,
                    'worker_rank': self.rank,
                    'debug_info': {
                        'context_item_keys': list(context_item.keys()),
                        'success_flag': success_flag
                    },
                    'success': False
                }
            
            # Step 4: Extract fields with detailed logging
            query = context_item.get('query', '')
            context = context_item.get('context', '')
            extra_info = context_item.get('extra_info', '')
            retrieval_method = context_item.get('retrieval_method', 'unknown')
            
            logger.debug(f"Worker {self.rank} - Item {item_id}: Field extraction results:")
            logger.debug(f"  - query length: {len(query)}")
            logger.debug(f"  - extra_info length: {len(extra_info)}")
            logger.debug(f"  - context length: {len(context)}")
            logger.debug(f"  - retrieval_method: {retrieval_method}")
            logger.debug(f"  - query preview: '{query[:200]}...'")
            logger.debug(f"  - context preview: '{context[:200]}...'")
            
            # Step 5: Validate required fields
            if not query.strip():
                logger.error(f"Worker {self.rank} - Item {item_id}: Query validation failed")
                logger.error(f"Worker {self.rank} - Item {item_id}: Query value: '{query}'")
                logger.error(f"Worker {self.rank} - Item {item_id}: Available keys: {list(context_item.keys())}")
                logger.error(f"Worker {self.rank} - Item {item_id}: Full context item: {context_item}")
                return {
                    'id': item_id,
                    'error': f'No query found. Query value: "{query}", Available keys: {list(context_item.keys())}',
                    'processing_time': time.time() - start_time,
                    'worker_rank': self.rank,
                    'debug_info': {
                        'context_item_keys': list(context_item.keys()),
                        'query_length': len(query),
                        'query_stripped_length': len(query.strip())
                    },
                    'success': False
                }
            
            if not context.strip():
                logger.error(f"Worker {self.rank} - Item {item_id}: Context validation failed")
                logger.error(f"Worker {self.rank} - Item {item_id}: Context exists in dict: {'context' in context_item}")
                logger.error(f"Worker {self.rank} - Item {item_id}: Context value preview: '{context[:200]}...'")
                logger.error(f"Worker {self.rank} - Item {item_id}: Full context item: {context_item}")
                return {
                    'id': item_id,
                    'error': f'No context found. Context field exists: {"context" in context_item}, Context length: {len(context)}',
                    'processing_time': time.time() - start_time,
                    'worker_rank': self.rank,
                    'debug_info': {
                        'context_item_keys': list(context_item.keys()),
                        'context_field_exists': 'context' in context_item,
                        'context_length': len(context),
                        'context_stripped_length': len(context.strip())
                    },
                    'success': False
                }
            
            logger.info(f"Worker {self.rank} - Item {item_id}: Validation passed, starting inference")
            
            # Clear GPU cache before processing to avoid memory issues
            if TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    if CONSERVATIVE_CONFIG_AVAILABLE:
                        cleanup_gpu_memory(self.rank)
                        monitor_gpu_memory(self.rank, f"Before Inference - Item {item_id}")
                    else:
                        torch.cuda.empty_cache()
                        if torch.cuda.device_count() > 0:
                            device = torch.cuda.current_device()
                            memory_before = torch.cuda.memory_allocated(device) / 1024**3
                            logger.debug(f"Worker {self.rank} - Item {item_id}: Memory before inference: {memory_before:.2f}GB")
                except Exception as mem_e:
                    logger.warning(f"Worker {self.rank} - Item {item_id}: Memory monitoring failed: {mem_e}")
            
            # Generate answer using inference system (sequential processing)
            inference_start = time.time()
            
            try:
                # Log inference type for debugging
                inference_type = getattr(self.inference_system, 'inference_type', 'unknown')
                logger.debug(f"Worker {self.rank} - Item {item_id}: Using inference type: {inference_type}")
                
                answer = self.inference_system.generate_answer(query, context, retrieval_method, extra_info)
                
                # Validate answer for API calls
                if not answer or (isinstance(answer, str) and not answer.strip()):
                    logger.warning(f"Worker {self.rank} - Item {item_id}: Empty answer received from inference")
                    answer = "[Empty response from model]"
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Handle different types of errors
                if TORCH_AVAILABLE and torch.cuda.is_available() and "out of memory" in error_str:
                    # Handle GPU OOM error for local inference
                    logger.error(f"Worker {self.rank} - Item {item_id}: GPU OOM error: {e}")
                    torch.cuda.empty_cache()
                    return {
                        'id': item_id,
                        'error': f'GPU Out of Memory: {str(e)}',
                        'processing_time': time.time() - start_time,
                        'worker_rank': self.rank,
                        'inference_type': getattr(self.inference_system, 'inference_type', 'unknown'),
                        'success': False
                    }
                elif "api" in error_str or "rate limit" in error_str or "quota" in error_str:
                    # Handle API-related errors
                    logger.error(f"Worker {self.rank} - Item {item_id}: API error: {e}")
                    return {
                        'id': item_id,
                        'error': f'API Error: {str(e)}',
                        'processing_time': time.time() - start_time,
                        'worker_rank': self.rank,
                        'inference_type': getattr(self.inference_system, 'inference_type', 'unknown'),
                        'success': False
                    }
                elif "timeout" in error_str or "connection" in error_str:
                    # Handle network/timeout errors
                    logger.error(f"Worker {self.rank} - Item {item_id}: Network/timeout error: {e}")
                    return {
                        'id': item_id,
                        'error': f'Network/Timeout Error: {str(e)}',
                        'processing_time': time.time() - start_time,
                        'worker_rank': self.rank,
                        'inference_type': getattr(self.inference_system, 'inference_type', 'unknown'),
                        'success': False
                    }
                else:
                    # Generic inference error
                    logger.error(f"Worker {self.rank} - Item {item_id}: Inference error: {e}")
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return {
                        'id': item_id,
                        'error': f'Inference error: {str(e)}',
                        'processing_time': time.time() - start_time,
                        'worker_rank': self.rank,
                        'inference_type': getattr(self.inference_system, 'inference_type', 'unknown'),
                        'success': False
                    }
                
            inference_time = time.time() - inference_start
            
            # Clear GPU cache after processing and monitor memory usage
            if TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    if CONSERVATIVE_CONFIG_AVAILABLE:
                        monitor_gpu_memory(self.rank, f"After Inference - Item {item_id}")
                        cleanup_gpu_memory(self.rank)
                        monitor_gpu_memory(self.rank, f"After Cleanup - Item {item_id}")
                    else:
                        if torch.cuda.device_count() > 0:
                            device = torch.cuda.current_device()
                            memory_after = torch.cuda.memory_allocated(device) / 1024**3
                            torch.cuda.empty_cache()
                            memory_cleaned = torch.cuda.memory_allocated(device) / 1024**3
                            logger.debug(f"Worker {self.rank} - Item {item_id}: Memory after inference: {memory_after:.2f}GB, "
                                       f"after cleanup: {memory_cleaned:.2f}GB")
                except Exception as mem_e:
                    logger.warning(f"Worker {self.rank} - Item {item_id}: Memory cleanup failed: {mem_e}")
            
            total_time = time.time() - start_time
            
            # Prepare result
            result = {
                'id': item_id,
                'original_item_id': context_item.get('original_item_id', ''),
                'query_index': context_item.get('query_index', 0),
                'dedup_key': context_item.get('dedup_key', ''),
                'query': query,
                'target_api': context_item.get('target_api', ''),
                'dependencies': context_item.get('dependencies', {}),
                'retrieval_method': retrieval_method,
                'context_length': len(context),
                'answer': answer,
                'retrieval_time': context_item.get('retrieval_time', 0),
                'inference_time': inference_time,
                'total_time': context_item.get('retrieval_time', 0) + inference_time,
                'worker_rank': self.rank,
                'inference_type': getattr(self.inference_system, 'inference_type', 'unknown'),
                'success': True
            }
            
            logger.info(f"Worker {self.rank} - Item {item_id}: inference completed in {inference_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.stats['errors'] += 1
            error_msg = str(e)
            
            # Detailed error logging
            logger.error(f"Worker {self.rank} - EXCEPTION in item {item_index} (ID: {item_id})")
            logger.error(f"Worker {self.rank} - Exception type: {type(e).__name__}")
            logger.error(f"Worker {self.rank} - Exception message: {error_msg}")
            logger.error(f"Worker {self.rank} - Full traceback:")
            logger.error(traceback.format_exc())
            
            # Log context item for debugging if exception occurred during processing
            try:
                logger.error(f"Worker {self.rank} - Context item that caused exception: {context_item}")
            except Exception as log_e:
                logger.error(f"Worker {self.rank} - Could not log context item: {log_e}")
            
            # Log system state
            if TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    if torch.cuda.device_count() > 0:
                        device = torch.cuda.current_device()
                        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
                        logger.error(f"Worker {self.rank} - GPU memory at exception: {memory_allocated:.2f}GB")
                except Exception as gpu_e:
                    logger.error(f"Worker {self.rank} - Could not get GPU info: {gpu_e}")
            
            return {
                'id': item_id,
                'error': f'Exception: {error_msg}',
                'exception_type': type(e).__name__,
                'processing_time': time.time() - start_time,
                'worker_rank': self.rank,
                'debug_info': {
                    'item_index': item_index,
                    'exception_details': traceback.format_exc(),
                    'context_item_available': context_item is not None
                },
                'success': False
            }
    
    def process_contexts(self, contexts: List[Dict], max_samples: Optional[int] = None) -> None:
        """
        Process contexts for assigned data slice
        
        Args:
            contexts: Full list of contexts
            max_samples: Maximum number of samples to process (per worker)
        """
        # Step 1: Filter out already processed contexts BEFORE distributing to workers
        filtered_contexts = self.filter_existing_contexts(contexts)
        
        # Step 2: Get data slice for this worker from filtered contexts
        worker_contexts = self.get_data_slice(filtered_contexts)
        
        # Step 3: Apply max_samples limit if specified
        if max_samples and len(worker_contexts) > max_samples:
            worker_contexts = worker_contexts[:max_samples]
            logger.info(f"Worker {self.rank}: Limited to {max_samples} samples")
        
        self.stats['total_contexts'] = len(worker_contexts)
        
        logger.info(f"Worker {self.rank}: Processing {len(worker_contexts)} contexts")
        
        # Step 4: Process contexts with progress bar
        total_inference_time = 0
        
        # Use safe append instead of opening file for writing
        for i, context_item in enumerate(tqdm(worker_contexts, desc=f"Worker {self.rank} inference")):
            result = self.process_context_item(context_item, i)
            
            # Accumulate timing statistics
            if result.get('success', False):
                self.stats['successful_inferences'] += 1
                total_inference_time += result.get('inference_time', 0)
            else:
                self.stats['errors'] += 1
            
            # Write result to file using safe append with file locking
            success = safe_append_to_jsonl(self.output_file, result)
            if not success:
                logger.error(f"Worker {self.rank}: Failed to write result for item {result.get('id', 'unknown')}")
        
        # Calculate final statistics
        if self.stats['successful_inferences'] > 0:
            self.stats['avg_inference_time'] = total_inference_time / self.stats['successful_inferences']
    
    def print_statistics(self):
        """Print worker statistics"""
        print(f"\n{'='*60}")
        print(f"WORKER {self.rank} INFERENCE STATISTICS")
        print(f"{'='*60}")
        print(f"Total contexts processed: {self.stats['total_contexts']}")
        print(f"Successful inferences: {self.stats['successful_inferences']}")
        print(f"Errors: {self.stats['errors']}")
        
        if self.stats['total_contexts'] > 0:
            success_rate = self.stats['successful_inferences'] / self.stats['total_contexts']
            print(f"Success rate: {success_rate:.1%}")
        
        if self.stats['avg_inference_time'] > 0:
            print(f"Average inference time: {self.stats['avg_inference_time']:.2f}s")
        
        if self.resume_from_existing and len(self.existing_pairs) > 0:
            print(f"Existing results found: {len(self.existing_pairs)}")
        
        print(f"Results saved to: {self.output_file}")


def safe_append_to_jsonl(file_path: str, data: Dict) -> bool:
    """Safely append data to JSONL file with file locking"""
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            # Acquire file lock
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(data, f, ensure_ascii=False)
                f.write('\n')
                f.flush()  # Ensure data is written to disk
            finally:
                # Release file lock
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return True
    except Exception as e:
        if logger:
            logger.error(f"Error writing to file {file_path}: {e}")
        else:
            print(f"Error writing to file {file_path}: {e}")
        return False


def merge_worker_outputs(base_output_file: str, world_size: int):
    """
    Merge outputs from all workers into a single file
    
    Args:
        base_output_file: Base output file path
        world_size: Number of workers
    """
    if world_size == 1:
        return
    
    if logger:
        logger.info("Merging worker outputs...")
    else:
        print("Merging worker outputs...")
    
    base_name, ext = os.path.splitext(base_output_file)
    
    with open(base_output_file, 'w', encoding='utf-8') as outfile:
        total_lines = 0
        for rank in range(world_size):
            worker_file = f"{base_name}_rank{rank}{ext}"
            if os.path.exists(worker_file):
                with open(worker_file, 'r', encoding='utf-8') as infile:
                    lines_written = 0
                    for line in infile:
                        outfile.write(line)
                        lines_written += 1
                    total_lines += lines_written
                    if logger:
                        logger.info(f"Merged {lines_written} lines from worker {rank}")
                    else:
                        print(f"Merged {lines_written} lines from worker {rank}")
                
                # Clean up worker file
                os.remove(worker_file)
            else:
                if logger:
                    logger.warning(f"Worker file not found: {worker_file}")
                else:
                    print(f"Worker file not found: {worker_file}")
    
    if logger:
        logger.info(f"Merged output saved to {base_output_file} ({total_lines} total lines)")
    else:
        print(f"Merged output saved to {base_output_file} ({total_lines} total lines)")


def load_contexts_from_file(contexts_file: str) -> List[Dict]:
    """
    Load contexts from JSONL file
    
    Args:
        contexts_file: Path to contexts file
        
    Returns:
        List of context dictionaries
    """
    if logger:
        logger.info(f"Loading contexts from: {contexts_file}")
    else:
        print(f"Loading contexts from: {contexts_file}")
    
    try:
        contexts = []
        total_lines = 0
        successful_contexts = 0
        failed_contexts = 0
        invalid_json_lines = 0
        
        with open(contexts_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    total_lines += 1
                    try:
                        context_item = json.loads(line)
                        
                        # Log first few items for debugging
                        if line_num <= 3:
                            if logger:
                                logger.debug(f"Sample context item {line_num}: {context_item}")
                            else:
                                print(f"Sample context item {line_num}: {context_item}")
                        
                        # Check if context is successful
                        if context_item.get('success', False):
                            contexts.append(context_item)
                            successful_contexts += 1
                        else:
                            failed_contexts += 1
                            error_msg = context_item.get('error', 'No error message')
                            if failed_contexts <= 5:  # Log first few failures
                                if logger:
                                    logger.warning(f"Failed context on line {line_num}: {error_msg}")
                                else:
                                    print(f"Failed context on line {line_num}: {error_msg}")
                            
                    except json.JSONDecodeError as e:
                        invalid_json_lines += 1
                        if invalid_json_lines <= 5:  # Log first few JSON errors
                            if logger:
                                logger.warning(f"Invalid JSON on line {line_num}: {e}")
                            else:
                                print(f"Invalid JSON on line {line_num}: {e}")
                        continue
        
        summary_lines = [
            f"Context loading summary:",
            f"  Total lines processed: {total_lines}",
            f"  Successful contexts: {successful_contexts}",
            f"  Failed contexts: {failed_contexts}",
            f"  Invalid JSON lines: {invalid_json_lines}",
            f"  Final contexts loaded: {len(contexts)}"
        ]
        
        for line in summary_lines:
            if logger:
                logger.info(line)
            else:
                print(line)
        
        if len(contexts) == 0:
            error_lines = [
                "CRITICAL: No successful contexts found! This might be the main issue.",
                "This means either:",
                "  1. All contexts have 'success': false",
                "  2. Context format is different than expected",
                "  3. File is empty or corrupted"
            ]
            
            for line in error_lines:
                if logger:
                    logger.error(line)
                else:
                    print(line)
            
            # Let's also try loading all contexts (including failed ones) for debugging
            debug_msg = "Attempting to load all contexts (including failed ones) for debugging..."
            if logger:
                logger.info(debug_msg)
            else:
                print(debug_msg)
                
            all_contexts = []
            with open(contexts_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            context_item = json.loads(line)
                            all_contexts.append(context_item)
                            if line_num <= 5:  # Show first 5 items
                                debug_lines = [
                                    f"DEBUG sample {line_num}: keys={list(context_item.keys())}",
                                    f"DEBUG sample {line_num}: success={context_item.get('success', 'N/A')}",
                                    f"DEBUG sample {line_num}: query={context_item.get('query', 'N/A')[:100]}...",
                                    f"DEBUG sample {line_num}: context_length={len(context_item.get('context', ''))}"
                                ]
                                for debug_line in debug_lines:
                                    if logger:
                                        logger.error(debug_line)
                                    else:
                                        print(debug_line)
                        except json.JSONDecodeError as json_e:
                            if line_num <= 3:
                                error_line = f"JSON decode error on line {line_num}: {json_e}"
                                if logger:
                                    logger.error(error_line)
                                else:
                                    print(error_line)
                            continue
            
            final_msg = f"Total contexts in file (including failed): {len(all_contexts)}"
            if logger:
                logger.error(final_msg)
            else:
                print(final_msg)
                
            if len(all_contexts) > 0:
                debug_msg = "Using all contexts (including failed ones) for debugging"
                if logger:
                    logger.error(debug_msg)
                else:
                    print(debug_msg)
                return all_contexts  # Return all contexts for debugging
            else:
                error_msg = "No contexts found at all - file might be empty or corrupted"
                if logger:
                    logger.error(error_msg)
                else:
                    print(error_msg)
                raise ValueError(f"No contexts found in file: {contexts_file}")
        
        return contexts
        
    except Exception as e:
        error_msg = f"Error loading contexts from {contexts_file}: {e}"
        traceback_msg = f"Traceback: {traceback.format_exc()}"
        
        if logger:
            logger.error(error_msg)
            logger.error(traceback_msg)
        else:
            print(error_msg)
            print(traceback_msg)
        raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Multi-worker inference for contexts with fixed GPU allocation via CUDA_VISIBLE_DEVICES")
    
    # Input/Output arguments
    parser.add_argument("--contexts_file", type=str, required=True, help="Path to contexts JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file for results")
    
    # Worker configuration
    parser.add_argument("--rank", type=int, default=0, help="Worker rank (0-based)")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of workers")
    
    # Processing arguments
    parser.add_argument("--max_samples_per_worker", type=int, help="Maximum number of samples per worker")
    
    # Inference arguments
    parser.add_argument("--model_path", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="Model path or name")
    parser.add_argument("--inference_type", type=str, default="local", choices=["local", "huggingface", "togetherai","qdd"], help="Inference type")
    parser.add_argument("--api_key", type=str, help="API key for remote inference")
    parser.add_argument("--api_model_name", type=str, help="Model name for API inference")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    
    # Control arguments
    parser.add_argument("--verbose","-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--resume_from_existing", action="store_true", help="Resume inference from existing output file")
    
    args = parser.parse_args()
    
    # Setup logging with worker-specific files
    global logger
    logger = setup_logging(rank=args.rank, verbose=args.verbose)
    
    logger.info(f"Worker {args.rank}: Starting multi-worker inference")
    logger.info(f"Worker {args.rank}: Arguments: {vars(args)}")
    
    # Log GPU environment
    if TORCH_AVAILABLE and torch.cuda.is_available():
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'all')
        gpu_count = torch.cuda.device_count()
        logger.info(f"Worker {args.rank}: CUDA_VISIBLE_DEVICES = {visible_devices}")
        logger.info(f"Worker {args.rank}: Available GPUs = {gpu_count}")
    
        # Apply conservative PyTorch settings if available
        if CONSERVATIVE_CONFIG_AVAILABLE:
            try:
                set_conservative_torch_settings()
                logger.info(f"Worker {args.rank}: Conservative PyTorch settings applied")
            except Exception as e:
                logger.warning(f"Worker {args.rank}: Failed to apply conservative settings: {e}")
    else:
        logger.info(f"Worker {args.rank}: Running in CPU mode")
    
    try:
        # Load contexts (all workers load the same data)
        contexts = load_contexts_from_file(args.contexts_file)
        
        if not contexts:
            logger.error("No contexts loaded, exiting")
            return
        
        # Configure conservative model loading
        if CONSERVATIVE_CONFIG_AVAILABLE:
            try:
                configure_conservative_model_loading()
                logger.info(f"Worker {args.rank}: Conservative model loading configured")
            except Exception as e:
                logger.warning(f"Worker {args.rank}: Failed to configure conservative model loading: {e}")
        
        # Initialize inference system
        try:
            inference_system = QueryBasedInference(
                model_path=args.model_path,
                inference_type=args.inference_type,
                api_key=args.api_key,
                api_model_name=args.api_model_name,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
            
            # Log model loading status for local inference
            if args.inference_type == "local" and TORCH_AVAILABLE and torch.cuda.is_available():
                if hasattr(inference_system, 'model') and inference_system.model is not None:
                    # Model should be automatically placed by HuggingFace device_map="auto"
                    device_map = getattr(inference_system.model, 'hf_device_map', None)
                    logger.info(f"Worker {args.rank}: Model loaded with device_map: {device_map}")
                    
                    # Monitor GPU memory if available
                    if CONSERVATIVE_CONFIG_AVAILABLE:
                        try:
                            monitor_gpu_memory(args.rank, "Model Loading")
                        except Exception as e:
                            logger.warning(f"Worker {args.rank}: Memory monitoring failed: {e}")
                    else:
                        # Basic memory logging for all available devices
                        for device_id in range(torch.cuda.device_count()):
                            try:
                                memory_allocated = torch.cuda.memory_allocated(device_id) / 1024**3
                                memory_reserved = torch.cuda.memory_reserved(device_id) / 1024**3
                                logger.info(f"Worker {args.rank}: GPU {device_id} - "
                                           f"Memory allocated: {memory_allocated:.2f}GB, "
                                           f"Memory reserved: {memory_reserved:.2f}GB")
                            except Exception as e:
                                logger.warning(f"Worker {args.rank}: Could not get memory info for GPU {device_id}: {e}")
                    
                if hasattr(inference_system, 'tokenizer') and inference_system.tokenizer is not None:
                    logger.info(f"Worker {args.rank}: Tokenizer ready")
            elif args.inference_type in ["huggingface", "togetherai", "qdd"]:
                # API-based inference
                logger.info(f"Worker {args.rank}: Using {args.inference_type} API inference")
                logger.info(f"Worker {args.rank}: Model: {args.api_model_name or args.model_path}")
                logger.info(f"Worker {args.rank}: API key configured: {bool(args.api_key)}")
            else:
                # Other inference types or CPU mode
                logger.info(f"Worker {args.rank}: Using {args.inference_type} inference (non-GPU mode)")
                if hasattr(inference_system, 'model') and inference_system.model is not None:
                    logger.info(f"Worker {args.rank}: Model loaded successfully")
                if hasattr(inference_system, 'tokenizer') and inference_system.tokenizer is not None:
                    logger.info(f"Worker {args.rank}: Tokenizer ready")
                        
        except Exception as e:
            logger.error(f"Worker {args.rank}: Failed to initialize inference system: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return
        
        # Initialize multi-worker inference
        multi_inference = MultiWorkerInference(
            inference_system=inference_system,
            output_file=args.output_file,
            rank=args.rank,
            world_size=args.world_size,
            resume_from_existing=args.resume_from_existing
        )
        
        # Process contexts
        start_time = time.time()
        multi_inference.process_contexts(contexts, args.max_samples_per_worker)
        total_time = time.time() - start_time
        
        # Print statistics
        multi_inference.print_statistics()
        logger.info(f"Worker {args.rank} completed in {total_time:.2f} seconds")
        
        # Note: No longer need to merge outputs since all workers write to the same file
        logger.info(f"Worker {args.rank}: Results written directly to {args.output_file}")
        
    except Exception as e:
        logger.error(f"Worker {args.rank}: Fatal error: {e}")
        logger.error(f"Worker {args.rank}: Fatal error traceback: {traceback.format_exc()}")
        return


if __name__ == "__main__":
    main() 