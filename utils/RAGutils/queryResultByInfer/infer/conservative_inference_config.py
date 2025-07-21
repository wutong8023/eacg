#!/usr/bin/env python3
"""
Conservative Inference Configuration

This module provides conservative settings and utilities for GPU memory management
during multi-worker inference to prevent OOM errors and conflicts.
"""

import os
import logging
import torch
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Conservative configuration settings
CONSERVATIVE_CONFIGS = {
    'max_memory_per_gpu': 0.8,  # Use at most 80% of GPU memory
    'batch_size': 1,            # Process one item at a time
    'enable_gradient_checkpointing': True,
    'use_fp16': True,           # Use half precision to save memory
    'max_seq_length': 2048,     # Limit sequence length
    'cache_cleanup_frequency': 1,  # Clean cache after every inference
}

def setup_conservative_cuda_environment(rank: int, world_size: int, total_gpus: int) -> List[int]:
    """
    Setup conservative CUDA environment for a specific worker
    
    Args:
        rank: Worker rank
        world_size: Total number of workers
        total_gpus: Total number of available GPUs
        
    Returns:
        List of GPU device IDs allocated to this worker
    """
    gpus_per_worker = total_gpus // world_size
    
    if gpus_per_worker == 0:
        # Not enough GPUs, use round-robin allocation
        gpu_id = rank % total_gpus
        gpu_devices = [gpu_id]
        logger.warning(f"Worker {rank}: Not enough GPUs ({total_gpus}) for {world_size} workers, "
                      f"using shared GPU {gpu_id}")
    else:
        # Calculate GPU range for this worker
        start_gpu = rank * gpus_per_worker
        end_gpu = start_gpu + gpus_per_worker
        gpu_devices = list(range(start_gpu, end_gpu))
        logger.info(f"Worker {rank}: Allocated GPUs {gpu_devices}")
    
    # Set CUDA_VISIBLE_DEVICES for this worker to isolate GPU access
    worker_cuda_devices = ','.join(map(str, range(len(gpu_devices))))
    os.environ['CUDA_VISIBLE_DEVICES'] = worker_cuda_devices
    
    # Set primary GPU device (now it's device 0 in worker's view)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        logger.info(f"Worker {rank}: Set primary device to cuda:0 (physical GPU {gpu_devices[0]})")
    
    return gpu_devices

def configure_conservative_model_loading():
    """Configure conservative settings for model loading"""
    # Set conservative PyTorch settings
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Limit number of threads
    torch.set_num_threads(2)
    
    logger.info("Applied conservative model loading configuration")

def monitor_gpu_memory(rank: int, stage: str):
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        memory_free = (torch.cuda.get_device_properties(device).total_memory - 
                      torch.cuda.memory_reserved(device)) / 1024**3
        
        logger.debug(f"Worker {rank} - {stage}: "
                    f"Allocated: {memory_allocated:.2f}GB, "
                    f"Reserved: {memory_reserved:.2f}GB, "
                    f"Free: {memory_free:.2f}GB")

def cleanup_gpu_memory(rank: int):
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.debug(f"Worker {rank}: GPU memory cleaned up")

def set_conservative_torch_settings():
    """Set conservative PyTorch settings"""
    # Disable autograd for inference
    torch.autograd.set_grad_enabled(False)
    
    # Conservative memory settings
    torch.backends.cuda.max_split_size_mb = 512
    
    # Limit number of threads
    if torch.get_num_threads() > 4:
        torch.set_num_threads(4)
    
    logger.info("Applied conservative PyTorch settings")

def validate_worker_gpu_isolation(rank: int, world_size: int):
    """Validate that workers have proper GPU isolation"""
    if torch.cuda.is_available():
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        current_device = torch.cuda.current_device()
        device_count = torch.cuda.device_count()
        
        logger.info(f"Worker {rank}: GPU isolation check - "
                   f"CUDA_VISIBLE_DEVICES='{visible_devices}', "
                   f"current_device={current_device}, "
                   f"device_count={device_count}")

def get_conservative_model_kwargs():
    """
    Get conservative model loading kwargs
    """
    return {
        'device_map': {'': 0},  # Force model to use device 0 only
        'torch_dtype': torch.float16,  # Use FP16 to save memory
        'low_cpu_mem_usage': True,
        'offload_folder': None,  # Don't offload to CPU
        'use_cache': False,  # Disable KV cache to save memory
        'trust_remote_code': True,
    }

def get_conservative_generation_kwargs():
    """
    Get conservative generation kwargs
    """
    return {
        'do_sample': True,
        'num_beams': 1,  # No beam search
        'early_stopping': True,
        'use_cache': False,  # Disable KV cache
        'pad_token_id': None,  # Will be set by tokenizer
        'eos_token_id': None,  # Will be set by tokenizer
        'output_attentions': False,
        'output_hidden_states': False,
        'return_dict_in_generate': False,
    } 