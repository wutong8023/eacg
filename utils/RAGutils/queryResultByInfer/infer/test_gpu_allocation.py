#!/usr/bin/env python3
"""
Test GPU Allocation Functionality

This script tests the GPU allocation logic for multi-worker inference.
"""

import os
import sys

def test_gpu_allocation_logic():
    """Test GPU allocation logic for different configurations"""
    
    print("Testing GPU Allocation Logic")
    print("=" * 50)
    
    # Test cases: (total_gpus, num_workers)
    test_cases = [
        (8, 4),   # 8 GPUs, 4 workers -> 2 GPUs per worker
        (8, 2),   # 8 GPUs, 2 workers -> 4 GPUs per worker  
        (4, 4),   # 4 GPUs, 4 workers -> 1 GPU per worker
        (6, 3),   # 6 GPUs, 3 workers -> 2 GPUs per worker
        (8, 3),   # 8 GPUs, 3 workers -> 2 GPUs per worker (remainder)
        (4, 8),   # 4 GPUs, 8 workers -> insufficient GPUs
    ]
    
    for total_gpus, num_workers in test_cases:
        print(f"\nTest case: {total_gpus} GPUs, {num_workers} workers")
        print(f"{'='*40}")
        
        gpus_per_worker = total_gpus // num_workers
        remainder_gpus = total_gpus % num_workers
        
        if gpus_per_worker == 0:
            print(f"❌ Insufficient GPUs: Each worker would get 0 GPUs")
            print(f"   Fallback to shared GPU allocation needed")
            # Show fallback allocation
            for worker in range(num_workers):
                gpu_id = worker % total_gpus
                print(f"   Worker {worker}: GPU [{gpu_id}] (shared)")
        else:
            print(f"✅ GPU allocation: {gpus_per_worker} GPUs per worker")
            if remainder_gpus > 0:
                print(f"   Note: {remainder_gpus} GPUs will be unused")
            
            # Show detailed allocation
            for worker in range(num_workers):
                start_gpu = worker * gpus_per_worker
                end_gpu = start_gpu + gpus_per_worker
                gpu_list = list(range(start_gpu, end_gpu))
                print(f"   Worker {worker}: GPUs {gpu_list}")
            
            # Show unused GPUs if any
            used_gpus = num_workers * gpus_per_worker
            if used_gpus < total_gpus:
                unused_gpus = list(range(used_gpus, total_gpus))
                print(f"   Unused GPUs: {unused_gpus}")


def test_cuda_visible_devices():
    """Test CUDA_VISIBLE_DEVICES environment variable handling"""
    
    print("\n\nTesting CUDA_VISIBLE_DEVICES")
    print("=" * 50)
    
    # Test different CUDA_VISIBLE_DEVICES scenarios
    test_envs = [
        "0,1,2,3,4,5,6,7",  # All 8 GPUs
        "0,1,2,3",          # First 4 GPUs
        "4,5,6,7",          # Last 4 GPUs  
        "0,2,4,6",          # Even GPUs
        "1,3,5,7",          # Odd GPUs
    ]
    
    for cuda_env in test_envs:
        print(f"\nCUDA_VISIBLE_DEVICES={cuda_env}")
        
        # Parse GPU list
        available_gpus = [int(x) for x in cuda_env.split(',')]
        total_gpus = len(available_gpus)
        
        # Test with different worker counts
        for num_workers in [1, 2, 4]:
            gpus_per_worker = total_gpus // num_workers
            
            if gpus_per_worker > 0:
                print(f"  {num_workers} workers: {gpus_per_worker} GPUs each")
                for worker in range(num_workers):
                    start_idx = worker * gpus_per_worker
                    end_idx = start_idx + gpus_per_worker
                    worker_gpus = available_gpus[start_idx:end_idx]
                    print(f"    Worker {worker}: {worker_gpus}")
            else:
                print(f"  {num_workers} workers: Insufficient GPUs (shared mode)")


def demonstrate_optimal_configurations():
    """Demonstrate optimal GPU configurations"""
    
    print("\n\nOptimal GPU Configurations")
    print("=" * 50)
    
    configurations = [
        {
            "scenario": "High-throughput inference",
            "total_gpus": 8,
            "num_workers": 4,
            "gpus_per_worker": 2,
            "description": "Balanced parallelism with sufficient GPU memory per worker"
        },
        {
            "scenario": "Large model inference", 
            "total_gpus": 8,
            "num_workers": 2,
            "gpus_per_worker": 4,
            "description": "More GPUs per worker for larger models"
        },
        {
            "scenario": "Maximum parallelism",
            "total_gpus": 8, 
            "num_workers": 8,
            "gpus_per_worker": 1,
            "description": "One GPU per worker for maximum parallel processing"
        },
        {
            "scenario": "Development/Testing",
            "total_gpus": 4,
            "num_workers": 2, 
            "gpus_per_worker": 2,
            "description": "Smaller scale for development and testing"
        }
    ]
    
    for config in configurations:
        print(f"\n{config['scenario']}:")
        print(f"  Total GPUs: {config['total_gpus']}")
        print(f"  Workers: {config['num_workers']}")
        print(f"  GPUs per worker: {config['gpus_per_worker']}")
        print(f"  Description: {config['description']}")
        
        # Show allocation
        for worker in range(config['num_workers']):
            start_gpu = worker * config['gpus_per_worker'] 
            end_gpu = start_gpu + config['gpus_per_worker']
            gpu_list = list(range(start_gpu, end_gpu))
            print(f"    Worker {worker}: GPUs {gpu_list}")


def main():
    """Main test function"""
    
    # Test GPU allocation logic
    test_gpu_allocation_logic()
    
    # Test CUDA environment handling
    test_cuda_visible_devices()
    
    # Show optimal configurations
    demonstrate_optimal_configurations()
    
    print("\n\nTesting complete!")
    print("\nUsage recommendations:")
    print("1. For 8 GPUs with 4 workers: Each worker gets 2 GPUs")
    print("2. For 8 GPUs with 2 workers: Each worker gets 4 GPUs") 
    print("3. For 4 GPUs with 4 workers: Each worker gets 1 GPU")
    print("4. Set CUDA_VISIBLE_DEVICES to control which GPUs are available")
    print("5. Ensure NUM_WORKERS divides evenly into TOTAL_GPUS for optimal allocation")


if __name__ == "__main__":
    main() 