#!/usr/bin/env python3
"""
Test Multi-Worker Inference System

This script tests the multi-worker inference functionality with sample data.
"""

import json
import os
import tempfile
import subprocess
import sys
from typing import Dict, List

def create_sample_contexts(num_contexts: int = 20) -> List[Dict]:
    """Create sample contexts for testing"""
    contexts = []
    
    for i in range(num_contexts):
        context = {
            'id': f'test_item_{i}',
            'original_item_id': f'original_{i // 3}',  # Some items share original_item_id
            'query_index': i % 3,
            'dedup_key': f'query_{i % 10}_api_{i % 5}',  # Some duplicates
            'query': f'How to use function_{i % 5}?',
            'target_api': f'function_{i % 5}',
            'dependencies': {'numpy': '1.16.6', 'matplotlib': '2.0.2'},
            'retrieval_method': 'exact_api_match' if i % 2 == 0 else 'rag_fallback',
            'context': f'This is the documentation for function_{i % 5}. It does something useful with parameter_{i}.',
            'context_length': 50 + i * 10,
            'retrieval_time': 0.1 + (i % 5) * 0.05,
            'success': True
        }
        contexts.append(context)
    
    # Add some failed contexts
    for i in range(3):
        failed_context = {
            'id': f'failed_item_{i}',
            'error': f'Test error {i}',
            'success': False
        }
        contexts.append(failed_context)
    
    return contexts

def write_contexts_file(contexts: List[Dict], filepath: str):
    """Write contexts to JSONL file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for context in contexts:
            f.write(json.dumps(context, ensure_ascii=False) + '\n')

def test_single_worker():
    """Test single worker inference"""
    print("=== Testing Single Worker Inference ===")
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as contexts_file:
        contexts = create_sample_contexts(10)
        write_contexts_file(contexts, contexts_file.name)
        contexts_filepath = contexts_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as output_file:
        output_filepath = output_file.name
    
    try:
        # Run single worker inference
        cmd = [
            'python', 'utils/RAGutils/queryResultByInfer/infer/multi_worker_inference.py',
            '--contexts_file', contexts_filepath,
            '--output_file', output_filepath,
            '--model_path', 'mistralai/Mistral-7B-Instruct-v0.2',
            '--inference_type', 'local',
            '--max_new_tokens', '50',
            '--temperature', '0.1',
            '--max_samples_per_worker', '5',
            '--verbose'
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        
        # Check output
        if os.path.exists(output_filepath):
            with open(output_filepath, 'r', encoding='utf-8') as f:
                results = [json.loads(line) for line in f if line.strip()]
            
            print(f"Generated {len(results)} results")
            
            # Show sample result
            if results:
                print("Sample result:")
                sample = results[0]
                for key in ['id', 'query', 'answer', 'inference_time', 'success']:
                    print(f"  {key}: {sample.get(key, 'N/A')}")
        
        return result.returncode == 0
        
    finally:
        # Cleanup
        if os.path.exists(contexts_filepath):
            os.unlink(contexts_filepath)
        if os.path.exists(output_filepath):
            os.unlink(output_filepath)

def test_multi_worker():
    """Test multi-worker inference"""
    print("\n=== Testing Multi-Worker Inference ===")
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as contexts_file:
        contexts = create_sample_contexts(20)
        write_contexts_file(contexts, contexts_file.name)
        contexts_filepath = contexts_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as output_file:
        output_filepath = output_file.name
    
    try:
        # Run multi-worker inference
        cmd = [
            'torchrun', '--nproc_per_node=2',
            'utils/RAGutils/queryResultByInfer/infer/multi_worker_inference.py',
            '--contexts_file', contexts_filepath,
            '--output_file', output_filepath,
            '--model_path', 'mistralai/Mistral-7B-Instruct-v0.2',
            '--inference_type', 'local',
            '--max_new_tokens', '50',
            '--temperature', '0.1',
            '--max_samples_per_worker', '5',
            '--verbose'
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        
        # Check output
        if os.path.exists(output_filepath):
            with open(output_filepath, 'r', encoding='utf-8') as f:
                results = [json.loads(line) for line in f if line.strip()]
            
            print(f"Generated {len(results)} results")
            
            # Check worker distribution
            worker_counts = {}
            for result in results:
                worker_rank = result.get('worker_rank', 'unknown')
                worker_counts[worker_rank] = worker_counts.get(worker_rank, 0) + 1
            
            print("Results per worker:")
            for worker, count in sorted(worker_counts.items()):
                print(f"  Worker {worker}: {count} results")
            
            # Show sample result
            if results:
                print("Sample result:")
                sample = results[0]
                for key in ['id', 'query', 'answer', 'inference_time', 'worker_rank', 'success']:
                    print(f"  {key}: {sample.get(key, 'N/A')}")
        
        return result.returncode == 0
        
    finally:
        # Cleanup
        if os.path.exists(contexts_filepath):
            os.unlink(contexts_filepath)
        if os.path.exists(output_filepath):
            os.unlink(output_filepath)

def test_data_slicing():
    """Test data slicing logic"""
    print("\n=== Testing Data Slicing Logic ===")
    
    from utils.RAGutils.queryResultByInfer.infer.multi_worker_inference import MultiWorkerInference
    from utils.RAGutils.queryResultByInfer.query_based_retrieval_inference import QueryBasedInference
    
    # Create mock inference system
    inference_system = QueryBasedInference(
        model_path="mock",
        inference_type="local"
    )
    
    # Test data slicing with different configurations
    test_cases = [
        (10, 1),  # 10 items, 1 worker
        (10, 2),  # 10 items, 2 workers
        (10, 3),  # 10 items, 3 workers
        (10, 4),  # 10 items, 4 workers
        (7, 3),   # 7 items, 3 workers
    ]
    
    for total_items, num_workers in test_cases:
        print(f"\nTesting {total_items} items with {num_workers} workers:")
        
        # Create mock contexts
        contexts = [{'id': f'item_{i}'} for i in range(total_items)]
        
        # Test each worker's slice
        all_items = set()
        for rank in range(num_workers):
            # Mock GPU devices allocation
            gpu_devices = [rank * 2, rank * 2 + 1] if num_workers <= 4 else [rank]
            
            multi_inference = MultiWorkerInference(
                inference_system=inference_system,
                output_file="mock.jsonl",
                rank=rank,
                world_size=num_workers,
                gpu_devices=gpu_devices
            )
            
            worker_slice = multi_inference.get_data_slice(contexts)
            worker_items = {item['id'] for item in worker_slice}
            
            print(f"  Worker {rank}: {len(worker_slice)} items ({list(worker_items)})")
            
            # Check for overlaps
            overlap = all_items & worker_items
            if overlap:
                print(f"    ❌ Overlap detected: {overlap}")
            else:
                print(f"    ✅ No overlap")
            
            all_items.update(worker_items)
        
        # Check if all items are covered
        expected_items = {f'item_{i}' for i in range(total_items)}
        if all_items == expected_items:
            print(f"  ✅ All items covered correctly")
        else:
            missing = expected_items - all_items
            extra = all_items - expected_items
            print(f"  ❌ Coverage issue - Missing: {missing}, Extra: {extra}")

def main():
    """Main test function"""
    print("Testing Multi-Worker Inference System")
    print("=" * 50)
    
    # Test data slicing logic first (doesn't require model loading)
    test_data_slicing()
    
    # Test actual inference (requires model - may be slow)
    if '--skip-inference' not in sys.argv:
        print("\nTesting actual inference (this may take a while)...")
        print("Use --skip-inference to skip these tests")
        
        success1 = test_single_worker()
        success2 = test_multi_worker()
        
        print(f"\nTest Results:")
        print(f"Single worker: {'✅ PASS' if success1 else '❌ FAIL'}")
        print(f"Multi worker: {'✅ PASS' if success2 else '❌ FAIL'}")
    else:
        print("\nSkipping inference tests (--skip-inference specified)")
    
    print("\nTesting complete!")

if __name__ == "__main__":
    main() 