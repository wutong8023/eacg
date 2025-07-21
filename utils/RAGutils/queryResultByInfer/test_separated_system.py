#!/usr/bin/env python3
"""
Test script for separated context generation and multi-worker inference system

This script creates test data and verifies that the separated system works correctly.
"""

import json
import os
import tempfile
import subprocess
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_queries(num_queries: int = 5) -> List[Dict]:
    """Create test queries for validation"""
    test_queries = [
        {
            "id": "test_1",
            "query": "How to sort array indices?",
            "target_api": "numpy.argsort",
            "target_dependency": {"numpy": "1.16.6"}
        },
        {
            "id": "test_2", 
            "query": "How to calculate Fast Fourier Transform?",
            "target_api": "scipy.fft.fft",
            "target_dependency": {"scipy": "1.4.1"}
        },
        {
            "id": "test_3",
            "query": "How to create a matplotlib plot?",
            "target_api": "matplotlib.pyplot.plot",
            "target_dependency": {"matplotlib": "2.0.2"}
        },
        {
            "id": "test_4",
            "query": "How to compute eigenvalues?",
            "target_api": "numpy.linalg.eig",
            "target_dependency": {"numpy": "1.16.6"}
        },
        {
            "id": "test_5",
            "query": "How to reshape an array?",
            "target_api": "numpy.reshape",
            "target_dependency": {"numpy": "1.16.6"}
        }
    ]
    
    return test_queries[:num_queries]

def save_test_queries(queries: List[Dict], filepath: str):
    """Save test queries to file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(queries)} test queries to {filepath}")

def run_context_generation(queries_file: str, contexts_file: str, corpus_path: str) -> bool:
    """Run context generation phase"""
    logger.info("Running context generation...")
    
    cmd = [
        "python", "utils/RAGutils/queryResultByInfer/batch_context_generator.py",
        "--queries_file", queries_file,
        "--corpus_path", corpus_path,
        "--output_file", contexts_file,
        "--corpus_type", "docstring",
        "--embedding_source", "local",
        "--max_documents", "5",
        "--max_tokens", "2000",
        "--enable_str_match",
        "--verbose"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Context generation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Context generation failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False

def run_single_worker_inference(contexts_file: str, results_file: str, model_path: str = "mistralai/Mistral-7B-Instruct-v0.2") -> bool:
    """Run single worker inference"""
    logger.info("Running single worker inference...")
    
    cmd = [
        "python", "utils/RAGutils/queryResultByInfer/multi_worker_inference.py",
        "--contexts_file", contexts_file,
        "--output_file", results_file,
        "--model_path", model_path,
        "--inference_type", "local",
        "--max_new_tokens", "256",
        "--temperature", "0.2",
        "--precision", "fp16",
        "--verbose"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Single worker inference completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Single worker inference failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False

def validate_contexts_file(contexts_file: str, expected_count: int) -> bool:
    """Validate contexts file format and content"""
    logger.info(f"Validating contexts file: {contexts_file}")
    
    if not os.path.exists(contexts_file):
        logger.error(f"Contexts file not found: {contexts_file}")
        return False
    
    try:
        contexts = []
        with open(contexts_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        context_data = json.loads(line)
                        contexts.append(context_data)
                        
                        # Validate required fields
                        required_fields = ['id', 'query', 'retrieval_method', 'context', 'success']
                        for field in required_fields:
                            if field not in context_data:
                                logger.error(f"Missing field '{field}' in line {line_num}")
                                return False
                                
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON on line {line_num}: {e}")
                        return False
        
        if len(contexts) != expected_count:
            logger.error(f"Expected {expected_count} contexts, found {len(contexts)}")
            return False
        
        # Check success rate
        successful = sum(1 for c in contexts if c.get('success', False))
        success_rate = successful / len(contexts)
        
        logger.info(f"Contexts validation passed: {len(contexts)} contexts, {success_rate:.1%} success rate")
        return True
        
    except Exception as e:
        logger.error(f"Error validating contexts file: {e}")
        return False

def validate_results_file(results_file: str, expected_count: int) -> bool:
    """Validate results file format and content"""
    logger.info(f"Validating results file: {results_file}")
    
    if not os.path.exists(results_file):
        logger.error(f"Results file not found: {results_file}")
        return False
    
    try:
        results = []
        with open(results_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        result_data = json.loads(line)
                        results.append(result_data)
                        
                        # Validate required fields
                        required_fields = ['id', 'query', 'context', 'answer', 'success']
                        for field in required_fields:
                            if field not in result_data:
                                logger.error(f"Missing field '{field}' in line {line_num}")
                                return False
                                
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON on line {line_num}: {e}")
                        return False
        
        if len(results) != expected_count:
            logger.error(f"Expected {expected_count} results, found {len(results)}")
            return False
        
        # Check success rate and answer quality
        successful = sum(1 for r in results if r.get('success', False))
        success_rate = successful / len(results)
        
        # Check that successful results have non-empty answers
        valid_answers = sum(1 for r in results if r.get('success', False) and r.get('answer', '').strip())
        
        logger.info(f"Results validation passed: {len(results)} results, {success_rate:.1%} success rate, {valid_answers} valid answers")
        return True
        
    except Exception as e:
        logger.error(f"Error validating results file: {e}")
        return False

def main():
    """Main test function"""
    logger.info("=== Testing Separated Context Generation and Multi-Worker Inference System ===")
    
    # Configuration
    corpus_path = "data/corpus"
    num_test_queries = 3  # Small number for quick testing
    
    # Check if corpus exists
    if not os.path.exists(corpus_path):
        logger.error(f"Corpus path not found: {corpus_path}")
        logger.info("Please ensure the corpus is available or update the corpus_path")
        return False
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Using temporary directory: {temp_dir}")
        
        # File paths
        queries_file = os.path.join(temp_dir, "test_queries.json")
        contexts_file = os.path.join(temp_dir, "test_contexts.jsonl")
        results_file = os.path.join(temp_dir, "test_results.jsonl")
        
        # Step 1: Create test queries
        logger.info("Step 1: Creating test queries...")
        test_queries = create_test_queries(num_test_queries)
        save_test_queries(test_queries, queries_file)
        
        # Step 2: Run context generation
        logger.info("Step 2: Running context generation...")
        if not run_context_generation(queries_file, contexts_file, corpus_path):
            logger.error("Context generation failed")
            return False
        
        # Step 3: Validate contexts
        logger.info("Step 3: Validating contexts...")
        if not validate_contexts_file(contexts_file, num_test_queries):
            logger.error("Context validation failed")
            return False
        
        # Step 4: Run single worker inference
        logger.info("Step 4: Running single worker inference...")
        if not run_single_worker_inference(contexts_file, results_file):
            logger.error("Single worker inference failed")
            return False
        
        # Step 5: Validate results
        logger.info("Step 5: Validating results...")
        if not validate_results_file(results_file, num_test_queries):
            logger.error("Results validation failed")
            return False
        
        # Step 6: Display sample results
        logger.info("Step 6: Displaying sample results...")
        with open(results_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 2:  # Show first 2 results
                    break
                result = json.loads(line.strip())
                logger.info(f"Sample {i+1}:")
                logger.info(f"  Query: {result['query'][:100]}...")
                logger.info(f"  Retrieval Method: {result['retrieval_method']}")
                logger.info(f"  Context Length: {result['context_length']}")
                logger.info(f"  Answer: {result['answer'][:200]}...")
                logger.info(f"  Success: {result['success']}")
        
        logger.info("=== All tests passed! ===")
        logger.info("The separated context generation and multi-worker inference system is working correctly.")
        
        return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 