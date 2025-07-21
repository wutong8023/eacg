#!/usr/bin/env python3
"""
Test script for query deduplication and expansion functionality
"""

import json
import os
import sys
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from utils.RAGutils.queryResultByInfer.batch_context_generator import load_queries_from_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_query_loading_and_deduplication():
    """Test query loading and deduplication"""
    
    queries_file = "data/generated_queries/versibcb_vace_queries_deduplicated.json"
    
    logger.info("=== Testing Query Loading and Deduplication ===")
    logger.info(f"Loading from: {queries_file}")
    
    try:
        # Load queries with new deduplication logic
        queries = load_queries_from_file(queries_file)
        
        logger.info(f"Total unique queries loaded: {len(queries)}")
        
        # Analyze the results
        if queries:
            # Show first few queries
            logger.info("\nFirst 3 queries:")
            for i, query in enumerate(queries[:3]):
                logger.info(f"  {i+1}. ID: {query['id']}")
                logger.info(f"     Query: {query['query'][:80]}...")
                logger.info(f"     Target API: {query['target_api']}")
                logger.info(f"     Dependencies: {list(query['target_dependency'].keys())}")
                logger.info(f"     Dedup key: {query['dedup_key']}")
                logger.info("")
            
            # Count unique APIs
            unique_apis = set(q['target_api'] for q in queries if q['target_api'])
            logger.info(f"Unique target APIs: {len(unique_apis)}")
            
            # Count by dependency packages
            dependency_counts = {}
            for query in queries:
                for pkg in query['target_dependency'].keys():
                    dependency_counts[pkg] = dependency_counts.get(pkg, 0) + 1
            
            logger.info("Queries by dependency package:")
            for pkg, count in sorted(dependency_counts.items()):
                logger.info(f"  {pkg}: {count} queries")
            
            # Show some statistics
            total_chars = sum(len(q['query']) for q in queries)
            avg_query_length = total_chars / len(queries)
            logger.info(f"Average query length: {avg_query_length:.1f} characters")
            
            return True
        else:
            logger.error("No queries loaded")
            return False
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_data_slicing():
    """Test data slicing for multi-worker"""
    
    logger.info("\n=== Testing Data Slicing for Multi-Worker ===")
    
    # Create sample data
    sample_queries = [{'id': f'query_{i}', 'query': f'test query {i}'} for i in range(10)]
    
    def get_data_slice(queries, rank, world_size):
        """Simulate the data slicing logic"""
        total_queries = len(queries)
        queries_per_worker = total_queries // world_size
        remainder = total_queries % world_size
        
        if rank < remainder:
            start_idx = rank * (queries_per_worker + 1)
            end_idx = start_idx + queries_per_worker + 1
        else:
            start_idx = remainder * (queries_per_worker + 1) + (rank - remainder) * queries_per_worker
            end_idx = start_idx + queries_per_worker
        
        return queries[start_idx:end_idx], start_idx, end_idx
    
    # Test different worker configurations
    for world_size in [1, 2, 3, 4]:
        logger.info(f"\nTesting with {world_size} workers:")
        total_assigned = 0
        
        for rank in range(world_size):
            slice_data, start_idx, end_idx = get_data_slice(sample_queries, rank, world_size)
            logger.info(f"  Worker {rank}: indices {start_idx}-{end_idx-1} ({len(slice_data)} queries)")
            total_assigned += len(slice_data)
        
        logger.info(f"  Total assigned: {total_assigned}/{len(sample_queries)}")
        assert total_assigned == len(sample_queries), f"Data slicing error for {world_size} workers"
    
    logger.info("✅ Data slicing test passed")
    return True

if __name__ == "__main__":
    logger.info("🚀 Starting deduplication and slicing tests...")
    
    success1 = test_query_loading_and_deduplication()
    success2 = test_data_slicing()
    
    if success1 and success2:
        logger.info("✅ All tests passed!")
    else:
        logger.error("❌ Some tests failed!")
        sys.exit(1) 