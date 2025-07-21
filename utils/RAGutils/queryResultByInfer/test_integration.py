#!/usr/bin/env python3
"""
Integration test script for query-based retrieval and inference system
"""

import json
import os
import sys
import logging
import tempfile

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from utils.RAGutils.queryResultByInfer.query_based_retrieval_inference import QueryBasedRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_retrieval_modes():
    """Test different retrieval modes"""
    
    # Test parameters
    corpus_path = "/datanfs2/chenrongyi/data/docs"
    dependencies = {"matplotlib": "3.7.0", "pandas": "2.0.3", "seaborn": "0.13.2"}
    
    # Test query
    test_query = "How to create a barplot with specific title in matplotlib?"
    test_target_api = "matplotlib.pyplot.title"
    
    logger.info("=== Testing Different Retrieval Modes ===")
    logger.info(f"Query: {test_query}")
    logger.info(f"Target API: {test_target_api}")
    logger.info(f"Dependencies: {dependencies}")
    
    results = {}
    
    try:
        # Test 1: String matching enabled with fixed docs
        logger.info("\n--- Test 1: String matching + Fixed docs ---")
        retriever1 = QueryBasedRetriever(
            corpus_path=corpus_path,
            dependencies=dependencies,
            corpus_type="docstring",
            embedding_source="local",
            max_documents=5,
            max_tokens=2000,
            str_match=True,
            fixed_docs_per_query=1,
            jump_exact_match=False
        )
        
        context1, method1 = retriever1.retrieve_context(test_query, test_target_api)
        results['str_match_fixed'] = {
            'method': method1,
            'context_length': len(context1),
            'context_preview': context1[:200] if context1 else "No context"
        }
        logger.info(f"Method: {method1}, Context length: {len(context1)}")
        
        # Test 2: String matching with jump exact match
        logger.info("\n--- Test 2: String matching + Jump exact match ---")
        retriever2 = QueryBasedRetriever(
            corpus_path=corpus_path,
            dependencies=dependencies,
            corpus_type="docstring",
            embedding_source="local",
            max_documents=5,
            max_tokens=2000,
            str_match=True,
            fixed_docs_per_query=1,
            jump_exact_match=True
        )
        
        context2, method2 = retriever2.retrieve_context(test_query, test_target_api)
        results['str_match_jump'] = {
            'method': method2,
            'context_length': len(context2),
            'context_preview': context2[:200] if context2 else "No context"
        }
        logger.info(f"Method: {method2}, Context length: {len(context2)}")
        
        # Test 3: No string matching (traditional mode)
        logger.info("\n--- Test 3: Traditional mode (no string matching) ---")
        retriever3 = QueryBasedRetriever(
            corpus_path=corpus_path,
            dependencies=dependencies,
            corpus_type="docstring",
            embedding_source="local",
            max_documents=5,
            max_tokens=2000,
            str_match=False,
            fixed_docs_per_query=1,
            jump_exact_match=False
        )
        
        context3, method3 = retriever3.retrieve_context(test_query, test_target_api)
        results['traditional'] = {
            'method': method3,
            'context_length': len(context3),
            'context_preview': context3[:200] if context3 else "No context"
        }
        logger.info(f"Method: {method3}, Context length: {len(context3)}")
        
        # Print comparison
        logger.info("\n=== Comparison Results ===")
        for mode, result in results.items():
            logger.info(f"{mode}: {result['method']} (length: {result['context_length']})")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_query_format():
    """Test different query formats"""
    
    logger.info("\n=== Testing Query Format Processing ===")
    
    # Test query formats
    test_queries = [
        # Format 1: Direct query with target_api
        {
            'id': 'test_1',
            'query': 'How to create a plot?',
            'target_api': 'matplotlib.pyplot.plot',
            'target_dependency': {'matplotlib': '3.7.0'}
        },
        
        # Format 2: Nested queries format (like in the actual file)
        {
            'id': 'test_2',
            'queries': [
                {
                    'query': 'How to set title?',
                    'target_api': 'matplotlib.pyplot.title'
                }
            ],
            'original_data': {
                'target_dependency': {'matplotlib': '3.7.0'}
            }
        }
    ]
    
    from utils.RAGutils.queryResultByInfer.batch_context_generator import BatchContextGenerator
    
    # Create a temporary output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_output = f.name
    
    try:
        # Initialize retriever
        retriever = QueryBasedRetriever(
            corpus_path="/datanfs2/chenrongyi/data/docs",
            dependencies={"matplotlib": "3.7.0"},
            corpus_type="docstring",
            embedding_source="local",
            max_documents=3,
            max_tokens=1000,
            str_match=True,
            fixed_docs_per_query=1,
            jump_exact_match=False
        )
        
        # Initialize generator
        generator = BatchContextGenerator(retriever, temp_output)
        
        # Test each query format
        for i, query_item in enumerate(test_queries):
            logger.info(f"\nTesting query format {i+1}: {query_item.get('id', 'unknown')}")
            result = generator.generate_context_for_item(query_item, i)
            
            if result.get('success', False):
                logger.info(f"✅ Success: {result['retrieval_method']}, context length: {result['context_length']}")
            else:
                logger.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Query format test failed: {e}")
        return False
    
    finally:
        # Clean up
        if os.path.exists(temp_output):
            os.unlink(temp_output)

if __name__ == "__main__":
    logger.info("🚀 Starting integration tests...")
    
    success1 = test_retrieval_modes()
    success2 = test_query_format()
    
    if success1 and success2:
        logger.info("✅ All tests passed!")
    else:
        logger.error("❌ Some tests failed!")
        sys.exit(1) 