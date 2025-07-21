#!/usr/bin/env python3
"""
Test script for progressive retrieval functionality
"""

import os
import sys
import json
import logging

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_progressive_retrieval():
    """Test the progressive retrieval functionality"""
    try:
        # 导入必要的模块
        from utils.RAGutils.RAGRetriever import RAGContextRetriever
        from utils.RAGutils.RAGEmbedding import CustomEmbeddingFunction
        import chromadb
        from benchmark.config.code.config import (
            RAG_COLLECTION_BASE, LOCAL_EMBEDDING_MODEL,
            DOCSTRING_EMBEDDING_BASE_PATH, SRCCODE_EMBEDDING_BASE_PATH,
            DOCSTRING_CORPUS_PATH, SRCCODE_CORPUS_PATH
        )
        
        print("=== Testing Progressive Retrieval ===\n")
        
        # 1. 准备测试数据
        test_data = {
            "id": "88",  # 这个ID在generated_queries文件中存在
            "description": "Fix issue related to line break and marker positioning in Matplotlib when plotting data",
            "target_dependency": {
                "matplotlib": "3.7.5",
                "numpy": "1.24.4", 
                "scipy": "1.10.1"
            },
            "origin_dependency": {
                "matplotlib": "3.7.1",
                "numpy": "1.24.3",
                "scipy": "1.10.0"
            }
        }
        
        print(f"Test data: {test_data['id']}")
        print(f"Description: {test_data['description'][:100]}...")
        
        # 2. 设置embedding参数
        embedding_args = {
            'source': 'local',
            'model_name': LOCAL_EMBEDDING_MODEL,
            'together_client': None,
            'batch_size': 64
        }
        
        # 3. 初始化ChromaDB客户端
        corpus_type = "docstring"
        embed_model_name = LOCAL_EMBEDDING_MODEL.split('/')[-1]
        db_path = os.path.join(RAG_COLLECTION_BASE, corpus_type, embed_model_name)
        
        print(f"Database path: {db_path}")
        
        chroma_settings = chromadb.Settings(
            allow_reset=True,
            anonymized_telemetry=False,
            is_persistent=True,
        )
        
        chroma_client = chromadb.PersistentClient(path=db_path, settings=chroma_settings)
        
        # 4. 创建生成查询文件路径
        generated_queries_file = "data/generated_queries/versibcb_vace_queries.json"
        
        # 5. 初始化RAG检索器（启用生成的查询）
        print("\nInitializing RAG retriever with generated queries...")
        rag_retriever = RAGContextRetriever(
            chroma_client=chroma_client,
            embed_func_args=embedding_args,
            corpus_type=corpus_type,
            rag_collection_base=RAG_COLLECTION_BASE,
            knowledge_type=corpus_type,
            embedding_source='local',
            docstring_embedding_base_path=DOCSTRING_EMBEDDING_BASE_PATH,
            srccode_embedding_base_path=SRCCODE_EMBEDDING_BASE_PATH,
            max_dependency_num=None,
            append_srcDep=False,
            query_cache_dir="/tmp/test_rag_cache",
            rag_document_num=5,  # 减少数量以便测试
            docstring_corpus_path=DOCSTRING_CORPUS_PATH,
            srccode_corpus_path=SRCCODE_CORPUS_PATH,
            generated_queries_file=generated_queries_file,
            use_generated_queries=True
        )
        
        print("RAG retriever initialized successfully!")
        
        # 6. 测试查询获取
        print("\n=== Testing Query Generation ===")
        queries = rag_retriever._get_query_from_data(test_data, "VACE")
        print(f"Generated {len(queries)} queries:")
        for i, query in enumerate(queries, 1):
            print(f"  {i}. {query[:100]}...")
        
        # 7. 执行上下文检索
        print("\n=== Testing Context Retrieval ===")
        max_token_length = 1000  # 较小的token长度以便观察渐进式检索
        
        context = rag_retriever.retrieve_context(
            data=test_data,
            benchmark_type="VersiBCB", 
            task="VACE",
            max_token_length=max_token_length
        )
        
        if context:
            print(f"\nRetrieved context length: {len(context)} characters")
            print("Context preview:")
            print("-" * 50)
            print(context[:500] + "..." if len(context) > 500 else context)
            print("-" * 50)
        else:
            print("No context retrieved")
        
        # 8. 测试对比：关闭生成的查询
        print("\n=== Testing Fallback (without generated queries) ===")
        rag_retriever.use_generated_queries = False
        
        fallback_queries = rag_retriever._get_query_from_data(test_data, "VACE")
        print(f"Fallback generated {len(fallback_queries)} queries:")
        for i, query in enumerate(fallback_queries, 1):
            print(f"  {i}. {query[:100]}...")
        
        fallback_context = rag_retriever.retrieve_context(
            data=test_data,
            benchmark_type="VersiBCB",
            task="VACE", 
            max_token_length=max_token_length
        )
        
        if fallback_context:
            print(f"\nFallback context length: {len(fallback_context)} characters")
        else:
            print("No fallback context retrieved")
        
        print("\n=== Test Completed Successfully! ===")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_progressive_retrieval() 