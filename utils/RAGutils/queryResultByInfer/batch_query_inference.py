#!/usr/bin/env python3
"""
Batch Query-based Retrieval and Inference Script

This script processes multiple queries from a JSON file, performing retrieval and inference for each query.
Supports both exact API matching and RAG fallback strategies.

Usage:
    python batch_query_inference.py --queries_file data/generated_queries/versibcb_vace_queries_deduplicated.json --corpus_path data/corpus --output_file batch_results.jsonl
"""

import argparse
import json
import os
import logging
import time
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import traceback

# Import the query-based retrieval and inference classes
from query_based_retrieval_inference import QueryBasedRetriever, QueryBasedInference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchQueryProcessor:
    """
    Batch processor for multiple queries with retrieval and inference
    """
    
    def __init__(self, 
                 retriever: QueryBasedRetriever,
                 inference_system: QueryBasedInference,
                 output_file: str):
        """
        Initialize batch processor
        
        Args:
            retriever: QueryBasedRetriever instance
            inference_system: QueryBasedInference instance  
            output_file: Output file path for results
        """
        self.retriever = retriever
        self.inference_system = inference_system
        self.output_file = output_file
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'exact_matches': 0,
            'rag_fallbacks': 0,
            'errors': 0,
            'total_time': 0,
            'avg_retrieval_time': 0,
            'avg_inference_time': 0
        }
    
    def process_query_item(self, query_item: Dict, sample_index: int) -> Dict:
        """
        Process a single query item (now expects expanded format)
        
        Args:
            query_item: Query item dictionary (expanded format)
            sample_index: Index of the sample
            
        Returns:
            Result dictionary
        """
        start_time = time.time()
        
        try:
            # Extract query information from expanded format
            sample_id = query_item.get('id', f'sample_{sample_index}')
            query = query_item.get('query', '')
            target_api = query_item.get('target_api', '')
            dependencies = query_item.get('target_dependency', {})
            
            if not query.strip():
                return {
                    'id': sample_id,
                    'error': 'No query found',
                    'processing_time': time.time() - start_time
                }
            
            if not dependencies:
                return {
                    'id': sample_id,
                    'error': 'No dependencies found',
                    'processing_time': time.time() - start_time
                }
            
            # Update retriever dependencies for this query
            self.retriever.dependencies = dependencies
            
            # Retrieve context
            retrieval_start = time.time()
            context, retrieval_method = self.retriever.retrieve_context(query, target_api)
            retrieval_time = time.time() - retrieval_start
            
            # Generate answer
            inference_start = time.time()
            answer = self.inference_system.generate_answer(query, context, retrieval_method)
            inference_time = time.time() - inference_start
            
            # Update statistics
            if retrieval_method == "exact_api_match":
                self.stats['exact_matches'] += 1
            else:
                self.stats['rag_fallbacks'] += 1
            
            total_time = time.time() - start_time
            
            result = {
                'id': sample_id,
                'original_item_id': query_item.get('original_item_id', ''),
                'query_index': query_item.get('query_index', 0),
                'dedup_key': query_item.get('dedup_key', ''),
                'query': query,
                'target_api': target_api,
                'dependencies': dependencies,
                'retrieval_method': retrieval_method,
                'context_length': len(context),
                'answer': answer,
                'retrieval_time': retrieval_time,
                'inference_time': inference_time,
                'total_time': total_time,
                'success': True
            }
            
            logger.info(f"Sample {sample_id}: {retrieval_method}, "
                       f"retrieval: {retrieval_time:.2f}s, inference: {inference_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.stats['errors'] += 1
            error_msg = str(e)
            
            logger.error(f"Error processing sample {sample_index}: {error_msg}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'id': query_item.get('id', f'sample_{sample_index}'),
                'error': error_msg,
                'processing_time': time.time() - start_time,
                'success': False
            }
    
    def process_queries(self, queries: List[Dict], start_index: int = 0, max_samples: Optional[int] = None) -> None:
        """
        Process list of queries
        
        Args:
            queries: List of query dictionaries
            start_index: Starting index for processing
            max_samples: Maximum number of samples to process
        """
        # Determine processing range
        end_index = len(queries)
        if max_samples:
            end_index = min(start_index + max_samples, len(queries))
        
        queries_to_process = queries[start_index:end_index]
        self.stats['total_queries'] = len(queries_to_process)
        
        logger.info(f"Processing {len(queries_to_process)} queries (indices {start_index} to {end_index-1})")
        
        # Process queries with progress bar
        total_retrieval_time = 0
        total_inference_time = 0
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for i, query_item in enumerate(tqdm(queries_to_process, desc="Processing queries")):
                result = self.process_query_item(query_item, start_index + i)
                
                # Accumulate timing statistics
                if result.get('success', False):
                    total_retrieval_time += result.get('retrieval_time', 0)
                    total_inference_time += result.get('inference_time', 0)
                
                # Write result to file
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()  # Ensure results are written immediately
        
        # Calculate final statistics
        successful_queries = self.stats['exact_matches'] + self.stats['rag_fallbacks']
        if successful_queries > 0:
            self.stats['avg_retrieval_time'] = total_retrieval_time / successful_queries
            self.stats['avg_inference_time'] = total_inference_time / successful_queries
        
        self.stats['total_time'] = time.time()
    
    def print_statistics(self):
        """Print processing statistics"""
        print("\n" + "="*60)
        print("BATCH PROCESSING STATISTICS")
        print("="*60)
        print(f"Total queries processed: {self.stats['total_queries']}")
        print(f"Exact API matches: {self.stats['exact_matches']}")
        print(f"RAG fallbacks: {self.stats['rag_fallbacks']}")
        print(f"Errors: {self.stats['errors']}")
        
        if self.stats['total_queries'] > 0:
            success_rate = (self.stats['exact_matches'] + self.stats['rag_fallbacks']) / self.stats['total_queries']
            exact_match_rate = self.stats['exact_matches'] / self.stats['total_queries']
            print(f"Success rate: {success_rate:.1%}")
            print(f"Exact match rate: {exact_match_rate:.1%}")
        
        if self.stats['avg_retrieval_time'] > 0:
            print(f"Average retrieval time: {self.stats['avg_retrieval_time']:.2f}s")
            print(f"Average inference time: {self.stats['avg_inference_time']:.2f}s")
        
        print(f"Results saved to: {self.output_file}")


def load_queries_from_file(queries_file: str) -> List[Dict]:
    """
    Load queries from JSON file
    
    Args:
        queries_file: Path to queries file
        
    Returns:
        List of query dictionaries
    """
    logger.info(f"Loading queries from: {queries_file}")
    
    try:
        with open(queries_file, 'r', encoding='utf-8') as f:
            if queries_file.endswith('.jsonl'):
                # JSONL format
                queries = []
                for line in f:
                    line = line.strip()
                    if line:
                        queries.append(json.loads(line))
            else:
                # JSON format
                data = json.load(f)
                if isinstance(data, dict):
                    # Handle dictionary format with numeric keys
                    queries = []
                    for key, value in data.items():
                        if isinstance(value, dict) and 'queries' in value:
                            # Create a query item that preserves the original structure
                            # but adds the necessary metadata for processing
                            query_item = {
                                'id': key,
                                'queries': value['queries'],
                                'original_data': value.get('original_data', {}),
                                'target_dependency': value.get('original_data', {}).get('target_dependency', {})
                            }
                            queries.append(query_item)
                elif isinstance(data, list):
                    queries = data
                elif isinstance(data, dict) and 'queries' in data:
                    queries = data['queries']
                else:
                    queries = [data]  # Single query
        
        logger.info(f"Loaded {len(queries)} queries")
        return queries
        
    except Exception as e:
        logger.error(f"Error loading queries from {queries_file}: {e}")
        raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Batch query-based retrieval and inference")
    
    # Input/Output arguments
    parser.add_argument("--queries_file", type=str, required=True, help="Path to queries JSON/JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file for results")
    
    # Corpus arguments
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to corpus directory")
    parser.add_argument("--corpus_type", type=str, default="docstring", choices=["docstring", "srccodes"], help="Type of corpus")
    
    # Processing range arguments
    parser.add_argument("--start_index", type=int, default=0, help="Starting index for processing")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to process")
    
    # Retrieval arguments
    parser.add_argument("--embedding_source", type=str, default="local", choices=["local", "togetherai"], help="Embedding source")
    parser.add_argument("--max_documents", type=int, default=10, help="Maximum documents to retrieve")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Maximum token length for context")
    parser.add_argument("--enable_str_match", action="store_true", help="Enable string matching")
    parser.add_argument("--fixed_docs_per_query", type=int, default=1, help="Enable fixed number of documents per query (required for string matching)")
    parser.add_argument("--jump_exact_match", action="store_true", help="Jump exact match")
    
    # Inference arguments  
    parser.add_argument("--model_path", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="Model path or name")
    parser.add_argument("--inference_type", type=str, default="local", choices=["local", "huggingface", "togetherai"], help="Inference type")
    parser.add_argument("--api_key", type=str, help="API key for remote inference")
    parser.add_argument("--api_model_name", type=str, help="Model name for API inference")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    
    # Control arguments
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load queries
    queries = load_queries_from_file(args.queries_file)
    
    if not queries:
        logger.error("No queries loaded, exiting")
        return
    
    # Initialize retriever with default dependencies (will be updated per query)
    # default_dependencies = {"numpy": "1.16.6", "matplotlib": "2.0.2", "scipy": "1.4.1"}
    
    try:
        retriever = QueryBasedRetriever(
            corpus_path=args.corpus_path,
            dependencies=None,
            corpus_type=args.corpus_type,
            embedding_source=args.embedding_source,
            max_documents=args.max_documents,
            max_tokens=args.max_tokens,
            str_match=args.enable_str_match,
            fixed_docs_per_query=args.fixed_docs_per_query,
            jump_exact_match=args.jump_exact_match
        )
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {e}")
        return
    
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
    except Exception as e:
        logger.error(f"Failed to initialize inference system: {e}")
        return
    
    # Initialize batch processor
    processor = BatchQueryProcessor(retriever, inference_system, args.output_file)
    
    # Process queries
    start_time = time.time()
    processor.process_queries(queries, args.start_index, args.max_samples)
    total_time = time.time() - start_time
    
    processor.stats['total_time'] = total_time
    
    # Print statistics
    processor.print_statistics()
    
    logger.info(f"Batch processing completed in {total_time:.2f} seconds")


if __name__ == "__main__":
    main() 