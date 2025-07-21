#!/usr/bin/env python3
"""
Batch Context Generation Script

This script processes multiple queries from a JSON file and generates contexts only,
separating the context generation phase from the inference phase for better scalability.

Usage:
    python batch_context_generator.py --queries_file data/queries.json --corpus_path data/corpus --output_file contexts.jsonl
"""

import argparse
import json
import os
import logging
import time
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import traceback

# Import the query-based retrieval class
from utils.RAGutils.queryResultByInfer.query_based_retrieval_inference import QueryBasedRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchContextGenerator:
    """
    Batch context generator for multiple queries
    """
    
    def __init__(self, 
                 retriever: QueryBasedRetriever,
                 output_file: str):
        """
        Initialize batch context generator
        
        Args:
            retriever: QueryBasedRetriever instance
            output_file: Output file path for contexts
        """
        self.retriever = retriever
        self.output_file = output_file
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'exact_matches': 0,
            'rag_fallbacks': 0,
            'rag_string_matches': 0,
            'errors': 0,
            'total_time': 0,
            'avg_retrieval_time': 0
        }
    
    def generate_context_for_item(self, query_item: Dict, sample_index: int) -> Dict:
        """
        Generate context for a single query item (now expects expanded format)
        
        Args:
            query_item: Query item dictionary (expanded format)
            sample_index: Index of the sample
            
        Returns:
            Context result dictionary
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
            
            # Update statistics
            if retrieval_method == "exact_api_match":
                self.stats['exact_matches'] += 1
            elif retrieval_method == "rag_string_match":
                self.stats['rag_string_matches'] += 1
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
                'context': context,
                'context_length': len(context),
                'retrieval_time': retrieval_time,
                'total_time': total_time,
                'success': True
            }
            
            logger.info(f"Sample {sample_id}: {retrieval_method}, "
                       f"retrieval: {retrieval_time:.2f}s, context_length: {len(context)}")
            
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
    
    def generate_contexts(self, queries: List[Dict], start_index: int = 0, max_samples: Optional[int] = None) -> None:
        """
        Generate contexts for list of queries
        
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
        
        logger.info(f"Generating contexts for {len(queries_to_process)} queries (indices {start_index} to {end_index-1})")
        
        # Process queries with progress bar
        total_retrieval_time = 0
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for i, query_item in enumerate(tqdm(queries_to_process, desc="Generating contexts")):
                result = self.generate_context_for_item(query_item, start_index + i)
                
                # Accumulate timing statistics
                if result.get('success', False):
                    total_retrieval_time += result.get('retrieval_time', 0)
                
                # Write result to file
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()  # Ensure results are written immediately
        
        # Calculate final statistics
        successful_queries = self.stats['exact_matches'] + self.stats['rag_fallbacks'] + self.stats['rag_string_matches']
        if successful_queries > 0:
            self.stats['avg_retrieval_time'] = total_retrieval_time / successful_queries
        
        self.stats['total_time'] = time.time()
    
    def print_statistics(self):
        """Print processing statistics"""
        print("\n" + "="*60)
        print("BATCH CONTEXT GENERATION STATISTICS")
        print("="*60)
        print(f"Total queries processed: {self.stats['total_queries']}")
        print(f"Exact API matches: {self.stats['exact_matches']}")
        print(f"RAG string matches: {self.stats['rag_string_matches']}")
        print(f"RAG fallbacks: {self.stats['rag_fallbacks']}")
        print(f"Errors: {self.stats['errors']}")
        
        if self.stats['total_queries'] > 0:
            success_rate = (self.stats['exact_matches'] + self.stats['rag_fallbacks'] + self.stats['rag_string_matches']) / self.stats['total_queries']
            exact_match_rate = self.stats['exact_matches'] / self.stats['total_queries']
            string_match_rate = self.stats['rag_string_matches'] / self.stats['total_queries']
            print(f"Success rate: {success_rate:.1%}")
            print(f"Exact match rate: {exact_match_rate:.1%}")
            print(f"String match rate: {string_match_rate:.1%}")
        
        if self.stats['avg_retrieval_time'] > 0:
            print(f"Average retrieval time: {self.stats['avg_retrieval_time']:.2f}s")
        
        print(f"Contexts saved to: {self.output_file}")


def load_queries_from_file(queries_file: str) -> List[Dict]:
    """
    Load queries from JSON file and expand all queries with deduplication
    
    Args:
        queries_file: Path to queries file
        
    Returns:
        List of unique query dictionaries
    """
    logger.info(f"Loading queries from: {queries_file}")
    
    try:
        raw_data = []
        with open(queries_file, 'r', encoding='utf-8') as f:
            if queries_file.endswith('.jsonl'):
                # JSONL format
                raw_data = []
                for line in f:
                    line = line.strip()
                    if line:
                        raw_data.append(json.loads(line))
            else:
                # JSON format
                data = json.load(f)

        
        # Expand all queries and deduplicate
        all_queries = []
        seen_queries = set()  # For deduplication based on (query, target_api)
        
        for bcb_index, item in data.items():
            if isinstance(item, dict) and 'queries' in item:
                target_dependency = item.get('original_data', {}).get('target_dependency', {})
                
                for query_idx, query in enumerate(item['queries']):
                    if isinstance(query, dict):
                        query_text = query.get('query', '')
                        target_api = query.get('target_api', '')
                        
                        # Create deduplication key
                        dedup_key = (query_text.strip(), target_api.strip())
                        
                        if dedup_key not in seen_queries and query_text.strip():
                            seen_queries.add(dedup_key)
                            
                            query_item = {
                                'id': f"{item.get('id', bcb_index)}_{query_idx}",
                                'original_item_id': item.get('id', str(bcb_index)),
                                'query_index': query_idx,
                                'query': query_text,
                                'target_api': target_api,
                                'target_dependency': target_dependency,
                                'dedup_key': f"{hash(dedup_key) % 1000000:06d}"  # For tracking
                            }
                            all_queries.append(query_item)
        
        logger.info(f"Loaded {len(raw_data)} items, expanded to {len(all_queries)} unique queries")
        logger.info(f"Deduplication removed {len([item for item in raw_data for q in item.get('queries', [])]) - len(all_queries)} duplicate queries")
        
        return all_queries
        
    except Exception as e:
        logger.error(f"Error loading queries from {queries_file}: {e}")
        raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Batch context generation for queries")
    
    # Input/Output arguments
    parser.add_argument("--queries_file", type=str,  help="Path to queries JSON/JSONL file",default='data/generated_queries/versibcb_vscc_queries_with_target_dependency.json')
    parser.add_argument("--output_file", type=str, help="Output JSONL file for contexts",default='data/temp/contexts/contexts.jsonl')
    
    # Corpus arguments
    parser.add_argument("--corpus_path", type=str, help="Path to corpus directory",default='/datanfs4/chenrongyi/data/docs')
    parser.add_argument("--corpus_type", type=str, default="docstring", choices=["docstring", "srccodes"], help="Type of corpus")
    
    # Processing range arguments
    parser.add_argument("--start_index", type=int, default=0, help="Starting index for processing")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to process")
    
    # Retrieval arguments
    parser.add_argument("--embedding_source", type=str, default="local", choices=["local", "togetherai"], help="Embedding source")
    parser.add_argument("--max_documents", type=int, default=10, help="Maximum documents to retrieve")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Maximum token length for context")
    parser.add_argument("--enable_str_match", action="store_true", help="Enable string matching")
    parser.add_argument("--fixed_docs_per_query",type=int,default=1,help="Enable fixed number of documents per query (required for string matching)")
    parser.add_argument("--jump_exact_match",action="store_true",help="Jump exact match")
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
    default_dependencies = {"numpy": "1.16.6", "matplotlib": "2.0.2", "scipy": "1.4.1"}
    
    try:
        retriever = QueryBasedRetriever(
            corpus_path=args.corpus_path,
            dependencies=default_dependencies,
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
    
    # Initialize batch context generator
    generator = BatchContextGenerator(retriever, args.output_file)
    
    # Generate contexts
    start_time = time.time()
    generator.generate_contexts(queries, args.start_index, args.max_samples)
    total_time = time.time() - start_time
    
    generator.stats['total_time'] = total_time
    
    # Print statistics
    generator.print_statistics()
    
    logger.info(f"Batch context generation completed in {total_time:.2f} seconds")


if __name__ == "__main__":
    main() 