#!/usr/bin/env python3
"""
Multi-Worker Context Generation Script

This script supports distributed context generation across multiple workers/GPUs.
Each worker processes a slice of the data based on rank and world size.

Usage:
    # Single worker
    python multi_worker_context_generator.py --queries_file data/queries.json --output_file contexts.jsonl
    
    # Multi-worker with torchrun
    torchrun --nproc_per_node=4 multi_worker_context_generator.py --queries_file data/queries.json --output_file contexts.jsonl
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

# Distributed training imports
try:
    import torch
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import the query-based retrieval class
from batch_context_generator import QueryBasedRetriever, BatchContextGenerator, load_queries_from_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiWorkerContextGenerator:
    """
    Multi-worker context generator with data slicing support
    """
    
    def __init__(self, 
                 retriever: QueryBasedRetriever,
                 output_file: str,
                 rank: int = 0,
                 world_size: int = 1):
        """
        Initialize multi-worker context generator
        
        Args:
            retriever: QueryBasedRetriever instance
            output_file: Output file path for contexts
            rank: Worker rank (0-based)
            world_size: Total number of workers
        """
        self.retriever = retriever
        self.output_file = output_file
        self.rank = rank
        self.world_size = world_size
        
        # Modify output file name to include rank
        if world_size > 1:
            base_name, ext = os.path.splitext(output_file)
            self.output_file = f"{base_name}_rank{rank}{ext}"
        
        # Initialize base generator
        self.generator = BatchContextGenerator(retriever, self.output_file)
        
        logger.info(f"Worker {rank}/{world_size} initialized, output: {self.output_file}")
    
    def get_data_slice(self, queries: List[Dict]) -> List[Dict]:
        """
        Get data slice for this worker based on rank and world size
        
        Args:
            queries: Full list of queries
            
        Returns:
            Slice of queries for this worker
        """
        total_queries = len(queries)
        queries_per_worker = total_queries // self.world_size
        remainder = total_queries % self.world_size
        
        # Calculate start and end indices for this worker
        if self.rank < remainder:
            # First 'remainder' workers get one extra query
            start_idx = self.rank * (queries_per_worker + 1)
            end_idx = start_idx + queries_per_worker + 1
        else:
            # Remaining workers get standard allocation
            start_idx = remainder * (queries_per_worker + 1) + (self.rank - remainder) * queries_per_worker
            end_idx = start_idx + queries_per_worker
        
        data_slice = queries[start_idx:end_idx]
        
        logger.info(f"Worker {self.rank}: processing queries {start_idx}-{end_idx-1} "
                   f"({len(data_slice)} queries out of {total_queries} total)")
        
        return data_slice
    
    def generate_contexts(self, queries: List[Dict], max_samples: Optional[int] = None) -> None:
        """
        Generate contexts for assigned data slice
        
        Args:
            queries: Full list of queries
            max_samples: Maximum number of samples to process (per worker)
        """
        # Get data slice for this worker
        worker_queries = self.get_data_slice(queries)
        
        # Apply max_samples limit if specified
        if max_samples and len(worker_queries) > max_samples:
            worker_queries = worker_queries[:max_samples]
            logger.info(f"Worker {self.rank}: Limited to {max_samples} samples")
        
        # Generate contexts using the base generator
        self.generator.generate_contexts(worker_queries, start_index=0, max_samples=None)
    
    def print_statistics(self):
        """Print worker statistics"""
        print(f"\n{'='*60}")
        print(f"WORKER {self.rank} STATISTICS")
        print(f"{'='*60}")
        self.generator.print_statistics()


def setup_distributed():
    """Setup distributed training if available"""
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available, running in single-worker mode")
        return 0, 1
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Running with torchrun
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        # Initialize distributed backend
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
        
        # Set CUDA device for this worker
        if torch.cuda.is_available():
            torch.cuda.set_device(rank % torch.cuda.device_count())
            logger.info(f"Worker {rank}: Using CUDA device {torch.cuda.current_device()}")
        
        return rank, world_size
    else:
        # Single worker mode
        return 0, 1


def cleanup_distributed():
    """Cleanup distributed training"""
    if TORCH_AVAILABLE and dist.is_initialized():
        dist.destroy_process_group()


def merge_worker_outputs(base_output_file: str, world_size: int):
    """
    Merge outputs from all workers into a single file
    
    Args:
        base_output_file: Base output file path
        world_size: Number of workers
    """
    if world_size == 1:
        return
    
    logger.info("Merging worker outputs...")
    
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
                    logger.info(f"Merged {lines_written} lines from worker {rank}")
                
                # Clean up worker file
                os.remove(worker_file)
            else:
                logger.warning(f"Worker file not found: {worker_file}")
    
    logger.info(f"Merged output saved to {base_output_file} ({total_lines} total lines)")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Multi-worker context generation for queries")
    
    # Input/Output arguments
    parser.add_argument("--queries_file", type=str, default='data/generated_queries/versibcb_vace_queries_deduplicated.json', help="Path to queries JSON/JSONL file")
    parser.add_argument("--output_file", type=str, default='data/temp/contexts/contexts.jsonl', help="Output JSONL file for contexts")
    
    # Corpus arguments
    parser.add_argument("--corpus_path", type=str, default='/datanfs2/chenrongyi/data/docs', help="Path to corpus directory")
    parser.add_argument("--corpus_type", type=str, default="docstring", choices=["docstring", "srccodes"], help="Type of corpus")
    
    # Processing arguments
    parser.add_argument("--max_samples_per_worker", type=int, help="Maximum number of samples per worker")
    
    # Retrieval arguments
    parser.add_argument("--embedding_source", type=str, default="local", choices=["local", "togetherai"], help="Embedding source")
    parser.add_argument("--max_documents", type=int, default=10, help="Maximum documents to retrieve")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Maximum token length for context")
    parser.add_argument("--enable_str_match", action="store_true", help="Enable string matching")
    parser.add_argument("--fixed_docs_per_query", type=int, default=1, help="Fixed number of documents per query")
    parser.add_argument("--jump_exact_match", action="store_true", help="Jump exact match")
    
    # Control arguments
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup distributed training
    rank, world_size = setup_distributed()
    
    try:
        # Load queries (all workers load the same data)
        queries = load_queries_from_file(args.queries_file)
        
        if not queries:
            logger.error("No queries loaded, exiting")
            return
        
        # Initialize retriever
        try:
            retriever = QueryBasedRetriever(
                corpus_path=args.corpus_path,
                dependencies=None,  # Will be updated per query
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
        
        # Initialize multi-worker generator
        multi_generator = MultiWorkerContextGenerator(
            retriever=retriever,
            output_file=args.output_file,
            rank=rank,
            world_size=world_size
        )
        
        # Generate contexts
        start_time = time.time()
        multi_generator.generate_contexts(queries, args.max_samples_per_worker)
        total_time = time.time() - start_time
        
        # Print statistics
        multi_generator.print_statistics()
        logger.info(f"Worker {rank} completed in {total_time:.2f} seconds")
        
        # Synchronize workers before merging
        if TORCH_AVAILABLE and dist.is_initialized():
            dist.barrier()
        
        # Merge outputs (only rank 0)
        if rank == 0:
            merge_worker_outputs(args.output_file, world_size)
        
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main() 