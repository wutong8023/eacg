#!/usr/bin/env python3
"""
Complete RAG Pipeline

This module provides a unified interface for the complete RAG pipeline:
1. Query-based context generation (retrieval)
2. Context-based answer generation (inference)

The pipeline supports both single-step and multi-step execution,
with flexible configuration and multi-worker support.
"""

import argparse
import json
import os
import logging
import time
import sys
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import traceback
from tqdm import tqdm

# Import pipeline components
try:
    from utils.RAGutils.queryResultByInfer.context_build.batch_context_generator import (
        QueryBasedRetriever, BatchContextGenerator, load_queries_from_file
    )
    from utils.RAGutils.queryResultByInfer.query_based_retrieval_inference import QueryBasedInference
    PIPELINE_COMPONENTS_AVAILABLE = True
except ImportError as e:
    PIPELINE_COMPONENTS_AVAILABLE = False
    print(f"Pipeline components not available: {e}")

# Multi-worker components
try:
    from utils.RAGutils.queryResultByInfer.context_build.multi_worker_context_generator import MultiWorkerContextGenerator
    from utils.RAGutils.queryResultByInfer.infer.multi_worker_inference import MultiWorkerInference
    MULTI_WORKER_AVAILABLE = True
except ImportError:
    MULTI_WORKER_AVAILABLE = False
    print("Multi-worker components not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ContextGenerationConfig:
    """Configuration for context generation phase"""
    corpus_path: str
    corpus_type: str = "docstring"  # docstring, srccodes
    embedding_source: str = "local"  # local, togetherai
    max_documents: int = 10
    max_tokens: int = 4000
    enable_str_match: bool = True
    fixed_docs_per_query: int = 1
    jump_exact_match: bool = False


@dataclass
class InferenceConfig:
    """Configuration for inference phase"""
    model_path: str
    inference_type: str = "local"  # local, huggingface, togetherai
    api_key: Optional[str] = None
    api_model_name: Optional[str] = None
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95


@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    # Input/Output
    queries_file: str
    contexts_output: str
    final_output: str
    
    # Phase configurations
    context_config: ContextGenerationConfig
    inference_config: InferenceConfig
    
    # Multi-worker settings
    num_workers: int = 1
    max_samples_per_worker: Optional[int] = None
    
    # Control settings
    skip_context_generation: bool = False
    skip_inference: bool = False
    verbose: bool = False
    
    # Performance settings
    enable_progress_bar: bool = True
    save_intermediate_results: bool = True
    cleanup_worker_files: bool = True


class PipelineStatistics:
    """Statistics tracker for pipeline execution"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics"""
        self.context_generation = {
            'total_queries': 0,
            'successful_contexts': 0,
            'failed_contexts': 0,
            'total_time': 0,
            'avg_retrieval_time': 0
        }
        
        self.inference = {
            'total_contexts': 0,
            'successful_inferences': 0,
            'failed_inferences': 0,
            'total_time': 0,
            'avg_inference_time': 0
        }
        
        self.overall = {
            'pipeline_start_time': None,
            'pipeline_end_time': None,
            'total_pipeline_time': 0
        }
    
    def update_context_stats(self, stats_dict: Dict):
        """Update context generation statistics"""
        self.context_generation.update(stats_dict)
    
    def update_inference_stats(self, stats_dict: Dict):
        """Update inference statistics"""
        self.inference.update(stats_dict)
    
    def start_pipeline(self):
        """Mark pipeline start time"""
        self.overall['pipeline_start_time'] = time.time()
    
    def end_pipeline(self):
        """Mark pipeline end time and calculate total time"""
        self.overall['pipeline_end_time'] = time.time()
        if self.overall['pipeline_start_time']:
            self.overall['total_pipeline_time'] = (
                self.overall['pipeline_end_time'] - self.overall['pipeline_start_time']
            )
    
    def print_summary(self):
        """Print comprehensive statistics summary"""
        print("\n" + "="*80)
        print("COMPLETE PIPELINE STATISTICS")
        print("="*80)
        
        # Context Generation Stats
        print("\n📋 CONTEXT GENERATION:")
        ctx_stats = self.context_generation
        print(f"  Total queries: {ctx_stats['total_queries']}")
        print(f"  Successful contexts: {ctx_stats['successful_contexts']}")
        print(f"  Failed contexts: {ctx_stats['failed_contexts']}")
        if ctx_stats['total_queries'] > 0:
            success_rate = ctx_stats['successful_contexts'] / ctx_stats['total_queries']
            print(f"  Success rate: {success_rate:.1%}")
        if ctx_stats['total_time'] > 0:
            print(f"  Total time: {ctx_stats['total_time']:.2f}s")
        if ctx_stats['avg_retrieval_time'] > 0:
            print(f"  Average retrieval time: {ctx_stats['avg_retrieval_time']:.2f}s")
        
        # Inference Stats
        print("\n🧠 INFERENCE:")
        inf_stats = self.inference
        print(f"  Total contexts: {inf_stats['total_contexts']}")
        print(f"  Successful inferences: {inf_stats['successful_inferences']}")
        print(f"  Failed inferences: {inf_stats['failed_inferences']}")
        if inf_stats['total_contexts'] > 0:
            success_rate = inf_stats['successful_inferences'] / inf_stats['total_contexts']
            print(f"  Success rate: {success_rate:.1%}")
        if inf_stats['total_time'] > 0:
            print(f"  Total time: {inf_stats['total_time']:.2f}s")
        if inf_stats['avg_inference_time'] > 0:
            print(f"  Average inference time: {inf_stats['avg_inference_time']:.2f}s")
        
        # Overall Stats
        print("\n⏱️  OVERALL PIPELINE:")
        overall_stats = self.overall
        if overall_stats['total_pipeline_time'] > 0:
            print(f"  Total pipeline time: {overall_stats['total_pipeline_time']:.2f}s")
        
        # End-to-end success rate
        if (ctx_stats['total_queries'] > 0 and inf_stats['total_contexts'] > 0):
            end_to_end_success = inf_stats['successful_inferences'] / ctx_stats['total_queries']
            print(f"  End-to-end success rate: {end_to_end_success:.1%}")
        
        print("="*80)


class CompletePipeline:
    """
    Complete RAG Pipeline
    
    Integrates context generation and inference into a unified, modular system.
    Supports both single-worker and multi-worker execution.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the complete pipeline
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.stats = PipelineStatistics()
        
        # Components (will be initialized when needed)
        self.retriever = None
        self.context_generator = None
        self.inference_system = None
        
        # Multi-worker components
        self.multi_context_generator = None
        self.multi_inference = None
        
        # Setup logging
        if config.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        logger.info("Complete Pipeline initialized")
        logger.info(f"Configuration: {asdict(config)}")
    
    def _initialize_retriever(self):
        """Initialize the retrieval system"""
        if self.retriever is None:
            logger.info("Initializing retrieval system...")
            
            try:
                self.retriever = QueryBasedRetriever(
                    corpus_path=self.config.context_config.corpus_path,
                    dependencies=None,  # Will be updated per query
                    corpus_type=self.config.context_config.corpus_type,
                    embedding_source=self.config.context_config.embedding_source,
                    max_documents=self.config.context_config.max_documents,
                    max_tokens=self.config.context_config.max_tokens,
                    str_match=self.config.context_config.enable_str_match,
                    fixed_docs_per_query=self.config.context_config.fixed_docs_per_query,
                    jump_exact_match=self.config.context_config.jump_exact_match
                )
                logger.info("✅ Retrieval system initialized successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize retrieval system: {e}")
                raise
    
    def _initialize_context_generator(self):
        """Initialize the context generation system"""
        if self.context_generator is None:
            self._initialize_retriever()
            
            logger.info("Initializing context generator...")
            
            try:
                self.context_generator = BatchContextGenerator(
                    self.retriever, 
                    self.config.contexts_output
                )
                logger.info("✅ Context generator initialized successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize context generator: {e}")
                raise
    
    def _initialize_inference_system(self):
        """Initialize the inference system"""
        if self.inference_system is None:
            logger.info("Initializing inference system...")
            
            try:
                inf_config = self.config.inference_config
                self.inference_system = QueryBasedInference(
                    model_path=inf_config.model_path,
                    inference_type=inf_config.inference_type,
                    api_key=inf_config.api_key,
                    api_model_name=inf_config.api_model_name,
                    max_new_tokens=inf_config.max_new_tokens,
                    temperature=inf_config.temperature,
                    top_p=inf_config.top_p
                )
                logger.info("✅ Inference system initialized successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize inference system: {e}")
                raise
    
    def generate_contexts(self, queries: List[Dict], use_multi_worker: bool = None) -> List[Dict]:
        """
        Generate contexts for queries
        
        Args:
            queries: List of query dictionaries
            use_multi_worker: Whether to use multi-worker processing (auto-detect if None)
            
        Returns:
            List of context dictionaries
        """
        if self.config.skip_context_generation:
            logger.info("⏭️  Skipping context generation (configured to skip)")
            return self._load_existing_contexts()
        
        # Auto-detect multi-worker usage
        if use_multi_worker is None:
            use_multi_worker = self.config.num_workers > 1 and MULTI_WORKER_AVAILABLE
        
        logger.info(f"🔍 Starting context generation for {len(queries)} queries")
        logger.info(f"Multi-worker mode: {use_multi_worker} (workers: {self.config.num_workers})")
        
        start_time = time.time()
        
        try:
            if use_multi_worker:
                # Multi-worker context generation
                self._initialize_retriever()
                
                self.multi_context_generator = MultiWorkerContextGenerator(
                    retriever=self.retriever,
                    output_file=self.config.contexts_output,
                    rank=0,  # For single-process usage, always rank 0
                    world_size=1  # Will be handled by external process coordination
                )
                
                self.multi_context_generator.generate_contexts(
                    queries, 
                    self.config.max_samples_per_worker
                )
                
            else:
                # Single-worker context generation
                self._initialize_context_generator()
                
                self.context_generator.generate_contexts(
                    queries, 
                    start_index=0,
                    max_samples=self.config.max_samples_per_worker
                )
            
            generation_time = time.time() - start_time
            
            # Load and analyze results
            contexts = self._load_contexts_from_file(self.config.contexts_output)
            
            # Update statistics
            successful_contexts = sum(1 for ctx in contexts if ctx.get('success', False))
            failed_contexts = len(contexts) - successful_contexts
            
            self.stats.update_context_stats({
                'total_queries': len(queries),
                'successful_contexts': successful_contexts,
                'failed_contexts': failed_contexts,
                'total_time': generation_time,
                'avg_retrieval_time': generation_time / len(queries) if queries else 0
            })
            
            logger.info(f"✅ Context generation completed in {generation_time:.2f}s")
            logger.info(f"📊 Results: {successful_contexts} successful, {failed_contexts} failed")
            
            return contexts
            
        except Exception as e:
            logger.error(f"❌ Context generation failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def run_inference(self, contexts: List[Dict], use_multi_worker: bool = None) -> List[Dict]:
        """
        Run inference on contexts
        
        Args:
            contexts: List of context dictionaries
            use_multi_worker: Whether to use multi-worker processing (auto-detect if None)
            
        Returns:
            List of inference results
        """
        if self.config.skip_inference:
            logger.info("⏭️  Skipping inference (configured to skip)")
            return self._load_existing_results()
        
        # Filter successful contexts
        successful_contexts = [ctx for ctx in contexts if ctx.get('success', False)]
        
        if not successful_contexts:
            logger.warning("⚠️  No successful contexts found for inference")
            return []
        
        # Auto-detect multi-worker usage
        if use_multi_worker is None:
            use_multi_worker = self.config.num_workers > 1 and MULTI_WORKER_AVAILABLE
        
        logger.info(f"🧠 Starting inference for {len(successful_contexts)} contexts")
        logger.info(f"Multi-worker mode: {use_multi_worker} (workers: {self.config.num_workers})")
        
        start_time = time.time()
        
        try:
            if use_multi_worker:
                # Multi-worker inference
                self._initialize_inference_system()
                
                self.multi_inference = MultiWorkerInference(
                    inference_system=self.inference_system,
                    output_file=self.config.final_output,
                    rank=0,  # For single-process usage, always rank 0
                    world_size=1  # Will be handled by external process coordination
                )
                
                self.multi_inference.process_contexts(
                    successful_contexts,
                    self.config.max_samples_per_worker
                )
                
            else:
                # Single-worker inference
                self._initialize_inference_system()
                
                results = []
                
                # Process contexts with progress bar
                contexts_to_process = successful_contexts
                if self.config.max_samples_per_worker:
                    contexts_to_process = contexts_to_process[:self.config.max_samples_per_worker]
                
                iterator = tqdm(contexts_to_process, desc="Inference") if self.config.enable_progress_bar else contexts_to_process
                
                for i, context_item in enumerate(iterator):
                    try:
                        result = self._process_single_context(context_item, i)
                        results.append(result)
                        
                        # Save intermediate results
                        if self.config.save_intermediate_results:
                            self._save_result_to_file(result, self.config.final_output)
                            
                    except Exception as e:
                        logger.error(f"Error processing context {i}: {e}")
                        error_result = {
                            'id': context_item.get('id', f'item_{i}'),
                            'error': str(e),
                            'success': False
                        }
                        results.append(error_result)
                        
                        if self.config.save_intermediate_results:
                            self._save_result_to_file(error_result, self.config.final_output)
            
            inference_time = time.time() - start_time
            
            # Load and analyze results
            results = self._load_results_from_file(self.config.final_output)
            
            # Update statistics
            successful_inferences = sum(1 for result in results if result.get('success', False))
            failed_inferences = len(results) - successful_inferences
            
            self.stats.update_inference_stats({
                'total_contexts': len(successful_contexts),
                'successful_inferences': successful_inferences,
                'failed_inferences': failed_inferences,
                'total_time': inference_time,
                'avg_inference_time': inference_time / len(successful_contexts) if successful_contexts else 0
            })
            
            logger.info(f"✅ Inference completed in {inference_time:.2f}s")
            logger.info(f"📊 Results: {successful_inferences} successful, {failed_inferences} failed")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def run_complete_pipeline(self, queries: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Run the complete pipeline: context generation + inference
        
        Args:
            queries: List of queries (will load from file if None)
            
        Returns:
            List of final results
        """
        logger.info("🚀 Starting complete RAG pipeline")
        self.stats.start_pipeline()
        
        try:
            # Load queries if not provided
            if queries is None:
                logger.info(f"📂 Loading queries from: {self.config.queries_file}")
                queries = load_queries_from_file(self.config.queries_file)
                
                if not queries:
                    raise ValueError(f"No queries loaded from {self.config.queries_file}")
                
                logger.info(f"✅ Loaded {len(queries)} queries")
            
            # Phase 1: Context Generation
            logger.info("\n" + "="*60)
            logger.info("PHASE 1: CONTEXT GENERATION")
            logger.info("="*60)
            
            contexts = self.generate_contexts(queries)
            
            # Phase 2: Inference
            logger.info("\n" + "="*60)
            logger.info("PHASE 2: INFERENCE")
            logger.info("="*60)
            
            results = self.run_inference(contexts)
            
            # Pipeline completion
            self.stats.end_pipeline()
            
            logger.info("\n🎉 Complete pipeline finished successfully!")
            self.stats.print_summary()
            
            return results
            
        except Exception as e:
            self.stats.end_pipeline()
            logger.error(f"❌ Pipeline failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _process_single_context(self, context_item: Dict, item_index: int) -> Dict:
        """Process a single context item for inference"""
        start_time = time.time()
        
        try:
            # Extract fields
            item_id = context_item.get('id', f'item_{item_index}')
            query = context_item.get('query', '')
            context = context_item.get('context', '')
            retrieval_method = context_item.get('retrieval_method', 'unknown')
            
            # Validate
            if not query.strip():
                raise ValueError(f"No query found for item {item_id}")
            
            if not context.strip():
                raise ValueError(f"No context found for item {item_id}")
            
            # Generate answer
            answer = self.inference_system.generate_answer(query, context, retrieval_method)
            
            # Prepare result
            result = {
                'id': item_id,
                'original_item_id': context_item.get('original_item_id', ''),
                'query_index': context_item.get('query_index', 0),
                'dedup_key': context_item.get('dedup_key', ''),
                'query': query,
                'target_api': context_item.get('target_api', ''),
                'dependencies': context_item.get('dependencies', {}),
                'retrieval_method': retrieval_method,
                'context_length': len(context),
                'answer': answer,
                'retrieval_time': context_item.get('retrieval_time', 0),
                'inference_time': time.time() - start_time,
                'total_time': context_item.get('retrieval_time', 0) + (time.time() - start_time),
                'success': True
            }
            
            return result
            
        except Exception as e:
            return {
                'id': context_item.get('id', f'item_{item_index}'),
                'error': str(e),
                'processing_time': time.time() - start_time,
                'success': False
            }
    
    def _load_existing_contexts(self) -> List[Dict]:
        """Load existing contexts from file"""
        if not os.path.exists(self.config.contexts_output):
            raise FileNotFoundError(f"Contexts file not found: {self.config.contexts_output}")
        
        return self._load_contexts_from_file(self.config.contexts_output)
    
    def _load_existing_results(self) -> List[Dict]:
        """Load existing results from file"""
        if not os.path.exists(self.config.final_output):
            raise FileNotFoundError(f"Results file not found: {self.config.final_output}")
        
        return self._load_results_from_file(self.config.final_output)
    
    def _load_contexts_from_file(self, file_path: str) -> List[Dict]:
        """Load contexts from JSONL file"""
        contexts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    contexts.append(json.loads(line))
        return contexts
    
    def _load_results_from_file(self, file_path: str) -> List[Dict]:
        """Load results from JSONL file"""
        results = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results
    
    def _save_result_to_file(self, result: Dict, file_path: str):
        """Save a single result to JSONL file"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'a', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')
    
    def get_statistics(self) -> Dict:
        """Get current pipeline statistics"""
        return {
            'context_generation': self.stats.context_generation,
            'inference': self.stats.inference,
            'overall': self.stats.overall
        }
    
    def print_statistics(self):
        """Print current pipeline statistics"""
        self.stats.print_summary()


# Utility functions for easy configuration creation

def create_context_config(
    corpus_path: str,
    corpus_type: str = "docstring",
    embedding_source: str = "local",
    max_documents: int = 10,
    max_tokens: int = 4000,
    **kwargs
) -> ContextGenerationConfig:
    """Create context generation configuration with sensible defaults"""
    return ContextGenerationConfig(
        corpus_path=corpus_path,
        corpus_type=corpus_type,
        embedding_source=embedding_source,
        max_documents=max_documents,
        max_tokens=max_tokens,
        **kwargs
    )


def create_inference_config(
    model_path: str,
    inference_type: str = "local",
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    **kwargs
) -> InferenceConfig:
    """Create inference configuration with sensible defaults"""
    return InferenceConfig(
        model_path=model_path,
        inference_type=inference_type,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        **kwargs
    )


def create_pipeline_config(
    queries_file: str,
    contexts_output: str,
    final_output: str,
    context_config: ContextGenerationConfig,
    inference_config: InferenceConfig,
    num_workers: int = 1,
    **kwargs
) -> PipelineConfig:
    """Create pipeline configuration with sensible defaults"""
    return PipelineConfig(
        queries_file=queries_file,
        contexts_output=contexts_output,
        final_output=final_output,
        context_config=context_config,
        inference_config=inference_config,
        num_workers=num_workers,
        **kwargs
    )


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Complete RAG Pipeline")
    
    # Input/Output
    parser.add_argument("--queries_file", required=True, help="Path to queries file")
    parser.add_argument("--contexts_output", required=True, help="Path to contexts output file")
    parser.add_argument("--final_output", required=True, help="Path to final results output file")
    
    # Context generation
    parser.add_argument("--corpus_path", required=True, help="Path to corpus directory")
    parser.add_argument("--corpus_type", default="docstring", choices=["docstring", "srccodes"])
    parser.add_argument("--embedding_source", default="local", choices=["local", "togetherai"])
    parser.add_argument("--max_documents", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=4000)
    parser.add_argument("--enable_str_match", action="store_true")
    parser.add_argument("--fixed_docs_per_query", type=int, default=1)
    parser.add_argument("--jump_exact_match", action="store_true")
    
    # Inference
    parser.add_argument("--model_path", required=True, help="Path to model")
    parser.add_argument("--inference_type", default="local", choices=["local", "huggingface", "togetherai"])
    parser.add_argument("--api_key", help="API key for remote inference")
    parser.add_argument("--api_model_name", help="Model name for API inference")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    
    # Pipeline control
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_samples_per_worker", type=int)
    parser.add_argument("--skip_context_generation", action="store_true")
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Create configurations
    context_config = create_context_config(
        corpus_path=args.corpus_path,
        corpus_type=args.corpus_type,
        embedding_source=args.embedding_source,
        max_documents=args.max_documents,
        max_tokens=args.max_tokens,
        enable_str_match=args.enable_str_match,
        fixed_docs_per_query=args.fixed_docs_per_query,
        jump_exact_match=args.jump_exact_match
    )
    
    inference_config = create_inference_config(
        model_path=args.model_path,
        inference_type=args.inference_type,
        api_key=args.api_key,
        api_model_name=args.api_model_name,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    pipeline_config = create_pipeline_config(
        queries_file=args.queries_file,
        contexts_output=args.contexts_output,
        final_output=args.final_output,
        context_config=context_config,
        inference_config=inference_config,
        num_workers=args.num_workers,
        max_samples_per_worker=args.max_samples_per_worker,
        skip_context_generation=args.skip_context_generation,
        skip_inference=args.skip_inference,
        verbose=args.verbose
    )
    
    # Run pipeline
    pipeline = CompletePipeline(pipeline_config)
    results = pipeline.run_complete_pipeline()
    
    print(f"\n✅ Pipeline completed successfully!")
    print(f"📊 Final results: {len(results)} items")
    print(f"📁 Results saved to: {args.final_output}")


if __name__ == "__main__":
    main() 