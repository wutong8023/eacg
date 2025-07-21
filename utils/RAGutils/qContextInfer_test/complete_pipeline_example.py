#!/usr/bin/env python3
"""
Complete Pipeline Usage Examples

This file demonstrates how to use the CompletePipeline class
with different configurations and use cases.
"""

import os
import sys
import json
from pathlib import Path

# Add the utils directory to the path to enable imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.RAGutils.complete_pipeline import (
    CompletePipeline,
    ContextGenerationConfig,
    InferenceConfig, 
    PipelineConfig,
    create_context_config,
    create_inference_config,
    create_pipeline_config
)


def example_1_basic_usage():
    """Example 1: Basic pipeline usage with local model"""
    print("=" * 60)
    print("EXAMPLE 1: Basic Local Pipeline")
    print("=" * 60)
    
    # Configuration
    context_config = create_context_config(
        corpus_path="/datanfs4/chenrongyi/data/docs",
        corpus_type="docstring",
        embedding_source="local",
        max_documents=1,
        max_tokens=2000,
        enable_str_match=True,
        fixed_docs_per_query=1
    )
    
    inference_config = create_inference_config(
        model_path="/datanfs4/chenrongyi/models/Llama-3.1-8B-Instruct",
        inference_type="local",
        max_new_tokens=512,
        temperature=1e-5,
        top_p=0.95
    )
    
    pipeline_config = create_pipeline_config(
        queries_file="data/generated_queries/sample_queries.json",
        contexts_output="data/temp/example1_contexts.jsonl",
        final_output="data/temp/example1_results.jsonl",
        context_config=context_config,
        inference_config=inference_config,
        num_workers=1,
        verbose=True
    )
    
    # Run pipeline
    pipeline = CompletePipeline(pipeline_config)
    
    # Option 1: Run complete pipeline
    try:
        results = pipeline.run_complete_pipeline()
        print(f"✅ Pipeline completed with {len(results)} results")
        
        # Print statistics
        pipeline.print_statistics()
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return None
    
    return results


def example_2_step_by_step():
    """Example 2: Step-by-step execution with custom queries"""
    print("=" * 60)
    print("EXAMPLE 2: Step-by-Step Execution")
    print("=" * 60)
    
    # Custom queries for demonstration
    queries = [
        {
            "id": "example_1",
            "query": "How to use pandas DataFrame?",
            "target_api": "pandas.DataFrame",
            "dependencies": {"pandas": "latest"}
        },
        {
            "id": "example_2", 
            "query": "How to create numpy array?",
            "target_api": "numpy.array",
            "dependencies": {"numpy": "latest"}
        }
    ]
    
    # Configuration
    context_config = create_context_config(
        corpus_path="/datanfs4/chenrongyi/data/docs",
        corpus_type="docstring",
        max_documents=3,
        max_tokens=1500
    )
    
    inference_config = create_inference_config(
        model_path="/datanfs4/chenrongyi/models/Llama-3.1-8B-Instruct",
        max_new_tokens=256,
        temperature=0.3
    )
    
    pipeline_config = create_pipeline_config(
        queries_file="",  # Not needed for custom queries
        contexts_output="data/temp/example2_contexts.jsonl", 
        final_output="data/temp/example2_results.jsonl",
        context_config=context_config,
        inference_config=inference_config,
        verbose=True
    )
    
    # Initialize pipeline
    pipeline = CompletePipeline(pipeline_config)
    
    try:
        # Step 1: Generate contexts
        print("\n🔍 Step 1: Generating contexts...")
        contexts = pipeline.generate_contexts(queries)
        print(f"Generated {len(contexts)} contexts")
        
        # Step 2: Run inference
        print("\n🧠 Step 2: Running inference...")
        results = pipeline.run_inference(contexts)
        print(f"Generated {len(results)} results")
        
        # Print final statistics
        pipeline.print_statistics()
        
        return results
        
    except Exception as e:
        print(f"❌ Step-by-step execution failed: {e}")
        return None


def example_3_multi_worker():
    """Example 3: Multi-worker processing"""
    print("=" * 60)
    print("EXAMPLE 3: Multi-Worker Processing")
    print("=" * 60)
    
    # Configuration for multi-worker
    context_config = create_context_config(
        corpus_path="/datanfs4/chenrongyi/data/docs",
        corpus_type="docstring",
        max_documents=2,
        max_tokens=3000
    )
    
    inference_config = create_inference_config(
        model_path="/datanfs4/chenrongyi/models/Llama-3.1-8B-Instruct",
        max_new_tokens=512
    )
    
    pipeline_config = create_pipeline_config(
        queries_file="data/generated_queries/sample_queries.json",
        contexts_output="data/temp/example3_contexts.jsonl",
        final_output="data/temp/example3_results.jsonl",
        context_config=context_config,
        inference_config=inference_config,
        num_workers=4,  # Multi-worker
        max_samples_per_worker=100,  # Limit samples per worker
        verbose=True
    )
    
    # Note: Multi-worker mode typically requires external process coordination
    # This example shows the configuration, but actual multi-worker execution
    # would need to be handled by external scripts (like torchrun)
    
    pipeline = CompletePipeline(pipeline_config)
    
    try:
        # This will run in single-worker mode but with multi-worker configuration
        results = pipeline.run_complete_pipeline()
        print(f"✅ Multi-worker pipeline completed with {len(results)} results")
        
        return results
        
    except Exception as e:
        print(f"❌ Multi-worker pipeline failed: {e}")
        return None


def example_4_api_inference():
    """Example 4: Using API-based inference"""
    print("=" * 60)
    print("EXAMPLE 4: API-based Inference")
    print("=" * 60)
    
    # Configuration with API inference
    context_config = create_context_config(
        corpus_path="/datanfs4/chenrongyi/data/docs",
        corpus_type="docstring"
    )
    
    # API-based inference configuration
    inference_config = create_inference_config(
        model_path="",  # Not needed for API
        inference_type="togetherai",  # or "huggingface"
        api_key="your_api_key_here",  # Replace with actual API key
        api_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        max_new_tokens=512,
        temperature=0.7
    )
    
    pipeline_config = create_pipeline_config(
        queries_file="data/generated_queries/sample_queries.json",
        contexts_output="data/temp/example4_contexts.jsonl",
        final_output="data/temp/example4_results.jsonl",
        context_config=context_config,
        inference_config=inference_config,
        max_samples_per_worker=10,  # Limit for API usage
        verbose=True
    )
    
    pipeline = CompletePipeline(pipeline_config)
    
    print("⚠️  Note: This example requires a valid API key")
    print("     Update the api_key in the configuration before running")
    
    # Uncomment to run with valid API key
    # try:
    #     results = pipeline.run_complete_pipeline()
    #     print(f"✅ API pipeline completed with {len(results)} results")
    #     return results
    # except Exception as e:
    #     print(f"❌ API pipeline failed: {e}")
    #     return None
    
    return None


def example_5_partial_pipeline():
    """Example 5: Partial pipeline execution (skip phases)"""
    print("=" * 60)
    print("EXAMPLE 5: Partial Pipeline Execution")
    print("=" * 60)
    
    # Configuration with phase skipping
    context_config = create_context_config(
        corpus_path="/datanfs4/chenrongyi/data/docs"
    )
    
    inference_config = create_inference_config(
        model_path="/datanfs4/chenrongyi/models/Llama-3.1-8B-Instruct"
    )
    
    # Example 5a: Skip context generation (use existing contexts)
    pipeline_config_inference_only = create_pipeline_config(
        queries_file="",  # Not needed
        contexts_output="data/temp/existing_contexts.jsonl",  # Existing file
        final_output="data/temp/example5a_results.jsonl",
        context_config=context_config,
        inference_config=inference_config,
        skip_context_generation=True,  # Skip context generation
        verbose=True
    )
    
    # Example 5b: Context generation only
    pipeline_config_context_only = create_pipeline_config(
        queries_file="data/generated_queries/sample_queries.json",
        contexts_output="data/temp/example5b_contexts.jsonl",
        final_output="data/temp/example5b_results.jsonl",
        context_config=context_config,
        inference_config=inference_config,
        skip_inference=True,  # Skip inference
        verbose=True
    )
    
    print("\n📋 5a: Inference-only pipeline (skip context generation)")
    try:
        pipeline_inference = CompletePipeline(pipeline_config_inference_only)
        # This would load existing contexts and run inference only
        # results = pipeline_inference.run_complete_pipeline()
        print("✅ Configuration ready for inference-only pipeline")
    except Exception as e:
        print(f"❌ Inference-only configuration failed: {e}")
    
    print("\n🔍 5b: Context-generation-only pipeline (skip inference)")
    try:
        pipeline_context = CompletePipeline(pipeline_config_context_only)
        # This would generate contexts but skip inference
        # contexts = pipeline_context.run_complete_pipeline()
        print("✅ Configuration ready for context-only pipeline")
    except Exception as e:
        print(f"❌ Context-only configuration failed: {e}")


def example_6_custom_configuration():
    """Example 6: Advanced custom configuration"""
    print("=" * 60)
    print("EXAMPLE 6: Advanced Custom Configuration")
    print("=" * 60)
    
    # Advanced context configuration
    context_config = ContextGenerationConfig(
        corpus_path="/datanfs4/chenrongyi/data/docs",
        corpus_type="srccodes",  # Source code instead of docstrings
        embedding_source="togetherai",  # Use TogetherAI embeddings
        max_documents=5,
        max_tokens=5000,
        enable_str_match=True,
        fixed_docs_per_query=3,
        jump_exact_match=False
    )
    
    # Advanced inference configuration
    inference_config = InferenceConfig(
        model_path="/datanfs4/chenrongyi/models/Llama-3.1-8B-Instruct",
        inference_type="local",
        max_new_tokens=1024,
        temperature=0.1,  # Very low temperature for deterministic output
        top_p=0.9
    )
    
    # Advanced pipeline configuration
    pipeline_config = PipelineConfig(
        queries_file="data/generated_queries/sample_queries.json",
        contexts_output="data/temp/example6_contexts.jsonl",
        final_output="data/temp/example6_results.jsonl",
        context_config=context_config,
        inference_config=inference_config,
        num_workers=2,
        max_samples_per_worker=50,
        skip_context_generation=False,
        skip_inference=False,
        verbose=True,
        enable_progress_bar=True,
        save_intermediate_results=True,
        cleanup_worker_files=True
    )
    
    pipeline = CompletePipeline(pipeline_config)
    
    # Show configuration details
    print("📋 Advanced Configuration:")
    print(f"  Context: {context_config}")
    print(f"  Inference: {inference_config}")
    print(f"  Pipeline: {pipeline_config}")
    
    print("✅ Advanced configuration ready")
    
    # Uncomment to run
    # try:
    #     results = pipeline.run_complete_pipeline()
    #     return results
    # except Exception as e:
    #     print(f"❌ Advanced pipeline failed: {e}")
    #     return None


def create_sample_queries_file():
    """Create a sample queries file for testing"""
    sample_queries = [
        {
            "id": "sample_1",
            "query": "How to read a CSV file using pandas?",
            "target_api": "pandas.read_csv",
            "dependencies": {"pandas": "latest"}
        },
        {
            "id": "sample_2",
            "query": "How to create a NumPy array from a list?",
            "target_api": "numpy.array",
            "dependencies": {"numpy": "latest"}
        },
        {
            "id": "sample_3",
            "query": "How to plot a simple line chart with matplotlib?",
            "target_api": "matplotlib.pyplot.plot",
            "dependencies": {"matplotlib": "latest"}
        }
    ]
    
    # Create directory if it doesn't exist
    os.makedirs("data/generated_queries", exist_ok=True)
    
    # Save sample queries
    with open("data/generated_queries/sample_queries.json", "w", encoding="utf-8") as f:
        json.dump(sample_queries, f, indent=2, ensure_ascii=False)
    
    print("✅ Sample queries file created: data/generated_queries/sample_queries.json")


def main():
    """Run all examples"""
    print("🚀 Complete Pipeline Examples")
    print("=" * 80)
    
    # Create sample data first
    print("\n📝 Creating sample queries file...")
    create_sample_queries_file()
    
    # Create output directories
    os.makedirs("data/temp", exist_ok=True)
    
    # Run examples
    print("\n" + "="*80)
    print("RUNNING EXAMPLES")
    print("="*80)
    
    # Note: Most examples are configured but not run to avoid requiring
    # actual model files and corpus data. Uncomment the execution lines
    # in each example to run them with your actual data.
    
    try:
        # Example 1: Basic usage
        # results1 = example_1_basic_usage()
        example_1_basic_usage()
        
        # Example 2: Step-by-step
        # results2 = example_2_step_by_step()
        example_2_step_by_step()
        
        # Example 3: Multi-worker
        # results3 = example_3_multi_worker()
        example_3_multi_worker()
        
        # Example 4: API inference
        example_4_api_inference()
        
        # Example 5: Partial pipeline
        example_5_partial_pipeline()
        
        # Example 6: Custom configuration
        example_6_custom_configuration()
        
    except Exception as e:
        print(f"❌ Examples failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETED")
    print("="*80)
    print("\n📖 Usage Instructions:")
    print("1. Update file paths in the examples to match your environment")
    print("2. Ensure you have the required model files and corpus data")
    print("3. Uncomment the execution lines in each example to run them")
    print("4. For API examples, provide valid API keys")
    print("5. For multi-worker examples, use external process coordination")
    print("\n✨ The pipeline provides a clean, modular interface for RAG workflows!")


if __name__ == "__main__":
    main() 