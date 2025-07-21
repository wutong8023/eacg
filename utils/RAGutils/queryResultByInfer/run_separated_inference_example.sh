#!/bin/bash

# Example script for separated context generation and multi-worker inference
# This demonstrates the two-phase approach: 
# 1. Generate contexts once
# 2. Run inference with multiple workers

set -e

# Configuration
CORPUS_PATH="data/corpus"
MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.2"
INFERENCE_TYPE="local"
QUERIES_FILE="data/generated_queries/versibcb_vace_queries_deduplicated.json"
OUTPUT_DIR="data/temp/separated_inference"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=== Separated Context Generation and Multi-Worker Inference Example ==="
echo "Corpus path: $CORPUS_PATH"
echo "Model path: $MODEL_PATH"
echo "Queries file: $QUERIES_FILE"
echo "Output directory: $OUTPUT_DIR"

# Phase 1: Generate contexts (single process, no GPU needed)
echo -e "\n=== Phase 1: Context Generation ==="
echo "Generating contexts for all queries..."

python utils/RAGutils/queryResultByInfer/batch_context_generator.py \
    --queries_file "$QUERIES_FILE" \
    --corpus_path "$CORPUS_PATH" \
    --output_file "$OUTPUT_DIR/contexts.jsonl" \
    --corpus_type "docstring" \
    --embedding_source "local" \
    --max_documents 10 \
    --max_tokens 4000 \
    --enable_str_match \
    --max_samples 100 \
    --verbose

echo "Context generation completed. Results saved to: $OUTPUT_DIR/contexts.jsonl"

# Phase 2: Multi-worker inference (distributed across GPUs)
echo -e "\n=== Phase 2: Multi-Worker Inference ==="

# Example 1: Single worker inference
echo -e "\n1. Single worker inference"
python utils/RAGutils/queryResultByInfer/multi_worker_inference.py \
    --contexts_file "$OUTPUT_DIR/contexts.jsonl" \
    --output_file "$OUTPUT_DIR/results_single_worker.jsonl" \
    --model_path "$MODEL_PATH" \
    --inference_type "$INFERENCE_TYPE" \
    --max_new_tokens 512 \
    --temperature 0.2 \
    --precision "fp16" \
    --verbose

echo "Single worker inference completed. Results: $OUTPUT_DIR/results_single_worker.jsonl"

# Example 2: Multi-worker inference with torchrun (4 GPUs)
echo -e "\n2. Multi-worker inference (4 workers)"
if command -v torchrun &> /dev/null; then
    torchrun --nproc_per_node=4 \
        utils/RAGutils/queryResultByInfer/multi_worker_inference.py \
        --contexts_file "$OUTPUT_DIR/contexts.jsonl" \
        --output_file "$OUTPUT_DIR/results_multi_worker.jsonl" \
        --model_path "$MODEL_PATH" \
        --inference_type "$INFERENCE_TYPE" \
        --max_new_tokens 512 \
        --temperature 0.2 \
        --precision "fp16" \
        --verbose
    
    echo "Multi-worker inference completed. Results: $OUTPUT_DIR/results_multi_worker.jsonl"
else
    echo "torchrun not available, skipping multi-worker example"
fi

# Example 3: Multi-worker inference with different GPU allocation (2 GPUs)
echo -e "\n3. Multi-worker inference (2 workers)"
if command -v torchrun &> /dev/null; then
    torchrun --nproc_per_node=2 \
        utils/RAGutils/queryResultByInfer/multi_worker_inference.py \
        --contexts_file "$OUTPUT_DIR/contexts.jsonl" \
        --output_file "$OUTPUT_DIR/results_2_workers.jsonl" \
        --model_path "$MODEL_PATH" \
        --inference_type "$INFERENCE_TYPE" \
        --max_new_tokens 512 \
        --temperature 0.2 \
        --precision "fp16" \
        --verbose
    
    echo "2-worker inference completed. Results: $OUTPUT_DIR/results_2_workers.jsonl"
else
    echo "torchrun not available, skipping 2-worker example"
fi

# Example 4: Remote API inference (no GPU needed, can use many workers)
echo -e "\n4. Multi-worker remote API inference"
if [[ -n "$HUGGINGFACE_API_KEY" ]]; then
    torchrun --nproc_per_node=8 \
        utils/RAGutils/queryResultByInfer/multi_worker_inference.py \
        --contexts_file "$OUTPUT_DIR/contexts.jsonl" \
        --output_file "$OUTPUT_DIR/results_api_inference.jsonl" \
        --model_path "mistralai/Mistral-7B-Instruct-v0.1" \
        --inference_type "huggingface" \
        --api_key "$HUGGINGFACE_API_KEY" \
        --api_model_name "mistralai/Mistral-7B-Instruct-v0.1" \
        --max_new_tokens 512 \
        --temperature 0.2 \
        --verbose
    
    echo "API inference completed. Results: $OUTPUT_DIR/results_api_inference.jsonl"
else
    echo "HUGGINGFACE_API_KEY not set, skipping API inference example"
fi

echo -e "\n=== Summary ==="
echo "Context generation and inference separation completed!"
echo "Generated files:"
ls -la "$OUTPUT_DIR"

echo -e "\nBenefits of this approach:"
echo "1. Context generation is done once and can be reused"
echo "2. Inference can be distributed across multiple GPUs"
echo "3. Different inference configurations can use the same contexts"
echo "4. Better resource utilization and scalability"

# Optional: Compare results
echo -e "\n=== Optional: Result Analysis ==="
if [[ -f "$OUTPUT_DIR/results_single_worker.jsonl" && -f "$OUTPUT_DIR/results_multi_worker.jsonl" ]]; then
    echo "Comparing single vs multi-worker results..."
    
    single_count=$(wc -l < "$OUTPUT_DIR/results_single_worker.jsonl")
    multi_count=$(wc -l < "$OUTPUT_DIR/results_multi_worker.jsonl")
    
    echo "Single worker results: $single_count samples"
    echo "Multi-worker results: $multi_count samples"
    
    if [[ $single_count -eq $multi_count ]]; then
        echo "✓ Result counts match - multi-worker processing successful"
    else
        echo "⚠ Result counts differ - check for processing errors"
    fi
fi 