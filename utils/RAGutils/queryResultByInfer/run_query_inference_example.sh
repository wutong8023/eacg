#!/bin/bash

# Query-based Retrieval and Inference Example Script
# This script demonstrates different usage scenarios

echo "=== Query-based Retrieval and Inference Examples ==="

# Set common parameters
MODEL_PATH="/datanfs2/chenrongyi/models/Llama-3.1-8B"
CORPUS_PATH="/datanfs2/chenrongyi/data/docs"  # Adjust to your corpus path
INFERENCE_TYPE="local"  # Change to "huggingface" or "togetherai" for API inference

# Example 1: Exact API match scenario
echo -e "\n1. Example with exact API match (numpy.argsort)"
python utils/RAGutils/queryResultByInfer/query_based_retrieval_inference.py \
    --query "What is the equivalent of numpy.argsort in numpy of target dependency, which is used to return the indices that would sort an array?" \
    --target_api "numpy.argsort" \
    --corpus_path "$CORPUS_PATH" \
    --dependencies '{"numpy": "1.16.6", "matplotlib": "2.0.2", "scipy": "1.4.1"}' \
    --corpus_type "docstring" \
    --model_path "$MODEL_PATH" \
    --inference_type "$INFERENCE_TYPE" \
    --max_new_tokens 512 \
    --output_file "data/temp/testqueryinfer/results_exact_match.json" \
    --verbose

echo -e "\n" + "="*60

# Example 2: RAG fallback scenario (no target_api provided)
echo -e "\n2. Example with RAG fallback (no target API)"
python utils/RAGutils/queryResultByInfer/query_based_retrieval_inference.py \
    --query "How can I create a scatter plot with different colors for different categories?" \
    --corpus_path "$CORPUS_PATH" \
    --dependencies '{"matplotlib": "2.0.2", "numpy": "1.16.6"}' \
    --corpus_type "docstring" \
    --model_path "$MODEL_PATH" \
    --inference_type "$INFERENCE_TYPE" \
    --max_documents 8 \
    --max_tokens 3000 \
    --output_file "data/temp/testqueryinfer/results_rag_fallback.json" \
    --verbose

echo -e "\n" + "="*60

# Example 3: Target API not found, fallback to RAG
echo -e "\n3. Example with target API not found (fallback to RAG)"
python utils/RAGutils/queryResultByInfer/query_based_retrieval_inference.py \
    --query "How to calculate the Fast Fourier Transform?" \
    --target_api "numpy.nonexistent_function" \
    --corpus_path "$CORPUS_PATH" \
    --dependencies '{"numpy": "1.16.6", "scipy": "1.4.1"}' \
    --corpus_type "docstring" \
    --model_path "$MODEL_PATH" \
    --inference_type "$INFERENCE_TYPE" \
    --max_documents 5 \
    --output_file "data/temp/testqueryinfer/results_fallback_example.json" \
    --verbose

echo -e "\n" + "="*60

# Example 4: Using source code corpus
echo -e "\n4. Example using source code corpus"
python utils/RAGutils/queryResultByInfer/query_based_retrieval_inference.py \
    --query "Show me how to use matplotlib to create a line plot" \
    --target_api "matplotlib.pyplot.plot" \
    --corpus_path "$CORPUS_PATH" \
    --dependencies '{"matplotlib": "2.0.2"}' \
    --corpus_type "srccodes" \
    --model_path "$MODEL_PATH" \
    --inference_type "$INFERENCE_TYPE" \
    --output_file "data/temp/testqueryinfer/results_srccode.json" \
    --verbose

echo -e "\n" + "="*60

# Example 5: Remote API inference (uncomment and configure if needed)
# echo -e "\n5. Example with remote API inference"
# python query_based_retrieval_inference.py \
#     --query "How to compute eigenvalues of a matrix?" \
#     --target_api "numpy.linalg.eig" \
#     --corpus_path "$CORPUS_PATH" \
#     --dependencies '{"numpy": "1.16.6"}' \
#     --corpus_type "docstring" \
#     --inference_type "huggingface" \
#     --api_key "your_api_key_here" \
#     --api_model_name "mistralai/Mistral-7B-Instruct-v0.1" \
#     --output_file "results_remote_api.json" \
#     --verbose

echo -e "\n=== All examples completed! ==="
echo "Check the output JSON files for detailed results." 