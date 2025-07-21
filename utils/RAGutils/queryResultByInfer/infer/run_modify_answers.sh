#!/bin/bash

# Default paths
INPUT_FILE="data/temp/inference/pipeline_results.jsonl"
OUTPUT_FILE="data/temp/inference/modified_pipeline_results.jsonl"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_file)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run the Python script
python utils/RAGutils/queryResultByInfer/infer/modify_equivalent_answers.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" 