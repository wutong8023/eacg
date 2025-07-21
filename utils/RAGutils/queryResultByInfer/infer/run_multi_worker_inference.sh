#!/bin/bash

# Multi-Worker Inference Launcher Script (Fixed GPU Allocation via CUDA_VISIBLE_DEVICES)
# 
# This script launches multiple independent worker processes for inference,
# each with dedicated GPU allocation via CUDA_VISIBLE_DEVICES.
# Each worker uses HuggingFace's device_map="auto" internally.

# Parse command line arguments
SHORT=w:,g:,h
LONG=world_size:,gpus_per_worker:,help

PARSED=$(getopt --options $SHORT --longoptions $LONG --name "$0" -- "$@") || { echo "Invalid Arguments."; exit 2; }
eval set -- "$PARSED"

# Default parameters
world_size=8
gpus_per_worker=1

# Required parameters (will be set via command line)
CONTEXTS_FILE="data/temp/contexts/versibcb_vscc_contexts_staticAnalysis.jsonl"
OUTPUT_FILE="data/temp/inference/versibcb_vscc_contexts_staticAnalysis.jsonl"
MODEL_PATH="/datanfs4/chenrongyi/models/Llama-3.1-8B-Instruct"
INFERENCE_TYPE="qdd"
MAX_NEW_TOKENS=512
TEMPERATURE=0.2
TOP_P=0.95
MAX_SAMPLES_PER_WORKER=""
VERBOSE=false
API_KEY="YOUR_API_KEY_HERE"
API_MODEL_NAME="llama-3.1-8b-instruct"
RESUME_FROM_EXISTING=true
# Function to get free GPUs
get_free_gpus() {
    # Get GPU usage, consider GPUs with <10% memory usage as free
    free_gpus=($(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | \
                awk -F, '{if ($2/$3 < 0.1) print $1}'))
    echo "${free_gpus[@]}"
}

# Parse arguments
while true; do
    case "$1" in
        -h|--help) 
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Optional Arguments:"
            echo "  -w, --world_size <num>        Number of workers (default: 8)"
            echo "  -g, --gpus_per_worker <num>   GPUs per worker (default: 1)"
            echo "  --contexts_file <file>        Path to contexts JSONL file (default: pipeline_contexts.jsonl)"
            echo "  --output_file <file>          Output JSONL file for results (default: pipeline_results.jsonl)"
            echo "  --model_path <path>           Model path (default: /datanfs2/chenrongyi/models/Llama-3.1-8B)"
            echo "  --inference_type <type>       local/huggingface/togetherai (default: local)"
            echo "  --max_new_tokens <num>        Maximum new tokens (default: 512)"
            echo "  --temperature <float>         Sampling temperature (default: 0.2)"
            echo "  --top_p <float>               Top-p sampling (default: 0.95)"
            echo "  --max_samples_per_worker <num> Maximum samples per worker"
            echo "  --verbose                     Enable verbose logging"
            echo "  --api_key <key>               API key for remote inference"
            echo "  --api_model_name <name>       Model name for API inference"
            echo ""
            echo "Example:"
            echo "  $0 -w 8 -g 1"
            echo "  $0 --contexts_file my_contexts.jsonl --output_file my_results.jsonl -w 4 -g 2"
            exit 0 ;;
        -w|--world_size) world_size="$2"; shift 2 ;;
        -g|--gpus_per_worker) gpus_per_worker="$2"; shift 2 ;;
        --contexts_file) CONTEXTS_FILE="$2"; shift 2 ;;
        --output_file) OUTPUT_FILE="$2"; shift 2 ;;
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --inference_type) INFERENCE_TYPE="$2"; shift 2 ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top_p) TOP_P="$2"; shift 2 ;;
        --max_samples_per_worker) MAX_SAMPLES_PER_WORKER="$2"; shift 2 ;;
        --api_key) API_KEY="$2"; shift 2 ;;
        --api_model_name) API_MODEL_NAME="$2"; shift 2 ;;
        --verbose) VERBOSE=true; shift ;;
        --) shift; break ;;
        *) echo "Programming error"; exit 3 ;;
    esac
done

# Validate required files
if [ ! -f "$CONTEXTS_FILE" ]; then
    echo "Error: Contexts file not found: $CONTEXTS_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Get available GPUs
nvidia-smi
available_gpus=($(get_free_gpus))
total_available_gpus=${#available_gpus[@]}

if [ $total_available_gpus -eq 0 ]; then
    echo "Error: No free GPUs available"
    exit 1
fi

echo "Found $total_available_gpus free GPUs: ${available_gpus[*]}"

# Calculate total GPUs needed
total_gpus_needed=$((world_size * gpus_per_worker))

if [ $total_gpus_needed -gt $total_available_gpus ]; then
    echo "Error: Need $total_gpus_needed GPUs but only $total_available_gpus available"
    echo "Reduce world_size or gpus_per_worker"
    exit 1
fi

echo "Configuration:"
echo "  World size: $world_size"
echo "  GPUs per worker: $gpus_per_worker"
echo "  Total GPUs needed: $total_gpus_needed"
echo "  Contexts file: $CONTEXTS_FILE"
echo "  Output file: $OUTPUT_FILE"
echo "  Model path: $MODEL_PATH"
echo "  Inference type: $INFERENCE_TYPE"

# GPU allocation strategy
echo ""
echo "GPU allocation strategy:"
for ((worker=0; worker<world_size; worker++)); do
    start_gpu=$((worker * gpus_per_worker))
    worker_gpus=""
    for ((gpu=0; gpu<gpus_per_worker; gpu++)); do
        gpu_idx=$((start_gpu + gpu))
        physical_gpu=${available_gpus[gpu_idx]}
        if [ -n "$worker_gpus" ]; then
            worker_gpus="$worker_gpus,$physical_gpu"
        else
            worker_gpus="$physical_gpu"
        fi
    done
    echo "  Worker $worker: CUDA_VISIBLE_DEVICES=$worker_gpus (physical GPUs [$worker_gpus])"
done

echo ""
echo "Starting multi-worker inference..."

# Create a trap to kill all background processes when script is interrupted
trap 'kill $(jobs -p) 2>/dev/null; wait; echo "All workers terminated."; exit' SIGINT SIGTERM

# Start all workers in the background
pids=()

for ((worker=0; worker<world_size; worker++)); do
    # Calculate GPU allocation for this worker
    start_gpu=$((worker * gpus_per_worker))
    worker_gpus=""
    for ((gpu=0; gpu<gpus_per_worker; gpu++)); do
        gpu_idx=$((start_gpu + gpu))
        physical_gpu=${available_gpus[gpu_idx]}
        if [ -n "$worker_gpus" ]; then
            worker_gpus="$worker_gpus,$physical_gpu"
        else
            worker_gpus="$physical_gpu"
        fi
    done
    
    echo "Starting worker $worker with CUDA_VISIBLE_DEVICES=$worker_gpus"
    
    # Build command arguments
    cmd_args=""
    cmd_args="$cmd_args --contexts_file $CONTEXTS_FILE"
    cmd_args="$cmd_args --output_file $OUTPUT_FILE"
    cmd_args="$cmd_args --rank $worker"
    cmd_args="$cmd_args --world_size $world_size"
    cmd_args="$cmd_args --model_path $MODEL_PATH"
    cmd_args="$cmd_args --inference_type $INFERENCE_TYPE"
    cmd_args="$cmd_args --max_new_tokens $MAX_NEW_TOKENS"
    cmd_args="$cmd_args --temperature $TEMPERATURE"
    cmd_args="$cmd_args --top_p $TOP_P"
    cmd_args="$cmd_args --api_key $API_KEY"
    cmd_args="$cmd_args --api_model_name $API_MODEL_NAME"

    if [ "$RESUME_FROM_EXISTING" = true ]; then
        cmd_args="$cmd_args --resume_from_existing"
    fi

    if [ -n "$MAX_SAMPLES_PER_WORKER" ]; then
            cmd_args="$cmd_args --max_samples_per_worker $MAX_SAMPLES_PER_WORKER"
    fi

    if [ "$VERBOSE" = true ]; then
            cmd_args="$cmd_args --verbose"
    fi

    if [ -n "$API_KEY" ]; then
        cmd_args="$cmd_args --api_key $API_KEY"
    fi
    
    if [ -n "$API_MODEL_NAME" ]; then
        cmd_args="$cmd_args --api_model_name $API_MODEL_NAME"
    fi
    
    # Only rank 0 merges outputs
    # if [ $worker -eq 0 ]; then
    #     cmd_args="$cmd_args --merge_outputs"
    # fi
    
    # Launch worker with dedicated GPU allocation
    CUDA_VISIBLE_DEVICES=$worker_gpus PYTHONPATH=. python utils/RAGutils/queryResultByInfer/infer/multi_worker_inference.py $cmd_args &
    worker_pid=$!
    pids+=($worker_pid)
    
    echo "Worker $worker started with PID $worker_pid (CUDA_VISIBLE_DEVICES=$worker_gpus)"
    
    # Small delay between worker launches
    sleep 2
done

echo ""
echo "All workers started. PIDs: ${pids[*]}"
echo "Waiting for workers to complete..."

# Wait for all workers to complete
failed_workers=0
for i in "${!pids[@]}"; do
    pid=${pids[i]}
    echo "Waiting for worker $i (PID: $pid)..."
    
    if wait $pid; then
        echo "Worker $i completed successfully"
    else
        exit_code=$?
        echo "Worker $i failed with exit code $exit_code"
        failed_workers=$((failed_workers + 1))
    fi
done

echo ""
if [ $failed_workers -eq 0 ]; then
    echo "✅ All workers completed successfully!"
    echo "Results saved to: $OUTPUT_FILE"
    
    # Show output file statistics
    if [ -f "$OUTPUT_FILE" ]; then
        total_lines=$(wc -l < "$OUTPUT_FILE")
        echo "Total results: $total_lines lines"
        
        # Show success/error statistics if JSON format
        if command -v jq >/dev/null 2>&1; then
            success_count=$(grep '"success": *true' "$OUTPUT_FILE" 2>/dev/null | wc -l)
            error_count=$(grep '"success": *false\|"error":' "$OUTPUT_FILE" 2>/dev/null | wc -l)
            echo "Successful inferences: $success_count"
            echo "Failed inferences: $error_count"
        
            if [ $total_lines -gt 0 ]; then
                success_rate=$(echo "scale=1; $success_count * 100 / $total_lines" | bc -l 2>/dev/null || echo "N/A")
                echo "Success rate: ${success_rate}%"
        fi
        fi
    fi
    
    # Show log directory
    echo ""
    echo "Detailed logs available in: logs/multi_worker_inference/"
    echo "  worker_*.log - Full logs for each worker"
    echo "  worker_*_errors.log - Error-only logs for debugging"
        
    else
    echo "❌ $failed_workers worker(s) failed"
    echo "Check logs in logs/multi_worker_inference/ for details"
    exit 1
fi

echo ""
echo "Multi-worker inference completed." 