#!/bin/bash

# 多worker查询生成脚本
# 基于run_rag_multiworker.sh的逻辑适配

SHORT=w:,n:,h
LONG=world_size:,num_gpus_per_job:,help

PARSED=$(getopt --options $SHORT --longoptions $LONG --name "$0" -- "$@") || { echo "Invalid Arguments."; exit 2; }
eval set -- "$PARSED"

# 默认参数
num_gpus_per_job=1
PY_SCRIPT="utils/RAGutils/genQueries/generate_queries.py"

# ============== 配置参数区域 ==============
# 用户可以根据需要修改以下参数


# 基础参数
DATASET="VersiBCB"
TASK="VSCC"  # 可选: VSCC, VACE 等
MODEL="/datanfs4/chenrongyi/models/Llama-3.1-8B-Instruct"
INFERENCE_TYPE="local"  # 可选: local, togetherai, huggingface
PRECISION="fp16"  # 可选: fp16, fp32, bf16
TARGET_TASK_IDS_FILE="data/temp/taskids_filter.json"
# 生成参数
MAX_NEW_TOKENS=512
TEMPERATURE=0.7
TOP_P=0.9

# 错误信息参数
GENERATED_ERRORS_PATH="data/temp/combined_errors_vscc.json"
GENERATED_CODE_PATH="output/approach_eval/BASELINE/Llama-3.1-8B-Instruct/versibcb_vscc_Llama-3.1-8B-Instruct_maxdep10.jsonl"
ENABLE_GENERATED_CODE=true
# 输出参数
OUTPUT_DIR="data/generated_queries"
PROMPT_VERSION="review"
# OUTPUT_FILENAME="VersiBCB_VSCC_RAG_complete_v0.json"
# API参数（当inference_type不是local时需要）
# API_KEY="your_api_key_here"
# API_MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

# ============================================

# 获取GPU使用情况并返回空闲GPU的ID
get_free_gpus() {
    # 获取所有GPU的使用情况，内存使用率低于10%认为是空闲的
    free_gpus=($(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | \
                awk -F, '{if ($2/$3 < 0.1) print $1}'))
    echo "${free_gpus[@]}"
}

# 获取GPU数量
nvidia-smi
available_gpus=($(get_free_gpus))
if [ ${#available_gpus[@]} -eq 0 ]; then
    echo "Error: No free GPUs available"
    exit 1
fi

world_size=${#available_gpus[@]}
echo "Found ${world_size} free GPUs: ${available_gpus[*]}"

while true; do
    case "$1" in
        -h|--help) 
            echo "Usage: $0 [-w <world_size>] [-n <num_gpus_per_job>] [--help]"
            echo "  -w, --world_size: number of GPUs to use (default: all available)"
            echo "  -n, --num_gpus_per_job: GPUs per job (default: 1)"
            echo ""
            echo "Generate queries using multiple workers for parallel processing"
            exit ;;
        -w|--world_size) world_size="$2"; shift 2 ;;
        -n|--num_gpus_per_job) num_gpus_per_job="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "Programming error"; exit 3 ;;
    esac
done

worker_num=$((world_size/num_gpus_per_job))

# 验证参数
if [ $(( world_size )) -gt ${#available_gpus[@]} ]; then
    echo "Error: Requested world_size ($world_size) exceeds available GPUs (${#available_gpus[@]})"
    exit 1
fi

if [ $(( world_size % num_gpus_per_job )) -ne 0 ]; then
    echo "Error: world_size ($world_size) must be divisible by num_gpus_per_job ($num_gpus_per_job)"
    exit 1
fi

echo "World size: $world_size"
echo "GPUs per job: $num_gpus_per_job"
echo "Total workers: $worker_num"

echo "Starting multi-worker query generation..."

trap 'kill $(jobs -p)' SIGINT

# 启动多个worker
if [ $(( world_size % num_gpus_per_job )) -eq 0 ]; then
    
    for i in $(seq 0 $((num_gpus_per_job)) $((world_size-1))); do
        rank=$((i/num_gpus_per_job))
        
        # 构建GPU设备字符串
        gpu_devices=""
        for j in $(seq 0 $((num_gpus_per_job-1))); do
            gpu_idx=$((i+j))
            if [ -n "$gpu_devices" ]; then
                gpu_devices="${gpu_devices},${available_gpus[gpu_idx]}"
            else
                gpu_devices="${available_gpus[gpu_idx]}"
            fi
        done
        
        echo "Starting rank ${rank} with GPUs ${gpu_devices}"

        # 构建参数
        args=""
        args="$args --dataset $DATASET"
        args="$args --task $TASK"
        args="$args --model $MODEL"
        args="$args --inference_type $INFERENCE_TYPE"
        args="$args --precision $PRECISION"
        args="$args --max_new_tokens $MAX_NEW_TOKENS"
        args="$args --temperature $TEMPERATURE"
        args="$args --top_p $TOP_P"
        args="$args --output_dir $OUTPUT_DIR"
        args="$args -pv $PROMPT_VERSION"
        if [ "$TARGET_TASK_IDS_FILE" ]; then
            args="$args --target_task_ids_file $TARGET_TASK_IDS_FILE"
        fi
        if [ "$ENABLE_GENERATED_CODE" = true ]; then
            args="$args --enable_generated_code"
        fi
        if [ "$GENERATED_CODE_PATH" ]; then
            args="$args --generated_code_path $GENERATED_CODE_PATH"
        fi
        if [ -n "$GENERATED_ERRORS_PATH" ]; then
            args="$args --generated_errors_path $GENERATED_ERRORS_PATH"
        fi
        if [ -n "$OUTPUT_FILENAME" ]; then
            args="$args --output_filename $OUTPUT_FILENAME"
        fi
        # 添加API参数（如果需要）
        if [ -n "$API_KEY" ]; then
            args="$args --api_key $API_KEY"
        fi
        if [ -n "$API_MODEL_NAME" ]; then
            args="$args --api_model_name $API_MODEL_NAME"
        fi
        
        # 添加多worker参数
        args="$args --rank $rank"
        args="$args --world_size $worker_num"
        
        # 启动worker
        CUDA_VISIBLE_DEVICES=$gpu_devices python "$PY_SCRIPT" $args &
    done
    
    # 等待所有worker完成
    wait
    echo "Query generation completed successfully!"
    echo "Results saved to: ${OUTPUT_DIR}"
    
else
    echo "Error: world_size ${world_size} is not divisible by ${num_gpus_per_job}."
    exit 1
fi