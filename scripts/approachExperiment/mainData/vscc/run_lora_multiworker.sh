#!/bin/bash

# 简化的多worker LoRA预测脚本
# 完全按照run_emllm.sh的逻辑

SHORT=w:,n:,h
LONG=world_size:,num_gpus_per_job:,help

PARSED=$(getopt --options $SHORT --longoptions $LONG --name "$0" -- "$@") || { echo "Invalid Arguments."; exit 2; }
eval set -- "$PARSED"

# 默认参数
num_gpus_per_job=1
PYTHON_SCRIPT="benchmark/pred_lora.py"

# 其他默认参数
MAX_DEPENDENCY_NUM=9
TEMPERATURE=1e-5
TOP_P=1.0
PRECISION="fp16"
MODEL_NAME="/datanfs4/chenrongyi/models/Llama-3.1-8B-Instruct"
KNOWLEDGE_TYPE="srccodes"
ADAPTOR_BASE_PATH="/datanfs4/chenrongyi/models/loraadaptors/"
MERGED_ADAPTOR_PATH="/datanfs4/chenrongyi/models/loraadaptors/merged_lora_model"
BASE_OUTPUT_DIR="output/approach_eval/"
FORCE_GPU=true  # 默认不强制使用GPU
ADOPT_IFT_CHECKPOINT=false
# SPECIFIED_BENCH_PATH="data/VersiBCB_Benchmark/vace_datas_for_general.json"
# task2maskpacks参数
ENABLE_TASK2MASKPACKS=false
TASK2MASKPACKS_FILE="data/task2maskpacks.json"
ONLY_TASK2MASKPACKS_IDS=false

# filter 参数
ENABLE_TASK_ID_FILTER=false
TASK_ID_FILTER_FILE="data/temp/taskids_filter.json"
# IFT_ENABLED_PACKAGES=""
# IFT_CARDS="/datanfs2/chenrongyi/codes/EM-LLM-model/configs/ift_cards/ift_meta_same_minor_version_distributed_asus-2024_4e42742e.yaml"
# 任务列表
TASKS=("vscc") # vace vscc
DEPRECATION_FLAGS=(false) # true false
# 获取GPU使用情况并返回空闲GPU的ID
get_free_gpus() {
    # 获取所有GPU的使用情况，内存使用率低于1%认为是空闲的
    free_gpus=($(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | \
                awk -F, '{if ($2/$3 < 0.30) print $1}'))
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


echo "Starting LoRA multi-worker prediction..."

trap 'kill $(jobs -p)' SIGINT

# 启动所有任务组合的worker
for task in "${TASKS[@]}"; do
    for ban_dep in "${DEPRECATION_FLAGS[@]}"; do
        echo "Starting workers for task: $task, ban_deprecation: $ban_dep"
        
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
                
                echo "Starting rank ${rank} with GPUs ${gpu_devices} for task ${task}"

                # 构建参数
                args=""
                args="$args --max_dependency_num $MAX_DEPENDENCY_NUM"
                args="$args --temperature $TEMPERATURE"
                args="$args --top_p $TOP_P"
                args="$args --precision $PRECISION"
                args="$args --model_name $MODEL_NAME"
                args="$args --knowledge_type $KNOWLEDGE_TYPE"
                args="$args --loraadaptor_save_path_base $ADAPTOR_BASE_PATH"
                args="$args --base_output_dir $BASE_OUTPUT_DIR"
                args="$args --rank $rank"
                args="$args --world_size $worker_num"
                args="$args --task $task"
                args="$args --tempmerged_adaptor_path $MERGED_ADAPTOR_PATH"
                if [ -n "$SPECIFIED_BENCH_PATH" ]; then
                    args="$args --specified_bench_path $SPECIFIED_BENCH_PATH"
                fi
                if [ "$ENABLE_TASK2MASKPACKS" = true ]; then
                    args="$args --enable_task2maskpacks"

                fi
                args="$args --task2maskpacks_file $TASK2MASKPACKS_FILE"
                if [ "$ONLY_TASK2MASKPACKS_IDS" = true ]; then
                    args="$args --only_task2maskpacks_ids"
                fi
                if [ "$ENABLE_TASK_ID_FILTER" = true ]; then
                    args="$args --enable_task_id_filter"
                    args="$args --task_id_filter_file $TASK_ID_FILTER_FILE"
                fi
                # args="$args --ift_enabled_packages $IFT_ENABLED_PACKAGES"
                # args="$args --ift_card $IFT_CARDS"
                if [ "$ban_dep" = true ]; then
                    args="$args --Ban_Deprecation"
                fi
                
                if [ "$FORCE_GPU" = true ]; then
                    args="$args --force_gpu"
                fi
                if [ "$ADOPT_IFT_CHECKPOINT" = true ]; then
                    args="$args --adopt_IFT_checkpoint"
                fi
                CUDA_VISIBLE_DEVICES=$gpu_devices python "$PYTHON_SCRIPT" $args &
            done
            
            # 等待当前任务组合的所有worker完成
            wait
            echo "Completed task: $task, ban_deprecation: $ban_dep"
        else
            echo "Error: world_size ${world_size} is not divisible by ${num_gpus_per_job}."
            exit 1
        fi
    done
done

echo "All tasks completed successfully!"
echo "Results saved to: $BASE_OUTPUT_DIR" 