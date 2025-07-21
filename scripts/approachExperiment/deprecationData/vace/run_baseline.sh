#!/bin/bash

# 多worker RAG预测脚本
# 完全按照run_lora_multiworker.sh的逻辑，适配RAG任务

SHORT=w:,n:,h
LONG=world_size:,num_gpus_per_job:,help

PARSED=$(getopt --options $SHORT --longoptions $LONG --name "$0" -- "$@") || { echo "Invalid Arguments."; exit 2; }
eval set -- "$PARSED"

# 默认参数
num_gpus_per_job=1
PYTHON_SCRIPT="benchmark/pred_rag.py"
# 是否追加context到输出,以及对应的自定义输出路径
APPENDCONTEXT=true
# OUTPUT_PATH="output/testge/RAG/testRAG.jsonl"
# RAG预测参数
DATASET="VersiBCB"
APPROACH="BASELINE" # RAG or BASELINE
MODEL="/datanfs4/chenrongyi/models/Llama-3.1-8B-Instruct" # codegemma-7b-it/Llama-3.1-8B
KNOWLEDGE_TYPE="docstring"
PRECISION="fp16"
LOCAL_EMBEDDING_MODEL="all-MiniLM-L6-v2"
TOGETHERAI_EMBEDDING_MODEL="togethercomputer/m2-bert-80M-32k-retrieval"
RAG_COLLECTION_BASE="/datanfs4/chenrongyi/data/RAG/chroma_data"
DOCSTRING_EMBEDDING_BASE_PATH="/datanfs4/chenrongyi/data/RAG/docs_embeddings"
SRCCODE_EMBEDDING_BASE_PATH="/datanfs4/chenrongyi/data/RAG/srccodes_embeddings"
# SPECIFIED_BENCH_PATH="data/VersiBCB_Benchmark/vace_datas_for_general.json"
INFERENCE_TYPE="local"
API_KEY="asd"
API_MODEL_NAME="meta-llama/Meta-Llama-3.1-8B" # meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo mistralai/Mistral-7B-Instruct-v0.2
NUM_WORKERS=1
TEMPERATURE=1e-5
TOP_P=1.0
BASE_OUTPUT_DIR="output/approach_eval"
MAX_TOKENS=2048
# 任务列表
TASKS=("VACE")
DEPRECATION_FLAGS=(true)
MAX_DEPENDENCY_NUMS=(10)
SKIP_GENERATION=false
GENERATED_QUERIES_FILE="data/generated_queries/withTargetCode/versibcb_vace_queries_with_code_v1.json"
USE_GENERATED_QUERIES=true
# 每个query获取doc数量
FIXED_DOCS_PER_QUERY=1
# 对于collection的filter
# ENABLE_COLLECTION_FILTERING=true
# 对于查询层面的filter
ENABLE_QUERY_FILTERING=true
QUERY_FILTER_STRICT_MODE=true # 目前默认，只需Enable_query_filter即可

# 知识压缩参数
ENABLE_KNOWLEDGE_COMPRESSION=false  # 默认关闭，需要时启用
# 对于doc的压缩
MAX_DOC_TOKENS=200
DOC_TRUNCATE_TOKENIZER="/datanfs2/chenrongyi/models/Llama-3.1-8B"
# 对于API名称字符串匹配
ENABLE_API_NAME_STR_MATCH=false
# QA缓存上下文
QACacheContext_path="data/temp/inference/modified_pipeline_results_fixed_improved.jsonl"
# target_task_ids_file="scripts/target_task_id.json"

get_free_gpus() {
    # 获取所有GPU的使用情况，内存使用率低于1%认为是空闲的
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

echo "Starting RAG multi-worker prediction..."

trap 'kill $(jobs -p)' SIGINT

# 启动所有任务组合的worker
for task in "${TASKS[@]}"; do
    for ban_dep in "${DEPRECATION_FLAGS[@]}"; do
        for max_dependency_num in "${MAX_DEPENDENCY_NUMS[@]}"; do
            echo "Starting workers for task: $task, ban_deprecation: $ban_dep, max_dependency_num: $max_dependency_num"
            
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
                    
                    echo "Starting rank ${rank} with GPUs ${gpu_devices} for task ${task}, ban_dep ${ban_dep}, max_dep ${max_dependency_num}"

                    # 构建参数
                    args=""
                    # 添加对应的参数
                    if $SKIP_GENERATION; then
                        args="$args --skip_generation"
                    fi
                    if $USE_GENERATED_QUERIES; then
                        args="$args --use_generated_queries"
                    fi
                    if [ -n "$target_task_ids_file" ]; then
                        args="$args --target_task_ids_file $target_task_ids_file"
                    fi
                    # doc截断参数
                    args="$args --max_doc_tokens $MAX_DOC_TOKENS"
                    args="$args --doc_truncate_tokenizer $DOC_TRUNCATE_TOKENIZER"
                    # 添加其他参数
                    args="$args --dataset $DATASET"
                    args="$args --task $task"
                    args="$args --Ban_Deprecation $ban_dep"
                    args="$args --approach $APPROACH"
                    args="$args --model $MODEL"
                    args="$args --knowledge_type $KNOWLEDGE_TYPE"
                    args="$args --precision $PRECISION"
                    args="$args --local_embedding_model $LOCAL_EMBEDDING_MODEL"
                    args="$args --togetherai_embedding_model $TOGETHERAI_EMBEDDING_MODEL"
                    args="$args --rag_collection_base $RAG_COLLECTION_BASE"
                    args="$args --docstring_embedding_base_path $DOCSTRING_EMBEDDING_BASE_PATH"
                    args="$args --srccode_embedding_base_path $SRCCODE_EMBEDDING_BASE_PATH"
                    args="$args --inference_type $INFERENCE_TYPE"
                    args="$args --api_key $API_KEY"
                    args="$args --api_model_name $API_MODEL_NAME"
                    args="$args --num_workers $NUM_WORKERS"
                    args="$args --temperature $TEMPERATURE"
                    args="$args --top_p $TOP_P"
                    args="$args --max_dependency_num $max_dependency_num"
                    args="$args --base_output_dir $BASE_OUTPUT_DIR"
                    args="$args --max_tokens $MAX_TOKENS"
                    args="$args --generated_queries_file $GENERATED_QUERIES_FILE"
                    if [ -n "$SPECIFIED_BENCH_PATH" ]; then
                        args="$args --specified_bench_path $SPECIFIED_BENCH_PATH"
                    fi
                    if [ "$REVIEW_ON_OUTPUT" = "true" ]; then
                        args="$args --review_on_output"
                        args="$args --generated_target_code_path $GENERATED_TARGET_CODE_PATH"
                    fi
                    # 添加多worker参数
                    args="$args --rank $rank"
                    args="$args --world_size $worker_num"
                    # collection filter参数
                    if [ -n "$FIXED_DOCS_PER_QUERY" ]; then
                        args="$args --fixed_docs_per_query $FIXED_DOCS_PER_QUERY"
                    fi
                    if [ "$ENABLE_QUERY_FILTERING" = "true" ]; then
                        args="$args --enable_query_dependency_filtering"
                        args="$args --query_filter_strict_mode"
                    fi
                    # 知识压缩参数
                    if [ "$ENABLE_KNOWLEDGE_COMPRESSION" = "true" ]; then
                        args="$args --enable_knowledge_compression"
                    fi
                    # API名称字符串匹配参数
                    if [ "$ENABLE_API_NAME_STR_MATCH" = "true" ]; then
                        args="$args --api_name_str_match"
                    fi
                    # error info参数
                    if [ -n "$ERROR_INFO_FILEPATH" ]; then
                        args="$args --errorinfos_filepath $ERROR_INFO_FILEPATH"
                    fi
                    # 自定义输出路径
                    if [ -n "$OUTPUT_PATH" ]; then
                        args="$args --selfdefined_OutputPath $OUTPUT_PATH"
                    fi
                    # retrieval参数
                    if [ -n "$RETRIEVED_INFO_FILEPATH" ]; then
                        args="$args --retrieved_info_filepath $RETRIEVED_INFO_FILEPATH"
                    fi
                    # QA缓存上下文
                    if [ -n "$QACacheContext_path" ]; then
                        args="$args --QACacheContext_path $QACacheContext_path"
                    fi
                    if $APPENDCONTEXT; then
                        args="$args --appendContext"
                    fi
                    # 启动worker
                    CUDA_VISIBLE_DEVICES=$gpu_devices python "$PYTHON_SCRIPT" $args &
                done
                
                # 等待当前任务组合的所有worker完成
                wait
                echo "Completed task: $task, ban_deprecation: $ban_dep, max_dependency_num: $max_dependency_num"
            else
                echo "Error: world_size ${world_size} is not divisible by ${num_gpus_per_job}."
                exit 1
            fi
        done
    done
done

echo "All RAG tasks completed successfully!"
echo "Results saved to: ${BASE_OUTPUT_DIR}/${APPROACH}/" 