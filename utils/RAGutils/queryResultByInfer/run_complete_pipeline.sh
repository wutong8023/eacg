#!/bin/bash

# 完整的端到端流水线脚本
# 包含 Context Generation 和 Inference 两个阶段，支持多worker并行处理

# =============================================================================
# 配置参数
# =============================================================================

# 输入输出文件
QUERIES_FILE="data/generated_queries/versibcb_vscc_queries_with_target_dependency.json"
CORPUS_PATH="/datanfs4/chenrongyi/data/docs" # /datanfs2/chenrongyi/data/docs
CONTEXTS_OUTPUT="data/temp/contexts/versibcb_vscc_contexts.jsonl"
FINAL_OUTPUT="data/temp/inference/versibcb_vscc_results.jsonl"

# Context Generation 参数
CORPUS_TYPE="docstring"
EMBEDDING_SOURCE="local"
MAX_DOCUMENTS=1
MAX_TOKENS=2000
ENABLE_STR_MATCH=true
FIXED_DOCS_PER_QUERY=1
JUMP_EXACT_MATCH=false

# Inference 参数
MODEL_PATH="/datanfs4/chenrongyi/models/Llama-3.1-8B-Instruct"
INFERENCE_TYPE="local"
MAX_NEW_TOKENS=512
TEMPERATURE=1e-5
TOP_P=0.95

# 多worker配置
NUM_WORKERS=8
TOTAL_GPUS=8  # 总GPU数量
GPUS_PER_WORKER=$((TOTAL_GPUS / NUM_WORKERS))  # 每个worker分配的GPU数量
MAX_SAMPLES_PER_WORKER=1000  # 每个worker处理的最大样本数（用于测试）

# 生成GPU设备列表
GPU_LIST=""
for ((i=0; i<TOTAL_GPUS; i++)); do
    if [ $i -eq 0 ]; then
        GPU_LIST="$i"
    else
        GPU_LIST="$GPU_LIST,$i"
    fi
done

# 控制参数
VERBOSE=true
SKIP_CONTEXT_GENERATION=true  # 设置为true跳过context生成阶段
SKIP_INFERENCE=false           # 设置为true跳过推理阶段

# =============================================================================
# 函数定义
# =============================================================================

print_header() {
    echo
    echo "=============================================================================="
    echo "$1"
    echo "=============================================================================="
}

print_section() {
    echo
    echo "--- $1 ---"
}

check_file_exists() {
    if [ ! -f "$1" ]; then
        echo "❌ File not found: $1"
        return 1
    fi
    return 0
}

get_file_stats() {
    local file="$1"
    local description="$2"
    
    if [ -f "$file" ]; then
        local total_lines=$(wc -l < "$file")
        local success_lines=$(grep '"success": *true' "$file" 2>/dev/null | wc -l)
        local error_lines=$(grep '"success": *false' "$file" 2>/dev/null | wc -l)
        
        echo "$description:"
        echo "  Total lines: $total_lines"
        echo "  Successful: $success_lines"
        echo "  Errors: $error_lines"
        
        if [ "$total_lines" -gt 0 ] && [ "$success_lines" -gt 0 ]; then
            local success_rate=$(echo "scale=1; $success_lines * 100 / $total_lines" | bc -l 2>/dev/null || echo "N/A")
            echo "  Success rate: ${success_rate}%"
        fi
    else
        echo "$description: File not found"
    fi
}

# =============================================================================
# 主流程
# =============================================================================

print_header "Complete Pipeline: Context Generation + Inference"

echo "Configuration:"
echo "  Queries file: $QUERIES_FILE"
echo "  Corpus path: $CORPUS_PATH"
echo "  Contexts output: $CONTEXTS_OUTPUT"
echo "  Final output: $FINAL_OUTPUT"
echo "  Number of workers: $NUM_WORKERS"
echo "  Total GPUs: $TOTAL_GPUS"
echo "  GPUs per worker: $GPUS_PER_WORKER"
echo "  Max samples per worker: $MAX_SAMPLES_PER_WORKER"

# 显示GPU分配策略
echo "GPU allocation strategy:"
for ((worker=0; worker<NUM_WORKERS; worker++)); do
    start_gpu=$((worker * GPUS_PER_WORKER))
    end_gpu=$((start_gpu + GPUS_PER_WORKER - 1))
    worker_gpus=""
    for ((gpu=start_gpu; gpu<=end_gpu; gpu++)); do
        if [ "$worker_gpus" = "" ]; then
            worker_gpus="$gpu"
        else
            worker_gpus="$worker_gpus,$gpu"
        fi
    done
    echo "  Worker $worker: GPUs [$worker_gpus]"
done
echo

# 检查输入文件
if ! check_file_exists "$QUERIES_FILE"; then
    echo "Please ensure the queries file exists."
    exit 1
fi

if [ ! -d "$CORPUS_PATH" ]; then
    echo "❌ Corpus directory not found: $CORPUS_PATH"
    exit 1
fi

# 创建输出目录
mkdir -p $(dirname "$CONTEXTS_OUTPUT")
mkdir -p $(dirname "$FINAL_OUTPUT")

# =============================================================================
# 阶段 1: Context Generation
# =============================================================================

if [ "$SKIP_CONTEXT_GENERATION" = false ]; then
    print_header "PHASE 1: Context Generation"
    
    # 构建context generation命令
    CONTEXT_CMD="utils/RAGutils/queryResultByInfer/context_build/multi_worker_context_generator.py \
        --queries_file \"$QUERIES_FILE\" \
        --corpus_path \"$CORPUS_PATH\" \
        --output_file \"$CONTEXTS_OUTPUT\" \
        --corpus_type \"$CORPUS_TYPE\" \
        --embedding_source \"$EMBEDDING_SOURCE\" \
        --max_documents $MAX_DOCUMENTS \
        --max_tokens $MAX_TOKENS \
        --fixed_docs_per_query $FIXED_DOCS_PER_QUERY"
    
    # 添加可选参数
    if [ "$ENABLE_STR_MATCH" = true ]; then
        CONTEXT_CMD="$CONTEXT_CMD --enable_str_match"
    fi
    
    if [ "$JUMP_EXACT_MATCH" = true ]; then
        CONTEXT_CMD="$CONTEXT_CMD --jump_exact_match"
    fi
    
    if [ -n "$MAX_SAMPLES_PER_WORKER" ]; then
        CONTEXT_CMD="$CONTEXT_CMD --max_samples_per_worker $MAX_SAMPLES_PER_WORKER"
    fi
    
    if [ "$VERBOSE" = true ]; then
        CONTEXT_CMD="$CONTEXT_CMD --verbose"
    fi
    
    # 执行context generation
    if [ "$NUM_WORKERS" -eq 1 ]; then
        print_section "Running single worker context generation"
        
        export CUDA_VISIBLE_DEVICES="$GPU_LIST"
        
        CMD="python $CONTEXT_CMD"
        echo "Executing: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $CMD"
        eval $CMD
        
    else
        print_section "Running multi-worker context generation"
        
        export CUDA_VISIBLE_DEVICES="$GPU_LIST"
        
        CMD="torchrun --nproc_per_node=$NUM_WORKERS $CONTEXT_CMD"
        echo "Executing: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $CMD"
        eval $CMD
    fi
    
    # 检查context generation结果
    if [ $? -eq 0 ]; then
        print_section "Context Generation Results"
        get_file_stats "$CONTEXTS_OUTPUT" "Context generation"
        
        # 检查是否有成功的contexts
        SUCCESS_CONTEXTS=$(grep '"success": *true' "$CONTEXTS_OUTPUT" 2>/dev/null | wc -l)
        if [ "$SUCCESS_CONTEXTS" -eq 0 ]; then
            echo "❌ No successful contexts generated. Cannot proceed to inference."
            exit 1
        fi
        
        echo "✅ Context generation completed successfully"
    else
        echo "❌ Context generation failed"
        exit 1
    fi
    
else
    print_section "Skipping context generation (SKIP_CONTEXT_GENERATION=true)"
    
    # 检查contexts文件是否存在
    if ! check_file_exists "$CONTEXTS_OUTPUT"; then
        echo "❌ Contexts file not found and context generation is skipped"
        exit 1
    fi
    
    get_file_stats "$CONTEXTS_OUTPUT" "Existing contexts"
fi

# =============================================================================
# 阶段 2: Inference
# =============================================================================

if [ "$SKIP_INFERENCE" = false ]; then
    print_header "PHASE 2: Inference"
    
    # 构建inference命令
    INFERENCE_CMD="utils/RAGutils/queryResultByInfer/infer/multi_worker_inference.py \
        --contexts_file \"$CONTEXTS_OUTPUT\" \
        --output_file \"$FINAL_OUTPUT\" \
        --model_path \"$MODEL_PATH\" \
        --inference_type \"$INFERENCE_TYPE\" \
        --max_new_tokens $MAX_NEW_TOKENS \
        --temperature $TEMPERATURE \
        --top_p $TOP_P"
    
    # 添加可选参数
    if [ -n "$MAX_SAMPLES_PER_WORKER" ]; then
        INFERENCE_CMD="$INFERENCE_CMD --max_samples_per_worker $MAX_SAMPLES_PER_WORKER"
    fi
    
    if [ "$VERBOSE" = true ]; then
        INFERENCE_CMD="$INFERENCE_CMD --verbose"
    fi
    
    # 执行inference
    if [ "$NUM_WORKERS" -eq 1 ]; then
        print_section "Running single worker inference"
        print_section "Single worker will use all $TOTAL_GPUS GPUs"
        
        export CUDA_VISIBLE_DEVICES="$GPU_LIST"
        
        CMD="python $INFERENCE_CMD"
        echo "Executing: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $CMD"
        eval $CMD
        
    else
        print_section "Running multi-worker inference with GPU allocation"
        print_section "Each worker will get $GPUS_PER_WORKER GPU(s)"
        
        # 检查GPU分配是否合理
        if [ $GPUS_PER_WORKER -eq 0 ]; then
            echo "❌ Not enough GPUs for the requested number of workers"
            echo "   Available GPUs: $TOTAL_GPUS, Requested workers: $NUM_WORKERS"
            echo "   Try reducing NUM_WORKERS or increasing TOTAL_GPUS"
            exit 1
        fi
        
        export CUDA_VISIBLE_DEVICES="$GPU_LIST"
        
        CMD="torchrun --nproc_per_node=$NUM_WORKERS $INFERENCE_CMD"
        echo "Executing: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $CMD"
        eval $CMD
    fi
    
    # 检查inference结果
    if [ $? -eq 0 ]; then
        print_section "Inference Results"
        get_file_stats "$FINAL_OUTPUT" "Final inference"
        
        echo "✅ Inference completed successfully"
    else
        echo "❌ Inference failed"
        exit 1
    fi
    
else
    print_section "Skipping inference (SKIP_INFERENCE=true)"
fi

# =============================================================================
# 最终总结
# =============================================================================

print_header "PIPELINE COMPLETE"

echo "Final Results Summary:"
echo

if [ -f "$CONTEXTS_OUTPUT" ]; then
    get_file_stats "$CONTEXTS_OUTPUT" "Context Generation"
fi

echo

if [ -f "$FINAL_OUTPUT" ]; then
    get_file_stats "$FINAL_OUTPUT" "Final Inference"
    
    print_section "Sample Output Preview"
    echo "First result:"
    head -n 1 "$FINAL_OUTPUT" | jq '.' 2>/dev/null || head -n 1 "$FINAL_OUTPUT"
    
    # 推理时间统计
    print_section "Performance Statistics"
    if command -v jq >/dev/null 2>&1; then
        echo "Inference time statistics:"
        jq -r '.inference_time // empty' "$FINAL_OUTPUT" 2>/dev/null | head -n 100 | \
        awk '{sum+=$1; count++} END {if(count>0) printf "Average inference time: %.2f seconds\n", sum/count}'
        
        echo "Retrieval time statistics:"
        jq -r '.retrieval_time // empty' "$FINAL_OUTPUT" 2>/dev/null | head -n 100 | \
        awk '{sum+=$1; count++} END {if(count>0) printf "Average retrieval time: %.2f seconds\n", sum/count}'
    fi
fi

echo
echo "Output files:"
echo "  Contexts: $CONTEXTS_OUTPUT"
echo "  Final results: $FINAL_OUTPUT"

print_header "ALL DONE!" 