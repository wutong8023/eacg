#!/bin/bash

# Multi-Worker Context Generation 运行脚本
# 支持指定GPU和worker数量
PYTHONSCRIPT="utils/RAGutils/queryResultByInfer/context_build/multi_worker_context_generator.py"
# 默认参数
QUERIES_FILE="data/temp/versibcb_format_queries.json"
CORPUS_PATH="/datanfs4/chenrongyi/data/docs"
OUTPUT_FILE="data/temp/contexts/versibcb_vscc_contexts_staticAnalysis.jsonl"
CORPUS_TYPE="docstring"
EMBEDDING_SOURCE="local"
MAX_DOCUMENTS=1
MAX_TOKENS=2000
ENABLE_STR_MATCH=true
FIXED_DOCS_PER_QUERY=1
JUMP_EXACT_MATCH=false
MAX_SAMPLES_PER_WORKER=""
VERBOSE=true

# 多worker参数
NUM_WORKERS=8
CUDA_DEVICES="0,1,2,3,4,5,6,7"  # 指定使用的GPU

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --cuda_devices)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --max_samples_per_worker)
            MAX_SAMPLES_PER_WORKER="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# 创建输出目录
mkdir -p $(dirname "$OUTPUT_FILE")

echo "=== Multi-Worker Context Generation ==="
echo "Number of workers: $NUM_WORKERS"
echo "CUDA devices: $CUDA_DEVICES"
echo "Output file: $OUTPUT_FILE"
if [ ! -z "$MAX_SAMPLES_PER_WORKER" ]; then
    echo "Max samples per worker: $MAX_SAMPLES_PER_WORKER"
fi
echo

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

# 构建命令
CMD="torchrun --nproc_per_node=$NUM_WORKERS \
    $PYTHONSCRIPT \
    --queries_file \"$QUERIES_FILE\" \
    --corpus_path \"$CORPUS_PATH\" \
    --output_file \"$OUTPUT_FILE\" \
    --corpus_type \"$CORPUS_TYPE\" \
    --embedding_source \"$EMBEDDING_SOURCE\" \
    --max_documents $MAX_DOCUMENTS \
    --max_tokens $MAX_TOKENS \
    --fixed_docs_per_query $FIXED_DOCS_PER_QUERY"

# 添加可选参数
if [ "$ENABLE_STR_MATCH" = true ]; then
    CMD="$CMD --enable_str_match"
fi

if [ "$JUMP_EXACT_MATCH" = true ]; then
    CMD="$CMD --jump_exact_match"
fi

if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

if [ ! -z "$MAX_SAMPLES_PER_WORKER" ]; then
    CMD="$CMD --max_samples_per_worker $MAX_SAMPLES_PER_WORKER"
fi

# 打印将要执行的命令
echo "Executing command:"
echo "$CMD"
echo

# 执行命令
eval $CMD

# 检查结果
if [ $? -eq 0 ]; then
    echo
    echo "=== Multi-Worker Results ==="
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Output file created successfully: $OUTPUT_FILE"
        echo "Number of lines in output:"
        wc -l "$OUTPUT_FILE"
        echo
        echo "First result preview:"
        head -n 1 "$OUTPUT_FILE" | jq '.' 2>/dev/null || head -n 1 "$OUTPUT_FILE"
        echo
        echo "Sample statistics:"
        echo "Unique dedup_keys:"
        grep -o '"dedup_key":"[^"]*"' "$OUTPUT_FILE" | sort | uniq | wc -l
        echo "Total contexts generated:"
        grep '"success":true' "$OUTPUT_FILE" | wc -l
    else
        echo "❌ Output file not created"
    fi
else
    echo "❌ Command failed with exit code $?"
fi

echo
echo "Usage examples:"
echo "  # Use 2 workers on GPUs 0,1"
echo "  $0 --num_workers 2 --cuda_devices \"0,1\""
echo
echo "  # Limit samples per worker for testing"
echo "  $0 --num_workers 4 --max_samples_per_worker 100" 