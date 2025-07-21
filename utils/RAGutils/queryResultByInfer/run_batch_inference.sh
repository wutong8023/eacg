#!/bin/bash

# Batch Query Inference 运行脚本
# 包含 context generation 和 inference 的完整流程

# 设置参数
QUERIES_FILE="data/generated_queries/versibcb_vace_queries_deduplicated.json"
CORPUS_PATH="/datanfs2/chenrongyi/data/docs"
OUTPUT_FILE="data/temp/inference/batch_inference_results.jsonl"
CORPUS_TYPE="docstring"
EMBEDDING_SOURCE="local"
MAX_DOCUMENTS=1
MAX_TOKENS=2000

MAX_SAMPLES=10000  # 测试用少量样本
VERBOSE=true
# 获取context参数
ENABLE_STR_MATCH=true
FIXED_DOCS_PER_QUERY=1
JUMP_EXACT_MATCH=false
# 推理参数
MODEL_PATH="/datanfs2/chenrongyi/models/Llama-3.1-8B"
INFERENCE_TYPE="local"
MAX_NEW_TOKENS=512
TEMPERATURE=1e-5
TOP_P=0.95

# 创建输出目录
mkdir -p $(dirname "$OUTPUT_FILE")

echo "=== Batch Query Inference ==="
echo "Processing $MAX_SAMPLES samples for testing..."
echo "Output file: $OUTPUT_FILE"
echo

# 构建命令
CMD="python utils/RAGutils/queryResultByInfer/batch_query_inference.py \
    --queries_file \"$QUERIES_FILE\" \
    --corpus_path \"$CORPUS_PATH\" \
    --output_file \"$OUTPUT_FILE\" \
    --corpus_type \"$CORPUS_TYPE\" \
    --embedding_source \"$EMBEDDING_SOURCE\" \
    --max_documents $MAX_DOCUMENTS \
    --max_tokens $MAX_TOKENS \
    --fixed_docs_per_query $FIXED_DOCS_PER_QUERY \
    --model_path \"$MODEL_PATH\" \
    --inference_type \"$INFERENCE_TYPE\" \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --max_samples $MAX_SAMPLES"

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

# 打印将要执行的命令
echo "Executing command:"
echo "$CMD"
echo

# 执行命令
eval $CMD

# 检查结果
if [ $? -eq 0 ]; then
    echo
    echo "=== Inference Results ==="
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Output file created successfully: $OUTPUT_FILE"
        echo "Number of lines in output:"
        wc -l "$OUTPUT_FILE"
        echo
        echo "First result preview:"
        head -n 1 "$OUTPUT_FILE" | jq '.' 2>/dev/null || head -n 1 "$OUTPUT_FILE"
    else
        echo "❌ Output file not created"
    fi
else
    echo "❌ Command failed with exit code $?"
fi 