#!/bin/bash

# 设置默认参数
QUERIES_FILE="data/generated_queries/versibcb_vace_queries_deduplicated.json"
CORPUS_PATH="/datanfs2/chenrongyi/data/docs"
OUTPUT_FILE="data/temp/contexts/contexts.jsonl"
CORPUS_TYPE="docstring"
EMBEDDING_SOURCE="local"
MAX_DOCUMENTS=10
MAX_TOKENS=4000
ENABLE_STR_MATCH=true
FIXED_DOCS_PER_QUERY=1
MAX_SAMPLES=""
VERBOSE=true
JUMP_EXACT_MATCH=true
# 创建输出目录
mkdir -p $(dirname "$OUTPUT_FILE")

# 构建命令
CMD="python utils/RAGutils/queryResultByInfer/batch_context_generator.py \
    --queries_file \"$QUERIES_FILE\" \
    --corpus_path \"$CORPUS_PATH\" \
    --output_file \"$OUTPUT_FILE\" \
    --corpus_type \"$CORPUS_TYPE\" \
    --embedding_source \"$EMBEDDING_SOURCE\" \
    --max_documents $MAX_DOCUMENTS \
    --max_tokens $MAX_TOKENS \
    --fixed_docs_per_query $FIXED_DOCS_PER_QUERY \
    "
# 添加可选参数
if [ "$ENABLE_STR_MATCH" = true ]; then
    CMD="$CMD --enable_str_match"
fi



if [ ! -z "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

if [ "$JUMP_EXACT_MATCH" = true ]; then
    CMD="$CMD --jump_exact_match"
fi

# 打印将要执行的命令
echo "Executing command:"
echo "$CMD"
echo

# 执行命令
eval $CMD 