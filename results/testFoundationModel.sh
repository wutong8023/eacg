#!/bin/bash

# 设置模型名称和基本路径
# MODEL_PATH="/datanfs2/chenrongyi/models/codegemma-7b-it"
# MODEL_NAME=$(basename "$MODEL_PATH")

MODEL_PATH='deepseek-chat' #模型位置或名称(如果调用本地模型的话就是位置)，用于指定推理模型
MODEL_NAME=$(basename "$MODEL_PATH")

JSON_PATH_VSCC="data/benchmark/vscc_datas.json" #vscc数据的位置
JSON_PATH_VACE="data/benchmark/vace_datas.json" #vace数据的位置
JSON_PATH_VSCC_BD="data/benchmark/vscc_datas_for_warning.json" #vscc_ban_deprecation数据的位置
JSON_PATH_VACE_BD="data/benchmark/vace_datas_for_warning.json" #vace_ban_deprecation数据的位置
API_TYPE="deepseek" #推理API类型
MAX_TOKENS=7000 #最大token数
DEVICE="auto"  # 改为auto自动选择空闲GPU
NUM_GPUS=2     # 默认使用1个GPU

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --num_gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    *)
      # 未知参数
      shift
      ;;
  esac
done

echo "Using device: $DEVICE"
echo "Number of GPUs to use: $NUM_GPUS"

# 确保输出目录存在
mkdir -p data/model_predictions/${MODEL_NAME}



# # 4. VACE, 禁用废弃函数
echo "运行测试: VACE, ban-deprecation"
python utils/model_predict/testFoundationModel.py \
    --task_type vace \
    --model_name_or_path ${MODEL_PATH} \
    --json_path ${JSON_PATH_VACE_BD} \
    --api_type ${API_TYPE} \
    --ban_deprecation True \
    --max_tokens ${MAX_TOKENS} \
    --device ${DEVICE} \
    --num_gpus ${NUM_GPUS}

echo "所有测试完成!"

# echo "开始本地${MODEL_NAME}模型测试..."

# 1. VSCC, 不禁用废弃函数
# echo "运行测试: VSCC, no-ban-deprecation"
# python utils/model_predict/testFoundationModel.py \
#     --task_type vscc \
#     --model_name_or_path ${MODEL_PATH} \
#     --json_path ${JSON_PATH_VSCC} \
#     --api_type ${API_TYPE} \
#     --ban_deprecation False \
#     --max_tokens ${MAX_TOKENS} \
#     --device ${DEVICE} \
#     --num_gpus ${NUM_GPUS}



# # 3. VACE, 不禁用废弃函数
# echo "运行测试: VACE, no-ban-deprecation"
# python utils/model_predict/testFoundationModel.py \
#     --task_type vace \
#     --model_name_or_path ${MODEL_PATH} \
#     --json_path ${JSON_PATH_VACE} \
#     --api_type ${API_TYPE} \
#     --ban_deprecation False \
#     --max_tokens ${MAX_TOKENS} \
#     --device ${DEVICE} \
#     --num_gpus ${NUM_GPUS}

# # 2. VSCC, 禁用废弃函数
# echo "运行测试: VSCC, ban-deprecation"
# python utils/model_predict/testFoundationModel.py \
#     --task_type vscc \
#     --model_name_or_path ${MODEL_PATH} \
#     --json_path ${JSON_PATH_VSCC_BD} \
#     --api_type ${API_TYPE} \
#     --ban_deprecation True \
#     --max_tokens ${MAX_TOKENS} \
#     --device ${DEVICE} \
#     --num_gpus ${NUM_GPUS}