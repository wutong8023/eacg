#!/bin/bash

# 设置默认值
NUM_GPUS=1
CONFIG_PATH="benchmark/config/code/config_lora.json"
PYTHON_SCRIPT="benchmark/train_lora.py"
ENABLE_MEMORY_DEBUG=false
MEMORY_DEBUG_LOG_DIR="logs/memory_debug_custom"
# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --script)
      PYTHON_SCRIPT="$2"
      shift 2
      ;;
    --enable_memory_debug)
      ENABLE_MEMORY_DEBUG=True
      shift 1
      ;;
    --memory_debug_log_dir)
      MEMORY_DEBUG_LOG_DIR="$2"
      shift 2
      ;;
    --help)
      echo "用法: $0 [选项]"
      echo "选项:"
      echo "  --gpus N       指定需要的GPU数量 (默认: 1)"
      echo "  --config PATH  指定配置文件路径 (默认: config.json)"
      echo "  --script PATH  指定Python脚本路径 (默认: tests/loraTrain/peftMoElora.py)"
      echo "  --help         显示此帮助信息"
      exit 0
      ;;
    *)
      echo "未知参数: $1"
      echo "使用 --help 查看帮助"
      exit 1
      ;;
  esac
done

# 检查是否为数字
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]]; then
  echo "错误: GPU数量必须是一个正整数"
  exit 1
fi

# 检查文件是否存在
if [ ! -f "$CONFIG_PATH" ]; then
  echo "错误: 配置文件 '$CONFIG_PATH' 不存在"
  exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
  echo "错误: Python脚本 '$PYTHON_SCRIPT' 不存在"
  exit 1
fi

# 获取可用GPU信息
echo "正在检查GPU状态..."
GPU_INFO=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)

if [ $? -ne 0 ]; then
  echo "错误: 无法获取GPU信息，请确保nvidia-smi可用"
  exit 1
fi

# 解析GPU信息，找出空闲GPU
declare -a FREE_GPUS
declare -a GPU_USAGE
declare -a GPU_MEM_USED

while IFS=, read -r idx mem_used mem_total util; do
  idx=$(echo $idx | tr -d ' ')
  mem_used=$(echo $mem_used | tr -d ' ')
  mem_total=$(echo $mem_total | tr -d ' ')
  util=$(echo $util | tr -d ' ')
  
  # 计算内存使用百分比
  mem_percent=$(awk "BEGIN {print ($mem_used/$mem_total)*100}")
  
  # 如果GPU利用率低于10%且内存使用低于20%，认为是空闲的
  if (( $(echo "$util < 10" | bc -l) )) && (( $(echo "$mem_percent < 20" | bc -l) )); then
    FREE_GPUS+=($idx)
    GPU_USAGE+=($util)
    GPU_MEM_USED+=($mem_percent)
  fi
done <<< "$GPU_INFO"

# 检查是否有足够的空闲GPU
if [ ${#FREE_GPUS[@]} -lt $NUM_GPUS ]; then
  echo "错误: 没有足够的空闲GPU。需要 $NUM_GPUS 个，但只找到 ${#FREE_GPUS[@]} 个"
  echo "所有GPU状态:"
  echo "$GPU_INFO" | tr ',' ' ' | awk '{printf "GPU %s: 使用率 %s%%, 内存使用 %s/%s MiB\n", $1, $4, $2, $3}'
  exit 1
fi

# 选择最空闲的N个GPU
if [ ${#FREE_GPUS[@]} -gt $NUM_GPUS ]; then
  echo "找到 ${#FREE_GPUS[@]} 个空闲GPU，将选择 $NUM_GPUS 个最空闲的GPU"
  
  # 创建一个包含GPU索引和使用率的数组
  declare -a GPU_COMBINED
  for i in "${!FREE_GPUS[@]}"; do
    # 组合GPU索引、使用率和内存使用率
    combined_score=$(awk "BEGIN {print ${GPU_USAGE[$i]} + ${GPU_MEM_USED[$i]}}")
    GPU_COMBINED+=("${combined_score}:${FREE_GPUS[$i]}")
  done
  
  # 按使用率排序
  IFS=$'\n' SORTED_GPUS=($(sort -n <<<"${GPU_COMBINED[*]}"))
  unset IFS
  
  # 选择前N个
  FREE_GPUS=()
  for i in $(seq 0 $(($NUM_GPUS-1))); do
    if [ $i -lt ${#SORTED_GPUS[@]} ]; then
      gpu_idx=${SORTED_GPUS[$i]#*:}
      FREE_GPUS+=($gpu_idx)
    fi
  done
fi

# 构建CUDA_VISIBLE_DEVICES环境变量
GPUS_TO_USE=$(IFS=,; echo "${FREE_GPUS[*]}")
echo "将使用以下物理GPU: $GPUS_TO_USE"

# ADJUSTED_GPUS=()
# for gpu in "${FREE_GPUS[@]}"; do
#   ADJUSTED_GPUS+=( $((gpu-1)) )
# done
# GPUS_TO_USE=$(IFS=,; echo "${ADJUSTED_GPUS[*]}")
# echo "将使用以下物理GPU(已减1): $GPUS_TO_USE"
# echo "注意：在脚本/代码内部，逻辑GPU id 0,1,2,... 实际对应物理GPU $GPUS_TO_USE"

# 执行Python脚本
echo "开始执行 $PYTHON_SCRIPT..."
args=""
if [ "$ENABLE_MEMORY_DEBUG" = "true" ]; then
  args="$args --enable_memory_debug --memory_debug_log_dir $MEMORY_DEBUG_LOG_DIR"
fi
CUDA_VISIBLE_DEVICES=$GPUS_TO_USE CUDA_LAUNCH_BLOCKING=1 python "$PYTHON_SCRIPT" $args
# 检查执行结果
if [ $? -eq 0 ]; then
  echo "脚本执行成功!"
else
  echo "脚本执行失败，退出代码: $?"
fi