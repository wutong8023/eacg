#!/bin/bash

# LoRA训练脚本 - 支持自动GPU检测和数量指定
# 使用方法:
#   ./train_lora_VersiBCB.sh --gpu-count 2                   # 自动选择2个最空闲的GPU
#   ./train_lora_VersiBCB.sh --manual-gpu 0,1,2             # 手动指定GPU
#   ./train_lora_VersiBCB.sh --help                          # 显示帮助

# 默认配置
PRECISION="bf16"
DATASET_TYPE="srccodes"
CORPUS_PATH="/datanfs2/chenrongyi/data/srccodes"
BENCHMARK_DATA_PATH="benchmark/data/VersiBCB_Benchmark/vace_datas.json"
GPU_COUNT="3"
BENCHMARK_DATA_PATHS=("benchmark/data/VersiBCB_Benchmark/vace_datas_for_warning.json")

# 标志变量：跟踪用户是否提供了GPU相关参数
USER_PROVIDED_GPU_COUNT=false
USER_PROVIDED_MANUAL_GPU=false

# GPU检测函数
detect_free_gpus() {
    local num_gpus=$1
    
    echo "=== 检测GPU状态 ==="
    
    # 检查nvidia-smi是否可用
    if ! command -v nvidia-smi &> /dev/null; then
        echo "错误: nvidia-smi 命令不可用"
        return 1
    fi
    
    # 先测试nvidia-smi基本功能
    if ! nvidia-smi > /dev/null 2>&1; then
        echo "错误: nvidia-smi无法正常运行"
        return 1
    fi
    
    # 获取GPU信息，按内存使用率排序
    # 注意：nvidia-smi的CSV输出可能包含空格，需要正确处理
    gpu_info=$(nvidia-smi --query-gpu=index,memory.used,memory.total,name --format=csv,noheader,nounits | \
        sed 's/^[ \t]*//;s/[ \t]*$//' | \
        awk -F', *' '{
            if (NF >= 4) {
                gpu_id=$1; used=$2; total=$3; name=$4;
                gsub(/^[ \t]+|[ \t]+$/, "", gpu_id);
                gsub(/^[ \t]+|[ \t]+$/, "", used);
                gsub(/^[ \t]+|[ \t]+$/, "", total);
                gsub(/^[ \t]+|[ \t]+$/, "", name);
                
                if (total > 0) {
                    usage_percent=int(used*100/total);
                    free_mb=total-used;
                    printf "%s,%d,%s,%d\n", gpu_id, usage_percent, name, free_mb;
                }
            }
        }' | sort -t, -k2,2n)  # 按使用率排序
    
    if [ -z "$gpu_info" ]; then
        echo "错误: 无法获取GPU信息或没有可用GPU"
        echo "尝试直接运行: nvidia-smi --query-gpu=index,memory.used,memory.total,name --format=csv,noheader,nounits"
        return 1
    fi
    
    echo "可用GPU状态:"
    echo "GPU_ID | 内存使用率 | 空闲内存(MB) | GPU名称"
    echo "-------|------------|--------------|--------"
    
    available_gpus=()
    while IFS=',' read -r gpu_id usage_percent gpu_name free_mb; do
        # 确保gpu_id是整数
        if [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
            printf "%-6s | %-10s | %-12s | %s\n" "$gpu_id" "${usage_percent}%" "${free_mb}MB" "$gpu_name"
            
            # 选择使用率低于50%且空闲内存大于4GB的GPU (使用整数比较)
            if [ "$usage_percent" -lt 50 ] && [ "$free_mb" -gt 4000 ]; then
                available_gpus+=($gpu_id)
                echo "    -> GPU $gpu_id 符合条件 (使用率${usage_percent}%, 空闲${free_mb}MB)"
            else
                echo "    -> GPU $gpu_id 不符合条件 (使用率${usage_percent}%, 空闲${free_mb}MB)"
            fi
        else
            echo "警告: GPU ID '$gpu_id' 不是有效的整数，跳过"
        fi
    done <<< "$gpu_info"
    
    echo ""
    echo "空闲GPU (使用率<50%, 空闲内存>4GB): ${available_gpus[*]}"
    
    if [ ${#available_gpus[@]} -eq 0 ]; then
        echo "警告: 没有找到空闲的GPU"
        echo "所有GPU状态:"
        while IFS=',' read -r gpu_id usage_percent gpu_name free_mb; do
            echo "  GPU $gpu_id: ${usage_percent}% 使用率, ${free_mb}MB 空闲, $gpu_name"
        done <<< "$gpu_info"
        return 1
    fi
    
    if [ ${#available_gpus[@]} -lt $num_gpus ]; then
        echo "警告: 只找到 ${#available_gpus[@]} 个空闲GPU，但需要 $num_gpus 个"
        echo "将使用所有可用的空闲GPU: ${available_gpus[*]}"
        selected_gpus=$(IFS=,; echo "${available_gpus[*]}")
    else
        # 选择前num_gpus个最空闲的GPU
        selected_gpus=""
        for ((i=0; i<$num_gpus; i++)); do
            if [ $i -eq 0 ]; then
                selected_gpus="${available_gpus[$i]}"
            else
                selected_gpus="$selected_gpus,${available_gpus[$i]}"
            fi
        done
    fi
    
    echo "选择的GPU: $selected_gpus"
    echo "$selected_gpus"
    return 0
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu-count)
            GPU_COUNT="$2"
            USER_PROVIDED_GPU_COUNT=true
            shift 2
            ;;
        --manual-gpu)
            MANUAL_GPUS="$2"
            USER_PROVIDED_MANUAL_GPU=true
            shift 2
            ;;
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        --dataset-type)
            DATASET_TYPE="$2"
            shift 2
            ;;
        --corpus-path)
            CORPUS_PATH="$2"
            shift 2
            ;;
        --benchmark-data-path)
            BENCHMARK_DATA_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpu-count NUM               自动选择NUM个最空闲的GPU (推荐)"
            echo "  --manual-gpu GPU_IDS          手动指定GPU IDs (例如: 0,1,2)"
            echo "  --precision PRECISION         模型精度 (fp16|fp32|bf16，默认: bf16)"
            echo "  --dataset-type TYPE           数据集类型 (docstring|srccodes，默认: srccodes)"
            echo "  --corpus-path PATH            语料库路径"
            echo "  --benchmark-data-path PATH    基准数据路径"
            echo "  -h, --help                    显示此帮助信息"
            echo ""
            echo "Examples:"
            echo "  $0 --gpu-count 2              # 自动选择2个最空闲的GPU"
            echo "  $0 --gpu-count 1              # 自动选择1个最空闲的GPU"
            echo "  $0 --manual-gpu 0,1,2         # 手动指定使用GPU 0,1,2"
            echo "  $0 --gpu-count 3 --precision fp16 --dataset-type docstring"
            echo ""
            echo "GPU选择策略:"
            echo "  - 优先选择内存使用率<50%的GPU"
            echo "  - 要求空闲内存>4GB"
            echo "  - 按内存使用率从低到高排序选择"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=== LoRA训练配置 ==="
echo "精度: $PRECISION"
echo "数据集类型: $DATASET_TYPE"
echo "语料库路径: $CORPUS_PATH"
echo "基准数据路径: $BENCHMARK_DATA_PATH"

# GPU选择逻辑
if [ "$USER_PROVIDED_MANUAL_GPU" = true ]; then
    echo "GPU模式: 手动指定 ($MANUAL_GPUS)"
    SELECTED_GPUS="$MANUAL_GPUS"
elif [ "$USER_PROVIDED_GPU_COUNT" = true ]; then
    echo "GPU模式: 自动选择 $GPU_COUNT 个GPU"
    
    # 检查GPU_COUNT是否为正整数
    if ! [[ "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]]; then
        echo "错误: --gpu-count 必须是正整数"
        exit 1
    fi
    
    # 检查是否需要bc计算器
    if ! command -v bc &> /dev/null; then
        echo "警告: bc 计算器不可用，GPU检测功能可能受限"
        echo "请安装bc: sudo apt-get install bc"
    fi
    
    # 自动检测GPU
    SELECTED_GPUS=$(detect_free_gpus $GPU_COUNT)
    if [ $? -ne 0 ] || [ -z "$SELECTED_GPUS" ]; then
        echo "错误: GPU自动检测失败"
        exit 1
    fi
else
    echo "GPU模式: 使用默认GPU (5,6,7)"
    SELECTED_GPUS="5,6,7"
fi

echo ""
echo "=== 开始训练 ==="
echo "使用GPU: $SELECTED_GPUS"
echo "时间: $(date)"

# 调试信息：显示环境变量设置
echo "设置 CUDA_VISIBLE_DEVICES=$SELECTED_GPUS"
export CUDA_VISIBLE_DEVICES="$SELECTED_GPUS"
echo "验证 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# 测试GPU是否可见
echo "测试PyTorch GPU可见性:"
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_properties(i).name}')
else:
    print('No CUDA GPUs detected')
"

echo ""

# 执行训练
CUDA_VISIBLE_DEVICES="$SELECTED_GPUS" python benchmark/train_lora.py \
    --precision "$PRECISION" \
    --dataset_type "$DATASET_TYPE" \
    --corpus_path "$CORPUS_PATH" \
    --benchmark_data_path "$BENCHMARK_DATA_PATH" \
    --benchmark_paths "${BENCHMARK_DATA_PATHS[@]}"

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=== 训练完成 ==="
    echo "成功时间: $(date)"
    echo "使用的GPU: $SELECTED_GPUS"
else
    echo ""
    echo "=== 训练失败 ==="
    echo "失败时间: $(date)"
    echo "使用的GPU: $SELECTED_GPUS"
    exit 1
fi