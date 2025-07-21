SHORT=m:,b:,w:,n:,r:,o:,h:
LONG=model:,benchmark:,world_size:,num_gpus_per_job:,rank_offset:,allow_disk_offload:,help:

PARSED=$(getopt --options $SHORT --longoptions $LONG --name "$0" -- "$@") || { echo "Invalid Arguments."; exit 2; }
eval set -- "$PARSED"

model=llama31_instruct
benchmark=long-bench 
num_gpus_per_job=1 
rank_offset=0 
allow_disk_offload=False
corpus_types=("docstring") # docstring, srccodes
# 获取GPU使用情况并返回空闲GPU的ID
get_free_gpus() {
    # 获取所有GPU的使用情况，内存使用率低于20%认为是空闲的
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
        -h|--help) echo "Usage: $0 [--model <str>] [--benchmark <str>] [--world_size <int>] [--num_gpus_per_job <int>] [--rank_offset <int>] [--allow_disk_offload <bool>] [--help]"; exit ;;
        -m|--model) model="$2"; shift 2 ;;
        -b|--benchmark) benchmark="$2"; shift 2 ;;
		-w|--world_size) world_size="$2"; shift 2 ;;
        -n|--num_gpus_per_job) num_gpus_per_job="$2"; shift 2 ;;
        -r|--rank_offset) rank_offset="$2"; shift 2 ;;
        -o|--allow_disk_offload) allow_disk_offload="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "Programming error"; exit 3 ;;
    esac
done

# ============================ SETUP ============================ #
base_dir="."

cd "${base_dir}"

# install requirements
# python3 -m pip install --upgrade pip
# pip install -r "${base_dir}/requirements.txt"
# pip install -e "${base_dir}/."

if [ $(( world_size )) -eq $(nvidia-smi --list-gpus | wc -l) ] && [ $(( rank_offset )) -ne 0 ]; then
    world_size=$((world_size - rank_offset))
fi

echo "World size: $world_size"

if [ "$benchmark" = "long-bench" ]; then
#    datasets="2wikimqa,gov_report,hotpotqa,lcc,multi_news,multifieldqa_en,musique,narrativeqa,passage_retrieval_en,qasper,qmsum,repobench-p,samsum,trec,triviaqa"
	 datasets="VersiBCB_vscc" # VersiBCB_vscc_BD VersiBCB_vace_BD VersiBCB_vscc_GEN VersiBCB_vace_GEN
elif [ "$benchmark" = "infinite-bench" ]; then
    datasets="code_debug,math_find,kv_retrieval,passkey,number_string,longbook_choice_eng"
elif [ "$benchmark" = "passkey" ]; then
	datasets="passkey__long"
else
    echo "Error: Benchmark not recognized."
    exit 1
fi

output_dir_path="${base_dir}/benchmark/results/${model}/${benchmark}"
config_file="${model}.yaml"


# ============================ RUN ============================ #
echo "Evaluating on ${benchmark} using ${model}."
echo "Output directory: ${output_dir_path}"
echo "Disk offload allowed: ${allow_disk_offload}"

trap 'kill $(jobs -p)' SIGINT

if [ $(( world_size % num_gpus_per_job )) -eq 0 ]; then
    subworld_num=$((world_size/num_gpus_per_job))
    
    for corpus_type in "${corpus_types[@]}"; do
        # job_index=0
        for i in $(seq 0 $((num_gpus_per_job)) $((world_size-1)))
        do
            rank=$((i/num_gpus_per_job))
            
            # 构建GPU设备字符串
            gpu_devices=""
            for j in $(seq 0 $((num_gpus_per_job-1)))
            do
                gpu_idx=$((i+j))
                if [ -n "$gpu_devices" ]; then
                    gpu_devices="${gpu_devices},${available_gpus[gpu_idx]}"
                else
                    gpu_devices="${available_gpus[gpu_idx]}"
                fi
            done
            
            echo "Starting rank ${rank} with GPUs ${gpu_devices}"

                CUDA_VISIBLE_DEVICES=$gpu_devices python "${base_dir}/benchmark/pred.py" \
                    --config_path "${base_dir}/config/${config_file}" \
                    --output_dir_path "${output_dir_path}" \
                --datasets "${datasets}" \
                --world_size "${subworld_num}" \
                    --rank "${rank}" \
                    --allow_disk_offload "${allow_disk_offload}" \
                    --corpus_type "${corpus_type}" \
                    --max_token_length 4000 \
                    --precision "fp16" \
                    --base_output_dir "output/approach_eval" \
                    --useQAContext true \
                    --QAContext_path "data/temp/inference/versibcb_vscc_results_filtered.jsonl" &
            
            # job_index=$((job_index + 1))
        done
        wait
    done
else
    echo "Error: number of free GPUs ${world_size} is not divisible by ${num_gpus_per_job}."
    exit 1
fi

# DIRECTORY="${output_dir_path}/offload_data"
# if [ -d "$DIRECTORY" ]; then
#   rm -rf "$DIRECTORY"
#   echo "Directory $DIRECTORY and all its contents have been deleted."
# else
#   echo "Directory $DIRECTORY does not exist."
# fi

# python3 "${base_dir}/benchmark/eval.py" --dir_path "${output_dir_path}"
