modelscope download --model 'AI-ModelScope/CodeLlama-13b-Instruct-hf' --exclude '*.bin' --local_dir '/seu_nvme/home/qiguilin/230248998/LLMs/cry/'
modelscope download --model 'AI-ModelScope/CodeLlama-34b-Instruct-hf' --exclude '*.bin' --local_dir '/seu_nvme/home/qiguilin/230248998/LLMs/cry/CodeLlama-34b-Instruct-hf'
modelscope download --model 'AI-ModelScope/CodeLlama-34b-Instruct-hf' model-00004-of-00007.safetensors  --local_dir '/seu_nvme/home/qiguilin/230248998/LLMs/cry/CodeLlama-34b-Instruct-hf'
find /seu_nvme/home/qiguilin/230248998/LLMs/cry/ -maxdepth 1 -type f -exec mv {} /seu_nvme/home/qiguilin/230248998/LLMs/cry/CodeLlama-13b-Instruct-hf \;

modelscope download --model AI-ModelScope/CodeLlama-7b-Instruct-hf  --exclude '*.bin' --local_dir ./CodeLlama-7b-Instruct-hf
modelscope download --model AI-ModelScope/CodeLlama-7b-hf  --exclude '*.bin' --local_dir ./CodeLlama-7b-hf
modelscope download --model AI-ModelScope/CodeLlama-13b-hf  --exclude '*.bin' --local_dir ./CodeLlama-13b-hf