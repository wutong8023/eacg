"""
test llama3-70B
"""
import json
from together import Together
from openai import OpenAI
import os
import tiktoken
from tqdm import tqdm
import asyncio
import aiohttp
import logging
from datetime import datetime
from tqdm.asyncio import tqdm as async_tqdm
from enum import Enum
import time
from typing import List, Dict, Any, Tuple
import argparse
import subprocess
import torch
# encoding = tiktoken.get_encoding("gpt2")

# Import transformers for local models
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, StoppingCriteria, StoppingCriteriaList

with open("../../API_KEYSET/deepseek.txt","r") as f:
    api_key=f.read()
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
with open("../../API_KEYSET/togetherai.txt","r") as f:
    together_api_key=f.read()
together_client = Together(api_key=together_api_key)
# model_name = "meta-llama/Llama-3-8b-chat-hf"

# 配置logging
os.makedirs('logs/prediction_log', exist_ok=True)
logging.basicConfig(
    filename=f'logs/prediction_log/prediction_{datetime.now().strftime("%Y%m%d")}.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class APIType(Enum):
    DEEPSEEK = "deepseek"
    TOGETHER = "together"
    LOCAL = "local"  # 新增本地推理类型

class TaskType(Enum):
    VSCC = "vscc"
    VACE = "vace"

# 全局变量用于存储本地模型和tokenizer
local_model = None
local_tokenizer = None

def get_free_gpus(memory_threshold=3000):
    """
    获取空闲GPU的ID
    
    Args:
        memory_threshold: 空闲内存阈值(MB)，低于此值视为空闲
    
    Returns:
        List[int]: 空闲GPU的ID列表
    """
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,nounits,noheader'])
        result = result.decode('utf-8').strip()
        
        free_gpus = []
        for line in result.split('\n'):
            idx, memory_used = map(int, line.split(','))
            if memory_used < memory_threshold:
                free_gpus.append(idx)
        
        if free_gpus:
            print(f"Found free GPUs: {free_gpus}")
            return free_gpus
        else:
            print("No free GPUs found, all GPUs are busy")
            return []
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return []

def get_device_map(num_gpus=None):
    """
    创建device map配置
    
    Args:
        num_gpus: 要使用的GPU数量，None表示自动使用所有空闲GPU
    
    Returns:
        str或dict: device map配置
    """
    # 获取空闲GPU
    free_gpus = get_free_gpus()
    
    if not free_gpus:
        print("No free GPUs found, falling back to CPU")
        return "cpu"
    
    # 如果指定了GPU数量，则只使用指定数量的GPU
    if num_gpus is not None and num_gpus > 0:
        free_gpus = free_gpus[:num_gpus]
    
    # 如果只有一个GPU，直接使用cuda:id
    if len(free_gpus) == 1:
        return f"cuda:{free_gpus[0]}"
    
    # 多个GPU，创建device map
    print(f"Using {len(free_gpus)} GPUs: {free_gpus}")
    
    if len(free_gpus) > 1:
        # 返回auto以使用所有可见GPU，同时设置CUDA_VISIBLE_DEVICES
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, free_gpus))
        return "auto"
    
    return f"cuda:{free_gpus[0]}"

def load_local_model(model_name_or_path, device="cuda", num_gpus=None):
    """
    加载本地模型
    
    Args:
        model_name_or_path: 模型名称或路径
        device: 使用的设备，默认为cuda
        num_gpus: 要使用的GPU数量，None表示自动使用所有空闲GPU
    """
    global local_model, local_tokenizer
    
    print(f"Loading local model: {model_name_or_path}")
    
    # 如果指定了自动选择GPU
    if device == "auto":
        device_map = get_device_map(num_gpus)
        print(f"Automatically selected device map: {device_map}")
    else:
        device_map = device
    
    # 加载tokenizer并设置特殊token
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    # 打印当前tokenizer配置
    print(f"Model: {os.path.basename(model_name_or_path)}")
    print(f"  eos_token: {tokenizer.eos_token}")
    print(f"  eos_token_id: {tokenizer.eos_token_id}")
    print(f"  pad_token: {tokenizer.pad_token}")
    print(f"  pad_token_id: {tokenizer.pad_token_id}")
    
    # 根据模型类型设置特殊token
    model_name_lower = model_name_or_path.lower()
    
    # 为不同模型设置特殊token
    if "codellama" in model_name_lower:
        print("Detected CodeLlama model, configuring special tokens")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "</s>"
            print(f"Set pad_token to </s>")
        if tokenizer.eos_token is None:
            tokenizer.eos_token = "</s>"
            print(f"Set eos_token to </s>")
            
    elif "llama" in model_name_lower:
        print("Detected Llama model, configuring special tokens")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "</s>"
            print(f"Set pad_token to </s>")
        if tokenizer.eos_token is None:
            tokenizer.eos_token = "</s>"
            print(f"Set eos_token to </s>")
            
    elif "starcoder" in model_name_lower:
        print("Detected StarCoder model, configuring special tokens")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "<|endoftext|>"
            print(f"Set pad_token to <|endoftext|>")
            
    elif "deepseek" in model_name_lower:
        print("Detected DeepSeek model, configuring special tokens")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = " "
            print(f"Set pad_token to ")
        if tokenizer.eos_token is None:
            tokenizer.eos_token = " "
            print(f"Set eos_token to ")
            
    elif "codegemma" in model_name_lower:
        print("Detected CodeGemma model, configuring special tokens")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "<pad>"
            print(f"Set pad_token to <pad>")
        if tokenizer.eos_token is None:
            tokenizer.eos_token = "<eos>"
            print(f"Set eos_token to <eos>")
    
    # 检查所有模型，确保pad_token存在
    if tokenizer.pad_token is None:
        print("Warning: No specific model type detected, using default special token configuration")
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set pad_token to eos_token: {tokenizer.eos_token}")
        else:
            tokenizer.pad_token = "[PAD]"
            tokenizer.eos_token = "[EOS]"
            print("Set pad_token to [PAD] and eos_token to [EOS]")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        device_map=device_map
    )
    
    local_model = model
    local_tokenizer = tokenizer
    
    # 打印最终配置
    print(f"Final configuration:")
    print(f"  pad_token: {tokenizer.pad_token}")
    print(f"  pad_token_id: {tokenizer.pad_token_id}")
    print(f"  eos_token: {tokenizer.eos_token}")
    print(f"  eos_token_id: {tokenizer.eos_token_id}")
    
    return model, tokenizer

class EndOfFunctionCriteria(StoppingCriteria):
    """当检测到特定停止序列时停止生成"""
    
    def __init__(self, stop_sequences, input_length):
        self.stop_sequences = stop_sequences
        self.input_length = input_length  # 记录输入序列的长度
        self.min_new_tokens = 10  # 至少生成这么多token再检查停止条件
        
    def __call__(self, input_ids, scores, **kwargs):
        # 如果生成的新token太少，不考虑停止
        if input_ids.shape[1] < self.input_length + self.min_new_tokens:
            return False
            
        # 只解码新生成的部分
        generated_text = local_tokenizer.decode(input_ids[0][self.input_length:])
        
        # 检查生成的文本是否包含停止序列
        for stop_seq in self.stop_sequences:
            if stop_seq in generated_text:
                print(f"Found stop sequence: '{stop_seq}' in generated text")
                return True
                
        # 检查是否找到完整的代码封装
        if "<start>" in generated_text and "<end>" in generated_text:
            print("Found complete code block with <start> and <end>")
            return True
            
        return False

def local_predict(text: str, num_samples: int = 3, batch_size: int = 4):
    """
    使用本地模型进行预测 - 通过批处理方式高效生成多个样本
    
    Args:
        text: 输入文本
        num_samples: 采样次数
        batch_size: 每批处理的样本数量
    """
    global local_model, local_tokenizer
    
    if local_model is None or local_tokenizer is None:
        raise ValueError("Local model not loaded. Call load_local_model first.")
    
    # 确认特殊token的设置
    print(f"Generation using tokenizer with:")
    print(f"  pad_token: {local_tokenizer.pad_token}")
    print(f"  pad_token_id: {local_tokenizer.pad_token_id}")
    print(f"  eos_token: {local_tokenizer.eos_token}")
    print(f"  eos_token_id: {local_tokenizer.eos_token_id}")
    
    # 编码输入文本
    print(f"Generating {num_samples} samples with batch size {batch_size}...")
    inputs = local_tokenizer(text, return_tensors="pt").to(local_model.device)
    input_length = inputs["input_ids"].shape[1]
    
    # 计算需要的批次数
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    # 使用模型分批生成样本
    predictions = []
    try:
        for batch_idx in range(num_batches):
            # 计算当前批次需要生成的样本数量
            current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
            print(f"Processing batch {batch_idx+1}/{num_batches}, generating {current_batch_size} samples...")
            
            # 对当前批次进行生成
            with torch.no_grad():
                batch_outputs = local_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=1.0,
                    top_p=0.95,
                    do_sample=True,
                    num_return_sequences=current_batch_size,  # 当前批次的样本数
                    pad_token_id=local_tokenizer.pad_token_id,
                    eos_token_id=local_tokenizer.eos_token_id
                )
            
            # 处理当前批次生成的结果
            for i in range(current_batch_size):
                # 解码输出，去除输入部分
                predicted_text = local_tokenizer.decode(
                    batch_outputs[i][input_length:], 
                    skip_special_tokens=True
                )
                
                sample_idx = batch_idx * batch_size + i + 1
                print(f"Sample {sample_idx}/{num_samples} generated, length: {len(predicted_text)}")
                
                # 检查输出是否为空
                if not predicted_text or len(predicted_text.strip()) < 10:
                    print(f"Warning: Empty or too short prediction for sample {sample_idx}, trying again...")
                    
                    # 对于失败的样本，单独重新生成
                    with torch.no_grad():
                        single_output = local_model.generate(
                            **inputs,
                            max_new_tokens=512,
                            temperature=0.8,  # 降低温度以增加确定性
                            top_p=0.9,
                            do_sample=True
                        )
                    
                    predicted_text = local_tokenizer.decode(
                        single_output[0][input_length:],
                        skip_special_tokens=True
                    )
                    
                    print(f"Retry generation for sample {sample_idx}, length: {len(predicted_text)}")
                    
                    if not predicted_text or len(predicted_text.strip()) < 10:
                        raise ValueError(f"Failed to generate meaningful content for sample {sample_idx}")
                
                predictions.append(predicted_text)
                
                # 如果已经生成了足够的样本，提前结束
                if len(predictions) >= num_samples:
                    break
        
        return predictions
        
    except Exception as e:
        error_msg = f"Error in batch generation: {str(e)}"
        logging.error(error_msg)
        print(error_msg)
        raise

def truncate_text(text, max_tokens):
    # obtain tokenizer
    encoding = tiktoken.get_encoding("gpt2")
    disallowed_special = ()

    tokens = encoding.encode(text, disallowed_special=disallowed_special)
    print(len(tokens))

    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    truncated_text = encoding.decode(tokens)

    return truncated_text

def predict(text: str, model_name_or_path: str, api_type: APIType, num_samples: int = 3):
    """
    统一的预测接口
    
    Args:
        text: 输入文本
        model_name_or_path: 模型名称或路径
        api_type: API类型 (DEEPSEEK, TOGETHER 或 LOCAL)
        num_samples: 采样次数
    """
    if api_type == APIType.TOGETHER:
        response = together_client.chat.completions.create(
            model=model_name_or_path,
            messages=[{"role": "user", "content": text}],
            frequency_penalty=0.1,
            n=num_samples,
            presence_penalty=0.0,
            stop=None,
            stream=False,
            temperature=1,
            top_p=0.95
        )
        return [choice.message.content for choice in response.choices]
    elif api_type == APIType.DEEPSEEK:
        prediction = deepseek_predict(text, model_name_or_path)
        return prediction
    elif api_type == APIType.LOCAL:
        # 使用本地模型进行预测
        return local_predict(text, num_samples)
    else:
        raise ValueError(f"Unsupported API type: {api_type}")

def deepseek_predict(text: str, model_name_or_path: str):
    response = client.chat.completions.create(
        model=model_name_or_path,
        messages=[{"role": "user", "content": text}],
        frequency_penalty=0.1,
        logprobs=None,
        presence_penalty=0.0,
        stop=None,
        stream=False,
        temperature=1,  # 保持temperature=1以确保多样性
        top_p=0.95
    )
    response_text = response.choices[0].message.content
    return response_text

def build_prompt(dependency, description, task_type: TaskType, ban_deprecation=False, **kwargs) -> str:
    """
    构建不同任务类型的 prompt
    
    Args:
        dependency: 依赖包信息
        description: 任务描述
        task_type: 任务类型 (VSCC 或 VACE)
        ban_deprecation: 是否禁用废弃函数
        **kwargs: 额外参数，用于VACE任务
    """
    ban_deprecation_str = "Also note that you should not use deprecated functions or classes." if ban_deprecation else ""
    if task_type == TaskType.VSCC:

        prompt = f'''
            You are a professional Python engineer, and I will provide functional descriptions and versions of specified dependency packages. 
            You need to write code in Python to implement this feature based on the functional description and using the dependency package and version I specified. 
            Please note that you only need to return the code that implements the function, and do not return any other content. {ban_deprecation_str}
            Please use <start> and <end> to enclose the generated code. Here is an example:
            ###Function Description：
            The function of this code is to print the results predicted by calling the model using vllm.
            ###dependency and version：
            'vllm': '0.3.3'
            ###response:
            <start>
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print("Prompt,Generated text")
            <end>
            Given above example, please generate answer code for below input to create required function.
            ###Function Description：
            {description}
            ###dependency and version：
            {dependency}
            ###response:
        '''
    else:  # TaskType.VACE
        origin_code = kwargs.get('origin_code', '')
        target_dependency = kwargs.get('target_dependency', '')
        prompt = f'''
            You are now a professional Python programming engineer. I will provide you with a code snippet and a description of its functionality, 
            including the dependencies and versions used in the code. Then, I will provide the same dependencies but with a specified old version. 
            Your task is to refactor the code using the methods provided by the specified old version and return the refactored code.{ban_deprecation_str} 
            Please note that you only need to return the refactored code and enclose it with <start> and <end>:
            ###Functionality description of the code
            {description}
            ###Dependency and origin version
            {dependency}
            ###Origin code
            {origin_code}
            ###Dependency and target version
            {target_dependency}
            ###Refactored new code
        '''
    
    return prompt


async def retry_with_backoff(func, *args, max_retries=5, initial_delay=10, **kwargs):
    """
    带有退避机制的重试函数
    
    Args:
        func: 要重试的异步函数
        max_retries: 最大重试次数
        initial_delay: 初始等待时间（秒）
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if "rate_limit" in str(e).lower():
                print(f"Rate limit hit, waiting {delay} seconds before retry...")
                await asyncio.sleep(delay)
                delay *= 2  # 指数退避
            else:
                raise e
    
    raise last_exception

def load_existing_results(model_name_or_path: str, json_path: str) -> Dict[str, List[str]]:
    """
    加载已有的预测结果
    Params:
        model_name_or_path: 模型名称或路径
        json_path: 输入的json文件路径,使用最后一部分用来构造输出文件名
    Returns:
        Dict[str, List[str]]: id到predictions的映射
    """
    model_name = os.path.basename(model_name_or_path)
    save_folder_path = os.path.join('data/model_predictions', model_name)
    save_json_path = os.path.join(save_folder_path, json_path.split('/')[-1])
    
    existing_results = {}
    if os.path.exists(save_json_path):
        with open(save_json_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            for item in existing_data:
                if ("model_output" in item and 
                    isinstance(item["model_output"], list) and 
                    len(item["model_output"]) == 3 and
                    all(output for output in item["model_output"])):
                    existing_results[item["id"]] = item["model_output"]
    
    return existing_results

def process_predictions(data_list, model_name_or_path, max_tokens, json_path, api_type: APIType, task_type: TaskType, ban_deprecation=False, device="auto", num_gpus=None):
    """
    主处理函数
    Params:
        data_list: 数据列表
        model_name_or_path: 模型名称或路径
        max_tokens: 最大token数
        json_path: 输入的json文件路径,使用最后一部分用来构造输出文件名
        api_type: API类型 (DEEPSEEK, TOGETHER 或 LOCAL)
        task_type: 任务类型 (VSCC 或 VACE)
        ban_deprecation: 是否禁用废弃函数
        device: 使用的设备，默认为cuda
        num_gpus: 要使用的GPU数量，None表示自动使用所有空闲GPU
    Returns:
        List[Dict[str, Any]]: 处理后的数据列表
    """
    print(f"Starting async prediction processing for {task_type.value} task with {model_name_or_path} using {api_type.value} API")
    print(f"Total items to process: {len(data_list)}")
    
    # 如果是本地模型，先加载
    if api_type == APIType.LOCAL and (local_model is None or local_tokenizer is None):
        load_local_model(model_name_or_path, device=device, num_gpus=num_gpus)
    
    # 加载已有的预测结果
    existing_results = load_existing_results(model_name_or_path, json_path)
    
    # 更新data_list中的数据
    for data in data_list:
        if data['id'] in existing_results:
            data['model_output'] = existing_results[data['id']]
        elif 'model_output' in data:
            # 如果数据不在existing_results中，删除可能存在的无效model_output
            del data['model_output']
    
    # 检查每个数据项的状态
    needs_processing = []
    for data in data_list:
        print(f"Checking task {data['id']}: ", end='')
        if 'model_output' not in data:
            print("No model_output key found - needs processing")
            needs_processing.append(data['id'])
        elif not isinstance(data['model_output'], list) or len(data['model_output']) != 3:
            print(f"Invalid model_output format - needs processing")
            needs_processing.append(data['id'])
        else:
            print(f"Has valid model_output with {len(data['model_output'])} samples")
    
    # 检查已处理的数量
    processed = len(data_list) - len(needs_processing)
    print(f"Found valid existing results: {processed}/{len(data_list)}")
    print(f"Need to process {len(needs_processing)} items")
    
    if len(needs_processing) == 0:
        print("All items have been processed with valid results, skipping...")
        return data_list
    
    # 执行异步处理
    updated_data_list = asyncio.run(async_predictions(
        data_list,
        model_name_or_path,
        max_tokens,
        json_path,
        api_type,
        task_type,
        ban_deprecation,
        device,
        num_gpus
    ))
    
    print("Processing completed!")
    return updated_data_list

async def process_single_prediction(data, model_name_or_path, max_tokens, ban_deprecation, semaphore, api_type: APIType, task_type: TaskType, num_samples=3):
    """处理单个预测的异步函数"""
    if "model_output" in data and isinstance(data["model_output"], list):
        if len(data["model_output"]) == num_samples:
            print(f"Skipping already processed task {data.get('id', 'unknown')}")
            return data, None
        else:
            print(f"Reprocessing task {data.get('id', 'unknown')} due to incomplete samples")
    
    try:
        async with semaphore:
            # 构建指令部分(保持不变)
            if task_type == TaskType.VSCC:
                dependency = data['dependency']
                description = data['description']
                instruction = build_prompt(
                    dependency, 
                    description, 
                    task_type=task_type,
                    ban_deprecation=ban_deprecation
                )
            else:  # TaskType.VACE
                origin_dependency = data['origin_dependency']
                target_dependency = data['target_dependency']
                description = data['description']
                origin_code = data['origin_code']
                instruction = build_prompt(
                    origin_dependency,
                    description,
                    task_type=task_type,
                    origin_code=origin_code,
                    target_dependency=target_dependency
                )
            
            truncated_text = truncate_text(instruction, max_tokens)
            
            if api_type == APIType.TOGETHER:
                # 远程API调用保持不变，包括错误处理和重试
                predictions = await retry_with_backoff(
                    asyncio.to_thread,
                    predict, 
                    truncated_text, 
                    model_name_or_path,
                    api_type,
                    num_samples
                )
            elif api_type == APIType.LOCAL:
                # 本地模型异步预测 - 不捕获异常，让它直接传播
                print(f"Starting local prediction for task {data.get('id', 'unknown')}")
                # 注意：这里不使用retry_with_backoff，让异常直接传播
                predictions = await asyncio.to_thread(
                    local_predict,
                    truncated_text,
                    num_samples
                )
                print(f"Completed local prediction for task {data.get('id', 'unknown')}")
            else:  # APIType.DEEPSEEK
                # 远程API调用保持不变
                predictions = []
                for _ in range(num_samples):
                    prediction = await retry_with_backoff(
                        deepseek_predict_async,
                        truncated_text,
                        model_name_or_path
                    )
                    predictions.append(prediction)
            
            data['model_output'] = predictions
            return data, None
            
    except Exception as e:
        # 对于非本地模型，继续捕获异常并处理
        if api_type != APIType.LOCAL:
            error_msg = f"Error processing prediction for task {data.get('id', 'unknown')}: {str(e)}"
            logging.error(error_msg)
            return data, error_msg
        else:
            # 对于本地模型，直接向上抛出异常
            raise

def save_progress(data_list, model_name_or_path, json_path):
    """
    保存当前进度
    Params:
        data_list: 数据列表
        model_name_or_path: 模型名称或路径
        json_path: 输入的json文件路径,使用最后一部分用来构造输出文件名
    Returns:
        None
    """
    model_name = os.path.basename(model_name_or_path)
    save_folder_path = os.path.join('data/model_predictions', model_name)
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    save_json_path = os.path.join(save_folder_path, json_path.split('/')[-1])
    
    with open(save_json_path, 'w', encoding='utf-8') as fw:
        json.dump(data_list, fw, indent=4, ensure_ascii=False)

async def async_predictions(data_list, model_name_or_path, max_tokens, json_path, api_type: APIType, task_type: TaskType, ban_deprecation=False, device="auto", num_gpus=None):
    """异步处理预测"""
    # 根据API类型设置合适的并发限制
    if api_type == APIType.TOGETHER:
        semaphore = asyncio.Semaphore(3)
    elif api_type == APIType.LOCAL:
        semaphore = asyncio.Semaphore(1)
    else:
        semaphore = asyncio.Semaphore(10)
    
    # 创建实际的Task对象而非协程
    tasks = []
    for data in data_list:
        coro = process_single_prediction(
            data,
            model_name_or_path=model_name_or_path,
            max_tokens=max_tokens,
            ban_deprecation=ban_deprecation,
            semaphore=semaphore,
            api_type=api_type,
            task_type=task_type
        )
        # 创建Task对象，而不是简单地存储协程
        task = asyncio.create_task(coro)
        tasks.append(task)
    
    # 使用tqdm显示进度
    results = []
    try:
        for task in async_tqdm.as_completed(tasks, desc="Processing predictions"):
            try:
                result = await task
                results.append(result)
                
                # 保存当前进度
                if len(results) % 1 == 0:
                    save_progress(data_list, model_name_or_path, json_path)
            except Exception as e:
                # 只有当使用本地模型时，才停止所有处理
                if api_type == APIType.LOCAL:
                    error_msg = f"Error in local model prediction: {str(e)}"
                    logging.error(error_msg)
                    print(f"\n{error_msg}")
                    print("\nStopping all processing due to error in local model")
                    
                    # 保存当前进度
                    save_progress(data_list, model_name_or_path, json_path)
                    
                    # 取消所有剩余任务 - 现在可以正确使用done()方法
                    for remaining_task in tasks:
                        if not remaining_task.done():
                            remaining_task.cancel()
                    
                    # 向上抛出异常
                    raise
                else:
                    # 非本地模型的错误，记录后继续处理
                    error_msg = f"Error in task: {str(e)}"
                    logging.error(error_msg)
                    print(f"\nError in task: {str(e)}")
        
        # 更新数据列表
        for i, (result, error) in enumerate(results):
            if error:
                print(f"Error in item {i+1}: {error}")
            data_list[i] = result
        
        # 最终保存
        save_progress(data_list, model_name_or_path, json_path)
            
    except asyncio.CancelledError:
        # 任务被取消
        print("Tasks were cancelled")
        save_progress(data_list, model_name_or_path, json_path)
    except Exception as e:
        # 捕获所有异常
        error_msg = f"Fatal error in processing: {str(e)}"
        logging.error(error_msg)
        print(f"\n{error_msg}")
        save_progress(data_list, model_name_or_path, json_path)
        raise
        
    return data_list

# 异步版本的API调用函数
async def deepseek_predict_async(text, model_name_or_path):
    """
    异步版本的deepseek_predict
    需要根据实际的API调用方式来实现
    """
    # 这里需要实现异步版本的API调用
    # 如果原始API不支持异步，可以使用asyncio.to_thread包装
    return await asyncio.to_thread(deepseek_predict, text, model_name_or_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate predictions for VSCC/VACE tasks using batch processing')
    parser.add_argument('--task_type', type=str, choices=['vscc', 'vace'], required=True,
                       help='Type of task to process (vscc or vace)')
    parser.add_argument('--model_name_or_path', type=str, required=True,
                       help='Name or path of the model to use')
    parser.add_argument('--json_path', type=str, required=True,
                       help='Path to the input JSON file')
    parser.add_argument('--api_type', type=str, choices=['deepseek', 'together', 'local', 'api2d', 'openai', 'anthropic', 'gemini'], required=True,
                       help='Type of API to use')
    parser.add_argument('--ban_deprecation', type=bool, required=True,
                       help='Whether to ban deprecated functions')
    parser.add_argument('--max_tokens', type=int, default=7000,
                       help='Maximum number of tokens for input')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for local model inference (cuda, cpu, or auto for automatic GPU selection)')
    parser.add_argument('--num_gpus', type=int, default=None,
                       help='Number of GPUs to use, only effective when device is auto')
    
    args = parser.parse_args()
    
    # 转换参数
    task_type = TaskType(args.task_type)
    api_type = APIType(args.api_type)
    
    # 读取数据
    with open(args.json_path, 'r', encoding='utf-8') as fr:
        data_list = json.load(fr)
    
    # 处理预测
    data_list = process_predictions(
        data_list,
        model_name_or_path=args.model_name_or_path,
        max_tokens=args.max_tokens,
        json_path=args.json_path,
        api_type=api_type,
        task_type=task_type,
        ban_deprecation=args.ban_deprecation,
        device=args.device,
        num_gpus=args.num_gpus
    )


