import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utils.loraTrain.getVersicodeData import getVersicodeBenchData,constructQAPairsFromBenchData
from utils.getPrompt import formatInput
from tqdm import tqdm
import json
from utils.loraTrain.loraTrainUtils import load_config,inference, check_lora_params
from utils.loraPathConfigure import pathConfigurator
from benchmark.config.code.config_lora import LORA_CONFIG_PATH
from peft import PeftModel
import os

def get_last_subdir(path: str) -> str:
    """获取路径字符串的最后一个子目录名，即使路径以 '/' 结尾也能正确处理。"""
    if not path:
        return ""
    # 去除末尾的斜杠（如果存在）
    cleaned_path = path.rstrip('/')
    if not cleaned_path:
        return ""  # 如果路径只有 '/', 则返回空字符串
    return cleaned_path.split('/')[-1]

def loadModelandPredict(adaptor_path=None, output_file=None, use_santacoder=False, santa_dataset_path=None):
    config = load_config(LORA_CONFIG_PATH)
    
    # Override config with function parameters if provided
    if santa_dataset_path:
        config["santa_dataset_path"] = santa_dataset_path
    
    # 检测可用GPU
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个GPU设备")
    
    # 设置明确的设备映射
    if num_gpus >= 2:
        device_map = "balanced"
    else:
        device_map = "auto"
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.get("model_name"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 量化配置
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # 加载量化基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        config.get("model_name"),
        device_map=device_map
    )
    
    print(f"基础模型设备映射: {base_model.hf_device_map}")
    
    # 加载LoRA适配器
    adaptor_path = adaptor_path if adaptor_path else pathConfigurator().getPath(config, "allEvolveRelatedInfo", "all")
    lora_model = PeftModel.from_pretrained(
        base_model,
        adaptor_path,
        device_map=device_map
    )
    
    print(f"LoRA模型设备映射: {lora_model.hf_device_map if hasattr(lora_model, 'hf_device_map') else '不可用'}")
    
    # 处理SantaCoder数据集
    if use_santacoder or config.get("santa_dataset_path"):
        print("开始处理SantaCoder数据集...")
        from utils.loraTrain.buildandloadData import getSantaCoderFIMDataset
        fim_dataset = getSantaCoderFIMDataset(
            tokenizer, 
            config.get("santa_dataset_path"),
            for_eval=True  # 使用专门为推理设计的FIMEvalDataset
        )
        
        if fim_dataset is not None:
            santa_results = []
            print(f"开始对SantaCoder数据集进行推理，共{len(fim_dataset)}个样本")
            
            for i, data in tqdm(enumerate(fim_dataset)):
                try:
                    # 获取输入序列（只包含prefix和suffix，不包含solution）
                    input_ids = data["input_ids"].unsqueeze(0).to(lora_model.device)
                    
                    # 保存原始前缀和黄金解决方案，用于后续评估
                    original_prefix = data["original_prefix"]
                    gold_solution = data["gold_solution"]
    
    # 执行推理
                    with torch.no_grad():
                        outputs = lora_model.generate(
                            input_ids=input_ids,
                            max_new_tokens=512,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            do_sample=True,
                            temperature=0.2,
                            top_p=0.95,
                            top_k=50,
                            num_beams=1,
                            no_repeat_ngram_size=2,
                            repetition_penalty=1.2,
                            length_penalty=1.0,
                            early_stopping=True
                        )
                    
                    # 解码生成的全部文本
                    full_generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # 从生成文本中提取出模型输出的部分（去除输入部分）
                    # 首先解码输入序列
                    input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    
                    # 判断输入是否包含在生成文本的开头
                    if full_generated_text.startswith(input_text):
                        # 提取真正的生成部分（不包含输入）
                        generated_only = full_generated_text[len(input_text):]
                    else:
                        # 如果不能准确匹配，则标记生成可能出现异常
                        generated_only = full_generated_text
                        print(f"警告：样本{i}的生成文本可能不包含原始输入")
                    
                    # 保存结果
                    santa_results.append({
                        "id": i,
                        "input_prefix": original_prefix,  # 保存原始输入前缀
                        "generated": generated_only,      # 只保存生成的部分
                        "gold_solution": gold_solution    # 保存黄金标准解决方案，便于后续评估
                    })
                    
                    # 每10个样本保存一次结果
                    if (i + 1) % 10 == 0:
                        # 使用参数指定的输出文件路径或者默认路径
                        santa_output_path = output_file if output_file else pathConfigurator().getPredictionPath(config, "SantaCoder", "predictions")
                        with open(santa_output_path, "w", encoding="utf-8") as f:
                            json.dump(santa_results, f, ensure_ascii=False, indent=2)
                        print(f"已保存前{i+1}个样本的预测结果")
                        
                except Exception as e:
                    print(f"处理样本{i}时出错: {str(e)}")
                    continue
            
            # 保存最终结果
            santa_output_path = output_file if output_file else pathConfigurator().getPredictionPath(config, "SantaCoder", "predictions")
            with open(santa_output_path, "w", encoding="utf-8") as f:
                json.dump(santa_results, f, ensure_ascii=False, indent=2)
            print(f"SantaCoder数据集推理完成，结果已保存至: {santa_output_path}")
            
            # 返回结果路径
            return santa_output_path
    # 返回结果
    return None

def loadModelandPredictVersiBench(base_model_path=None,adaptor_path=None,output_path_base=None, output_file=None,isVersiBCB=True,only_base_model=False,max_new_tokens=512,sample_num=1,precision='fp16',temperature=0.2,load_existing_results=False):
    '''
    在VersiBCB基准测试上进行预测
    
    Args:
        adaptor_path: str, LoRA适配器的路径，如果为None则使用默认路径
        output_file: str, 保存预测结果的路径，如果为None则使用默认路径，默认名称会根据base_model_name和adaptor_path生成，放在output_path_base下
        precision: str, 模型精度选择 ('fp32', 'fp16', 'bf16', 'int8')
        load_existing_results: bool, 是否加载已有结果避免重复预测
    
    Returns:
        str: 保存结果的文件路径
    '''
    if output_path_base is not None:
        only_base_model_str = 'base' if only_base_model else 'lora'
        adaptor_name = get_last_subdir(adaptor_path)
        base_model_name = get_last_subdir(base_model_path)
        output_file = f"{output_path_base}/{base_model_name}_{adaptor_name}_{only_base_model_str}_lora_pred_result.json"

    config = load_config(LORA_CONFIG_PATH)
    
    # 检测可用GPU
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个GPU设备")
    
    # 设置明确的设备映射
    if num_gpus >= 2:
        device_map = "balanced"
    else:
        device_map = "auto"
    
    # 根据precision设置torch_dtype
    if precision == "fp32":
        torch_dtype = torch.float32
    elif precision == "fp16":
        torch_dtype = torch.float16
    elif precision == "bf16":
        torch_dtype = torch.bfloat16
    elif precision == "int8":
        torch_dtype = torch.int8
    else:
        torch_dtype = torch.float16  # 默认使用fp16
    
    print(f"使用精度: {precision} (torch_dtype: {torch_dtype})")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    slow_tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("设置 pad_token = eos_token")
    
    # 量化配置 - 正确位置：在加载基础模型时应用，而不是在加载PEFT模型时
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # 首先加载量化基础模型
    print(f"加载基础模型: {config.get('model_name')}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.get("model_name") if base_model_path is None else base_model_path,
        # quantization_config=quantization_config,  # 在这里应用量化配置
        device_map=device_map,
        torch_dtype=torch_dtype  # 使用指定的精度
    )
    
    print(f"基础模型设备映射: {base_model.hf_device_map}")
    
    # 检查模型是否包含LoRA参数
    # has_lora, lora_params = check_lora_params(base_model)
    # if has_lora:
    #     print("警告: 基础模型包含LoRA参数。")
    #     for param in lora_params:
    #         print(f"Parameter: {param['name']}, Shape: {param['shape']}")
    # else:
    #     print("基础模型不包含LoRA参数。")
    
    # 然后加载LoRA适配器到已量化的基础模型上
    if not only_base_model:
        adaptor_path = adaptor_path if adaptor_path else pathConfigurator().getPath(config, "allEvolveRelatedInfo", "all")
        print(f"加载LoRA适配器: {adaptor_path}")
        lora_model = PeftModel.from_pretrained(
            base_model,       # 传入已经量化的基础模型
            adaptor_path,     # 适配器路径
            device_map=device_map  # 确保适配器也使用相同的设备映射
        )
        
        print(f"LoRA模型设备映射: {lora_model.hf_device_map if hasattr(lora_model, 'hf_device_map') else '不可用'}")
        
        # 再次检查LoRA模型的参数
        # has_lora_after, lora_params_after = check_lora_params(lora_model)
        # if has_lora_after:
        #     print("LoRA模型包含以下LoRA参数:")
        #     for param in lora_params_after:
        #         print(f"Parameter: {param['name']}, Shape: {param['shape']}")
        
    # 加载已有结果（如果启用）
    existing_results = []
    processed_ids = set()
    
    if load_existing_results and output_file and os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            processed_ids = {result['id'] for result in existing_results if 'id' in result}
            print(f"已加载 {len(existing_results)} 条已有结果，跳过 {len(processed_ids)} 个已处理样本")
        except Exception as e:
            print(f"加载已有结果时出错: {e}")
            existing_results = []
            processed_ids = set()
    
    # 初始化结果列表
    results = existing_results.copy()  # 从已有结果开始
    
    if isVersiBCB:
        # 执行推理
        benchmark = "Versicode_Benchmark" if config["dataset"] == "versicode" else "VersiBCB_Benchmark"
        filename = 'vscc_datas' if config["task"] == "vscc" else 'vace_datas'
        filename = filename + "_for_warning" if config["removeDeprecationData"] else filename
    
        # 确定输出文件路径
        # if output_file is None:
        #     output_file = f"output/{benchmark}/{filename}_lora_pred_result.json"
        

        
        # 加载测试数据
        print(f"加载测试数据: benchmark/data/{benchmark}/{filename}.json")
        with open(f"benchmark/data/{benchmark}/{filename}.json", "r") as f:
            datas = json.load(f)
        print(f"开始在{benchmark}上进行推理，共{len(datas)}个样本")
    else:
        benchmark_path = 'benchmark/data/Versicode_Benchmark/code_completion/numpy/downstream_application_code_token_numpy.json'
        with open(benchmark_path, "r") as f:
            datas = json.load(f)
        print(f"开始在{benchmark_path}上进行推理，共{len(datas)}个样本")
    # 开始推理
    new_results_count = 0  # 记录新处理的样本数量

    for i, data in tqdm(enumerate(datas)):    
        # 检查样本是否在指定范围内
        # if i < VSCC_LOW_BOUND or i > VSCC_HIGH_BOUND:
        #     continue
        
        # 检查是否已经处理过这个样本
        sample_id = data.get("id", f"sample_{i}")
        if load_existing_results and sample_id in processed_ids:
            print(f"跳过已处理的样本: {sample_id}")
            continue
        
        try:
            # 构建输入提示
            input = formatInput(data, config)
            
            # 执行推理
            results_list = []
            for _ in range(sample_num):
                result = inference(base_model, slow_tokenizer, input,max_new_tokens,temperature) if only_base_model else inference(lora_model, slow_tokenizer, input,max_new_tokens,temperature)
                results_list.append(result)
            # 保存结果
            if isVersiBCB:
                new_result = {"id": data["id"], "answer": results_list}
                results.append(new_result)
            else:
                if config['granularity'] == 'line':
                    new_result = {"id": data["id"], "model_prediction": results_list,"solution": data["masked_line"],"solution_api":data['answer']}
                    results.append(new_result)
                elif config['granularity'] == 'token':
                    new_result = {"id": data["id"], "model_prediction": results_list,"solution": data["answer"]}
                    results.append(new_result)
                else:
                    raise ValueError(f"Granularity {config['granularity']} not supported")
            
            new_results_count += 1
            
            # 定期保存结果（每2个新样本）
            if new_results_count % 2 == 0:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"已保存 {new_results_count} 个新处理样本的预测结果，总共 {len(results)} 个结果")
            
        except Exception as e:
            print(f"处理样本{i}时出错: {str(e)}")
            error_result = {"id": data["id"], "answer": ""}
            results.append(error_result)
            new_results_count += 1
            
            # 清理GPU缓存
            try:
                torch.cuda.empty_cache()
            except:
                try:
                    torch.cuda.reset_device()
                except:
                    pass
            continue
    
    # 保存最终结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"推理完成，结果已保存至: {output_file}")
    
    # 返回结果文件路径
    return output_file
