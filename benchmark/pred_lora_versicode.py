import torch
import os
from peft import get_peft_model,PeftModel
import json
from benchmark.pred_other import get_version
# from benchmark.config.code.config import CORPUS_PATH
from tqdm import tqdm
import warnings
from utils.loraPathConfigure import pathConfigurator
from utils.getDatasetPacks import getPackVersions
from utils.loraTrain.loraTrainUtils import loraModelExists,get_dataloader,buildandTrainLoraModel,getDataExistence,load_lora_model,load_config,create_lora_config,save_lora_model,load_lora_model_withPeft,load_base_model,merge_lora_weights,inference
from transformers import AutoTokenizer
from benchmark.config.code.config import VSCC_LOW_BOUND,VSCC_HIGH_BOUND
from benchmark.config.code.config_lora import LORA_CONFIG_PATH
from utils.loraTrain.buildandloadData import collate_fn,getDataset,QADataset,DocstringDataset,getCodeParrotDataset,getSantaCoderFIMDataset
from utils.loraTrain.loraTrainUtils import getEquipAdaptorModel,train_lora_model_withPEFT
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from utils.getPrompt import get_prompt,formatInput
from utils.loraTrain.loraPred import loadModelandPredictVersiBench,loadModelandPredict
from utils.loraTrain.dataset_loader import load_dataset
from utils.loraTrain.getVersicodeData import getPkgDocstringItems
# warnings.filterwarnings("error",category=UserWarning)

def get_lora_pred(data,dependencies):
    '''
    Description:
        获取lora预测结果。
        1.尝试load对应package的lora权重，若没有，则进行训练，获得对应权重
        2.使用训练好的lora权重进行推理
    Args:
        data: dict,数据
        dependencies: list,依赖
    Returns:
        lora_pred: str,lora预测结果
    '''
    config = load_config(LORA_CONFIG_PATH)
    base_model, tokenizer = load_base_model(config.get("model_name"), config.get("device_map"))
    lora_models_path = {}
    model_name = config["model_name"].split("/")[-1]  # Extract model name for path generation
    knowledge_type = config.get("knowledge_type", "docstring")  # Get knowledge type from config
    for pkg,version in dependencies.items():
        try:
            print(f"加载{pkg}的{version}的lora模型")
            lora_model_path = load_lora_model(pkg,version,config,knowledge_type,None)
            lora_models_path[pkg] = lora_model_path
            print(f"加载{pkg}的{version}的lora模型成功")
        except Exception as e:
            print(f"加载{pkg}的{version}的lora模型失败: {str(e)}")
            dataloader = get_dataloader(config,pkg,version,tokenizer)
            # 使用配置中的精度，如果没有设置则默认为fp16
            precision = config.get("precision", "fp16")
            lora_model = buildandTrainLoraModel(config, dataloader, precision, pkg, version, knowledge_type)
            pathConfig = pathConfigurator()
            path = pathConfig.getPath(config,pkg,version,model_name,knowledge_type)
            lora_model.save_pretrained(path)
            lora_models_path[pkg] = path
    try:
        lora_pred = combineLoraWeights_and_predict(data,base_model,tokenizer,lora_models_path,config)
        return {"id":data["id"],"answer":lora_pred}
    except Exception as e:
        print(f"预测过程中出现错误: {str(e)}")
        raise e





def combineLoraWeights_and_predict(data, base_model, tokenizer, lora_models_path, config):
    try:
        # 合并模型权重
        lora_models_path_list = list(lora_models_path.values())
        merged_model = merge_lora_weights(base_model, lora_models_path_list)
        
        # 保存合并后的模型
        save_path_base = config["save_path_base"]
        merged_save_path = f"{save_path_base}/merged_lora_model"
        save_lora_model(merged_model, merged_save_path)
        print(f"合并模型已保存到: {merged_save_path}")
        
        # 构建输入提示
        if config["dataset"] == 'versicode':
            input_prompt = (config.get("versicode_vscc_prompt") 
                          if config["task"] == 'vscc' 
                          else config.get("versicode_vace_prompt"))
            
            if config["task"] == 'vscc':
                input = input_prompt.format(
                    description=data["description"],
                    dependency=data["dependency"],
                    current_version=data["current_version"]
                )
            else:
                input = input_prompt.format(
                    description=data["description"],
                    dependency=data["dependency"]
                )
        elif config["dataset"] == "versiBCB":
            input_prompt = config.get("versiBCB_vace_prompt") if config["task"] == "vace" else config.get("versiBCB_vscc_prompt")
            if config["task"] == "vscc":
                input = input_prompt.format(
                    description=data["description"],
                    dependency=data["dependency"]
                )
            else:
                input = input_prompt.format(
                    description=data["description"],
                    origin_dependency=data["origin_dependency"],
                    origin_code=data["origin_code"],
                    target_dependency=data["target_dependency"]
                )
        else:
            raise ValueError(f"数据集不存在: {config['dataset']}")
        # 加载合并后的模型进行推理
        loaded_model = load_lora_model_withPeft(base_model, merged_save_path)
        
        # 使用更稳定的推理设置
        result = inference(loaded_model, tokenizer, input)
        
        return result
        
    except Exception as e:
        print(f"预测过程中出现错误: {str(e)}")
        raise e
        # return ""  # 返回空字符串作为错误处理

def get_lora_pred_with_cleanup(data, dependencies):
    try:
        return get_lora_pred(data, dependencies)
    except RuntimeError as e:
        print(f"发生错误: {str(e)}")
        try:
            # 尝试清理 GPU 缓存
            torch.cuda.empty_cache()
        except:
            # 如果清理失败，尝试重置 CUDA 设备
            try:
                torch.cuda.reset_device()
            except:
                pass
        # 返回空结果
        return {"id": data["id"], "answer": ""}

def versiBCB_lora_merge_pred():
    config = load_config(LORA_CONFIG_PATH)
    lora_pred_result = []
    benchmark = "Versicode_Benchmark" if config["dataset"] == "versicode" else "VersiBCB_Benchmark"
    filename = 'vscc_datas' if config["task"] == "vscc" else 'vace_datas'
    filename = filename + "_for_warning" if config["removeDeprecationData"] else filename
    
    with open(f"benchmark/data/{benchmark}/{filename}.json", "r") as f:
        datas = json.load(f)
    
    for i,data in tqdm(enumerate(datas)):    
        if i < VSCC_LOW_BOUND or i > VSCC_HIGH_BOUND:
            continue
        if config["dataset"] == "versicode":
            pack = data["dependency"]
            version = get_version(data["version"]) if config["task"] == "vscc" else get_version(data["target_version"])
            dependencies = {pack:version}
            if not getDataExistence(dependencies):
                continue
        elif config["dataset"] == "versiBCB":
            dependencies = data["target_dependency"] if config["task"] == "vace" else data["dependency"]
        else:
            raise ValueError(f"数据集不存在: {config['dataset']}")
            
        try:
            lora_pred = get_lora_pred_with_cleanup(data, dependencies)
            lora_pred_result.append(lora_pred)
        except Exception as e:
            print(f"处理 ID {data['id']} 时发生错误: {str(e)}")
            lora_pred_result.append({"id": data["id"], "answer": ""})
            try:
                torch.cuda.empty_cache()
            except:
                try:
                    torch.cuda.reset_device()
                except:
                    pass
            continue
            
        # 每处理一个样本就保存一次结果
        with open(f"output/{benchmark}/{filename}_lora_pred_result.json", "w", encoding="utf-8") as f:
            json.dump(lora_pred_result, f, ensure_ascii=False)


if __name__ == "__main__":
    # trainLoraModels()
    # trainNormalLoraModels()
    
    # 可以直接调用无参函数，使用默认配置
    # loadModelandPredict()
    
    # 也可以指定参数
    import argparse
    
    parser = argparse.ArgumentParser(description='Run prediction with LoRA model')
    # 现在其实只用上面6项
    parser.add_argument('--base_model_path', type=str, help='Path to the base model')
    parser.add_argument('--adaptor_path', type=str, help='Path to the LoRA model')
    parser.add_argument('--output_path_base', type=str, help='Path to save prediction results')
    parser.add_argument('--output_file', type=str, default='benchmark/results/1.json', help='Path to save prediction results')
    parser.add_argument('--isVersiBCB', action='store_true', help='Whether to run on VersiBCB benchmark')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum number of new tokens to generate')
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp32', 'fp16', 'bf16', 'int8'], help='Precision for model (fp32, fp16, bf16, int8)')
    parser.add_argument('--load_existing_results', action='store_true', help='Whether to load existing results and skip already processed samples')

    parser.add_argument('--only_base_model', action='store_true', help='Whether to only use the base model')


    parser.add_argument('--use_santacoder', action='store_true', help='Whether to use SantaCoder dataset')
    parser.add_argument('--santa_dataset_path', type=str, help='Path to SantaCoder dataset')
    parser.add_argument('--use_versibcb', action='store_true', help='Whether to run on VersiBCB benchmark')
    parser.add_argument('--temperature', type=float, default=0.2, help='Temperature for model')
    args = parser.parse_args()
    
    if True:
        # 执行在VersiBCB上的预测
        loadModelandPredictVersiBench(
            base_model_path=args.base_model_path,
            adaptor_path=args.adaptor_path,
            output_path_base=args.output_path_base,
            output_file=args.output_file, 
            isVersiBCB=args.isVersiBCB,
            only_base_model=args.only_base_model,
            max_new_tokens=args.max_new_tokens,
            precision=args.precision,
            load_existing_results=args.load_existing_results,
            temperature=args.temperature
        )
    elif args.adaptor_path or args.output_file or args.use_santacoder or args.santa_dataset_path:
        loadModelandPredict(
            adaptor_path=args.adaptor_path,
            output_file=args.output_file,
            use_santacoder=args.use_santacoder,
            santa_dataset_path=args.santa_dataset_path
        )
    else:
        # 如果没有提供任何参数，则使用默认设置
        # loadModelandPredict('/datanfs2/chenrongyi/models/versiBCB/checkpoints/checkpoint_20250428_223536_epoch8')
        from benchmark.config.code.config_lora import load_config
        lora_config = load_config(LORA_CONFIG_PATH)
        enable_description = 'desc' if lora_config["enable_description"] else 'nodesc'
        description_instruct_format = 'instruct' if lora_config["description_instruct_format"] else 'noinstruct'
        code_start = 'code' if lora_config["code_start"] else 'nocode'
        loadModelandPredictVersiBench(
            base_model_path='/datanfs2/chenrongyi/models/Llama-2-7b-hf',
            adaptor_path='/datanfs2/chenrongyi/models/starcoder-lora-rank-64-20B-tokens/',
            output_file=f'output/VersiBCB_Benchmark/llama2-7b/vscc_datas_for_token_lora_pred_result_basic.json',
            isVersiBCB=False,
            only_base_model=True
        )