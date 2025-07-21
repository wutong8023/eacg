import torch
import os
from peft import get_peft_model,PeftModel
import json
from benchmark.pred_other import get_version
# from benchmark.config.code.config import CORPUS_PATH
from tqdm import tqdm
import warnings
import argparse
import traceback
import logging
import sys
from datetime import datetime
from utils.loraPathConfigure import pathConfigurator
from utils.getDatasetPacks import getPackVersions
from utils.loraTrain.loraTrainUtils import loraModelExists,get_dataloader,buildandTrainLoraModel,getDataExistence,load_lora_model,load_config,create_lora_config,save_lora_model,load_lora_model_withPeft,load_base_model,merge_lora_weights,inference
from utils.loraTrain.loraMerge import uniform_lora_merging
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
from utils.getDependencyUtils import dict_to_pkg_ver_tuples
from utils.output_manager import OutputManager
from omegaconf import OmegaConf
from utils.iftModelManager import get_default_manager as get_ift_manager
from utils.iftCardManager import get_default_card_manager
# warnings.filterwarnings("error",category=UserWarning)

def setup_logging(args):
    """
    设置日志配置，创建带时间戳的日志文件夹
    """
    # 创建日志根目录
    log_base_dir = "logs"
    os.makedirs(log_base_dir, exist_ok=True)
    
    # 创建带时间戳的日志子目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_info = f"{args.task}_{args.knowledge_type}"
    if hasattr(args, 'rank') and args.world_size > 1:
        log_dir = os.path.join(log_base_dir, f"pred_lora_{task_info}_{timestamp}_rank{args.rank}")
    else:
        log_dir = os.path.join(log_base_dir, f"pred_lora_{task_info}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置日志文件路径
    log_file = os.path.join(log_dir, f"pred_lora_{task_info}.log")
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # 记录配置信息
    logging.info(f"日志保存到: {log_file}")
    logging.info(f"预测配置: task={args.task}, knowledge_type={args.knowledge_type}")
    logging.info(f"model_name={args.model_name}")
    logging.info(f"precision={args.precision}")
    logging.info(f"max_dependency_num={args.max_dependency_num}")
    logging.info(f"loraadaptor_save_path_base={args.loraadaptor_save_path_base}")
    if hasattr(args, 'rank'):
        logging.info(f"多worker模式: rank={args.rank}, world_size={args.world_size}")
    
    return log_dir

def get_lora_pred(data,dependencies,args):
    '''
    Description:
        获取lora预测结果。
        1.尝试load对应package的lora权重，若没有，则进行训练，获得对应权重
        2.使用训练好的lora权重进行推理
    Args:
        data: dict,数据
        dependencies: list,依赖
        args: 参数对象，包含所有必要的配置信息
    Returns:
        dict: 包含id、answer和adaptor加载信息的字典
    '''
    # 记录是否启用了IFT checkpoint功能
    adopt_ift = getattr(args, 'adopt_IFT_checkpoint', False)
    if adopt_ift:
        logging.info("🚀 IFT Checkpoint模式已启用 - 将优先加载IFT模型")
    else:
        logging.info("📝 标准LoRA模式 - 将加载普通LoRA模型")
    
    # 从args中获取配置信息，而不是外部加载config
    # 检查是否有强制GPU选项
    force_gpu = getattr(args, 'force_gpu', False)
    base_model, tokenizer = load_base_model(args.model_name, args.device_map, args.precision, force_gpu=force_gpu)
    lora_models_path = {}
    attempted_packages = []
    successful_packages = []
    failed_packages = []
    
    for pkg,version in dict_to_pkg_ver_tuples(dependencies):
        attempted_packages.append(f"{pkg}-{version}")
        try:
            logging.info(f"加载{pkg}的{version}的lora模型")
            # 使用args中的配置信息
            # config_dict = {
            #     "model_name": args.model_name,
            #     "save_path_base": args.adaptor_base_path,
            #     "device_map": args.device_map
            # }
            lora_model_path = load_lora_model(pkg, version, args.lora_config, args.knowledge_type, args)
            if pkg not in lora_models_path:
                lora_models_path[pkg] = [lora_model_path]
            else:
                lora_models_path[pkg].append(lora_model_path)
            successful_packages.append(f"{pkg}-{version}")
            logging.info(f"成功加载{pkg}的{version}的lora模型: {lora_model_path}")
        except Exception as e:
            failed_packages.append(f"{pkg}-{version}")
            logging.error(f"加载{pkg}的{version}的lora模型失败: {str(e)}")
            traceback.print_exc()
            continue
    
    # 记录加载统计信息
    total_attempted = len(attempted_packages)
    total_successful = len(successful_packages)
    total_failed = len(failed_packages)
    
    logging.info(f"LoRA package loading summary:")
    logging.info(f"  Attempted: {total_attempted} packages: {attempted_packages}")
    logging.info(f"  Successful: {total_successful} packages: {successful_packages}")
    if failed_packages:
        logging.warning(f"  Failed: {total_failed} packages: {failed_packages}")
    
    # 准备adaptor加载信息
    adaptor_info = {
        "attempted_packages": attempted_packages,
        "successful_packages": successful_packages,
        "failed_packages": failed_packages,
        "total_attempted": total_attempted,
        "total_successful": total_successful,
        "total_failed": total_failed
    }
    
    # if len(lora_models_path)<1: 
    #     print("No LoRA models loaded successfully, returning empty answer")
    #     return {
    #         "id": data["id"],
    #         "answer": "",
    #         "adaptor_info": adaptor_info,
    #         "merged_adaptor_count": 0
    #     }
    try:
        lora_pred, merged_count = combineLoraWeights_and_predict(data, base_model, tokenizer, lora_models_path, args)
        return {
            "id": data["id"],
            "answer": lora_pred,
            "adaptor_info": adaptor_info,
            "merged_adaptor_count": merged_count
        }
    except Exception as e:
        logging.error(f"预测过程中出现错误: {str(e)}")
        traceback.print_exc()
        raise e





def combineLoraWeights_and_predict(data, base_model, tokenizer, lora_models_path, args):
    '''
    1.合并模型权重
    2.构建输入提示
    3.加载合并后的模型进行推理
    4.清理临时合并的模型，节省磁盘空间
    Args:
        data: dict,数据
        base_model: 基础模型
    constraints:
        对于lora_models_path，如果为空的情况，会直接启用base_model进行推理
    '''
    try:
        # 合并模型权重
        # 展平路径列表，因为lora_models_path[pkg]是一个包含路径的列表
        lora_models_path_list = []
        total_requested_paths = 0
        for pkg, paths in lora_models_path.items():
            if isinstance(paths, list):
                lora_models_path_list.extend(paths)
                total_requested_paths += len(paths)
            else:
                lora_models_path_list.append(paths)
                total_requested_paths += 1
                
        logging.info(f"Attempting to merge {total_requested_paths} LoRA adapters for packages: {list(lora_models_path.keys())}")
 
        # merged_model = merge_lora_weights(base_model, lora_models_path_list)

        # # 保存合并后的模型
        # save_path_base = args.loraadaptor_save_path_base
        # merged_save_path = f"{save_path_base}/merged_lora_model"
        # save_lora_model(merged_model, merged_save_path)
        # print(f"合并模型已保存到: {merged_save_path}")

        save_path_base = args.loraadaptor_save_path_base
        # 添加rank_id到merged model路径，避免多worker冲突
        rank = getattr(args, 'rank', 0)
        if args.tempmerged_adaptor_path is None:
            merged_save_path = f"{save_path_base}/merged_lora_model_rank{rank}"
        else:
            merged_save_path = f"{args.tempmerged_adaptor_path}/merged_lora_model_rank{rank}"
        count = uniform_lora_merging(lora_models_path_list, merged_save_path, 'cuda')
        
        logging.info(f"Successfully merged {count}/{total_requested_paths} LoRA adapters")
        
        if count == 0:
            logging.warning("No adapters were successfully merged, using base model")
        elif count < total_requested_paths:
            logging.warning(f"Only {count} out of {total_requested_paths} adapters were successfully merged")

        # 构建输入提示
        if args.dataset == 'versicode':
            raise ValueError("versicode not supported")
            # input_prompt = (args.versicode_vscc_prompt 
            #               if args.task == 'vscc' 
            #               else args.versicode_vace_prompt)
            
            # if args.task == 'vscc':
            #     input = input_prompt.format(
            #         description=data["description"],
            #         dependency=data["dependency"],
            #         current_version=data["current_version"]
            #     )
            # else:
            #     input = input_prompt.format(
            #         description=data["description"],
            #         dependency=data["dependency"]
            #     )
        #TODO:prompt目前是临时之策，需要稍等后换回ban_deprecation的prompt
        elif args.dataset == "versiBCB":
            # if args.Ban_Deprecation:
            #     input_prompt = args.versiBCB_vace_bd_prompt if args.task == "vace" else args.versiBCB_vscc_bd_prompt
            # else:
            input_prompt = args.versiBCB_vace_prompt if args.task == "vace" else args.versiBCB_vscc_prompt

                
            if args.task == "vscc":
                input = input_prompt.format(
                    description=data["description"],
                    dependency=data["dependency"] if "true_dependency" not in data else data["true_dependency"]
                )
            else:
                input = input_prompt.format(
                    description=data["description"],
                    origin_dependency=data["origin_dependency"],
                    origin_code=data["origin_code"],
                    target_dependency=data["target_dependency"] if "true_target_dependency" not in data else data["true_target_dependency"]
                )
        else:
            raise ValueError(f"数据集不存在: {args.dataset}")
        
        # 加载合并后的模型进行推理
        if count == 0:
            loaded_model = base_model
            logging.info("Using base model for inference (no adapters loaded)")
        else:
            try:
                loaded_model = PeftModel.from_pretrained(base_model, merged_save_path)
                logging.info(f"Using merged LoRA model with {count} adapters for inference from {merged_save_path}")
            except Exception as load_error:
                logging.error(f"Error loading merged LoRA model from {merged_save_path}: {load_error}")
                logging.info("Falling back to base model for inference")
                loaded_model = base_model
                count = 0  # Reset count since we're using base model
        
        # 使用args中的推理设置
        result = inference(loaded_model, tokenizer, input, max_new_tokens=1024, 
                         temperature=args.temperature, 
                         top_p=args.top_p)
        
        # 清理临时合并的模型文件，节省磁盘空间
        if count > 0:
            try:
                import shutil
                if os.path.exists(merged_save_path):
                    shutil.rmtree(merged_save_path)
                    logging.info(f"Cleaned up temporary merged model at {merged_save_path}")
            except Exception as cleanup_error:
                logging.warning(f"Warning: Failed to cleanup merged model at {merged_save_path}: {cleanup_error}")
        
        return result, count
        
    except Exception as e:
        logging.error(f"预测过程中出现错误: {str(e)}")
        traceback.print_exc()
        raise e

def get_lora_pred_with_cleanup(data, dependencies,args):
    try:
        return get_lora_pred(data, dependencies,args)
    except RuntimeError as e:
        logging.error(f"发生错误: {str(e)}")
        traceback.print_exc()
        try:
            # 尝试清理 GPU 缓存
            torch.cuda.empty_cache()
        except:
            # 如果清理失败，尝试重置 CUDA 设备
            try:
                torch.cuda.reset_device()
            except:
                pass
        # 返回空结果，包含adaptor信息
        return {
            "id": data["id"], 
            "answer": "",
            "adaptor_info": {
                "attempted_packages": [],
                "successful_packages": [],
                "failed_packages": [],
                "total_attempted": 0,
                "total_successful": 0,
                "total_failed": 0,
                "error": str(e)
            },
            "merged_adaptor_count": 0
        }

def load_model_from_ift_meta_config(card_id_or_path, args):
    """
    基于IFT元配置卡片加载模型配置，但具体的预测数据由args.task确定
    
    Args:
        card_id_or_path: str, IFT元配置卡片ID或文件路径
        args: argparse参数，包含task等信息
        
    Returns:
        dict: 包含模型配置的字典
    """
    try:
        card_manager = get_default_card_manager()
        
        # 加载IFT卡片
        if card_id_or_path.endswith('.yaml') or card_id_or_path.endswith('.yml'):
            ift_card = card_manager.load_ift_card(card_id_or_path)
        else:
            ift_card = card_manager.get_card_by_id(card_id_or_path)
        
        if not ift_card:
            raise ValueError(f"找不到IFT卡片: {card_id_or_path}")
        
        # 验证卡片
        if not card_manager.validate_card(ift_card):
            raise ValueError("IFT配置卡片无效")
        
        # 提取预测配置
        prediction_config = card_manager.extract_prediction_config(ift_card)
        
        # 检查任务兼容性
        compatible_tasks = prediction_config.get("compatible_tasks", ["vace", "vscc"])
        if args.task not in compatible_tasks:
            logging.warning(f"任务 {args.task} 可能与此IFT配置不兼容，支持的任务: {compatible_tasks}")
        
        # 应用IFT配置到args（元配置方式）
        args.model_name = prediction_config["model_name"]
        if prediction_config.get("tokenizer_name"):
            args.tokenizer_name = prediction_config["tokenizer_name"]
        args.knowledge_type = prediction_config["knowledge_type"]
        args.precision = prediction_config["precision"]
        args.adopt_IFT_checkpoint = True  # 启用IFT模式
        args.ift_type = prediction_config["ift_type"]
        args.ift_data_strategy = prediction_config["data_strategy"]
        args.device_map = prediction_config.get("device_strategy", "auto")
        
        # 从路径策略中推断LoRA基础路径
        path_strategy = prediction_config.get("path_strategy", {})
        lora_base_path = path_strategy.get("lora_base_path_pattern")
        if lora_base_path and (not hasattr(args, 'loraadaptor_save_path_base') or args.loraadaptor_save_path_base is None):
            args.loraadaptor_save_path_base = lora_base_path
        
        # 保存IFT配置信息供后续使用
        args.ift_card_config = prediction_config
        
        logging.info(f"✅ 已加载IFT元配置: {prediction_config['card_id']}")
        logging.info(f"🎯 IFT类型: {prediction_config['ift_type']}")
        logging.info(f"📊 数据策略: {prediction_config['data_strategy']}")
        logging.info(f"🏷️ 知识类型: {prediction_config['knowledge_type']}")
        logging.info(f"🎮 当前任务: {args.task}")
        
        return prediction_config
        
    except Exception as e:
        logging.error(f"加载IFT元配置失败: {e}")
        raise


def apply_ift_config_to_lora_training_args(prediction_config, args):
    """
    将IFT元配置应用到LoRA训练参数中
    
    Args:
        prediction_config: dict, 从IFT卡片提取的预测配置
        args: argparse参数对象
    """
    # 应用LoRA配置
    lora_config = prediction_config.get("lora_config", {})
    if lora_config:
        args.lora_r = lora_config.get("r", 64)
        args.lora_alpha = lora_config.get("alpha", 128)
        args.lora_dropout = lora_config.get("dropout", 0.05)
        args.lora_target_modules = lora_config.get("target_modules", [])
        
        logging.info(f"应用LoRA配置: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    
    # 应用IFT训练配置
    ift_training_config = prediction_config.get("ift_training_config", {})
    if ift_training_config:
        args.ift_learning_rate = ift_training_config.get("learning_rate", 1e-6)
        args.ift_num_epochs = ift_training_config.get("num_epochs", 16)
        args.ift_batch_size = ift_training_config.get("batch_size", 8)
        
        logging.info(f"应用IFT训练配置: lr={args.ift_learning_rate}, epochs={args.ift_num_epochs}, batch_size={args.ift_batch_size}")
    
    # 应用数据源配置
    data_sources = prediction_config.get("data_sources", {})
    if data_sources:
        args.ift_data_sources = data_sources
        logging.info(f"应用数据源配置: {data_sources.keys()}")

def apply_task2maskpacks_filter(dependency, task_id, task2maskpacks):
    """
    根据task2maskpacks配置过滤dependency
    
    Args:
        dependency: dict, 原始依赖字典 {package: version}
        task_id: str/int, 任务ID
        task2maskpacks: dict, 任务ID到掩码包列表的映射
        
    Returns:
        dict: 过滤后的依赖字典
    """
    if task2maskpacks is None or not dependency:
        return dependency
    
    # 将task_id转换为字符串以匹配JSON文件中的格式
    task_id_str = str(task_id)
    
    # 如果该task_id不在掩码配置中，返回原始dependency
    if task_id_str not in task2maskpacks:
        logging.debug(f"Task ID {task_id} 不在掩码配置中，保持原始dependency")
        return dependency
    
    # 获取该任务应该保留的包列表
    mask_packages = task2maskpacks[task_id_str]
    
    # 如果掩码包列表为空，返回空的dependency
    if not mask_packages:
        logging.info(f"Task ID {task_id} 的掩码包列表为空，返回空dependency")
        return {}
    
    # 过滤dependency，只保留在掩码列表中的包
    filtered_dependency = {}
    for package, version in dependency.items():
        if package in mask_packages:
            filtered_dependency[package] = version
        else:
            logging.debug(f"Task ID {task_id}: 过滤掉包 {package} (不在掩码列表中)")
    
    logging.info(f"Task ID {task_id}: 应用掩码过滤，{len(dependency)} -> {len(filtered_dependency)} 个包")
    logging.debug(f"Task ID {task_id}: 原始包: {list(dependency.keys())}")
    logging.debug(f"Task ID {task_id}: 掩码包: {mask_packages}")
    logging.debug(f"Task ID {task_id}: 过滤后包: {list(filtered_dependency.keys())}")
    
    return filtered_dependency


def versiBCB_lora_merge_pred(args, task="vace", removeDeprecationData=False):
    from omegaconf import OmegaConf
    import fcntl
    
    # 获取rank和world_size参数，默认为单worker模式
    rank = getattr(args, 'rank', 0)
    world_size = getattr(args, 'world_size', 1)
    
    logging.info(f"Worker {rank}/{world_size} starting task: {task}, removeDeprecation: {removeDeprecationData}")
    
    # 加载task2maskpacks掩码数据（如果启用）
    task2maskpacks = None
    valid_task_ids = None
    task_id_filter_set = None
    
    if getattr(args, 'enable_task2maskpacks', False):
        task2maskpacks_file = getattr(args, 'task2maskpacks_file', 'data/task2maskpacks.json')
        try:
            with open(task2maskpacks_file, 'r', encoding='utf-8') as f:
                task2maskpacks = json.load(f)
            logging.info(f"✅ 成功加载task2maskpacks数据: {len(task2maskpacks)} 个任务掩码配置")
            
            # 如果启用了only_task2maskpacks_ids选项，提取有效的task id列表
            if getattr(args, 'only_task2maskpacks_ids', False):
                # 将字符串task id转换为整数，用于后续过滤
                valid_task_ids = set()
                for task_id_str in task2maskpacks.keys():
                    try:
                        task_id_int = int(task_id_str)
                        valid_task_ids.add(task_id_int)
                    except ValueError:
                        logging.warning(f"⚠️ 跳过无效的task id: {task_id_str} (无法转换为整数)")
                
                logging.info(f"🔍 提取到 {len(valid_task_ids)} 个有效的task ids用于过滤")
                logging.debug(f"有效task ids示例: {sorted(list(valid_task_ids))[:10]}")
                
        except Exception as e:
            logging.error(f"❌ 加载task2maskpacks文件失败 {task2maskpacks_file}: {e}")
            logging.info("🔄 将继续执行但不应用掩码")
            task2maskpacks = None
            valid_task_ids = None
    else:
        logging.info("📝 task2maskpacks掩码功能未启用")
    
    # 加载task id过滤文件（如果启用）
    if getattr(args, 'enable_task_id_filter', False):
        task_id_filter_file = getattr(args, 'task_id_filter_file')
        if task_id_filter_file is None:
            logging.error("❌ --enable_task_id_filter 已启用，但 --task_id_filter_file 未指定")
            raise ValueError("必须指定 --task_id_filter_file 路径")
        
        try:
            with open(task_id_filter_file, 'r', encoding='utf-8') as f:
                task_id_filter_data = json.load(f)
            
            # 确保数据是列表格式
            if isinstance(task_id_filter_data, list):
                task_id_filter_set = set(task_id_filter_data)
            else:
                logging.error(f"❌ task id过滤文件格式错误，期望列表格式，得到: {type(task_id_filter_data)}")
                raise ValueError("task id过滤文件必须是整数列表格式")
            
            logging.info(f"✅ 成功加载task id过滤文件: {task_id_filter_file}")
            logging.info(f"🔍 过滤task ids数量: {len(task_id_filter_set)}")
            logging.debug(f"过滤task ids示例: {sorted(list(task_id_filter_set))[:10]}")
            
        except Exception as e:
            logging.error(f"❌ 加载task id过滤文件失败 {task_id_filter_file}: {e}")
            raise
    else:
        logging.info("📝 task id过滤功能未启用")
    
    # 如果指定了IFT元配置卡片，则使用元配置模式
    if hasattr(args, 'ift_card') and args.ift_card:
        logging.info(f"🚀 使用IFT元配置模式: {args.ift_card}")
        
        try:
            # 加载IFT元配置
            prediction_config = load_model_from_ift_meta_config(args.ift_card, args)
            
            # 应用IFT配置到训练参数
            apply_ift_config_to_lora_training_args(prediction_config, args)
            
            # 从IFT配置加载lora_config（如果没有通过其他方式指定）
            if not hasattr(args, 'lora_config') or args.lora_config is None:
                # 🔧 修复：首先加载基础配置文件作为默认配置
                logging.info("📝 加载基础LoRA配置文件作为默认配置")
                base_lora_config = OmegaConf.load(LORA_CONFIG_PATH)
                
                # 然后用命令行参数覆盖基础配置
                base_lora_config.temperature = args.temperature
                base_lora_config.top_p = args.top_p
                base_lora_config.max_dependency_num = args.max_dependency_num
                base_lora_config.append_srcDep = args.append_srcDep
                
                # 用IFT卡片配置覆盖基础配置中的相应字段
                if prediction_config.get("data_sources", {}).get("dataset"):
                    base_lora_config.dataset = prediction_config["data_sources"]["dataset"]
                else:
                    base_lora_config.dataset = "versiBCB"
                    
                # 确保model_name字段存在
                base_lora_config.model_name = args.model_name
                
                # 确保关键路径配置存在
                if args.loraadaptor_save_path_base:
                    base_lora_config.loraadaptor_save_path_base = args.loraadaptor_save_path_base
                if args.device_map:
                    base_lora_config.device_map = args.device_map
                if args.knowledge_type:
                    base_lora_config.knowledge_type = args.knowledge_type
                if args.precision:
                    base_lora_config.precision = args.precision
                # 从IFT卡片获取prompt配置（如果存在的话）
                prompt_configs = {
                    "versiBCB_vace_prompt": prediction_config.get("data_sources", {}).get("versiBCB_vace_prompt", ""),
                    "versiBCB_vscc_prompt": prediction_config.get("data_sources", {}).get("versiBCB_vscc_prompt", ""),
                    "versiBCB_vscc_bd_prompt": prediction_config.get("data_sources", {}).get("versiBCB_vscc_bd_prompt", ""),
                    "versiBCB_vace_bd_prompt": prediction_config.get("data_sources", {}).get("versiBCB_vace_bd_prompt", ""),
                    "versicode_vscc_prompt": prediction_config.get("data_sources", {}).get("versicode_vscc_prompt", ""),
                    "versicode_vace_prompt": prediction_config.get("data_sources", {}).get("versicode_vace_prompt", "")
                }
                
                # 只有当IFT卡片中有具体的prompt配置时才覆盖，否则保持基础配置
                for prompt_key, prompt_value in prompt_configs.items():
                    if prompt_value:  # 只有非空值才覆盖
                        setattr(base_lora_config, prompt_key, prompt_value)
                
                args.lora_config = base_lora_config
                logging.info("✅ 基础配置与IFT配置合并完成")
            
            logging.info("✅ IFT元配置应用完成")
            
        except Exception as e:
            logging.error(f"❌ 应用IFT元配置失败: {e}")
            logging.info("🔄 回退到标准配置模式")
            # 回退到标准配置加载
            lora_config = OmegaConf.load(LORA_CONFIG_PATH)
            lora_config.temperature = args.temperature
            lora_config.top_p = args.top_p
            lora_config.max_dependency_num = args.max_dependency_num
            lora_config.append_srcDep = args.append_srcDep
            lora_config.model_name = args.model_name
            args.lora_config = lora_config
    else:
        # 标准配置模式
        logging.info("📝 使用标准LoRA配置模式")
        
        # 加载 lora 配置
        lora_config = OmegaConf.load(LORA_CONFIG_PATH)
        
        # 将命令行参数合并到配置中
        lora_config.temperature = args.temperature
        lora_config.top_p = args.top_p
        lora_config.max_dependency_num = args.max_dependency_num
        lora_config.append_srcDep = args.append_srcDep
        lora_config.model_name = args.model_name
        # 添加 lora_config 到 args 中
        args.lora_config = lora_config
    
    # 转换为普通字典用于后续处理
    config = OmegaConf.to_container(args.lora_config, resolve=True)
    
    # 将所有必要的配置信息添加到args中，使prediction过程完全依赖args
    args.dataset = config.get("dataset", "versiBCB")
    args.task = task
    args.Ban_Deprecation = removeDeprecationData
    args.lora_config_path = LORA_CONFIG_PATH
    
    # 如果args中没有设置loraadaptor_save_path_base或为None，则使用config中的值
    if not hasattr(args, 'loraadaptor_save_path_base') or args.loraadaptor_save_path_base is None:
        args.loraadaptor_save_path_base = config.get("loraadaptor_save_path_base", "/datanfs2/chenrongyi/models/loraadaptors/docstring/")
    
    if not hasattr(args, 'device_map') or args.device_map is None:
        args.device_map = config.get("device_map", "auto")
    
    # 如果命令行没有指定knowledge_type，则使用配置文件中的值
    if not hasattr(args, 'knowledge_type') or args.knowledge_type is None:
        args.knowledge_type = config.get("knowledge_type", "doc")
    
    # 添加prompt信息到args
    args.versiBCB_vace_prompt = config.get("versiBCB_vace_prompt", "")
    args.versiBCB_vscc_prompt = config.get("versiBCB_vscc_prompt", "")
    args.versiBCB_vscc_bd_prompt = config.get("versiBCB_vscc_bd_prompt", "")
    args.versiBCB_vace_bd_prompt = config.get("versiBCB_vace_bd_prompt", "")
    args.versicode_vscc_prompt = config.get("versicode_vscc_prompt", "")
    args.versicode_vace_prompt = config.get("versicode_vace_prompt", "")
    
    # 提取LoRA card info用于文件名
    lora_cardinfo = {
        'r': config.get('r', 64),
        'alpha': config.get('alpha', 128),
        'lr': config.get('learning_rate', 1e-6),
        'bs': config.get('batch_size', 8),
        'epochs': config.get('num_epochs', 4)
    }
    
    # 根据args.task确定要加载的数据（而不是从IFT卡片中获取）
    benchmark = "Versicode_Benchmark" if args.dataset == "versicode" else "VersiBCB_Benchmark"
    filename = 'vscc_datas' if task == "vscc" else 'vace_datas'
    filename = filename + "_for_warning" if removeDeprecationData else filename
    
    if args.specified_bench_path is None:
        with open(f"data/{benchmark}/{filename}.json", "r") as f:
            datas = json.load(f)
    else:
        with open(args.specified_bench_path, "r") as f:
            datas = json.load(f)
    
    logging.info(f"📊 加载数据集: {benchmark}/{filename}.json")
    logging.info(f"📈 数据条目数量: {len(datas)}")
    
    # 创建输出管理器
    output_manager = OutputManager(args.base_output_dir)
    
    # 生成基础文件名（包含IFT信息如果使用了IFT配置）
    dataset_name = f"{args.dataset}_{task}{'_BD' if removeDeprecationData else ''}"
    model_name = args.model_name.split('/')[-1]
    
    # 如果使用了IFT配置，在文件名中体现
    approach_name = "LoRA"
    if hasattr(args, 'ift_card_config') and args.ift_card_config:
        ift_type = args.ift_card_config.get('ift_type', 'default')
        data_strategy = args.ift_card_config.get('data_strategy', 'default')
        approach_name = f"LoRA_IFT_{ift_type}_{data_strategy}"
        logging.info(f"🏷️ 使用IFT增强方法: {approach_name}")
    
    base_filename = output_manager.generate_base_filename(
        dataset=dataset_name,
        model_name=model_name,
        approach=approach_name,
        corpus_type=args.knowledge_type,
        max_dependency_num=args.max_dependency_num,
        append_srcDep=args.append_srcDep,
        lora_cardinfo=lora_cardinfo
    )
    
    # 获取输出路径和配置路径
    output_path, config_path = output_manager.get_output_path_and_config(
        approach="LoRA",
        base_filename=base_filename,
        model_name_or_path=args.model_name,
        newWhenExist=args.newWhenExist
    )
    if args.specified_output_path is not None:
        output_path = args.specified_output_path
    if args.specified_config_path is not None:
        config_path = args.specified_config_path
    
    # 处理已存在结果的情况 - 改进的逻辑
    processed_ids = set()
    
    # 步骤1：加载已有结果
    if os.path.exists(output_path) and not args.overwrite:
        logging.info(f"Found existing output file: {output_path}")
        logging.info("Loading existing results to avoid duplicates...")
        
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            result = json.loads(line)
                            if "id" in result:
                                processed_ids.add(result["id"])
                        except json.JSONDecodeError as e:
                            logging.warning(f"Warning: Invalid JSON on line {line_num}: {e}")
                            continue
            
            logging.info(f"Found {len(processed_ids)} already processed IDs")
            
        except Exception as e:
            logging.error(f"Error loading existing results: {e}")
            processed_ids = set()
    
    elif args.overwrite and os.path.exists(output_path):
        logging.info(f"Worker {rank}: Overwriting existing file: {output_path}")
        # 只有rank 0 负责清空文件，其他worker等待
        if rank == 0:
            open(output_path, 'w').close()
            logging.info(f"Worker {rank}: File cleared")
        else:
            # 其他worker等待一下，确保文件被清空
            import time
            time.sleep(2)
            logging.info(f"Worker {rank}: Waiting for file to be cleared...")
    
    # 步骤2：过滤出需要处理的数据（在VSCC范围内且未被处理）
    unprocessed_data = []
    skipped_by_task2maskpacks_filter = 0
    skipped_by_task_id_filter = 0
    
    for i, data in enumerate(datas):
        if VSCC_LOW_BOUND <= i <= VSCC_HIGH_BOUND:
            if data["id"] not in processed_ids:
                # 如果启用了only_task2maskpacks_ids选项，检查该task id是否在有效列表中
                if valid_task_ids is not None:
                    task_id = data.get("id")
                    if task_id not in valid_task_ids:
                        logging.debug(f"构建阶段跳过Task ID {task_id}: 不在task2maskpacks文件中")
                        skipped_by_task2maskpacks_filter += 1
                        continue
                
                # 如果启用了task id过滤，检查该task id是否在过滤列表中
                if task_id_filter_set is not None:
                    task_id = data.get("id")
                    if task_id not in task_id_filter_set:
                        logging.debug(f"构建阶段跳过Task ID {task_id}: 不在task id过滤文件中")
                        skipped_by_task_id_filter += 1
                        continue
                
                unprocessed_data.append((i, data))

    logging.info(f"Total data in range [{VSCC_LOW_BOUND}, {VSCC_HIGH_BOUND}]: {VSCC_HIGH_BOUND - VSCC_LOW_BOUND + 1}")
    logging.info(f"Already processed: {len(processed_ids)}")
    if valid_task_ids is not None:
        logging.info(f"Skipped by task2maskpacks filter: {skipped_by_task2maskpacks_filter}")
        logging.info(f"Valid task2maskpacks IDs count: {len(valid_task_ids)}")
    if task_id_filter_set is not None:
        logging.info(f"Skipped by task id filter: {skipped_by_task_id_filter}")
        logging.info(f"Task id filter count: {len(task_id_filter_set)}")
    logging.info(f"Remaining unprocessed: {len(unprocessed_data)}")
    
    # 步骤3：对未处理的数据进行均匀分配
    if world_size > 1 and len(unprocessed_data) > 0:
        # 按rank分片未处理的数据
        total_unprocessed = len(unprocessed_data)
        samples_per_worker = total_unprocessed // world_size
        remainder = total_unprocessed % world_size
        
        # 计算当前worker的数据范围
        start_idx = rank * samples_per_worker + min(rank, remainder)
        end_idx = start_idx + samples_per_worker + (1 if rank < remainder else 0)
        
        worker_data = unprocessed_data[start_idx:end_idx]
        logging.info(f"Worker {rank} processing {len(worker_data)} unprocessed samples (global indices {start_idx}-{end_idx-1} of unprocessed data)")
        
        if len(worker_data) == 0:
            logging.info(f"Worker {rank} has no data to process, exiting...")
            return 0, 0
    else:
        # 单worker模式或没有未处理数据
        worker_data = unprocessed_data
        if world_size == 1:
            logging.info(f"Single worker processing {len(worker_data)} unprocessed samples")
        else:
            logging.info(f"No unprocessed data remaining for workers to handle")
            
        if len(worker_data) == 0:
            logging.info(f"No unprocessed data, task completed")
            return 0, 0
    
    # 辅助函数：安全地追加到文件
    def safe_append_to_file(file_path, data):
        """使用文件锁安全地追加数据到文件"""
        try:
            with open(file_path, "a", encoding="utf-8") as f:
                # 获取文件锁
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(data, f, ensure_ascii=False)
                    f.write('\n')
                    f.flush()  # 确保数据写入磁盘
                finally:
                    # 释放文件锁
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return True
        except Exception as e:
            logging.error(f"Worker {rank}: Error writing to file: {e}")
            return False
    
    def append2jsonl(file_path,data):
        try:
            with open(file_path,'a+',encoding='utf-8') as f:
                json.dump(data,f,ensure_ascii=False)
                f.write('\n')
                f.flush()
            logging.info(f"successful write to {file_path}")
            return True
            
        except Exception as e:
            logging.error(f"Worker {rank}: Error writing to file: {e}")
            return False
        
    # 处理分配给当前worker的数据
    processed_count = 0
    lora_pred_result = []
    
    for original_idx, data in tqdm(worker_data, desc=f"Worker {rank}"):
        if args.dataset == "versicode":
            pack = data["dependency"]
            version = get_version(data["version"]) if args.task == "vscc" else get_version(data["target_version"])
            dependency = {pack:version}
            if not getDataExistence(dependency):
                continue
            
            # 应用task2maskpacks掩码过滤
            if task2maskpacks is not None:
                original_dependency = dependency.copy()
                dependency = apply_task2maskpacks_filter(dependency, data["id"], task2maskpacks)
                
                # 记录掩码应用情况
                if len(dependency) != len(original_dependency):
                    logging.info(f"Task ID {data['id']}: 掩码过滤生效，{len(original_dependency)} -> {len(dependency)} 个包")
        elif args.dataset == "versiBCB":
            dependency = data["target_dependency"] if task == "vace" else data["dependency"]
            src_dependency = data["origin_dependency"] if task == "vace" else data["dependency"]
            from utils.getDependencyUtils import getSubsetDep,combineDep
            if args.max_dependency_num is not None:
                dependency = getSubsetDep(dependency, args.max_dependency_num)
                src_dependency = getSubsetDep(src_dependency, args.max_dependency_num)
            if args.append_srcDep:
                if task=='vscc':
                    raise ValueError("vscc任务不支持添加源依赖")
                dependency = combineDep(dependency,src_dependency)
            
            # 应用task2maskpacks掩码过滤
            if task2maskpacks is not None:
                original_dependency = dependency.copy()
                dependency = apply_task2maskpacks_filter(dependency, data["id"], task2maskpacks)
                
                # 记录掩码应用情况
                if len(dependency) != len(original_dependency):
                    logging.info(f"Task ID {data['id']}: 掩码过滤生效，{len(original_dependency)} -> {len(dependency)} 个包")
        else:
            raise ValueError(f"数据集不存在: {args.dataset}")
            
        try:
            lora_pred = get_lora_pred_with_cleanup(data, dependency, args)
            processed_count += 1
            
            # 安全地追加到共享文件
            if append2jsonl(output_path, lora_pred):
                # 加入到内存列表用于统计
                lora_pred_result.append(lora_pred)
                # 更新已处理ID集合
                processed_ids.add(data["id"])
            else:
                logging.error(f"Worker {rank}: Failed to write result for ID {data['id']}")
                
            # 每处理10个样本报告一次进度
            if processed_count % 10 == 0:
                logging.info(f"Worker {rank}: Processed {processed_count}/{len(worker_data)} samples")
                
        except Exception as e:
            logging.error(f"Worker {rank}: 处理 ID {data['id']} 时发生错误: {str(e)}")
            traceback.print_exc()
            error_result = {
                "id": data["id"], 
                "answer": "",
                "adaptor_info": {
                    "attempted_packages": [],
                    "successful_packages": [],
                    "failed_packages": [],
                    "total_attempted": 0,
                    "total_successful": 0,
                    "total_failed": 0,
                    "error": f"Processing error: {str(e)}"
                },
                "merged_adaptor_count": 0
            }
            processed_count += 1
            continue
    
    # 计算当前worker的adaptor使用统计
    total_samples = len(lora_pred_result)
    samples_with_adaptors = 0
    total_successful_adaptors = 0
    total_attempted_adaptors = 0
    samples_with_full_success = 0
    
    for result in lora_pred_result:
        if "adaptor_info" in result:
            adaptor_info = result["adaptor_info"]
            total_attempted_adaptors += adaptor_info.get("total_attempted", 0)
            total_successful_adaptors += adaptor_info.get("total_successful", 0)
            if adaptor_info.get("total_successful", 0) > 0:
                samples_with_adaptors += 1
            if (adaptor_info.get("total_attempted", 0) > 0 and 
                adaptor_info.get("total_successful", 0) == adaptor_info.get("total_attempted", 0)):
                samples_with_full_success += 1
    
    logging.info(f"\n=== Worker {rank} LoRA Adaptor Usage Statistics ===")
    logging.info(f"Worker {rank} processed samples: {total_samples}")
    logging.info(f"Samples with at least 1 adaptor loaded: {samples_with_adaptors}/{total_samples}")
    logging.info(f"Samples with all adaptors loaded successfully: {samples_with_full_success}/{total_samples}")
    logging.info(f"Total adaptors attempted: {total_attempted_adaptors}")
    logging.info(f"Total adaptors loaded successfully: {total_successful_adaptors}/{total_attempted_adaptors}")
    if total_attempted_adaptors > 0:
        success_rate = (total_successful_adaptors / total_attempted_adaptors) * 100
        logging.info(f"Adaptor loading success rate: {success_rate:.2f}%")
    logging.info(f"===============================================")
    
    # 只有rank 0负责保存配置文件，避免并发写入冲突
    if rank == 0:
        logging.info(f"Worker {rank}: Saving configuration file...")
        try:
            # 生成并保存配置文件
            config_data = output_manager.generate_config(
                approach="LoRA",
                args=args
            )
            # 添加多worker信息到配置中
            if world_size > 1:
                config_data["experiment_info"]["multi_worker"] = {
                    "world_size": world_size,
                    "data_parallel": True,
                    "unprocessed_data_distribution": True
                }
            output_manager.save_config(config_path, config_data)
            logging.info(f"Worker {rank}: Config saved to: {config_path}")
        except Exception as e:
            logging.error(f"Worker {rank}: Error saving config: {e}")
    else:
        logging.info(f"Worker {rank}: Skipping config save (handled by rank 0)")
    
    logging.info(f"Worker {rank}: Results appended to: {output_path}")
    logging.info(f"Worker {rank}: Task {task} completed with {processed_count} processed samples")
    
    # 返回处理的样本数量，用于后续统计
    return processed_count, 0  # 第二个返回值改为0，因为没有跳过样本的概念了


if __name__ == "__main__":
    # trainLoraModels()
    # trainNormalLoraModels()
    
    # 可以直接调用无参函数，使用默认配置
    # loadModelandPredict()
    args = argparse.ArgumentParser()
    args.add_argument("--max_dependency_num", type=int, default=10,help="最大依赖数量")
    args.add_argument("--append_srcDep", action="store_true", help="是否添加源依赖")
    args.add_argument("--temperature", type=float, default=0.2, help="Temperature for sampling")
    args.add_argument("--top_p", type=float, default=0.95, help="Top-p for nucleus sampling")
    args.add_argument("--precision", type=str, default="fp16", help="Precision for model prediction")
    args.add_argument("--loraadaptor_save_path_base", type=str, default="/datanfs2/chenrongyi/models/loraadaptors/", help="adaptor_base_path")
    args.add_argument("--tempmerged_adaptor_path", type=str, default=None, help="存放临时merge的adaptor点")
    args.add_argument("--model_name", type=str, default="/datanfs2/chenrongyi/hf/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/3ad372fc79158a2148299e3318516c786aeded6c", help="model_name")
    args.add_argument("--knowledge_type", type=str, default="docstring", help="knowledge_type,override对应的conf") # 对应名字中需包含对应的knowledge_type才会成功加载
    args.add_argument("--newWhenExist", action="store_true", help="当同名输出文件存在时，是否创建新文件")
    args.add_argument("--overwrite", action="store_true", help="当同名输出文件存在时，是否清空，还是load原有结果并继续未进行的推理")
    args.add_argument("--rank", type=int, default=0, help="rank")
    args.add_argument("--world_size", type=int, default=1, help="world_size")
    args.add_argument("--task", type=str, default="vace", help="task")
    args.add_argument("--specified_bench_path", type=str, default=None, help="指定要预测的benchmark路径")
    args.add_argument("--Ban_Deprecation", action="store_true", help="是否禁用deprecation")
    args.add_argument("--force_gpu", action="store_true", help="是否强制使用GPU")
    args.add_argument("--base_output_dir", type=str, default="output/approach_eval", help="Base output directory")
    args.add_argument("--adopt_IFT_checkpoint", action="store_true", help="是否使用对应目录下IFT的checkpoint")
    args.add_argument("--ift_enabled_packages", type=str, nargs='+', default=None,
                     help="指定启用IFT的包名列表，只有列表中的包会使用IFT checkpoint，其他包使用普通LoRA模型。例如：--ift_enabled_packages matplotlib numpy pandas")
    args.add_argument("--ift_data_strategy", type=str, default=None, 
                     choices=["same_minor_version", "all_versions", "closest_n"],
                     help="偏好的IFT数据策略")
    args.add_argument("--ift_type", type=str, default=None,
                     help="偏好的IFT类型标识")
    args.add_argument("--list_ift_models", action="store_true", help="列出所有可用的IFT模型然后退出")
    args.add_argument("--ift_model_id", type=str, default=None, help="直接指定IFT模型ID")
    args.add_argument("--ift_card", type=str, default=None, help="指定IFT配置卡片文件路径")
    args.add_argument("--list_ift_cards", action="store_true", help="列出所有可用的IFT配置卡片然后退出")
    args.add_argument("--enable_task2maskpacks", action="store_true", help="启用task2maskpacks掩码功能，根据taskid掩码对应dependency的pack")
    args.add_argument("--task2maskpacks_file", type=str, default="data/task2maskpacks.json", help="task2maskpacks掩码文件路径")
    args.add_argument("--only_task2maskpacks_ids", action="store_true", help="仅处理task2maskpacks文件中定义的task ids")
    args.add_argument("--enable_task_id_filter", action="store_true", help="启用task id过滤功能，仅处理指定文件中的task ids")
    args.add_argument("--task_id_filter_file", type=str, default=None, help="task id过滤文件路径，应为包含整数列表的JSON文件")
    # 自定义输出文件
    args.add_argument("--specified_output_path", type=str, default=None, help="指定输出文件路径")
    args.add_argument("--specified_config_path", type=str, default=None, help="指定配置文件路径")
    args = args.parse_args()
    
    # 设置日志
    log_dir = setup_logging(args)
    
    # 新增：task2maskpacks功能状态日志
    if getattr(args, 'enable_task2maskpacks', False):
        logging.info("=== Task2MaskPacks 掩码功能已启用 ===")
        logging.info(f"🎯 掩码文件路径: {getattr(args, 'task2maskpacks_file', 'data/task2maskpacks.json')}")
        logging.info("📝 将根据taskid掩码对应dependency的pack")
        if getattr(args, 'only_task2maskpacks_ids', False):
            logging.info("🔍 仅处理task2maskpacks文件中定义的task ids")
    else:
        logging.info("=== Task2MaskPacks 掩码功能未启用 ===")
        logging.info("📝 将使用原始dependency配置")
        if getattr(args, 'only_task2maskpacks_ids', False):
            logging.warning("⚠️ --only_task2maskpacks_ids 已设置，但 --enable_task2maskpacks 未启用，该选项将被忽略")
    
    # 新增：task id过滤功能状态日志
    if getattr(args, 'enable_task_id_filter', False):
        logging.info("=== Task ID 过滤功能已启用 ===")
        logging.info(f"🎯 过滤文件路径: {getattr(args, 'task_id_filter_file', '未指定')}")
        logging.info("📝 将仅处理过滤文件中指定的task ids")
    else:
        logging.info("=== Task ID 过滤功能未启用 ===")
    
    # 🎯 新增：包级别IFT控制的日志记录和验证
    if hasattr(args, 'ift_enabled_packages') and args.ift_enabled_packages:
        logging.info("=== 包级别IFT控制已启用 ===")
        logging.info(f"📦 指定的IFT启用包列表: {args.ift_enabled_packages}")
        logging.info(f"💡 只有以下包会尝试加载IFT checkpoint:")
        for pkg in args.ift_enabled_packages:
            logging.info(f"   - {pkg}")
        logging.info(f"📝 其他包将直接使用普通LoRA模型")
        
        # 如果同时设置了adopt_IFT_checkpoint，给出提示
        if args.adopt_IFT_checkpoint:
            logging.info(f"⚠️ 注意: 同时设置了--adopt_IFT_checkpoint和--ift_enabled_packages")
            logging.info(f"   实际行为: 按包级别控制，只对指定包启用IFT")
    elif args.adopt_IFT_checkpoint:
        logging.info("=== 全局IFT模式已启用 ===")
        logging.info("🚀 所有包都将尝试加载IFT checkpoint")
    else:
        logging.info("=== 标准LoRA模式 ===")
        logging.info("📝 所有包都将使用普通LoRA模型")
    
    # 如果请求列出IFT模型，则执行并退出
    if args.list_ift_models:
        ift_manager = get_ift_manager()
        models = ift_manager.list_models()
        
        if not models:
            print("没有找到任何注册的IFT模型")
        else:
            print(f"找到 {len(models)} 个已注册的IFT模型:")
            print()
            for model in models:
                print(f"模型ID: {model['model_id']}")
                print(f"  包-版本: {model['pkg']}-{model['version']}")
                print(f"  基础模型: {model['base_model'].split('/')[-1]}")
                print(f"  知识类型: {model['knowledge_type']}")
                print(f"  数据策略: {model['data_strategy']}")
                print(f"  IFT类型: {model['ift_type'] or 'default'}")
                print(f"  模型路径: {model['ift_model_path']}")
                print(f"  存在性: {'✓' if model.get('model_exists', False) else '✗'}")
                print(f"  注册时间: {model['registered_at']}")
                print("-" * 60)
        print("退出程序")
        sys.exit(0)
    
    # 如果请求列出IFT配置卡片，则执行并退出
    if args.list_ift_cards:
        card_manager = get_default_card_manager()
        cards = card_manager.list_ift_cards()
        
        if not cards:
            print("没有找到任何IFT配置卡片")
        else:
            print(f"找到 {len(cards)} 个IFT配置卡片:")
            print()
            for card in cards:
                print(f"卡片ID: {card['card_id']}")
                print(f"  文件名: {card['filename']}")
                print(f"  包-版本: {card['pkg']}-{card['version']}")
                print(f"  IFT类型: {card['ift_type']}")
                print(f"  数据策略: {card['data_strategy']}")
                print(f"  创建时间: {card['created_at']}")
                print(f"  描述: {card['description']}")
                print(f"  使用方式: python benchmark/pred_lora.py --ift_card {card['filename']}")
                print("-" * 80)
        print("退出程序")
        sys.exit(0)
    
    # 如果指定了IFT配置卡片，则加载并应用配置
    if args.ift_card:
        try:
            card_manager = get_default_card_manager()
            ift_card = card_manager.load_ift_card(args.ift_card)
            
            # 验证配置卡片
            if not card_manager.validate_card(ift_card):
                logging.error(f"IFT配置卡片无效: {args.ift_card}")
                sys.exit(1)
            
            # 从配置卡片中提取配置
            prediction_config = card_manager.extract_prediction_config(ift_card)
            
            # 应用配置卡片中的设置到args
            args.model_name = prediction_config["model_name"]
            args.knowledge_type = prediction_config["knowledge_type"]
            args.precision = prediction_config["precision"]
            args.adopt_IFT_checkpoint = True  # 强制启用IFT模式
            args.ift_type = prediction_config["ift_type"]
            args.ift_data_strategy = prediction_config["data_strategy"]
            
            # 如果没有指定loraadaptor_save_path_base，从卡片中推断
            if not hasattr(args, 'loraadaptor_save_path_base') or args.loraadaptor_save_path_base is None:
                # 从base_lora_path推断基础路径
                base_lora_path = prediction_config["base_lora_path"]
                # 假设路径结构是 .../base_path/pkg/version/model_name/knowledge_type/
                import os
                parts = base_lora_path.split(os.sep)
                if len(parts) >= 4:
                    # 找到knowledge_type之前的路径作为基础路径
                    knowledge_type = prediction_config["knowledge_type"]
                    try:
                        if knowledge_type in parts:
                            kt_index = parts.index(knowledge_type)
                            base_path = os.sep.join(parts[:kt_index])
                            args.loraadaptor_save_path_base = base_path
                            logging.info(f"从IFT配置卡片推断基础路径: {base_path}")
                    except:
                        # 如果推断失败，使用默认路径
                        pass
            
            logging.info(f"✅ 已加载IFT配置卡片: {args.ift_card}")
            logging.info(f"🎯 目标包-版本: {prediction_config['pkg']}-{prediction_config['version']}")
            logging.info(f"📋 IFT类型: {prediction_config['ift_type']}")
            logging.info(f"📊 数据策略: {prediction_config['data_strategy']}")
            logging.info(f"🏷️ 卡片ID: {prediction_config['card_id']}")
            
        except Exception as e:
            logging.error(f"加载IFT配置卡片失败: {e}")
            sys.exit(1)
    
    # 详细的CUDA环境检测和诊断
    logging.info("=== CUDA环境诊断 ===")
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    logging.info(f"CUDA_VISIBLE_DEVICES={cuda_devices}")
    
    # 检查CUDA是否可用
    logging.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        visible_gpu_count = torch.cuda.device_count()
        logging.info(f"PyTorch可见GPU数量: {visible_gpu_count}")
        
        if visible_gpu_count > 0:
            for i in range(visible_gpu_count):
                gpu_name = torch.cuda.get_device_properties(i).name
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logging.info(f"GPU {i}: {gpu_name} ({total_memory:.1f}GB)")
        else:
            logging.warning("PyTorch检测到CUDA可用但GPU数量为0")
    else:
        logging.warning("PyTorch检测CUDA不可用")
        
        # 如果设置了CUDA_VISIBLE_DEVICES但CUDA不可用，给出详细诊断
        if cuda_devices != 'Not set':
            logging.error(f"环境变量CUDA_VISIBLE_DEVICES={cuda_devices}，但PyTorch无法使用CUDA")
            logging.error("可能的原因：")
            logging.error("1. PyTorch没有CUDA支持 - 检查：python -c 'import torch; print(torch.version.cuda)'")
            logging.error("2. CUDA驱动/运行时版本不匹配")
            logging.error("3. GPU设备编号不存在")
            logging.error("4. GPU被其他进程占用")
            
            # 尝试运行nvidia-smi检查GPU状态
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    logging.info("nvidia-smi输出:")
                    for line in result.stdout.strip().split('\n'):
                        logging.info(f"  {line}")
                else:
                    logging.error(f"nvidia-smi执行失败: {result.stderr}")
            except Exception as e:
                logging.error(f"无法运行nvidia-smi: {e}")
    
    logging.info("=== 开始LoRA预测任务 ===")
    logging.info(f"日志保存目录: {log_dir}")
    
    versiBCB_lora_merge_pred(args, task=args.task, removeDeprecationData=args.Ban_Deprecation)
    # versiBCB_lora_merge_pred(args, task="vscc", removeDeprecationData=False)
    # versiBCB_lora_merge_pred(args, task="vace", removeDeprecationData=True)
    # versiBCB_lora_merge_pred(args, task="vscc", removeDeprecationData=True)
    
    logging.info("=== LoRA预测任务完成 ===")
    print("-" * 60)
    sys.exit(0)