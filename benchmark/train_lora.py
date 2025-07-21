import torch
import json
import os
import shutil
import argparse
import deepspeed
import logging
import sys
from datetime import datetime
from peft import PeftModel,TaskType
from torch.utils.data import DataLoader
from utils.loraTrain.buildandloadData import DocstringDataset, collate_fn, LazyDocstringDataset,QADataset,TextDataset,DocstringDataset1,SrccodeDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from benchmark.config.code.config_lora import LORA_CONFIG_PATH, load_config
from utils.loraTrain.getVersicodeData import getPkgDocstringItems,GetQAPairsFromBenchData,GetQAPairsFromFlatIFTData
from utils.getDatasetPacks import getPackVersions
from peft import get_peft_model, LoraConfig
from utils.loraPathConfigure import pathConfigurator
from utils.loraTrain.dataset_loader import load_dataset
from utils.loraTrain.loraTrainUtils import getEquipAdaptorModel
from utils.loraTrain.loraTrainUtils import loraModelExists,get_dataloader,buildandTrainLoraModel,getDataExistence,load_lora_model,load_config,create_lora_config,save_lora_model,load_lora_model_withPeft,load_base_model,merge_lora_weights,inference,train_lora_model_withPEFT
from utils.data_statistics.getStatistics import load_and_aggregate_package_versions
from utils.loraTrain.log import setup_logging
import traceback

# 导入内存调试模块
try:
    from utils.memoryDebug.memoryCheck import GPUMemoryProfiler
    from utils.memoryDebug.trainWithMemoryDebug import debug_lora_training, TrainingMemoryProfiler
    from utils.memoryDebug.quickMemoryDebug import QuickMemoryDebugger
    MEMORY_DEBUG_AVAILABLE = True
except ImportError as e:
    logging.warning(f"内存调试模块导入失败: {e}")
    MEMORY_DEBUG_AVAILABLE = False
profiler = GPUMemoryProfiler()
# def trainNormalLoraModel(config,tokenizer):
#     # 由原始数据源构建dataset
#     corpus_path = ''
#     dataset = getDataset(corpus_path,tokenizer)
#     # 构建dataloader
#     dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,collate_fn=collate_fn)
#     # 使用dataloader训练模型，应用配置中的精度
#     precision = config.get("precision", "fp16")
#     lora_model = buildandTrainLoraModel(config, dataloader, precision)
#     return lora_model
def trainLoraModelForPack(config,pkg,version,tokenizer,corpus_path=None,dataset_type='docstring',precision='fp16',enable_memory_debug=False,memory_debug_log_dir=None):
    files_info = []
    data_file_path = f"{corpus_path}/{pkg}/{version}.jsonl"
    
    logging.info(f"正在加载包 {pkg}-{version} 的训练数据: {data_file_path}")
    
    try:
        with open(data_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_data = json.loads(line)
                files_info.append(line_data)
    except FileNotFoundError:
        logging.error(f"数据文件不存在: {data_file_path}")
        return None
    except Exception as e:
        logging.error(f"读取数据文件时出错: {e}")
        return None
        
    # 根据配置的训练数据百分比截取数据
    original_count = len(files_info)
    if config["override_data_percentage"] is not None:
        files_info = files_info[:int(len(files_info)*float(config["override_data_percentage"]))]
    else:
        files_info = files_info[:int(len(files_info)*config["traindata_percentage"])]
    used_count = len(files_info)
    
    logging.info(f"包 {pkg}-{version}: 总数据量={original_count}, 使用数据量={used_count} ({config['traindata_percentage']*100:.1f}%)")
    
    if len(files_info) == 0:
        logging.warning(f"包 {pkg}-{version} 没有可用的训练数据")
        return None
        
    # 根据数据集类型创建对应的数据集
    if dataset_type == 'docstring':
        dataset = DocstringDataset1(files_info, tokenizer, block_size=128,pkg=pkg,version=version)
    elif dataset_type == 'docs':
        dataset = DocstringDataset1(files_info, tokenizer, block_size=128,pkg=pkg,version=version)
    elif dataset_type == 'srccodes':
        dataset = SrccodeDataset(files_info, tokenizer, block_size=128,pkg=pkg,version=version)
    else:
        logging.error(f"不支持的数据集类型: {dataset_type}")
        raise ValueError(f"Invalid dataset type: {dataset_type}")
        
    logging.info(f"创建数据集完成，数据集大小: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,collate_fn=lambda batch: collate_fn(batch, tokenizer))
    logging.info(f"创建DataLoader完成，batch_size={config['batch_size']}")

    logging.info(f"开始训练包 {pkg}-{version} 的LoRA模型，精度: {precision}")
    
    # 根据是否启用内存调试选择不同的训练方式
    if enable_memory_debug and MEMORY_DEBUG_AVAILABLE:
        logging.info(f"🔍 启用内存调试模式训练包 {pkg}-{version}")
        
        # 设置内存调试日志目录
        if memory_debug_log_dir is None:
            memory_debug_log_dir = f"logs/memory_debug/{pkg}_{version}"
        
        # 使用内存调试训练
        lora_model, debug_results = debug_lora_training(
            config, dataloader, precision, pkg, version, dataset_type, memory_debug_log_dir
        )
        
        # 记录内存调试结果到日志
        logging.info("="*80)
        logging.info(f"📊 包 {pkg}-{version} 内存调试结果:")
        logging.info("="*80)
        
        # 🆕 首先显示简洁的内存报告摘要
        if 'json_reports' in debug_results:
            logging.info(f"\n📄 简洁内存报告摘要:")
            for stage, report in debug_results['json_reports'].items():
                if 'brief' in stage and 'memory_breakdown' in report:
                    logging.info(f"\n  🔍 阶段: {stage}")
                    
                    breakdown = report['memory_breakdown']
                    total_memory = report['total_memory_mb']
                    gpu_utilization = report['gpu_utilization_percentage']
                    
                    logging.info(f"    总内存占用: {total_memory:.2f} MB")
                    logging.info(f"    GPU利用率: {gpu_utilization:.2f}%")
                    logging.info(f"    各组件内存占用:")
                    
                    # 按内存占用排序
                    sorted_components = sorted(breakdown.items(), key=lambda x: x[1]["memory_mb"], reverse=True)
                    
                    for i, (component, info) in enumerate(sorted_components):
                        memory_mb = info["memory_mb"]
                        percentage = info["percentage"]
                        
                        # 中文组件名映射
                        component_names = {
                            "base_model": "基础模型",
                            "lora_parameters": "LoRA参数",
                            "gradients": "梯度",
                            "optimizer_states": "优化器状态",
                            "activations": "激活值"
                        }
                        
                        display_name = component_names.get(component, component)
                        logging.info(f"      {i+1}. {display_name}: {memory_mb:.2f} MB ({percentage:.1f}%)")
        
        # 记录综合分析摘要
        if 'summary' in debug_results:
            summary = debug_results['summary']
            logging.info(f"\n🎯 综合分析摘要:")
            logging.info(f"  分析阶段数: {summary.get('total_stages_analyzed', 0)}")
            logging.info(f"  分析的阶段: {', '.join(summary.get('stages', []))}")
            logging.info(f"  峰值内存使用阶段: {summary.get('peak_memory_stage', 'N/A')}")
            logging.info(f"  峰值内存使用量: {summary.get('peak_memory_mb', 0):.2f} MB")
            
        # 记录详细的内存使用情况（仅显示关键信息）
        if 'json_reports' in debug_results:
            logging.info(f"\n📊 详细内存使用情况:")
            for stage, report in debug_results['json_reports'].items():
                if 'brief' not in stage and 'memory_summary' in report:
                    memory_summary = report['memory_summary']
                    occupancy = report.get('occupancy_percentages', {})
                    
                    logging.info(f"\n  🔍 阶段: {stage}")
                    logging.info(f"    总内存使用: {memory_summary['total_memory_mb']:.2f} MB")
                    logging.info(f"    GPU占用率: {occupancy.get('total_used_percentage', 0):.2f}%")
                    
                    # 只显示内存占用最高的3个组件
                    component_breakdown = memory_summary.get('component_breakdown', {})
                    if component_breakdown:
                        sorted_components = sorted(component_breakdown.items(), key=lambda x: x[1], reverse=True)
                        logging.info(f"    主要组件内存占用:")
                        for i, (component, memory_mb) in enumerate(sorted_components[:3]):
                            component_percentage = occupancy.get('component_percentages', {}).get(component, 0)
                            logging.info(f"      {i+1}. {component}: {memory_mb:.2f} MB ({component_percentage:.2f}%)")
        
        # 记录内存问题检测结果
        if 'memory_issues' in debug_results:
            memory_issues = debug_results['memory_issues']
            logging.info(f"\n🚨 内存问题检测:")
            has_issues = False
            for issue_type, details in memory_issues.items():
                if details:
                    has_issues = True
                    logging.warning(f"  ⚠️  {issue_type}: {details}")
            if not has_issues:
                logging.info("  ✅ 未检测到内存问题")
        
        # 生成内存优化建议
        if 'json_reports' in debug_results:
            try:
                from utils.memoryDebug.trainWithMemoryDebug import generate_memory_optimization_suggestions
                suggestions = generate_memory_optimization_suggestions(debug_results['json_reports'])
                
                if any(suggestion_list for suggestion_list in suggestions.values()):
                    logging.info(f"\n💡 内存优化建议:")
                    for category, suggestion_list in suggestions.items():
                        if suggestion_list:
                            category_names = {
                                "general_suggestions": "通用建议",
                                "parameter_optimization": "参数优化",
                                "gradient_optimization": "梯度优化",
                                "optimizer_optimization": "优化器优化",
                                "training_optimization": "训练优化"
                            }
                            display_category = category_names.get(category, category)
                            logging.info(f"  📝 {display_category}:")
                            for suggestion in suggestion_list[:2]:  # 只显示前2个建议
                                logging.info(f"    • {suggestion}")
                else:
                    logging.info(f"\n💡 内存优化建议: 暂无特别建议，内存使用正常")
            except Exception as e:
                logging.warning(f"生成优化建议时出错: {e}")
        
        logging.info(f"\n📂 详细JSON报告已保存到: {memory_debug_log_dir}")
        logging.info("="*80)
        
    else:
        # 使用标准训练方式
        if enable_memory_debug and not MEMORY_DEBUG_AVAILABLE:
            logging.warning("⚠️  内存调试模块不可用，使用标准训练模式")
            
        lora_model=buildandTrainLoraModel(config,dataloader,precision,pkg,version,knowledge_type=dataset_type)
    
    logging.info(f"包 {pkg}-{version} 的LoRA模型训练完成")
    
    return lora_model



def trainLoraModelsForVersiBCB(benchmark_data_path=None,corpus_path="/datanfs2/chenrongyi/data/docs",knowledge_type='docstring',model_config=None,precision='fp16',pack_versions=None,pre_filtered=False,enable_memory_debug=False,memory_debug_log_dir=None):
    """
    训练LoRA模型for VersiBCB
    
    Args:
        benchmark_data_path: str, 单个benchmark数据文件路径（用于向后兼容）
        corpus_path: str, 语料库路径
        knowledge_type: str, 知识类型
        model_config: dict, 模型配置
        precision: str, 精度
        pack_versions: dict, 预先汇总的包版本信息 {pkg: [versions]}，如果提供则忽略benchmark_data_path
        pre_filtered: bool, 包版本是否已经预过滤（多worker模式下为True）
        enable_memory_debug: bool, 是否启用内存调试
        memory_debug_log_dir: str, 内存调试日志目录
    """
    if pack_versions is not None:
        # 使用预先汇总的包版本信息
        packVersions = pack_versions
        logging.info(f"使用预先汇总的包版本信息: {len(packVersions)} 个包")
    else:
        # 向后兼容：使用单个benchmark文件
        if benchmark_data_path is None:
            raise ValueError("必须提供 benchmark_data_path 或 pack_versions 其中之一")
        
        logging.info(f"从单个benchmark文件加载包版本信息: {benchmark_data_path}")
        with open(benchmark_data_path, "r") as f:
            datas = json.load(f)
        packVersions = getPackVersions(datas)
    # base_model, tokenizer = load_base_model(config.get("model_name"), config.get("device_map"))
    tokenizer = AutoTokenizer.from_pretrained(model_config.get("model_name"))
    model_name = model_config.get("model_name").split("/")[-1]
    
    # 内存调试相关日志
    if enable_memory_debug:
        if MEMORY_DEBUG_AVAILABLE:
            logging.info("🔍 启用内存调试模式")
            if memory_debug_log_dir:
                logging.info(f"📂 内存调试日志目录: {memory_debug_log_dir}")
        else:
            logging.warning("⚠️  内存调试模块不可用，将使用标准训练模式")
    
    logging.info(f"开始训练LoRA模型，数据集类型: {knowledge_type}")
    logging.info(f"总计 {len(packVersions)} 个包需要处理")
    
    trained_count = 0
    skipped_count = 0
    error_count = 0
    
    for pkg,versions in packVersions.items():
        for version in versions:
            if pre_filtered:
                # 多worker模式下，包版本已经预过滤，跳过存在性检查
                logging.info(f"开始训练包 {pkg}-{version} 的LoRA模型（预过滤模式）")
            else:
                # 传统模式，需要检查LoRA模型是否存在
                logging.info(f"检查包 {pkg}-{version} 的LoRA模型是否存在")
                if loraModelExists(pkg,version,model_name,model_config,knowledge_type=knowledge_type):
                    logging.info(f"包 {pkg}-{version} 的LoRA模型已存在，跳过训练")
                    skipped_count += 1
                    continue
                    
            try:
                if not pre_filtered:
                    logging.info(f"开始训练包 {pkg}-{version} 的LoRA模型")
                    
                # 设置每个包的内存调试日志目录
                pkg_memory_debug_log_dir = None
                if enable_memory_debug and memory_debug_log_dir:
                    pkg_memory_debug_log_dir = os.path.join(memory_debug_log_dir, f"{pkg}_{version}")
                    
                lora_model=trainLoraModelForPack(
                    model_config,pkg,version,tokenizer,corpus_path,
                    dataset_type=knowledge_type,precision=precision,
                    enable_memory_debug=enable_memory_debug,
                    memory_debug_log_dir=pkg_memory_debug_log_dir
                )
                lora_save_path = pathConfigurator().getPath(model_config,pkg,version,model_name,knowledge_type=knowledge_type)
                lora_model.save_pretrained(lora_save_path)
                logging.info(f"成功训练并保存包 {pkg}-{version} 的LoRA模型到: {lora_save_path}")
                trained_count += 1
            except Exception as e:
                logging.error(f"训练包 {pkg}-{version} 时出错: {e}")
                logging.error(f"错误信息: {traceback.format_exc()}")
                error_count += 1
                continue
    
    logging.info(f"训练完成统计: 训练={trained_count}, 跳过={skipped_count}, 错误={error_count}")
    
    # 返回统计信息以供多worker模式使用
    return {
        'trained': trained_count,
        'skipped': skipped_count,
        'failed': error_count,
        'total': trained_count + skipped_count + error_count
    }

if __name__ == "__main__":
    model_config = load_config(LORA_CONFIG_PATH)
    args = argparse.ArgumentParser()
    # ？似乎搞错了，args.precision会覆盖config_lora中的precision
    args.add_argument("--precision", type=str, default="bf16",help="precision of the model,但会被config_lora中的precision覆盖",choices=["fp16","fp32","bf16"])
    args.add_argument("--dataset_type", type=str, default="docstring",help="dataset type, docstring or srccodes,用于存储文件的最终命名",choices=["docstring","srccodes"])
    args.add_argument("--corpus_path", type=str, default="/datanfs4/chenrongyi/data/docs",help="corpus path，必须与dataset_type一致，不然会训练出错误的对象")
    args.add_argument("--benchmark_data_path", type=str, default="data/VersiBCB_Benchmark/vace_datas.json",help="benchmark data path (single file, for backward compatibility)")
    args.add_argument("--benchmark_paths", type=str, nargs='+', default=None, help="multiple benchmark data paths (will override benchmark_data_path if provided)")
    args.add_argument("--loraadaptor_save_path_base", type=str, default="/datanfs2/chenrongyi/models/loraadaptors/",help="lora adaptor save path base")
    args.add_argument("--model_name", type=str, default="/datanfs2/chenrongyi/models/Llama-3.1-8B",help="model name")
    args.add_argument("--log_dir", type=str, default=None, help="指定日志目录，如果不指定则自动生成")
    args.add_argument("--override_data_percentage", type=str, default=None, help="override the data percentage,但是不修改文件名")
    # 均衡设备映射参数
    args.add_argument("--use_balanced_device_map", type=bool, default=True, help="是否使用均衡设备映射")
    args.add_argument("--force_balance", type=bool, default=True, help="是否强制均衡分配")
    args.add_argument("--exclude_cpu", type=bool, default=True, help="是否排除CPU设备")
    args.add_argument("--check_r_consistency", type=bool, default=True, help="是否检查r值一致性")
    args.add_argument("--strict_r_check", type=bool, default=False, help="是否严格检查r值一致性") 
    # 动态设备映射参数
    args.add_argument("--use_dynamic_device_map", type=bool, default=False, help="是否使用动态设备映射策略")
    args.add_argument("--balance_threshold", type=float, default=0.3, help="动态映射的均衡阈值(0.0-1.0)")
    # 设备管理override参数，用于override前部设备映射配置
    args.add_argument("--device_map_strategy", type=str, default="balanced", 
                     choices=["auto", "balanced", "dynamic"], help="设备映射策略选择")
    
    # 多worker参数（新增，模仿train_lora_ift.py）
    args.add_argument("--rank", type=int, default=0,
                     help="当前worker的rank（用于多worker训练）")
    args.add_argument("--world_size", type=int, default=1,
                     help="总worker数量（用于多worker训练）")
    
    # 内存调试参数
    args.add_argument("--enable_memory_debug", action="store_true", default=False,
                     help="启用内存调试模式，输出详细的参数与显存占用对照表")
    args.add_argument("--memory_debug_log_dir", type=str, default=None,
                     help="内存调试日志目录，如果不指定则自动生成")
    
    args = args.parse_args()
    
    # 设置日志
    log_dir = setup_logging(args)
    
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
            
            # 当有GPU可用时，更新配置使用GPU
            logging.info("检测到GPU，设置模型使用GPU")
            model_config["device_map"] = "auto"
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
    
    logging.info("=== 配置信息 ===")
    logging.info(f"最终device_map: {model_config.get('device_map', 'auto')}")
    
    # override base_configs using args
    model_config["precision"] = args.precision
    model_config["knowledge_type"] = args.dataset_type
    model_config["corpus_path"] = args.corpus_path
    model_config["benchmark_data_path"] = args.benchmark_data_path
    model_config["loraadaptor_save_path_base"] = args.loraadaptor_save_path_base
    model_config["model_name"] = args.model_name
    model_config["override_data_percentage"] = args.override_data_percentage
    model_config["use_balanced_device_map"] = args.use_balanced_device_map
    model_config["force_balance"] = args.force_balance
    model_config["exclude_cpu"] = args.exclude_cpu
    model_config["check_r_consistency"] = args.check_r_consistency
    model_config["strict_r_check"] = args.strict_r_check
    
    # 动态设备映射参数
    model_config["use_dynamic_device_map"] = args.use_dynamic_device_map
    model_config["balance_threshold"] = args.balance_threshold
    
    # 根据device_map_strategy自动设置映射参数
    if args.device_map_strategy == "dynamic":
        model_config["use_dynamic_device_map"] = True
        model_config["use_balanced_device_map"] = False
    elif args.device_map_strategy == "balanced":
        model_config["use_dynamic_device_map"] = False
        model_config["use_balanced_device_map"] = True
    elif args.device_map_strategy == "auto":
        model_config["use_dynamic_device_map"] = False
        model_config["use_balanced_device_map"] = False
        model_config["device_map"] = "auto"
    
    # 处理内存调试参数
    if args.enable_memory_debug:
        if not MEMORY_DEBUG_AVAILABLE:
            logging.error("❌ 内存调试模块不可用，请检查utils.memoryDebug模块是否正确安装")
            logging.error("内存调试功能将被禁用")
            args.enable_memory_debug = False
        else:
            logging.info("🔍 启用内存调试模式")
            
            # 设置内存调试日志目录
            if args.memory_debug_log_dir is None:
                # 自动生成内存调试日志目录
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                args.memory_debug_log_dir = os.path.join(log_dir, f"memory_debug_{timestamp}")
                
            # 确保内存调试日志目录存在
            os.makedirs(args.memory_debug_log_dir, exist_ok=True)
            logging.info(f"📂 内存调试日志将保存到: {args.memory_debug_log_dir}")
            
            # 检查CUDA可用性（内存调试需要GPU）
            if not torch.cuda.is_available():
                logging.warning("⚠️  CUDA不可用，内存调试功能可能受限")
            else:
                logging.info(f"✅ 检测到 {torch.cuda.device_count()} 个GPU，内存调试功能已就绪")
    # 确定使用哪种方式加载包版本信息
    pack_versions = None
    if args.benchmark_paths is not None:
        # 使用多个benchmark文件
        logging.info("=== 使用多个benchmark文件模式 ===")
        logging.info(f"Benchmark文件列表: {args.benchmark_paths}")
        pack_versions = load_and_aggregate_package_versions(args.benchmark_paths)
        benchmark_data_path = None  # 不传递单个文件路径
    else:
        # 使用单个benchmark文件（向后兼容）
        logging.info("=== 使用单个benchmark文件模式 ===") 
        logging.info(f"Benchmark文件: {args.benchmark_data_path}")
        benchmark_data_path = args.benchmark_data_path
    
    # 🚀 多worker模式的包版本分配（新增）
    if pack_versions is not None and args.world_size > 1:
        logging.info(f"🚀 多worker模式: rank={args.rank}, world_size={args.world_size}")
        
        # 🔍 第一步：预过滤出真正需要训练的包版本组合
        logging.info("📋 第一步：预过滤真正需要训练的包版本...")
        
        # 准备模型相关信息
        model_name = model_config.get("model_name").split("/")[-1] if model_config.get("model_name") else "unknown"
        knowledge_type = model_config.get("knowledge_type", "docstring")
        
        # 收集所有包版本组合
        all_pkg_versions = []
        for pkg, versions in pack_versions.items():
            for version in versions:
                all_pkg_versions.append((pkg, version))
        
        total_combinations = len(all_pkg_versions)
        logging.info(f"总计 {total_combinations} 个包版本组合待检查")
        
        # 过滤出真正需要训练的组合
        need_training_combinations = []
        already_trained_count = 0
        no_data_count = 0
        
        logging.info("🔍 开始检查每个包版本的训练状态...")
        for i, (pkg, version) in enumerate(all_pkg_versions):
            if i % 20 == 0 and i > 0:  # 每20个打印一次进度
                logging.info(f"检查进度: {i}/{total_combinations} ({i/total_combinations*100:.1f}%)")
            
            # 检查LoRA模型是否已存在（使用相同的逻辑）
            try:
                from utils.loraTrain.loraTrainUtils import loraModelExists
                if loraModelExists(pkg, version, model_name, model_config, knowledge_type=knowledge_type):
                    already_trained_count += 1
                    logging.debug(f"跳过已训练: {pkg}-{version}")
                    continue
            except Exception as e:
                logging.debug(f"检查LoRA模型存在性时出错 {pkg}-{version}: {e}")
                # 如果检查失败，保守地加入训练列表
                pass
            
            # 检查是否有训练数据（简化检查，只验证数据文件是否存在）
            try:
                corpus_path = model_config.get("corpus_path", "/datanfs2/chenrongyi/data/docs")
                data_file_path = f"{corpus_path}/{pkg}/{version}.jsonl"
                if not os.path.exists(data_file_path):
                    no_data_count += 1
                    logging.debug(f"跳过无数据: {pkg}-{version}")
                    continue
                
                # 检查文件是否为空
                if os.path.getsize(data_file_path) == 0:
                    no_data_count += 1
                    logging.debug(f"跳过空数据: {pkg}-{version}")
                    continue
                    
            except Exception as e:
                logging.debug(f"检查训练数据时出错 {pkg}-{version}: {e}")
                # 如果检查失败，保守地加入训练列表
                pass
            
            # 如果通过所有检查，加入需要训练的列表
            need_training_combinations.append((pkg, version))
        
        logging.info(f"📊 预过滤统计:")
        logging.info(f"  原始组合数: {total_combinations}")
        logging.info(f"  已训练跳过: {already_trained_count}")
        logging.info(f"  无数据跳过: {no_data_count}")
        logging.info(f"  需要训练: {len(need_training_combinations)}")
        logging.info(f"  过滤率: {(already_trained_count + no_data_count)/total_combinations*100:.1f}%")
        
        if len(need_training_combinations) == 0:
            logging.info(f"Worker {args.rank}: 没有需要训练的包版本组合，退出")
            sys.exit(0)
        
        # 🎯 第二步：将真正需要训练的组合分配给workers
        logging.info(f"🎯 第二步：分配 {len(need_training_combinations)} 个真正需要训练的组合...")
        
        # 计算当前worker负责的包版本范围（基于需要训练的组合）
        real_combinations = len(need_training_combinations)
        combinations_per_worker = real_combinations // args.world_size
        remainder = real_combinations % args.world_size
        
        # 计算当前worker的起始和结束索引
        start_idx = args.rank * combinations_per_worker + min(args.rank, remainder)
        end_idx = start_idx + combinations_per_worker + (1 if args.rank < remainder else 0)
        
        # 分配给当前worker的包版本组合（只包含需要训练的）
        worker_pkg_versions = need_training_combinations[start_idx:end_idx]
        
        logging.info(f"Worker {args.rank} 分配到 {len(worker_pkg_versions)} 个需要训练的包版本组合:")
        logging.info(f"  全局索引范围: [{start_idx}, {end_idx}) (基于需要训练的组合)")
        logging.info(f"  预期训练工作量均衡: ✅")
        
        # 显示分配详情（前5个）
        for i, (pkg, version) in enumerate(worker_pkg_versions[:5]):
            logging.info(f"    {i+1}. {pkg}-{version}")
        if len(worker_pkg_versions) > 5:
            logging.info(f"    ... 还有 {len(worker_pkg_versions) - 5} 个")
        
        # 重新构建当前worker的pack_versions字典
        worker_pack_versions = {}
        for pkg, version in worker_pkg_versions:
            if pkg not in worker_pack_versions:
                worker_pack_versions[pkg] = []
            worker_pack_versions[pkg].append(version)
        
        # 使用分配后的包版本信息
        pack_versions = worker_pack_versions
        
        if len(pack_versions) == 0:
            logging.info(f"Worker {args.rank} 没有分配到任何需要训练的包版本，退出")
            sys.exit(0)
        
        # 显示最终的负载分配信息
        logging.info(f"💪 Worker {args.rank} 最终分配:")
        logging.info(f"  包数量: {len(worker_pack_versions)}")
        logging.info(f"  包版本组合数: {len(worker_pkg_versions)}")
        logging.info(f"  预期都需要实际训练: ✅")
            
    elif pack_versions is not None:
        logging.info("单worker模式")
    elif args.world_size > 1:
        logging.info(f"🚀 多worker模式: rank={args.rank}, world_size={args.world_size}")
        logging.info("注意: 多worker模式当前仅支持使用--benchmark_paths参数的批量模式")
        logging.info("单个benchmark文件模式将在所有worker上运行相同的训练任务")
    
    if pack_versions is not None:
        logging.info(f"当前worker将处理 {len(pack_versions)} 个包，共 {sum(len(versions) for versions in pack_versions.values())} 个包版本组合")
    
    logging.info("开始LoRA模型训练...")
    try:
        stats = trainLoraModelsForVersiBCB(
            benchmark_data_path=benchmark_data_path,
            model_config=model_config,
            precision=args.precision,
            knowledge_type=args.dataset_type,
            corpus_path=args.corpus_path,
            pack_versions=pack_versions,
            pre_filtered=pack_versions is not None and args.world_size > 1,
            enable_memory_debug=args.enable_memory_debug,
            memory_debug_log_dir=args.memory_debug_log_dir
        )
        if args.world_size > 1:
            logging.info(f"Worker {args.rank} LoRA模型训练任务完成!")
            logging.info(f"Worker {args.rank} 统计: 训练={stats['trained']}, 跳过={stats['skipped']}, 错误={stats['failed']}, 总计={stats['total']}")
        else:
            logging.info("LoRA模型训练成功完成!")
            logging.info(f"最终统计: 训练={stats['trained']}, 跳过={stats['skipped']}, 错误={stats['failed']}, 总计={stats['total']}")
    except Exception as e:
        if args.world_size > 1:
            logging.error(f"Worker {args.rank} 训练过程中发生错误: {e}")
        else:
            logging.error(f"训练过程中发生错误: {e}")
        raise
    
    logging.info(f"所有日志已保存到: {log_dir}")
    pass

# =============================================================================
# 使用示例 (Usage Examples)
# =============================================================================
#
# 1. 使用单个benchmark文件（向后兼容）:
# python benchmark/train_lora.py \
#     --precision fp16 \
#     --dataset_type docstring \
#     --corpus_path /datanfs2/chenrongyi/data/docs \
#     --benchmark_data_path benchmark/data/VersiBCB_Benchmark/vace_datas.json \
#     --model_name /datanfs2/chenrongyi/models/Llama-3.1-8B
#
# 2. 使用多个benchmark文件（新功能）:
# python benchmark/train_lora.py \
#     --precision fp16 \
#     --dataset_type docstring \
#     --corpus_path /datanfs2/chenrongyi/data/docs \
#     --benchmark_paths \
#         benchmark/data/VersiBCB_Benchmark/vace_datas.json \
#         benchmark/data/VersiBCB_Benchmark/vscc_datas.json \
#         benchmark/data/VersiBCB_Benchmark/vace_datas_for_warning.json \
#     --model_name /datanfs2/chenrongyi/models/Llama-3.1-8B
#
# 3. 启用内存调试模式（新功能）:
# python benchmark/train_lora.py \
#     --precision fp16 \
#     --dataset_type docstring \
#     --corpus_path /datanfs2/chenrongyi/data/docs \
#     --benchmark_data_path benchmark/data/VersiBCB_Benchmark/vace_datas.json \
#     --model_name /datanfs2/chenrongyi/models/Llama-3.1-8B \
#     --enable_memory_debug \
#     --memory_debug_log_dir logs/memory_debug_custom
#
# 4. 内存调试 + 多文件批量训练:
# python benchmark/train_lora.py \
#     --precision fp16 \
#     --dataset_type docstring \
#     --corpus_path /datanfs2/chenrongyi/data/docs \
#     --benchmark_paths \
#         benchmark/data/VersiBCB_Benchmark/vace_datas.json \
#         benchmark/data/VersiBCB_Benchmark/vscc_datas.json \
#     --model_name /datanfs2/chenrongyi/models/Llama-3.1-8B \
#     --enable_memory_debug
#
# 注意: 
# - 如果同时提供 --benchmark_paths 和 --benchmark_data_path，
#   --benchmark_paths 会覆盖 --benchmark_data_path
# - 内存调试模式会输出详细的参数与显存占用对照表到日志文件
# - 如果不指定 --memory_debug_log_dir，会自动生成时间戳目录
# ============================================================================= 
