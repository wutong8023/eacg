import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import json
import os
import shutil
import argparse
import logging
import sys
from datetime import datetime
from peft import PeftModel, TaskType
from torch.utils.data import DataLoader
from utils.loraTrain.buildandloadData import DocstringDataset, collate_fn, LazyDocstringDataset, QADataset, TextDataset, DocstringDataset1, SrccodeDataset
from benchmark.config.code.config_lora import MODEL_SOURCE
if MODEL_SOURCE == "MODELSCOPE":
    from modelscope import AutoTokenizer
else:
    from transformers import AutoTokenizer
from benchmark.config.code.config_lora import LORA_CONFIG_PATH, load_config
from utils.loraTrain.getVersicodeData import getPkgDocstringItems, GetQAPairsFromBenchData, GetQAPairsFromFlatIFTData
from utils.getDatasetPacks import getPackVersions
from peft import get_peft_model, LoraConfig
from utils.loraPathConfigure import pathConfigurator
from utils.loraTrain.dataset_loader import load_dataset
from utils.loraTrain.loraTrainUtils import getEquipAdaptorModel, loraModelExists, get_dataloader, buildandTrainLoraModel, getDataExistence, load_lora_model, load_config, create_lora_config, save_lora_model, load_lora_model_withPeft, load_base_model, merge_lora_weights, inference, train_lora_model_withPEFT
from utils.data_statistics.getStatistics import load_and_aggregate_package_versions
from utils.loraTrain.log import setup_logging
from utils.data_distribution_checker import create_distributed_dataloader_with_checking, check_batch_distribution
from utils.clean_resource import log_gpu_memory_usage, clear_model_outputs_and_cache, clear_optimizer_states, force_clear_cuda_context, comprehensive_memory_cleanup, force_memory_reset_device, force_cleanup_memory,  find_and_clear_lingering_tensors,log_detailed_gpu_memory_report
from utils.loraTrain.dataFilter import getTrainDataItems, apply_prefilter_to_package_versions
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))





import matplotlib
from utils.plotting.training_plots import save_training_plots, create_loss_summary, print_loss_summary


# 设置环境变量以禁用tokenizers并行化，避免分布式训练中的死锁警告

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def init_distributed_training():
    """
    初始化分布式训练环境
    """
    # 检查是否在torchrun环境中
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # 从环境变量中获取rank和world_size
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # 设置CUDA设备
        torch.cuda.set_device(local_rank)
        
        # 初始化分布式进程组
        dist.init_process_group(backend='nccl', init_method='env://')
        
        logging.info(f"分布式训练初始化完成 - Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
        
        return rank, world_size, local_rank
    else:
        # 非分布式环境
        logging.info("非分布式训练环境")
        return 0, 1, 0

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import numpy as np
import torch.distributed as dist
from utils.loraTrain.buildandloadData import collate_fn

def create_kfold_dataloader(dataset, batch_size, rank, world_size, shuffle=True):
    """
    创建K-Fold分布式数据加载器，确保每个rank数据总量一致且正交
    
    参数:
        dataset: 完整数据集,目前的格式是DocstringDataset1，每条数据是str，通过collate_fn转换为dict
        batch_size: 每个batch的大小
        rank: 当前进程的rank (0到world_size-1)
        world_size: 总进程数(即K值)
        shuffle: 是否在划分fold前打乱数据顺序
    
    返回:
        DataLoader: 为当前rank创建的DataLoader
    """
    # 确保world_size等于进程数且大于1
    if world_size < 2:
        # 单进程情况，直接返回完整数据集
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            collate_fn=lambda batch: collate_fn(batch, dataset.tokenizer if hasattr(dataset, 'tokenizer') else None)
        )
        return dataloader
    
    # 1. 创建全局索引
    indices = np.arange(len(dataset))
    # 打乱的选项
    # if shuffle:
    #     rng = np.random.RandomState(42)  # 固定随机种子保证各rank一致
    #     rng.shuffle(indices)
    
    # 2. 将数据划分为K个fold，确保每个fold大小相等
    fold_size = len(dataset) // world_size
    fold_indices = []
    
    for i in range(world_size):
        start = i * fold_size
        end = (i + 1) * fold_size
        fold_indices.append(indices[start:end].tolist())
        logging.info(f"fold_indices length for fold {i}: {len(fold_indices[i])}")
    # logging.info(f"fold_indices: {fold_indices}")
    # logging.info(f"fold_indices length: {len(fold_indices)}")
    
    # 3. 为当前rank创建子集
    current_fold_indices = fold_indices[rank]
    subset = Subset(dataset, current_fold_indices)

    # 5. 验证数据分配并打印索引信息
    logging.info(f"Rank {rank}: 分配到 {len(subset)} 个样本")
    if world_size > 1:
        all_subsets = [None] * world_size
        dist.gather_object(
            current_fold_indices,
            all_subsets if rank == 0 else None,
            dst=0
        )
        for i in range(world_size):
            if rank == 0:
                logging.info(f"Rank {i}: 根据subset分配到 {len(all_subsets[i])} 个样本")

    
    # logging.info(f"Rank {rank}: 索引范围 {min(current_fold_indices)}-{max(current_fold_indices)}")
    # logging.info(f"Rank {rank}: 前10个索引: {current_fold_indices[:10]}")
    # logging.info(f"Rank {rank}: 后10个索引: {current_fold_indices[-10:]}")
        
    # 4. 创建DataLoader
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,  
        pin_memory=True,
        drop_last=False,  # 确保所有rank的batch数一致
        collate_fn=lambda batch: collate_fn(batch, dataset.tokenizer if hasattr(dataset, 'tokenizer') else None)
    )
    

    # 6. 收集所有rank的索引并验证正交性
    if world_size > 1:
        # 收集所有rank的索引
        all_indices = [None] * world_size
        dist.gather_object(
            current_fold_indices,
            all_indices if rank == 0 else None,
            dst=0
        )
        
        # 在rank 0上验证正交性
        if rank == 0:
            logging.info(f"\n=== K-Fold数据分配验证 ===")
            logging.info(f"总数据集大小: {len(dataset)}")
            
            # 检查每个fold的大小
            for i, fold_idx in enumerate(all_indices):
                logging.info(f"Fold {i}: {len(fold_idx)} 个样本, 最小索引: {min(fold_idx)}, 最大索引: {max(fold_idx)}")
            
            # 检查正交性（无重叠）
            overlap_found = False
            for i in range(world_size):
                for j in range(i + 1, world_size):
                    set_i = set(all_indices[i])
                    set_j = set(all_indices[j])
                    overlap = set_i & set_j
                    if overlap:
                        logging.info(f"❌ Fold {i} 和 Fold {j} 有 {len(overlap)} 个重叠索引")
                        overlap_found = True
            
            if not overlap_found:
                logging.info("✅ 所有fold之间完全正交（无重叠）")
            
            # 检查覆盖率
            all_used_indices = set()
            for fold_idx in all_indices:
                all_used_indices.update(fold_idx)
            
            total_used = len(all_used_indices)
            expected_used = world_size * fold_size
            logging.info(f"数据覆盖情况: 使用了 {total_used} / {len(dataset)} 个样本")
            logging.info(f"预期使用: {expected_used} 个样本")
            
            if total_used == expected_used:
                logging.info("✅ 数据覆盖率正确")
            else:
                logging.info(f"⚠️  数据覆盖率问题: 丢失了 {len(dataset) - total_used} 个样本")
    
    return dataloader


def verify_dataloader_indices(dataloader, rank, world_size):
    """
    验证DataLoader实际加载的数据索引
    """
    print(f"\n=== Rank {rank} DataLoader索引验证 ===")
    
    # 收集前几个batch的实际索引
    batch_indices = []
    for i, batch in enumerate(dataloader):
        if i >= 3:  # 只检查前3个batch
            break
        
        # 尝试从batch中提取原始索引信息
        if hasattr(batch, 'indices'):
            indices = batch.indices
        elif isinstance(batch, dict) and 'indices' in batch:
            indices = batch['indices']
        else:
            # 如果没有直接的索引信息，使用batch的顺序推断
            batch_size = len(batch['input_ids']) if isinstance(batch, dict) and 'input_ids' in batch else len(batch)
            start_idx = i * batch_size
            indices = list(range(start_idx, start_idx + batch_size))
        
        batch_indices.extend(indices)
        print(f"Rank {rank}, Batch {i}: 索引 {indices[:5]}...{indices[-5:] if len(indices) > 5 else indices}")
    
    print(f"Rank {rank}: 前3个batch总共涉及 {len(batch_indices)} 个样本")
    
    # 收集所有rank的batch索引
    if world_size > 1:
        all_batch_indices = [None] * world_size
        dist.gather_object(
            batch_indices,
            all_batch_indices if rank == 0 else None,
            dst=0
        )
        
        # 在rank 0上验证batch级别的正交性
        if rank == 0:
            print(f"\n=== Batch级别正交性验证 ===")
            overlap_found = False
            for i in range(world_size):
                for j in range(i + 1, world_size):
                    set_i = set(all_batch_indices[i])
                    set_j = set(all_batch_indices[j])
                    overlap = set_i & set_j
                    if overlap:
                        print(f"❌ Rank {i} 和 Rank {j} 的batch有 {len(overlap)} 个重叠索引")
                        overlap_found = True
            
            if not overlap_found:
                print("✅ 所有rank的batch数据完全正交")


def trainLoraModelForPack_DDP(config, pkg, version, tokenizer, corpus_path=None, dataset_type='docstring', precision='fp16', rank=0, world_size=1, local_rank=0, gpus_per_process=1, enable_data_checking=True,no_save=False):
    """
    分布式训练单个包的LoRA模型（带数据分布检查）
    """
    if rank == 0:
        logging.info(f"正在加载包 {pkg}-{version} 的原始训练数据")
        
    # files_info = []
    # data_file_path = f"{corpus_path}/{pkg}/{version}.jsonl"
    # try:
    #     with open(data_file_path, "r", encoding="utf-8") as f:
    #         for line in f:
    #             files_info.append(json.loads(line))
    # except FileNotFoundError:
    #     if rank == 0:
    #         logging.error(f"数据文件不存在: {data_file_path}")
    #     return None
        
    # original_count = len(files_info)
    # if config["override_data_percentage"] is not None:
    #     files_info = files_info[:int(len(files_info)*float(config["override_data_percentage"]))]
    # else:
    #     files_info = files_info[:int(len(files_info)*config["traindata_percentage"])]
    original_count,files_info = getTrainDataItems(corpus_path,pkg,version,config)
    if rank == 0:
        logging.info(f"包 {pkg}-{version}: 总数据量={original_count}, 使用数据量={len(files_info)} ({config['traindata_percentage']*100:.1f}%)")

    # --- 数据集单点生成与分发 ---
    processed_input_ids = None
    
    if rank == 0:
        # 只有 rank 0 进行数据预处理
        logging.info("Rank 0: 正在进行分词和切块...")
        try:
            temp_dataset = DocstringDataset1(files_info, tokenizer, block_size=128, pkg=pkg, version=version)
            processed_input_ids = temp_dataset.input_ids
            logging.info(f"Rank 0: 数据处理完成，生成 {len(processed_input_ids)} 条样本")
            
            if len(processed_input_ids) == 0:
                logging.warning("Rank 0: 警告 - 处理后的数据集为空")
        except Exception as e:
            logging.error(f"Rank 0: 数据处理失败: {e}")
            processed_input_ids = []
    
    # 使用 broadcast_object_list 分发处理好的数据
    if world_size > 1:
        logging.info(f"Rank {rank}: 开始数据分发...")
        
        # 创建要广播的对象列表
        object_list = [processed_input_ids] if rank == 0 else [None]
        
        try:
            # 广播数据
            dist.broadcast_object_list(object_list, src=0)
            
            # 所有进程都从广播后的数据中获取 input_ids
            final_input_ids = object_list[0]
            
            if final_input_ids is None:
                logging.error(f"Rank {rank}: 接收到空数据")
                return None
            
            logging.info(f"Rank {rank}: 数据分发完成，接收到 {len(final_input_ids)} 条样本")
        except Exception as e:
            logging.error(f"Rank {rank}: 数据分发失败: {e}")
            return None
    else:
        # 单进程情况
        final_input_ids = processed_input_ids
        if final_input_ids is None or len(final_input_ids) == 0:
            logging.warning(f"Rank {rank}: 单进程模式，数据为空")
            return None
        logging.info(f"Rank {rank}: 单进程模式，数据量 {len(final_input_ids)} 条样本")
    
    # 所有进程使用相同的数据创建数据集实例
    dataset = DocstringDataset1(
        items=None,  # 不需要原始数据
        tokenizer=tokenizer,
        block_size=128,
        pkg=pkg,
        version=version,
        input_ids=final_input_ids  # 使用分发的数据
    )
    
    # 验证所有rank的数据集大小一致
    if world_size > 1:
        dataset_sizes = [None] * world_size
        dist.gather_object(
            len(dataset),
            dataset_sizes if rank == 0 else None,
            dst=0
        )
        
        if rank == 0:
            logging.info("=== 数据分发后的Dataset长度验证 ===")
            for i, size in enumerate(dataset_sizes):
                logging.info(f"Rank {i}: 最终数据集大小 {size} 条样本")
            
            # 检查是否所有rank的数据集大小一致
            unique_sizes = set(dataset_sizes)
            if len(unique_sizes) == 1:
                logging.info("✅ 所有rank的数据集大小完全一致")
            else:
                logging.error(f"❌ 数据分发后仍然不一致: {unique_sizes}")
                
            logging.info("=== 数据分发验证完成 ===")
    else:
        logging.info(f"Rank {rank}: 单进程模式，最终数据集大小 {len(dataset)} 条样本")

    if rank == 0:
        logging.info(f"所有进程均已同步数据集，总大小: {len(dataset)}")
    
    # 创建K-fold分布式数据加载器
    dataloader = create_kfold_dataloader(dataset, config["batch_size"], rank, world_size, shuffle=True)
    
    if rank == 0:
        logging.info(f"创建K-fold DataLoader完成，batch_size={config['batch_size']}, world_size={world_size}")
        if enable_data_checking:
            logging.info("数据分布检查功能已启用")
    
    # 验证DataLoader的实际索引分布
    if enable_data_checking:
        verify_dataloader_indices(dataloader, rank, world_size)
    
    if rank == 0:
        logging.info(f"开始训练包 {pkg}-{version} 的LoRA模型，精度: {precision}")
    
    # 调用分布式训练函数
    lora_model = buildandTrainLoraModel_DDP(
        config, dataloader, precision, pkg, version, 
        knowledge_type=dataset_type, rank=rank, world_size=world_size, 
        local_rank=local_rank, gpus_per_process=gpus_per_process,
        no_save=no_save
    )
    
    if rank == 0:
        logging.info(f"包 {pkg}-{version} 的LoRA模型训练完成")
    
    # 清理数据集和数据加载器
    if 'dataset' in locals():
        del dataset
    if 'dataloader' in locals():
        del dataloader
    if 'final_input_ids' in locals():
        del final_input_ids
    if 'processed_input_ids' in locals():
        del processed_input_ids
    if 'files_info' in locals():
        del files_info
    
    # 使用新的清理函数
    # comprehensive_memory_cleanup(rank, "包训练结束", f"{pkg}-{version}")
    
    return lora_model





def trainLoraModelsForVersiBCB_DDP(benchmark_data_path=None, corpus_path="/datanfs2/chenrongyi/data/docs", knowledge_type='docstring', model_config=None, precision='fp16', pack_versions=None, pre_filtered=False, rank=0, world_size=1, local_rank=0, gpus_per_process=1, enable_data_checking=True, no_save=False, enable_prefilter=True):
    """
    分布式训练LoRA模型for VersiBCB,指向所有包版本
    
    Args:
        enable_prefilter: 是否启用预过滤功能，提前过滤掉不需要训练的包版本
    """
    if pack_versions is not None:
        # 使用预先汇总的包版本信息
        packVersions = pack_versions
        if rank == 0:
            logging.info(f"使用预先汇总的包版本信息: {len(packVersions)} 个包")
    else:
        # 向后兼容：使用单个benchmark文件
        if benchmark_data_path is None:
            raise ValueError("必须提供 benchmark_data_path 或 pack_versions 其中之一")
        
        if rank == 0:
            logging.info(f"从单个benchmark文件加载包版本信息: {benchmark_data_path}")
        
        with open(benchmark_data_path, "r") as f:
            datas = json.load(f)
        packVersions = getPackVersions(datas)
    
    # 只在rank 0上加载tokenizer
    if rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(model_config.get("model_name"))
        model_name = model_config.get("model_name").split("/")[-1]
        
        logging.info(f"开始分布式训练LoRA模型，数据集类型: {knowledge_type}")
        
        # 预过滤功能（仅在rank 0上执行）
        if enable_prefilter and not pre_filtered:
            logging.info("🔍 启用预过滤功能，检查训练需求...")
            
            original_total = sum(len(versions) for versions in packVersions.values())
            
            filtered_packVersions, prefilter_stats = apply_prefilter_to_package_versions(
                packVersions, model_config, corpus_path, knowledge_type, log_details=True
            )
            
            # 更新包版本列表为过滤后的版本
            packVersions = filtered_packVersions
            
            new_total = sum(len(versions) for versions in packVersions.values())
            logging.info(f"预过滤完成: {original_total} -> {new_total} 个包版本需要训练")
            
            if new_total == 0:
                logging.info("🎉 预过滤后没有包版本需要训练，所有任务已完成!")
                return {
                    'trained': 0,
                    'skipped': prefilter_stats['lora_exists'] + prefilter_stats['no_data'],
                    'failed': prefilter_stats['errors'],
                    'total': prefilter_stats['total'],
                    'prefilter_stats': prefilter_stats
                }
        else:
            if pre_filtered:
                logging.info("使用预过滤的包版本列表，跳过预过滤步骤")
            else:
                logging.info("预过滤功能已禁用")
        
        logging.info(f"总计 {len(packVersions)} 个包需要处理")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_config.get("model_name"))
        model_name = model_config.get("model_name").split("/")[-1]
    
    trained_count = 0
    skipped_count = 0
    error_count = 0
    
    for pkg, versions in packVersions.items():
        for version in versions:
            if pre_filtered or enable_prefilter:
                # 预过滤模式或已过滤模式下，跳过存在性检查（已在预过滤中完成）
                if rank == 0:
                    logging.info(f"开始训练包 {pkg}-{version} 的LoRA模型（预过滤模式）")
            else:
                # 传统模式，需要检查LoRA模型是否存在
                if rank == 0:
                    logging.info(f"检查包 {pkg}-{version} 的LoRA模型是否存在")
                
                if loraModelExists(pkg, version, model_name, model_config, knowledge_type=knowledge_type):
                    if rank == 0:
                        logging.info(f"包 {pkg}-{version} 的LoRA模型已存在，跳过训练")
                    skipped_count += 1
                    continue
            
            # 获取训练数据（根据是否启用预过滤选择不同的处理方式）
            if enable_prefilter:
                # 预过滤模式：简化的数据检查，因为预过滤已经验证过数据存在性
                try:
                    original_count, files_info = getTrainDataItems(corpus_path, pkg, version, model_config)
                    if rank == 0:
                        logging.info(f"包 {pkg}-{version}: 使用数据量={len(files_info)} (预过滤模式)")
                except Exception as e:
                    if rank == 0:
                        logging.error(f"获取训练数据失败 {pkg}-{version}: {e}")
                    error_count += 1
                    continue
            else:
                # 传统模式：完整的数据检查
                original_count, files_info = getTrainDataItems(corpus_path, pkg, version, model_config)
                
                if rank == 0:
                    logging.info(f"包 {pkg}-{version}: 总数据量={original_count}, 使用数据量={len(files_info)} ({model_config['traindata_percentage']*100:.1f}%)")
                if len(files_info) == 0:
                    if rank == 0:
                        logging.info(f"包 {pkg}-{version} 的训练数据为空，跳过训练")
                    skipped_count += 1
                    continue
            
            try:
                if not pre_filtered and rank == 0:
                    logging.info(f"开始训练包 {pkg}-{version} 的LoRA模型")
                
                lora_model = trainLoraModelForPack_DDP(
                    model_config, pkg, version, tokenizer, corpus_path, 
                    dataset_type=knowledge_type, precision=precision,
                    rank=rank, world_size=world_size, local_rank=local_rank,
                    gpus_per_process=gpus_per_process,
                    enable_data_checking=enable_data_checking,
                    no_save=no_save
                )
                
                # 只在rank 0上保存模型（除非指定不保存）
                if rank == 0 and lora_model is not None and not no_save:
                    lora_save_path = pathConfigurator().getPath(model_config, pkg, version, model_name, knowledge_type=knowledge_type)
                    lora_model.save_pretrained(lora_save_path)
                    logging.info(f"成功训练并保存包 {pkg}-{version} 的LoRA模型到: {lora_save_path}")
                elif rank == 0 and lora_model is not None and no_save:
                    logging.info(f"成功训练包 {pkg}-{version} 的LoRA模型（未保存，no-save模式）")
                
                # 清理模型对象
                if lora_model is not None:
                    del lora_model
                
                trained_count += 1
                
                # 每个package训练完成后强制清理内存
                # comprehensive_memory_cleanup(rank, "package完成", f"{pkg}-{version}")
                
                # 如果内存仍然没有完全释放，使用设备重置
                if rank == 0 and torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    if allocated > 0.1:  # 如果还有超过100MB的分配内存
                        logging.warning(f"内存仍有 {allocated:.2f}GB 未释放，执行设备重置")
                        force_memory_reset_device(rank, "package完成_设备重置", f"{pkg}-{version}")
                
                # ---- NEW DIAGNOSTICS ----
                if rank == 0:
                    logging.info("--- Running Post-Cleanup Diagnostics ---")
                    # find_and_clear_lingering_tensors(rank, f"after training {pkg}-{version}")
                    # log_detailed_gpu_memory_report(rank, f"after training {pkg}-{version}")

            except Exception as e:
                if rank == 0:
                    logging.error(f"训练包 {pkg}-{version} 时出错: {e}")
                error_count += 1
                
                # 即使出错也要清理内存
                # comprehensive_memory_cleanup(rank, "package出错", f"{pkg}-{version}")
                
                # 出错后也执行设备重置，确保下一个包能正常训练
                # force_memory_reset_device(rank, "package出错_设备重置", f"{pkg}-{version}")

                # ---- NEW DIAGNOSTICS ----
                if rank == 0:
                    logging.info("--- Running Post-Error Diagnostics ---")
                    # find_and_clear_lingering_tensors(rank, f"after error in {pkg}-{version}")
                    log_detailed_gpu_memory_report(rank, f"after error in {pkg}-{version}")
                
                continue
    
    if rank == 0:
        logging.info(f"训练完成统计: 训练={trained_count}, 跳过={skipped_count}, 错误={error_count}")
    
    # 返回统计信息以供多worker模式使用
    return {
        'trained': trained_count,
        'skipped': skipped_count,
        'failed': error_count,
        'total': trained_count + skipped_count + error_count
    }


def buildandTrainLoraModel_DDP(config, dataloader, precision='fp16', pkg=None, version=None, knowledge_type=None, rank=0, world_size=1, local_rank=0, gpus_per_process=1,no_save=False):
    """
    分布式训练LoRA模型
    """
    # 记录训练开始和配置信息
    if rank == 0:
        logging.info("=" * 60)
        logging.info(f"buildandTrainLoraModel_DDP - 开始训练: {pkg}-{version}")
        logging.info(f"分布式训练参数: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        logging.info("=" * 60)
    
    model_name = config["model_name"].split("/")[-1]
    if pkg and version:
        pathConfig = pathConfigurator()
        output_adaptor_path = pathConfig.getPath(config, pkg, version, model_name, knowledge_type=knowledge_type)
        if rank == 0:
            logging.info(f"输出路径: {output_adaptor_path}")
    else:
        if rank == 0:
            logging.error("pkg和version必须提供")
        raise ValueError("pkg and version must be provided")
    
    # 记录训练开始前的内存使用情况
    log_gpu_memory_usage(rank, "新一个pkgVersion训练开始前", f"{pkg}-{version}")
    
    try:
        # 清理现有的GPU缓存
        torch.cuda.empty_cache()
        log_gpu_memory_usage(rank, "新一个pkgVersion训练开始前对于GPU缓存后占用,期望为0", f"{pkg}-{version}")
        # 计算每个进程应该使用的GPU范围
        gpu_start = rank * gpus_per_process
        gpu_end = (rank + 1) * gpus_per_process
        process_gpu_list = list(range(gpu_start, gpu_end))
        
        if rank == 0:
            logging.info(f"进程GPU分配: rank={rank}, gpu_start={gpu_start}, gpu_end={gpu_end}, process_gpu_list={process_gpu_list}")
        
        # 根据每个进程的GPU数量决定设备映射策略
        if gpus_per_process == 1:
            # 每个进程只使用一个GPU，使用简单的设备映射
            device_map = gpu_start  # 使用分配给当前进程的GPU
            if rank == 0:
                logging.info(f"单GPU模式，使用设备: cuda:{gpu_start}")
        else:
            # 每个进程使用多个GPU，需要使用复杂的设备映射策略
            if rank == 0:
                logging.info(f"多GPU模式，每个进程使用 {gpus_per_process} 个GPU: {process_gpu_list}")
            
            # 设置CUDA_VISIBLE_DEVICES环境变量，限制当前进程只能看到分配给它的GPU
            original_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            gpu_ids_str = ','.join(str(gpu_id) for gpu_id in process_gpu_list)
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
            
            if rank == 0:
                logging.info(f"设置CUDA_VISIBLE_DEVICES={gpu_ids_str}")
            
            use_balanced_device_map = config.get("use_balanced_device_map", False)
            
            if use_balanced_device_map:
                from utils.loraTrain.loraTrainUtils import create_balanced_device_map
                force_balance = config.get("force_balance", False)
                exclude_cpu = config.get("exclude_cpu", True)
                if rank == 0:
                    logging.info(f"均衡设备映射参数: force_balance={force_balance}, exclude_cpu={exclude_cpu}")
                    print("启用均衡设备映射...")
                
                device_map = create_balanced_device_map(
                    config["model_name"],
                    force_balance=force_balance,
                    exclude_cpu=exclude_cpu
                )
                if device_map is None:
                    if rank == 0:
                        logging.warning("均衡设备映射失败，回退到auto模式")
                        print("⚠️  均衡设备映射失败，回退到auto模式")
                    device_map = "auto"
                else:
                    if rank == 0:
                        logging.info(f"均衡设备映射创建成功: {type(device_map)}")
            else:
                # 使用配置中的设备映射，支持多GPU分布
                device_map = config.get("device_map", "auto")
                if rank == 0:
                    logging.info(f"使用标准设备映射: {device_map}")
        
        if rank == 0:
            print(f"使用设备映射: {device_map}")
        
        # 加载基础模型和tokenizer，使用指定的精度和设备映射
        if rank == 0:
            logging.info(f"开始加载基础模型: {config['model_name']}")
            logging.info(f"使用精度: {precision}")
        
        base_model, tokenizer = load_base_model(config["model_name"], device_map, precision)
        
        if rank == 0:
            logging.info("创建LoRA配置...")
            logging.info(f"LoRA配置参数: target_modules={config['target_modules']}, target_layers={config['target_layers']}, r={config['r']}, alpha={config['alpha']}")
        
        lora_config = create_lora_config(
            config["target_modules"], 
            config["target_layers"], 
            config["r"], 
            config["alpha"]
        )
        
        # 创建LoRA模型
        if rank == 0:
            logging.info("创建LoRA模型...")
        
        lora_model = get_peft_model(base_model, lora_config)
        
        # 对于真正的分布式训练（多进程），使用DDP包装模型
        if world_size > 1:
            if gpus_per_process == 1:
                # 每个进程一个GPU，使用标准DDP，将模型移动到分配给它的GPU
                lora_model = lora_model.to(f'cuda:{gpu_start}')
                
                lora_model = DDP(
                    lora_model,
                    device_ids=[gpu_start],
                    output_device=gpu_start,
                    find_unused_parameters=False
                )
                
                if rank == 0:
                    logging.info(f"模型已用DDP包装，设备: cuda:{gpu_start}")
            else:
                # 每个进程多个GPU，模型已经通过device_map分布到多个GPU
                # 不使用DDP，因为模型已经在多个GPU上分布
                if rank == 0:
                    logging.info(f"多GPU模式，模型已通过device_map分布到GPU {process_gpu_list}，不使用DDP包装")
        else:
            # 单进程训练，不需要DDP包装
            if rank == 0:
                logging.info("单进程训练，不使用DDP包装")
        
        # 分析模型参数分布（仅在rank 0上）
        if rank == 0:
            devices_found = {}
            model_for_analysis = lora_model.module if hasattr(lora_model, 'module') else lora_model
            
            for name, param in model_for_analysis.named_parameters():
                device = param.device
                if device not in devices_found:
                    devices_found[device] = []
                devices_found[device].append(name)
            
            logging.info("训练模型参数分布:")
            print("训练模型参数分布:")
            for device, params in devices_found.items():
                device_info = f"{device}: {len(params)} parameters"
                logging.info(f"- {device_info}")
                print(f"- {device_info}")
        
        # 训练模型
        if rank == 0:
            logging.info("开始分布式训练模型")
            logging.info(f"训练参数: epochs={config['num_epochs']}, lr={config['learning_rate']}")
            print(f"开始分布式训练模型")
        
        lora_model = train_lora_model_DDP(
            lora_model, 
            dataloader, 
            config["num_epochs"], 
            config["learning_rate"],
            config,
            output_adaptor_path,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            gpus_per_process=gpus_per_process,
            process_gpu_list=process_gpu_list,
            no_save=no_save
        )
        
        # 清理分布式训练资源
        if world_size > 1:
            dist.barrier()  # 确保所有进程都完成训练
            
            # 提取原始模型（去掉DDP包装）
            if hasattr(lora_model, 'module'):
                lora_model = lora_model.module
        
        # 将模型移到CPU并分离计算图（仅在rank 0上）
        if rank == 0:
            logging.info("训练完成，将模型移动到CPU...")
            lora_model = lora_model.cpu()
            for param in lora_model.parameters():
                param.requires_grad = False
        
        # 恢复原始的CUDA_VISIBLE_DEVICES环境变量
        if gpus_per_process > 1:
            if original_cuda_visible_devices:
                os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible_devices
            else:
                del os.environ['CUDA_VISIBLE_DEVICES']
        
        # 清理不需要的对象
        if rank == 0:
            logging.info("清理训练资源...")
        
        del base_model
        del tokenizer
        
        # 使用新的清理函数
        # comprehensive_memory_cleanup(rank, "训练完成后", f"{pkg}-{version}")
        
        if rank == 0:
            logging.info(f"训练完成: {pkg}-{version}")
            logging.info("=" * 60)
        
        return lora_model if rank == 0 else None
        
    except Exception as e:
        if rank == 0:
            logging.error(f"训练过程中出错: {str(e)}")
            logging.error("错误详情:")
            import traceback
            logging.error(traceback.format_exc())
            
            print(f"创建LoRA模型时出错: {str(e)}")
            traceback.print_exc()
        
        # 确保清理所有可能的资源
        if 'base_model' in locals():
            del base_model
        if 'tokenizer' in locals():
            del tokenizer
        if 'lora_model' in locals():
            del lora_model
        
        # 使用新的清理函数
        comprehensive_memory_cleanup(rank, "异常处理", f"{pkg}-{version}")
        
        # 异常情况下也执行设备重置
        force_memory_reset_device(rank, "异常处理_设备重置", f"{pkg}-{version}")
        
        raise e


def train_lora_model_DDP(lora_model, dataloader, num_epochs=10, learning_rate=1e-3, train_config=None, output_adaptor_path=None, rank=0, world_size=1, local_rank=0, gpus_per_process=1, process_gpu_list=None,no_save=False):
    """
    分布式训练LoRA模型（带数据分布检查）
    """
    from llmfoundry.optim import DecoupledLionW
    import os
    
    # 如果没有提供process_gpu_list，则根据rank和gpus_per_process计算
    if process_gpu_list is None:
        gpu_start = rank * gpus_per_process
        gpu_end = (rank + 1) * gpus_per_process
        process_gpu_list = list(range(gpu_start, gpu_end))
        
        if rank == 0:
            print(f"train_lora_model_DDP: 计算进程GPU列表: {process_gpu_list}")
    
    if rank == 0:
        print(f"train_lora_model_DDP: 使用GPU列表: {process_gpu_list}")
    
    
    # 尝试设置matplotlib后端，如果失败则禁用绘图功能
    try:

        matplotlib.use('Agg')  # 设置为非交互式后端
        plotting_enabled = True
    except Exception:
        if rank == 0:
            print("警告：无法设置matplotlib后端，损失曲线绘图功能将被禁用。")
        plotting_enabled = False
    
    # 使用 DecoupledLionW 优化器
    optimizer = DecoupledLionW(
        lora_model.parameters(),
        lr=float(learning_rate),
        betas=(0.9, 0.95),
        weight_decay=1e-6
    )
    
    # 梯度累积设置
    actual_batch_size = dataloader.batch_size
    target_batch_size = 16  # 默认目标批次大小
    
    # 如果提供了config，从中读取目标批次大小
    if train_config and "target_batch_size" in train_config:
        target_batch_size = train_config["target_batch_size"]
        if rank == 0:
            print(f"从配置中读取目标批次大小: {target_batch_size}")
    
    accumulation_steps = max(1, target_batch_size // actual_batch_size)
    if rank == 0:
        print(f"使用梯度累积: 实际批次大小={actual_batch_size}, 目标批次大小={target_batch_size}, 累积步数={accumulation_steps}")
    
    # 创建保存日志和模型的目录（仅在rank 0上）
    if rank == 0:
        save_path_base = train_config.get("save_path_base", "/datanfs2/chenrongyi/models/versiBCB") if train_config else "/datanfs2/chenrongyi/models/versiBCB"
        log_dir = os.path.join(output_adaptor_path, "training_logs")
        checkpoint_dir = os.path.join(output_adaptor_path, "checkpoints")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 为本次训练创建唯一的时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建日志文件路径和checkpoint路径基础
        log_file = os.path.join(log_dir, f"training_log_{timestamp}_rank{rank}.csv")
        log_plot = os.path.join(log_dir, f"loss_curve_{timestamp}_rank{rank}.png")
        checkpoint_base = os.path.join(checkpoint_dir, f"checkpoint_{timestamp}")
        
        # 初始化损失记录
        epoch_losses = []
        step_losses = []
        batch_losses = []
        
        # 写入日志文件头
        try:
            with open(log_file, "w") as f:
                f.write("epoch,step,batch,loss\n")
        except Exception as e:
            if rank == 0:
                print(f"Error writing to log file: {e}")
    
    # 获取精度设置
    
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        valid_batch_count = 0
        step_count = 0
        
        # 设置分布式采样器的epoch（对于shuffle）
        if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
            
            # 检查采样器索引分布
            # if data_checker: # This line was removed as per the new_code, as data_checker is no longer passed
            #     data_checker.check_sampler_indices(dataloader.sampler, epoch + 1, batch_count)
        
        # 在每个epoch开始前清零梯度
        optimizer.zero_grad()
        
        epoch_batch_count = 0
        for batch in dataloader:
            epoch_batch_count += 1
            
            # 提取batch数据
            inputs = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]
            
            # 将数据移动到正确的设备
            if gpus_per_process == 1:
                # 单GPU模式，移动到分配给当前进程的GPU
                target_device = f'cuda:{process_gpu_list[0]}' if process_gpu_list else f'cuda:{local_rank}'
                inputs = inputs.to(target_device)
                labels = labels.to(target_device)
                attention_mask = attention_mask.to(target_device)
            else:
                # 多GPU模式，移动到进程的第一个GPU
                # 在设置了CUDA_VISIBLE_DEVICES后，这里使用cuda:0指向进程的第一个GPU
                target_device = 'cuda:0'
                inputs = inputs.to(target_device)
                labels = labels.to(target_device)
                attention_mask = attention_mask.to(target_device)
            
            try:
                # 前向传播
                # with torch.amp.autocast(device_type='cuda', dtype=torch_dtype, enabled=torch.cuda.is_available()):
                outputs = lora_model(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss = loss / accumulation_steps
                
                # 立即提取loss值，然后清理outputs对象
                current_loss = loss.item() * accumulation_steps
                
                # 清理模型输出对象，避免残留
                if hasattr(outputs, 'logits'):
                    outputs.logits = None
                if hasattr(outputs, 'past_key_values'):
                    outputs.past_key_values = None
                if hasattr(outputs, 'hidden_states'):
                    outputs.hidden_states = None
                if hasattr(outputs, 'attentions'):
                    outputs.attentions = None
                del outputs
                
                # 反向传播
                loss.backward()
                
                # 检查loss是否为NaN，只有非NaN值才计入平均值
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += current_loss
                    valid_batch_count += 1
                    
                    # 只在rank 0上记录损失
                    if rank == 0:
                        batch_losses.append(current_loss)
                else:
                    if rank == 0:
                        print(f"Warning: NaN or Inf loss detected in Epoch {epoch+1}, Batch {batch_count+1}. Ignoring this batch for average calculation.")
                
                batch_count += 1
                step_count += 1
                
                # 每个batch都输出当前batch的loss（仅在rank 0上）
                if rank == 0 and batch_count % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_count}/{len(dataloader)}, Current Batch Loss: {current_loss:.4f}")
                
                # 每accumulation_steps步更新一次参数
                if step_count % accumulation_steps == 0 or batch_count == len(dataloader):
                    # 梯度裁剪，防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(lora_model.parameters(), max_norm=1.0)
                    
                    # 参数更新
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # 每隔几步清理一次缓存，避免attention缓存积累
                    if step_count % (accumulation_steps * 10) == 0:
                        torch.cuda.empty_cache()
                    
                    # 计算当前步骤的平均损失，只使用有效的batch
                    if valid_batch_count > 0:
                        avg_loss = total_loss / valid_batch_count
                    else:
                        avg_loss = float('nan')
                    
                    # 只在rank 0上记录和输出
                    if rank == 0:
                        if not torch.isnan(torch.tensor(avg_loss)) and not torch.isinf(torch.tensor(avg_loss)):
                            step_losses.append(avg_loss)
                        
                        # 记录到日志文件
                        with open(log_file, "a") as f:
                            f.write(f"{epoch+1},{step_count // accumulation_steps},{batch_count},{avg_loss:.6f}\n")
                        
                        # 每2个更新步骤输出一次当前avg loss
                        if (step_count // accumulation_steps) % 2 == 0:
                            print(f"Epoch {epoch+1}/{num_epochs}, Step {step_count // accumulation_steps}, Batch {batch_count}/{len(dataloader)}, Avg Loss: {avg_loss:.4f}")
                
            except RuntimeError as e:
                if "expected all tensors to be on the same device" in str(e).lower():
                    if rank == 0:
                        print(f"设备不一致错误: {e}")
                        print("跳过此batch并继续训练...")
                    
                    batch_count += 1
                    step_count += 1
                    optimizer.zero_grad()
                    continue
                elif "out of memory" in str(e).lower():
                    if rank == 0:
                        print(f"GPU内存不足: {e}")
                        print("清理缓存并跳过此batch...")
                    torch.cuda.empty_cache()
                    
                    batch_count += 1
                    step_count += 1
                    optimizer.zero_grad()
                    continue
                else:
                    if rank == 0:
                        print(f"训练过程中出现RuntimeError: {e}")
                        import traceback
                        traceback.print_exc()
                    raise e
        
        # 在epoch结束时检查batch处理数量
        if rank == 0:
            print(f"Rank {rank}: Epoch {epoch + 1} 处理了 {epoch_batch_count} 个batches")
        
        # 收集所有rank的batch处理数量
        if world_size > 1:
            batch_counts = torch.tensor([epoch_batch_count], dtype=torch.long).cuda()
            all_batch_counts = [torch.zeros(1, dtype=torch.long).cuda() for _ in range(world_size)]
            dist.all_gather(all_batch_counts, batch_counts)
            
            if rank == 0:
                # 从张量列表中提取数值
                batch_counts_list = [tensor.cpu().item() for tensor in all_batch_counts]
                print(f"各rank本epoch处理的batch数: {batch_counts_list}")
                
                if len(set(batch_counts_list)) == 1:
                    print(f"✅ 所有rank处理了相同数量的batch: {batch_counts_list[0]}")
                else:
                    print(f"❌ 不同rank处理的batch数量不同: {batch_counts_list}")
        
        # 使用更健壮的同步，确保所有进程都到达这个点
        print(f"rank {rank} 训练进入同步点前")
        if world_size > 1:
            if rank == 0:
                print("所有rank的epoch训练循环完成，准备进入同步点...")
            dist.barrier()
            if rank == 0:
                print("所有rank已成功同步。")
        
        # 计算并存储此epoch的平均损失（仅在rank 0上）
        if rank == 0:
            avg_epoch_loss = total_loss / max(1, valid_batch_count)
            epoch_losses.append(avg_epoch_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss:.4f} (基于 {valid_batch_count} 个有效batch)")
            
            # 每2个epoch保存一次模型
            if not no_save:
                if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
                    checkpoint_path = f"{checkpoint_base}_epoch{epoch+1}"
                    os.makedirs(checkpoint_path, exist_ok=True)
                    print(f"保存中间检查点到 {checkpoint_path}")
                    
                    # 从DDP中提取原始模型进行保存
                    model_to_save = lora_model.module if hasattr(lora_model, 'module') else lora_model
                    model_to_save.save_pretrained(checkpoint_path)
                    
                    # 保存最新的损失曲线图
                    if plotting_enabled:
                        save_training_plots(epoch_losses, step_losses, batch_losses, log_plot, is_final=False)
        
    # 训练结束，保存最终的损失曲线和统计摘要（仅在rank 0上）
    if rank == 0:
        if plotting_enabled:
            save_training_plots(epoch_losses, step_losses, batch_losses, log_plot, is_final=True)
        else:
            print("训练完成。由于matplotlib不可用，损失曲线未绘制。")
        
        # 打印训练损失统计摘要
        if epoch_losses or step_losses or batch_losses:
            summary = create_loss_summary(epoch_losses, step_losses, batch_losses)
            print_loss_summary(summary,"训练完成")
    
    # 清理训练资源
    if rank == 0:
        logging.info("清理训练资源...")
    
    # 清理优化器状态
    if 'optimizer' in locals():
        clear_optimizer_states(optimizer, rank)
        del optimizer
    
    # 清理数据加载器
    if 'dataloader' in locals():
        del dataloader
    
    # 清理损失记录
    if 'epoch_losses' in locals():
        del epoch_losses
    if 'step_losses' in locals():
        del step_losses
    if 'batch_losses' in locals():
        del batch_losses
    
    # 使用新的清理函数
    comprehensive_memory_cleanup(rank, "训练函数结束", "")
    
    return lora_model


def main():
    """主函数"""
    # 初始化分布式训练
    rank, world_size, local_rank = init_distributed_training()
    
    model_config = load_config(LORA_CONFIG_PATH)
    args = argparse.ArgumentParser()
    
    # 原有参数
    args.add_argument("--precision", type=str, default="bf16", help="precision of the model", choices=["fp16", "fp32", "bf16"])
    args.add_argument("--dataset_type", type=str, default="docstring", help="dataset type", choices=["docstring", "srccodes"])
    args.add_argument("--corpus_path", type=str, default="/datanfs4/chenrongyi/data/docs", help="corpus path")
    args.add_argument("--benchmark_data_path", type=str, default="data/VersiBCB_Benchmark/vace_datas.json", help="benchmark data path")
    args.add_argument("--benchmark_paths", type=str, nargs='+', default=None, help="multiple benchmark data paths")
    args.add_argument("--loraadaptor_save_path_base", type=str, default="/datanfs4/chenrongyi/models/loraadaptors/", help="lora adaptor save path base")
    args.add_argument("--model_name", type=str, default="/datanfs2/chenrongyi/models/Llama-3.1-8B-Instruct", help="model name")
    args.add_argument("--log_dir", type=str, default=None, help="指定日志目录")
    args.add_argument("--override_data_percentage", type=str, default=None, help="override the data percentage")
    
    # 设备映射参数
    args.add_argument("--use_balanced_device_map", type=bool, default=False, help="是否使用均衡设备映射（分布式训练时建议关闭）")
    args.add_argument("--force_balance", type=bool, default=True, help="是否强制均衡分配")
    args.add_argument("--exclude_cpu", type=bool, default=True, help="是否排除CPU设备")
    args.add_argument("--check_r_consistency", type=bool, default=True, help="是否检查r值一致性")
    args.add_argument("--strict_r_check", type=bool, default=False, help="是否严格检查r值一致性")
    args.add_argument("--use_dynamic_device_map", type=bool, default=False, help="是否使用动态设备映射策略")
    args.add_argument("--balance_threshold", type=float, default=0.3, help="动态映射的均衡阈值")
    args.add_argument("--device_map_strategy", type=str, default="auto", choices=["auto", "balanced", "dynamic"], help="设备映射策略选择")
    args.add_argument("--gpus_per_process", type=int, default=1, help="每个进程使用的GPU数量")
    
    # 添加数据检查参数
    args.add_argument("--enable_data_checking", action="store_true", help="启用数据分布正交性检查")
    args.add_argument("--data_check_interval", type=int, default=10, help="数据检查间隔（batch数）")
    
    # 添加模型保存控制参数
    args.add_argument("--no_save", action="store_true", help="训练完成后不保存模型")
    
    # 添加单包训练参数
    args.add_argument("--single_package", type=str, default=None, help="训练单个包版本，格式: pkg:version")
    
    # 添加预过滤参数
    args.add_argument("--enable_prefilter", action="store_true", help="启用预过滤功能，提前过滤掉不需要训练的包版本")
    
    args = args.parse_args()
    
    # 设置日志（仅在rank 0上输出详细信息）
    if rank == 0:
        log_dir = setup_logging(args)
        logging.info(f"分布式训练初始化完成 - Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
    
    # 详细的CUDA环境检测和诊断（仅在rank 0上）
    if rank == 0:
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
        
        logging.info("=== 配置信息 ===")
        logging.info(f"分布式训练: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        logging.info(f"每个进程GPU数量: {args.gpus_per_process}")
    
    # 覆盖基础配置
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
    
    # 确定使用哪种方式加载包版本信息
    pack_versions = None
    benchmark_data_path = None
    
    if args.single_package:
        # 单包训练模式
        if rank == 0:
            logging.info("=== 单包训练模式 ===")
            logging.info(f"训练包: {args.single_package}")
        
        try:
            pkg, version = args.single_package.split(":")
            pack_versions = {pkg: [version]}
            if rank == 0:
                logging.info(f"单包训练: {pkg}-{version}")
        except ValueError:
            if rank == 0:
                logging.error(f"单包参数格式错误: {args.single_package}，应为 pkg:version")
            sys.exit(1)
    
    elif args.benchmark_paths is not None:
        # 使用多个benchmark文件
        if rank == 0:
            logging.info("=== 使用多个benchmark文件模式 ===")
            logging.info(f"Benchmark文件列表: {args.benchmark_paths}")
        pack_versions = load_and_aggregate_package_versions(args.benchmark_paths)
    else:
        # 使用单个benchmark文件
        if rank == 0:
            logging.info("=== 使用单个benchmark文件模式 ===")
            logging.info(f"Benchmark文件: {args.benchmark_data_path}")
        benchmark_data_path = args.benchmark_data_path
    
    # 数据分割（如果是多进程训练且不是单包模式）
    if pack_versions is not None and world_size > 1 and not args.single_package:
        if rank == 0:
            logging.info(f"分布式训练模式: 将数据分割给 {world_size} 个进程")
        
        # 收集所有包版本组合
        all_pkg_versions = []
        for pkg, versions in pack_versions.items():
            for version in versions:
                all_pkg_versions.append((pkg, version))
        
        # 按rank分割数据
        total_combinations = len(all_pkg_versions)
        combinations_per_rank = total_combinations // world_size
        remainder = total_combinations % world_size
        
        start_idx = rank * combinations_per_rank + min(rank, remainder)
        end_idx = start_idx + combinations_per_rank + (1 if rank < remainder else 0)
        
        rank_pkg_versions = all_pkg_versions[start_idx:end_idx]
        
        if rank == 0:
            logging.info(f"数据分割完成: 总计 {total_combinations} 个包版本组合")
            logging.info(f"每个进程分配: {combinations_per_rank}+{1 if rank < remainder else 0} 个组合")
        
        # 重新构建当前rank的pack_versions字典
        rank_pack_versions = {}
        for pkg, version in rank_pkg_versions:
            if pkg not in rank_pack_versions:
                rank_pack_versions[pkg] = []
            rank_pack_versions[pkg].append(version)
        
        pack_versions = rank_pack_versions
        
        if rank == 0:
            logging.info(f"Rank {rank} 分配到 {len(pack_versions)} 个包，共 {sum(len(versions) for versions in pack_versions.values())} 个包版本组合")
    
    if rank == 0:
        if args.no_save:
            logging.info("开始分布式LoRA模型训练（no-save模式，训练完成后不保存模型）...")
        else:
            logging.info("开始分布式LoRA模型训练...")
    
    try:
        stats = trainLoraModelsForVersiBCB_DDP(
            benchmark_data_path=benchmark_data_path,
            model_config=model_config,
            precision=args.precision,
            knowledge_type=args.dataset_type,
            corpus_path=args.corpus_path,
            pack_versions=pack_versions,
            pre_filtered=pack_versions is not None and world_size > 1 and not args.single_package,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            gpus_per_process=args.gpus_per_process,
            enable_data_checking=args.enable_data_checking,
            no_save=args.no_save,
            enable_prefilter=args.enable_prefilter
        )
        
        if rank == 0:
            logging.info("分布式LoRA模型训练成功完成!")
            logging.info(f"最终统计: 训练={stats['trained']}, 跳过={stats['skipped']}, 错误={stats['failed']}, 总计={stats['total']}")
            logging.info(f"所有日志已保存到: {log_dir}")
        if args.single_package:
            if stats['trained'] > 0:
                logging.info(f"单包训练成功完成! 训练了 {stats['trained']} 个包版本")
            else:
                if stats['failed'] > 0:
                    logging.info(f"单包训练失败! 失败导致")
                    raise Exception(f"单包训练失败! 失败导致")
                else:
                    logging.info(f"单包训练失败! 跳过，数据为空")
                    raise Exception(f"单包训练失败! 跳过，数据为空或者模型已存在")
                

    except Exception as e:
        if rank == 0:
            logging.error(f"分布式训练过程中发生错误: {e}")
        raise
    
    # 清理分布式训练
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main() 