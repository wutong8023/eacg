"""
该脚本用于一次性构建指定任务所需的所有collections。
相比于原有的buildCollection.py，该脚本：
1. 针对特定任务（如VACE、VSCC）构建所有需要的collections
2. 使用新的存储结构，每个dependency组合有独立的文件夹
3. 支持多种数据集和任务类型
"""

import json
import os
import logging
import argparse
import chromadb
from chromadb.config import Settings
from utils.RAGutils.CollectionBuilder import CollectionBuilder
from utils.RAGutils.collectionConfig.collectionBuild_config import (
    RAG_COLLECTION_BASE, KNOWLEDGE_TYPE, EMBEDDING_SOURCE, CORPUS_PATH, EMBEDDING_BASE_PATH,
    LOCAL_EMBEDDING_MODEL, TOGETHERAI_EMBEDDING_MODEL
)

def build_all_task_collections(dataset_path: str, 
                              dataset_name: str,
                              task_name: str,
                              ban_deprecation: bool = False,
                              force_rebuild: bool = False, 
                              batch_size: int = 250, 
                              num_processes: int = 1):
    """
    为指定任务构建所有需要的collections
    
    Args:
        dataset_path: 数据集文件路径
        dataset_name: 数据集名称 ("VersiBCB" 或 "VersiCode")
        task_name: 任务名称 ("VACE" 或 "VSCC")
        ban_deprecation: 是否使用ban deprecation版本的数据
        force_rebuild: 是否强制重建已有集合
        batch_size: 批处理大小
        num_processes: 使用的进程数量
    """
    # 设置日志
    log_filename = f'build_task_collections_{dataset_name}_{task_name}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='a'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("BuildTaskCollections")
    logger.info(f"Starting collection building for {dataset_name} {task_name} task")
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Ban deprecation: {ban_deprecation}")
    logger.info(f"Force rebuild: {force_rebuild}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Num processes: {num_processes}")
    
    # 验证数据集文件存在
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        return {}
    
    # 创建基础存储目录
    base_path = os.path.join(RAG_COLLECTION_BASE, KNOWLEDGE_TYPE, EMBEDDING_SOURCE)
    os.makedirs(base_path, exist_ok=True)
    logger.info(f"Base collection path: {base_path}")
    
    # 配置embedding参数
    embed_func_args = {
        'source': EMBEDDING_SOURCE,
        'model_name': LOCAL_EMBEDDING_MODEL if EMBEDDING_SOURCE == 'local' else TOGETHERAI_EMBEDDING_MODEL,
        'batch_size': 64
    }
    
    # 创建一个临时的ChromaDB客户端用于初始化CollectionBuilder
    # 注意：实际的collection会使用独立的客户端和路径
    temp_client = chromadb.PersistentClient(
        path=base_path,
        settings=Settings(
            allow_reset=True,
            anonymized_telemetry=False,
            is_persistent=True
        )
    )
    
    # 创建构建器
    builder = CollectionBuilder(
        chroma_client=temp_client,
        embed_func_args=embed_func_args,
        corpus_base=CORPUS_PATH,
        collection_base=RAG_COLLECTION_BASE,
        batch_size=batch_size,
        verbose=True,
        num_processes=num_processes
    )
    
    # 构建所有任务相关的collections
    try:
        collection_map = builder.build_all_task_collections(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            task_name=task_name,
            ban_deprecation=ban_deprecation,
            force_rebuild=force_rebuild,
            num_processes=num_processes
        )
        
        logger.info(f"Successfully built {len(collection_map)} collections for {task_name} task")
        
        # 保存collection映射到文件
        mapping_file = f"collection_mapping_{dataset_name}_{task_name}{'_BD' if ban_deprecation else ''}.json"
        with open(mapping_file, 'w') as f:
            json.dump(collection_map, f, indent=2)
        logger.info(f"Collection mapping saved to: {mapping_file}")
        
        return collection_map
        
    except Exception as e:
        logger.error(f"Error building collections: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}

def get_dataset_path(dataset_name: str, task_name: str, ban_deprecation: bool) -> str:
    """
    根据数据集名称、任务名称和ban_deprecation标志获取数据集路径
    
    Args:
        dataset_name: 数据集名称
        task_name: 任务名称
        ban_deprecation: 是否使用ban deprecation版本
        
    Returns:
        数据集文件路径
    """
    if dataset_name == "VersiBCB":
        data_path = 'data/VersiBCB_Benchmark'
        data_name = f"{task_name.lower()}_datas{'_for_warning' if ban_deprecation else ''}.json"
        return os.path.join(data_path, data_name)
    elif dataset_name == "VersiCode":
        # VersiCode数据集路径需要根据实际情况调整
        return "benchmark/data/VersiCode_Benchmark/blockcode_completion.json"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为指定任务构建所有需要的ChromaDB collections")
    parser.add_argument("--dataset", type=str, default="VersiBCB", 
                       choices=["VersiBCB", "VersiCode"],
                       help="数据集名称")
    parser.add_argument("--task", type=str, default="VACE",
                       choices=["VACE", "VSCC"],
                       help="任务名称")
    parser.add_argument("--dataset-path", type=str, default=None,
                       help="数据集文件路径（如果不指定，将根据dataset和task自动推断）")
    parser.add_argument("--ban-deprecation", action="store_true",
                       help="是否使用ban deprecation版本的数据")
    parser.add_argument("--force", action="store_true", 
                       help="强制重建已有集合")
    parser.add_argument("--batch-size", type=int, default=250,
                       help="每批处理的文档数量")
    parser.add_argument("--num-processes", type=int, default=1,
                       help="使用的进程数量，默认为1（不使用多进程）")
    
    args = parser.parse_args()
    
    # 确定数据集路径
    if args.dataset_path:
        dataset_path = args.dataset_path
    else:
        dataset_path = get_dataset_path(args.dataset, args.task, args.ban_deprecation)
    
    print(f"Building collections for {args.dataset} {args.task} task")
    print(f"Dataset path: {dataset_path}")
    print(f"Knowledge type: {KNOWLEDGE_TYPE}")
    print(f"Embedding source: {EMBEDDING_SOURCE}")
    print(f"Ban deprecation: {args.ban_deprecation}")
    print(f"Force rebuild: {args.force}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num processes: {args.num_processes}")
    print("-" * 50)
    
    # 构建所有collections
    collections = build_all_task_collections(
        dataset_path=dataset_path,
        dataset_name=args.dataset,
        task_name=args.task,
        ban_deprecation=args.ban_deprecation,
        force_rebuild=args.force,
        batch_size=args.batch_size,
        num_processes=args.num_processes
    )
    
    print(f"\n成功构建了 {len(collections)} 个collections")
    print("Collection构建完成！") 