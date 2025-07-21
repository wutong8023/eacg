import hashlib
import os
import json
import logging
import numpy as np
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from typing import Dict, List, Tuple, Any, Optional, Set, Union
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import torch
import time
import random
import concurrent
import sys
import fcntl
import tempfile

from utils.RAGutils.RAGEmbedding import PrecomputedEmbeddingsManager, CustomEmbeddingFunction
from utils.RAGutils.document_utils import get_version
from utils.getDependencyUtils import dict_to_pkg_ver_tuples

# 设置多进程启动方法为spawn
# 这是解决CUDA在多进程中的关键
multiprocessing.set_start_method('spawn', force=True)

# 添加GPU资源获取函数
def acquire_gpu_resource(logger, max_retries=60, retry_interval=5):
    """
    尝试获取GPU资源，如果GPU内存不足则等待
    
    Args:
        logger: 日志记录器
        max_retries: 最大重试次数
        retry_interval: 重试间隔(秒)
        
    Returns:
        bool: 是否成功获取GPU资源
    """
    for attempt in range(max_retries):
        try:
            if torch.cuda.is_available():
                # 检查GPU内存是否足够
                # 这里使用简化的判断方法，实际应用可能需要更复杂的逻辑
                torch.cuda.empty_cache()  # 尝试清理缓存
                
                # 可选: 尝试分配少量内存测试可用性
                test_tensor = torch.zeros(10, 10).cuda()
                del test_tensor
                
                # 成功分配，表示资源可用
                return True
            else:
                # 没有GPU可用，直接返回False
                return False
        except RuntimeError as e:
            # 捕获"CUDA out of memory"等错误
            if "CUDA out of memory" in str(e):
                # 增加随机时间，避免多进程同时重试
                jitter = random.uniform(0, 1.0)
                wait_time = retry_interval + jitter
                
                logger.info(f"Process {os.getpid()} waiting for GPU resources (attempt {attempt+1}/{max_retries}), retrying in {wait_time:.1f}s")
                time.sleep(wait_time)
            else:
                # 其他CUDA错误，可能是初始化问题
                logger.error(f"CUDA error: {e}")
                return False
    
    logger.error(f"Process {os.getpid()} failed to acquire GPU resources after {max_retries} attempts")
    return False

# 全局函数同上，但添加了GPU设备管理
def process_dependency(dep_tuple, chroma_client_path, embed_func_args, corpus_base, collection_base, batch_size, embedding_base_path, skip_existing=True):
    """
    处理单个依赖项的独立函数，用于多进程调用
    
    Args:
        dep_tuple: 依赖项元组
        chroma_client_path: ChromaDB客户端基础路径
        embed_func_args: 嵌入参数
        corpus_base: 语料库基础路径
        collection_base: 集合基础路径
        batch_size: 批处理大小
        embedding_base_path: 嵌入基础路径
        skip_existing: 是否跳过已存在的集合
        
    Returns:
        (collection_hash, collection_name) 元组
    """
    # 创建专用日志记录器并确保它输出到控制台和文件
    logger = logging.getLogger(f"CollectionBuilder-Process-{os.getpid()}")
    logger.handlers = []  # 清除已有的处理器
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler(stream=sys.stdout)  # 明确指定stdout
    console_formatter = logging.Formatter('%(asctime)s - [PID:%(process)d] - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 添加文件处理器
    file_handler = logging.FileHandler(f'collection_builder_proc_{os.getpid()}.log', mode='a')
    file_formatter = logging.Formatter('%(asctime)s - [PID:%(process)d] - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 不要向上传播日志事件
    
    # 输出初始信息确认日志正常工作
    logger.info(f"Process {os.getpid()} started for dependency processing")
    
    # 尝试获取GPU资源
    if not acquire_gpu_resource(logger):
        logger.error(f"Process {os.getpid()} could not acquire GPU resources and won't continue without GPU")
        return None, None
    
    # 设置GPU设备
    torch.cuda.set_device(0)  # 使用第一个GPU
    
    # 记录本进程的GPU信息
    device_info = f"CUDA device {torch.cuda.current_device()}: {torch.cuda.get_device_name()}"
    logger.info(f"Process {os.getpid()} acquired {device_info}")
    
    # 转换回字典
    dependencies = dict(dep_tuple)
    
    # 生成集合哈希和依赖文件夹名
    dep_str = "_".join([f"{pkg}_{ver}" for pkg, ver in dependencies.items() if ver is not None])
    collection_hash = hashlib.md5(dep_str.encode()).hexdigest()[:60]
    
    # 生成依赖文件夹名
    valid_deps = [(pkg, ver) for pkg, ver in dependencies.items() if ver is not None]
    sorted_deps = sorted(valid_deps)
    dependency_folder = "_".join([f"{pkg}_{ver}" for pkg, ver in sorted_deps])
    
    # 创建与单进程模式一致的路径结构
    collection_path = os.path.join(chroma_client_path, dependency_folder, collection_hash)
    os.makedirs(collection_path, exist_ok=True)
    
    # 创建独立的ChromaDB客户端实例
    client = chromadb.PersistentClient(
        path=collection_path,
        settings=Settings(
            allow_reset=True,
            anonymized_telemetry=False,
            is_persistent=True
        )
    )
    
    # 创建嵌入管理器
    embeddings_manager = PrecomputedEmbeddingsManager(embedding_base_path)
    
    # 检查是否应跳过
    if skip_existing:
        try:
            existing_collection = client.get_collection(name=collection_hash)
            if existing_collection.count() > 0:
                logger.info(f"Skipping existing collection: {collection_hash}")
                return collection_hash, collection_hash
        except Exception:
            # 集合不存在，继续创建
            pass
    
    logger.info(f"Building collection: {collection_hash}")
    
    # 创建embedding函数实例
    embedding_function = CustomEmbeddingFunction(**embed_func_args)
    
    # 创建集合
    try:
        collection = client.get_or_create_collection(
            name=collection_hash,
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_function
        )
        logger.info(f"Created collection: {collection_hash}")
    except Exception as e:
        logger.error(f"Error creating collection {collection_hash}: {e}")
        return collection_hash, None
    
    # 加载文档和嵌入
    all_documents = []
    all_embeddings = []
    
    for package, version in dependencies.items():
        if version is None:
            logger.info(f"Skipping {package} with None version")
            continue
        
        # 尝试加载预计算的嵌入
        cached_data = embeddings_manager.load_embeddings(package, version)
        documents_for_package = []
        
        # 如果有预计算的文档，直接使用
        if cached_data is not None and 'documents' in cached_data:
            logger.info(f"Using precomputed documents for {package} {version}")
            documents_for_package = cached_data['documents']
        else:
            # 否则从语料库加载
            corpus_path = os.path.join(corpus_base, package, version + ".jsonl")
            try:
                with open(corpus_path, "r") as f:
                    for line in f:
                        documents_for_package.append(line)
                logger.info(f"Loaded {len(documents_for_package)} documents for {package} {version}")
            except Exception as e:
                logger.error(f"Error loading corpus for {package} {version}: {e}")
                logger.error(f"Corpus path: {corpus_path}")
                continue
        
        # 如果没有文档，跳过处理
        if not documents_for_package:
            logger.warning(f"No documents loaded for {package} {version}, skipping")
            continue
        
        # 处理嵌入
        package_embeddings = None
        if cached_data is not None and 'embeddings' in cached_data and len(cached_data['embeddings']) == len(documents_for_package):
            logger.info(f"Using cached embeddings for {package} {version}")
            package_embeddings = cached_data['embeddings']
        else:
            logger.info(f"Computing embeddings for {package} {version}")
            package_embeddings = embeddings_manager.precompute_embeddings(
                package, version, documents_for_package,
                embedding_function,
                batch_size=batch_size
            )
        
        if package_embeddings is not None:
            all_documents.extend(documents_for_package)
            all_embeddings.extend(package_embeddings)
        else:
            logger.warning(f"Failed to get embeddings for {package} {version}")
    
    if not all_documents:
        logger.warning(f"No documents available for dependencies, collection remains empty")
        return collection_hash, collection_hash
    
    # 填充集合
    total_docs = len(all_documents)
    total_batches = (total_docs + batch_size - 1) // batch_size
    
    logger.info(f"Adding {total_docs} documents to collection {collection_hash} in {total_batches} batches")
    
    # 使用tqdm显示进度，设置position确保多进程下的进度条显示正常
    position = os.getpid() % 10  # 简单的position计算，避免重叠
    batch_range = range(total_batches)
    
    # 关键修改：使用file=sys.stdout确保在子进程中显示，设置position避免进度条重叠
    for batch_idx in tqdm(batch_range, 
                          desc=f"PID {os.getpid()} adding docs to {collection_hash[:8]}", 
                          unit="batch", 
                          position=position,
                          leave=True,
                          file=sys.stdout):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_docs)
        
        batch_documents = all_documents[start_idx:end_idx]
        batch_ids = [f"{collection_hash}_{idx}" for idx in range(start_idx, end_idx)]
        
        # 处理空文档
        processed_batch_documents = [(doc if doc and doc.strip() else " ") for doc in batch_documents]

        try:
            if all_embeddings:
                batch_embeddings = all_embeddings[start_idx:end_idx]
                if isinstance(batch_embeddings, np.ndarray):
                    batch_embeddings = batch_embeddings.tolist()
                elif isinstance(batch_embeddings, list) and batch_embeddings and isinstance(batch_embeddings[0], np.ndarray):
                    batch_embeddings = [emb.tolist() for emb in batch_embeddings]
                
                collection.upsert(
                    documents=processed_batch_documents,
                    ids=batch_ids,
                    embeddings=batch_embeddings
                )
            else:
                collection.upsert(
                    documents=processed_batch_documents,
                    ids=batch_ids
                )
        except Exception as e:
            error_msg = f"Error adding batch {batch_idx + 1} to collection {collection_hash}: {e}"
            logger.error(error_msg)
    
    logger.info(f"Collection {collection_hash} completed")
    return collection_hash, collection_hash


class CollectionBuildingLockedException(Exception):
    """当collection正在被其他进程构建时抛出的异常"""
    pass


class CollectionBuilder:
    """
    集合构建器，专门用于创建和管理ChromaDB集合。
    
    该类负责:
    1. 为给定的依赖项创建集合
    2. 批量处理文档添加
    3. 构建索引和优化集合性能
    初始化集合构建器
    
    Args:
        chroma_client: ChromaDB客户端实例
        embed_func_args: 嵌入函数的参数字典
        corpus_base: 语料库基础路径,语料库的结构为corpus_base/pkg/version.jsonl
        collection_base: 集合存储基础路径,结构为collection_base/knowledge_type/embed_model/collection_dependency/
        batch_size: 批处理大小
        verbose: 是否打印详细日志
        num_processes: 多进程处理数量，默认为1（不使用多进程）
    """
    
    def __init__(self, 
                 chroma_client: chromadb.PersistentClient,
                 embed_func_args: Dict[str, Any],
                 corpus_base: str = None, # 如果不存在，就会使用该参数
                 collection_base: str = None, # 如果不存在，就会使用该参数
                 batch_size: int = 250,
                 verbose: bool = True,
                 num_processes: int = 1,
                 knowledge_type: str = "docstring",
                 embedding_base_path: str = None):
        """
        初始化集合构建器
        
        Args:
            chroma_client: ChromaDB客户端实例
            embed_func_args: 嵌入函数的参数字典
            corpus_base: 语料库基础路径,其之下
            collection_base: 集合存储基础路径
            batch_size: 批处理大小
            verbose: 是否打印详细日志
            num_processes: 多进程处理数量，默认为1（不使用多进程）
        """
        self.chroma_client = chroma_client
        self.embed_func_args = embed_func_args
        self.corpus_base = corpus_base
        self.collection_base = collection_base
        self.batch_size = batch_size
        self.verbose = verbose
        self.num_processes = num_processes
        self.embedding_base_path = embedding_base_path
        self.embeddings_manager = PrecomputedEmbeddingsManager(self.embedding_base_path)
        self.logger = logging.getLogger("CollectionBuilder")
        self.knowledge_type = knowledge_type
        
        if verbose:
            # 确保日志配置已设置
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
    
    def get_collection_hash(self, dependencies: Dict[str, str]) -> str:
        """
        为依赖项列表生成唯一哈希
        支持两种格式：
        1. dict[pkg, ver] - 标准格式
        2. dict[pkg, list[ver]] - appendSrcDep后的格式
        
        Args:
            dependencies: 依赖名称和版本的字典
            
        Returns:
            哈希字符串
        """
        # 使用dict_to_pkg_ver_tuples处理两种格式
        pkg_ver_tuples = dict_to_pkg_ver_tuples(dependencies)
        
        # 过滤掉None版本的依赖项，然后排序并生成字符串
        valid_deps = [(pkg, ver) for pkg, ver in pkg_ver_tuples if ver is not None]
        sorted_deps = sorted(valid_deps)
        dep_str = "_".join([f"{pkg}_{ver}" for pkg, ver in sorted_deps])
        return hashlib.md5(dep_str.encode()).hexdigest()[:60]
    
    def load_documents(self, dependencies: Dict[str, str]) -> Tuple[List[str], List[Any]]:
        """
        加载指定依赖项的文档和嵌入
        支持两种格式：
        1. dict[pkg, ver] - 标准格式
        2. dict[pkg, list[ver]] - appendSrcDep后的格式
        
        Args:
            dependencies: 依赖名称和版本的字典
            
        Returns:
            (documents, embeddings)的元组
        """
        all_documents = []
        all_embeddings = []
        
        # 使用dict_to_pkg_ver_tuples处理两种格式
        pkg_ver_tuples = dict_to_pkg_ver_tuples(dependencies)
        
        for package, version in pkg_ver_tuples:
            if version is None:
                self.logger.info(f"Skipping {package} with None version")
                continue
            
            # 尝试加载预计算的嵌入
            cached_data = self.embeddings_manager.load_embeddings(package, version)
            documents_for_package = []
            
            # 如果有预计算的文档，直接使用
            if cached_data is not None and 'documents' in cached_data:
                self.logger.info(f"Using precomputed documents for {package} {version}")
                documents_for_package = cached_data['documents']
            else:
                # 否则从语料库加载
                corpus_path = os.path.join(self.corpus_base, package, version + ".jsonl")
                try:
                    with open(corpus_path, "r") as f:
                        for line in f:
                            documents_for_package.append(line)
                    self.logger.info(f"Loaded {len(documents_for_package)} documents for {package} {version}")
                except Exception as e:
                    self.logger.error(f"Error loading corpus for {package} {version}: {e}")
                    self.logger.error(f"Corpus path: {corpus_path}")
                    continue
            
            # 如果没有文档，跳过处理
            if not documents_for_package:
                self.logger.warning(f"No documents loaded for {package} {version}, skipping")
                continue
            
            # 处理嵌入
            package_embeddings = None
            if cached_data is not None and 'embeddings' in cached_data and len(cached_data['embeddings']) == len(documents_for_package):
                self.logger.info(f"Using cached embeddings for {package} {version}")
                package_embeddings = cached_data['embeddings']
            else:
                self.logger.info(f"Computing embeddings for {package} {version}")
                embedding_function = CustomEmbeddingFunction(**self.embed_func_args)
                package_embeddings = self.embeddings_manager.precompute_embeddings(
                    package, version, documents_for_package,
                    embedding_function,
                    batch_size=self.batch_size
                )
            
            if package_embeddings is not None:
                all_documents.extend(documents_for_package)
                all_embeddings.extend(package_embeddings)
            else:
                self.logger.warning(f"Failed to get embeddings for {package} {version}")
        
        return all_documents, all_embeddings
    
    def build_collection(self, 
                         dependencies: Union[Dict[str, str], Dict[str, List[str]]], 
                         collection_name: Optional[str] = None,
                         force_rebuild: bool = False) -> Optional[str]:
        """
        为给定的依赖项构建集合
        
        Args:
            dependencies: 依赖名称和版本的字典
            collection_name: 可选的集合名称，如果不提供则根据依赖项生成
            force_rebuild: 如果为True，即使集合已存在也重新构建
            
        Returns:
            集合名称或None（如果构建失败）
        """
        # 获取或生成集合名称
        if collection_name is None:
            collection_name = self.get_collection_hash(dependencies)
        
        self.logger.info(f"Building collection: {collection_name}")
        
        # 检查集合是否已存在
        try:
            if not force_rebuild:
                existing_collection = self.chroma_client.get_collection(name=collection_name)
                if existing_collection.count() > 0:
                    self.logger.info(f"Collection {collection_name} already exists with {existing_collection.count()} items")
                    return collection_name
        except Exception:
            # 集合不存在，继续创建
            pass
        
        # 创建embedding函数实例
        embedding_function = CustomEmbeddingFunction(**self.embed_func_args)
        
        # 创建集合
        try:
            collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=embedding_function
            )
            self.logger.info(f"Created collection: {collection_name}")
        except Exception as e:
            self.logger.error(f"Error creating collection {collection_name}: {e}")
            return None
        
        # 加载文档和嵌入
        documents, embeddings = self.load_documents(dependencies)
        
        if not documents:
            self.logger.warning(f"No documents available for dependencies, collection remains empty")
            return collection_name
        
        # 填充集合
        self._populate_collection(collection, collection_name, documents, embeddings)
        
        # 在新版本的ChromaDB中，索引是自动创建的，不需要手动调用create_index
        self.logger.info(f"Collection {collection_name} completed (index created automatically)")
        
        return collection_name
    
   
    def _populate_collection(self, collection, collection_name, documents, embeddings=None):
        """批量添加文档到集合"""
        if not documents:
            self.logger.warning(f"No documents to add to collection {collection_name}")
            return
        
        total_docs = len(documents)
        total_batches = (total_docs + self.batch_size - 1) // self.batch_size
        
        self.logger.info(f"Adding {total_docs} documents to collection {collection_name} in {total_batches} batches")
        
        # 使用tqdm显示进度
        for batch_idx in tqdm(range(total_batches), desc=f"Adding documents to {collection_name}", unit="batch"):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_docs)
            
            self._process_batch(collection, collection_name, documents, embeddings, start_idx, end_idx, batch_idx)
    
    def _process_batch(self, collection, collection_name, documents, embeddings, start_idx, end_idx, batch_idx):
        """处理单个批次的文档"""
        batch_documents = documents[start_idx:end_idx]
        batch_ids = [f"{collection_name}_{idx}" for idx in range(start_idx, end_idx)]
        
        # 处理空文档
        processed_batch_documents = [(doc if doc and doc.strip() else " ") for doc in batch_documents]

        try:
            if embeddings:
                batch_embeddings = self._prepare_embeddings(embeddings, start_idx, end_idx)
                collection.upsert(
                    documents=processed_batch_documents,
                    ids=batch_ids,
                    embeddings=batch_embeddings
                )
            else:
                collection.upsert(
                    documents=processed_batch_documents,
                    ids=batch_ids
                )
        except Exception as e:
            error_msg = f"Error adding batch {batch_idx + 1} to collection {collection_name}: {e}"
            self.logger.error(error_msg)
    
    def _prepare_embeddings(self, embeddings, start_idx, end_idx):
        """准备embedding数据，确保格式正确"""
        batch_embeddings = embeddings[start_idx:end_idx]
        if isinstance(batch_embeddings, np.ndarray):
            return batch_embeddings.tolist()
        elif isinstance(batch_embeddings, list) and batch_embeddings and isinstance(batch_embeddings[0], np.ndarray):
            return [emb.tolist() for emb in batch_embeddings]
        return batch_embeddings

    def get_dependency_folder_name(self, dependencies: Dict[str, str]) -> str:
        """
        为依赖项生成文件夹名称，格式为 "_".join(sorted(dependency.items()))
        支持两种格式：
        1. dict[pkg, ver] - 标准格式
        2. dict[pkg, list[ver]] - appendSrcDep后的格式
        
        Args:
            dependencies: 依赖名称和版本的字典
            
        Returns:
            文件夹名称字符串
        """
        # 使用dict_to_pkg_ver_tuples处理两种格式
        pkg_ver_tuples = dict_to_pkg_ver_tuples(dependencies)
        
        # 过滤掉None版本的依赖项，然后排序
        valid_deps = [(pkg, ver) for pkg, ver in pkg_ver_tuples if ver is not None]
        sorted_deps = sorted(valid_deps)
        return "_".join([f"{pkg}_{ver}" for pkg, ver in sorted_deps])

    def get_collection_path(self, dependencies: Dict[str, str]) -> str:
        """
        获取collection的完整存储路径
        
        Args:
            dependencies: 依赖名称和版本的字典
            
        Returns:
            collection存储路径
        """
        folder_name = self.get_dependency_folder_name(dependencies)
        collection_hash = self.get_collection_hash(dependencies)
        # 使用 knowledge_type/embed_model_name 作为路径结构
        embed_model_name = self.embed_func_args['model_name'].split('/')[-1]
        return os.path.join(self.collection_base, self.knowledge_type, embed_model_name, folder_name, collection_hash)

    def build_all_task_collections(self, 
                                  dataset_path: str, 
                                  dataset_name: str,
                                  task_name: str,
                                  force_rebuild: bool = False,
                                  num_processes: int = 1) -> Dict[str, str]:
        """
        为指定任务构建所有需要的collections
        
        Args:
            dataset_path: 数据集文件路径
            dataset_name: 数据集名称 ("VersiBCB" 或 "VersiCode")
            task_name: 任务名称 ("VACE" 或 "VSCC")
            ban_deprecation: 是否使用ban deprecation版本的数据
            force_rebuild: 是否强制重建已有集合
            num_processes: 使用的进程数量
            
        Returns:
            集合映射 {依赖项哈希: 集合名称}
        """
        self.logger.info(f"Building all collections for {dataset_name} {task_name} task")
        
        # 加载数据集
        try:
            with open(dataset_path, 'r') as f:
                data_items = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading dataset from {dataset_path}: {e}")
            return {}
        
        self.logger.info(f"Loaded {len(data_items)} items from dataset")
        
        # 根据数据集和任务确定依赖项字段
        dependency_fields = []
        if dataset_name == "VersiBCB":
            if task_name == "VACE":
                dependency_fields = ["target_dependency", "origin_dependency"]
            elif task_name == "VSCC":
                dependency_fields = ["dependency"]
        elif dataset_name == "VersiCode":
            dependency_fields = ["dependency"]
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # 收集所有唯一的依赖项集合
        all_dependencies = set()
        for item in data_items:
            for field in dependency_fields:
                if field in item and item[field]:
                    # 转换为不可变类型以便可以添加到集合中
                    dep_tuple = tuple(sorted(item[field].items()))
                    all_dependencies.add(dep_tuple)
        
        self.logger.info(f"Found {len(all_dependencies)} unique dependency sets across all fields: {dependency_fields}")
        
        # 为每个依赖项集合创建独立的ChromaDB客户端和collection
        collection_map = {}
        
        if num_processes <= 1:
            # 单进程处理
            for dep_tuple in tqdm(all_dependencies, desc=f"Building collections for {task_name}"):
                dependencies = dict(dep_tuple)
                collection_hash = self.get_collection_hash(dependencies)
                
                # 创建独立的存储路径
                collection_path = self.get_collection_path(dependencies)
                os.makedirs(collection_path, exist_ok=True)
                
                # 为每个collection创建独立的ChromaDB客户端
                collection_client = chromadb.PersistentClient(
                    path=collection_path,
                    settings=chromadb.Settings(
                        allow_reset=True,
                        anonymized_telemetry=False,
                        is_persistent=True
                    )
                )
                
                # 检查是否应跳过
                if not force_rebuild:
                    try:
                        existing_collection = collection_client.get_collection(name=collection_hash)
                        if existing_collection.count() > 0:
                            self.logger.info(f"Skipping existing collection: {collection_hash}")
                            collection_map[collection_hash] = collection_hash
                            continue
                    except Exception:
                        # 集合不存在，继续创建
                        pass
                
                # 使用独立客户端构建collection
                collection_name = self._build_single_collection(
                    collection_client, dependencies, collection_hash, force_rebuild
                )
                if collection_name:
                    collection_map[collection_hash] = collection_name
                    
        else:
            # 多进程处理 - 需要修改process_dependency函数以支持新的存储结构
            self.logger.info(f"Using {num_processes} processes for parallel collection building")
            collection_map = self._build_collections_multiprocess(
                all_dependencies, force_rebuild, num_processes
            )
        
        self.logger.info(f"Successfully built {len(collection_map)} collections for {task_name} task")
        return collection_map

    def _build_single_collection(self, 
                                client: chromadb.PersistentClient,
                                dependencies: Dict[str, str], 
                                collection_name: str,
                                force_rebuild: bool = False) -> Optional[str]:
        """
        使用指定客户端构建单个collection
        
        Args:
            client: ChromaDB客户端
            dependencies: 依赖项字典
            collection_name: collection名称
            force_rebuild: 是否强制重建
            
        Returns:
            collection名称或None
        """
        self.logger.info(f"Building collection: {collection_name}")
        
        # 检查集合是否已存在
        try:
            if not force_rebuild:
                existing_collection = client.get_collection(name=collection_name)
                if existing_collection.count() > 0:
                    self.logger.info(f"Collection {collection_name} already exists with {existing_collection.count()} items")
                    return collection_name
        except Exception:
            # 集合不存在，继续创建
            pass
        
        # 创建embedding函数实例
        embedding_function = CustomEmbeddingFunction(**self.embed_func_args)
        
        # 创建集合
        try:
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=embedding_function
            )
            self.logger.info(f"Created collection: {collection_name}")
        except Exception as e:
            self.logger.error(f"Error creating collection {collection_name}: {e}")
            return None
        
        # 加载文档和嵌入
        documents, embeddings = self.load_documents(dependencies)
        
        if not documents:
            self.logger.warning(f"No documents available for dependencies, collection remains empty")
            return collection_name
        
        # 填充集合
        self._populate_collection(collection, collection_name, documents, embeddings)
        
        # 在新版本的ChromaDB中，索引是自动创建的，不需要手动调用create_index
        self.logger.info(f"Collection {collection_name} completed (index created automatically)")
        
        return collection_name

    def _build_collections_multiprocess(self, 
                                       all_dependencies: set, 
                                       force_rebuild: bool, 
                                       num_processes: int) -> Dict[str, str]:
        """
        使用多进程构建collections（需要修改以支持新的存储结构）
        
        Args:
            all_dependencies: 所有依赖项集合
            force_rebuild: 是否强制重建
            num_processes: 进程数量
            
        Returns:
            collection映射
        """
        # TODO: 实现多进程版本，需要修改process_dependency函数以支持新的存储结构
        # 暂时使用单进程版本
        self.logger.warning("Multiprocess building with new storage structure not yet implemented, falling back to single process")
        
        collection_map = {}
        for dep_tuple in tqdm(all_dependencies, desc="Building collections (single process fallback)"):
            dependencies = dict(dep_tuple)
            collection_hash = self.get_collection_hash(dependencies)
            
            # 创建独立的存储路径
            collection_path = self.get_collection_path(dependencies)
            os.makedirs(collection_path, exist_ok=True)
            
            # 为每个collection创建独立的ChromaDB客户端
            collection_client = chromadb.PersistentClient(
                path=collection_path,
                settings=chromadb.Settings(
                    allow_reset=True,
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
            
            # 使用独立客户端构建collection
            collection_name = self._build_single_collection(
                collection_client, dependencies, collection_hash, force_rebuild
            )
            if collection_name:
                collection_map[collection_hash] = collection_name
                
        return collection_map

    def _get_lock_file_path(self, collection_name: str) -> str:
        """
        获取collection构建锁文件路径
        
        Args:
            collection_name: collection名称
            
        Returns:
            锁文件路径
        """
        # 使用系统临时目录存放锁文件
        lock_dir = os.path.join(tempfile.gettempdir(), "chroma_build_locks")
        os.makedirs(lock_dir, exist_ok=True)
        return os.path.join(lock_dir, f"chroma_build_{collection_name}.lock")
    
    def build_collection_with_lock(self, 
                                  dependencies: Union[Dict[str, str], Dict[str, List[str]]], 
                                  collection_name: Optional[str] = None,
                                  force_rebuild: bool = False,
                                  timeout: float = 0.1) -> Optional[str]:
        """
        使用文件锁机制构建collection，避免并发冲突
        
        Args:
            dependencies: 依赖名称和版本的字典
            collection_name: 可选的集合名称，如果不提供则根据依赖项生成
            force_rebuild: 如果为True，即使集合已存在也重新构建
            timeout: 锁等待超时时间（秒），0表示不等待
            
        Returns:
            集合名称或None（如果构建失败或被锁）
            
        Raises:
            CollectionBuildingLockedException: 当collection正在被其他进程构建时
        """
        # 获取或生成集合名称
        if collection_name is None:
            collection_name = self.get_collection_hash(dependencies)
        
        # 首先快速检查collection是否已存在
        try:
            if not force_rebuild:
                existing_collection = self.chroma_client.get_collection(name=collection_name)
                if existing_collection.count() > 0:
                    self.logger.info(f"Collection {collection_name} already exists with {existing_collection.count()} items")
                    return collection_name
        except Exception:
            # 集合不存在，继续创建
            pass
        
        lock_file_path = self._get_lock_file_path(collection_name)
        
        try:
            with open(lock_file_path, 'w') as lock_file:
                try:
                    # 尝试获取非阻塞独占锁
                    if timeout <= 0:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    else:
                        # 有限时间等待锁
                        import signal
                        
                        def timeout_handler(signum, frame):
                            raise TimeoutError("Lock acquisition timeout")
                        
                        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(int(timeout))
                        
                        try:
                            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                        finally:
                            signal.alarm(0)
                            signal.signal(signal.SIGALRM, old_handler)
                    
                    self.logger.info(f"Acquired lock for collection {collection_name}")
                    
                    # 获得锁后，再次检查collection是否已被其他进程创建
                    try:
                        if not force_rebuild:
                            existing_collection = self.chroma_client.get_collection(name=collection_name)
                            if existing_collection.count() > 0:
                                self.logger.info(f"Collection {collection_name} was created by another process while waiting for lock")
                                return collection_name
                    except Exception:
                        # 集合不存在，继续创建
                        pass
                    
                    # 安全地构建collection
                    self.logger.info(f"Building collection {collection_name} with lock protection")
                    result = self.build_collection(dependencies, collection_name, force_rebuild)
                    
                    if result:
                        self.logger.info(f"Successfully built collection {collection_name}")
                    else:
                        self.logger.error(f"Failed to build collection {collection_name}")
                    
                    return result
                    
                except (IOError, OSError) as e:
                    if e.errno == 11:  # EAGAIN - 资源暂时不可用
                        self.logger.info(f"Collection {collection_name} is being built by another process (lock conflict)")
                        raise CollectionBuildingLockedException(f"Collection {collection_name} is currently being built by another process")
                    else:
                        self.logger.error(f"Lock error for collection {collection_name}: {e}")
                        raise
                except TimeoutError:
                    self.logger.info(f"Timeout waiting for lock on collection {collection_name}")
                    raise CollectionBuildingLockedException(f"Timeout waiting for lock on collection {collection_name}")
                    
        except FileNotFoundError:
            self.logger.error(f"Cannot create lock file {lock_file_path}")
            # 如果无法创建锁文件，降级到无锁模式
            self.logger.warning(f"Falling back to lockless build for collection {collection_name}")
            return self.build_collection(dependencies, collection_name, force_rebuild)
        
        finally:
            # 清理锁文件
            try:
                if os.path.exists(lock_file_path):
                    os.remove(lock_file_path)
            except Exception as e:
                self.logger.warning(f"Failed to remove lock file {lock_file_path}: {e}")


# 简单使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build collections for dependencies")
    parser.add_argument("--dataset", type=str, help="Path to dataset JSON file", 
                       default="benchmark/data/VersiBCB_Benchmark/vace_datas.json")
    parser.add_argument("--dataset-name", type=str, help="Dataset name (VersiBCB or VersiCode)", 
                       default="VersiBCB")
    parser.add_argument("--task-name", type=str, help="Task name (VACE or VSCC)", 
                       default="VACE")
    parser.add_argument("--force", action="store_true", help="Force rebuild existing collections")
    parser.add_argument("--batch-size", type=int, default=250, help="Batch size for adding documents")
    parser.add_argument("--num-processes", type=int, default=1, help="Number of processes to use (1=no multiprocessing)")
    parser.add_argument("--embedding-source", type=str, default="local", help="Embedding source (local or togetherai)")

    parser.add_argument("--local-embedding-model", type=str, default="all-MiniLM-L6-v2", help="Local embedding model")
    parser.add_argument("--togetherai-embedding-model", type=str, default="togethercomputer/m2-bert-80M-8k-retrieval", help="TogetherAI embedding model")
    
    parser.add_argument("--corpus-base", type=str, default="/datanfs2/chenrongyi/data/docs", help="Corpus base path")
    
    parser.add_argument("--collection-base", type=str, default="/datanfs2/chenrongyi/data/RAG/chroma_data", help="Collection存放的基础路径，最终存储路径为collection_base/knowledge_type/embed_model_name/")
    parser.add_argument("--knowledge-type", type=str, default="docstring", help="Knowledge type (docstring or srccodes)")
    
    parser.add_argument("--embedding-base-path", type=str, default="/datanfs2/chenrongyi/data/RAG/docs_embeddings", help="Embedding base path")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('collection_builder.log', mode='a'),
            logging.StreamHandler()
        ]
    )
    
    # 配置嵌入参数
    embed_func_args = {
        'source': args.embedding_source,
        'model_name': args.local_embedding_model if args.embedding_source == 'local' else args.togetherai_embedding_model,
        'batch_size': 256
    }
    
    # 创建构建器 - 不需要预先创建客户端，因为build_all_task_collections会为每个collection创建独立的客户端
    builder = CollectionBuilder(
        chroma_client=None,  # 这里传None，因为build_all_task_collections不使用这个参数
        embed_func_args=embed_func_args,
        corpus_base=args.corpus_base,
        collection_base=args.collection_base,
        batch_size=args.batch_size,
        num_processes=args.num_processes,
        knowledge_type=args.knowledge_type,
        embedding_base_path=args.embedding_base_path
    )
    
    # 使用build_all_task_collections构建集合
    collection_map = builder.build_all_task_collections(
        dataset_path=args.dataset,
        dataset_name=args.dataset_name,
        task_name=args.task_name,
        force_rebuild=args.force,
        num_processes=args.num_processes
    )
    
    print(f"Built {len(collection_map)} collections for {args.dataset_name} {args.task_name} task") 