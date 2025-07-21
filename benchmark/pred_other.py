from together import Together
from datasets import load_from_disk
import json
import chromadb
import hashlib
import os
from sentence_transformers import SentenceTransformer
import re
import tiktoken
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.docTokenDistribute import distribute_doc_tokens,truncate_doc
from benchmark.config.code.config import (
VSCC_LOW_BOUND, VSCC_HIGH_BOUND, RAG_DOCUMENT_NUM,
    FC_MAX_TOKEN_LENGTH, RAG_MAX_TOKEN_LENGTH, TOGETHER_API_KEY_PATH,
 RAG_COLLECTION_BASE, KNOWLEDGE_TYPE,
 LOCAL_EMBEDDING_MODEL, TOGETHERAI_EMBEDDING_MODEL,EMBEDDING_SOURCE
)
import torch
from chromadb.api.types import Documents, EmbeddingFunction
import time
import pickle
import numpy as np
from utils.RAGutils.RAGEmbedding import PrecomputedEmbeddingsManager, CustomEmbeddingFunction
from utils.RAGutils.document_utils import truncate_context, truncate_BCB_context, get_version
from utils.prompt_utils import format_prompt
from utils.RAGutils.RAGRetriever import RAGContextRetriever,getKnowledgeDocs
from utils.loraTrain.loraTrainUtils import inference
import traceback  # 在文件顶部添加
from chromadb.config import  Settings  # 在文件顶部添加
import logging  # 在文件顶部添加
import multiprocessing
from utils.getDependencyUtils import dict_to_pkg_ver_tuples

# 在文件顶部添加日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chromadb_processing.log', mode='a'),
        logging.StreamHandler()
    ]
)

# Global variables for process-local clients (will be set by initializer)
process_chroma_client = None
process_together_client = None
process_rag_retriever = None  # 新增：每个进程的RAG检索器实例
process_local_model = None
process_local_tokenizer = None

# 初始化 Together AI 客户端
try:
    with open(TOGETHER_API_KEY_PATH, "r") as f:
        api_key = f.read()
    together_client = Together(api_key=api_key)
except Exception as e:
    logging.error(f"Error initializing Together AI client: {e}")
    together_client = None



# # 初始化 Chroma 客户端（持久化版）
# dep_path = os.path.join(RAG_COLLECTION_BASE, KNOWLEDGE_TYPE, EMBEDDING_SOURCE)  # 添加 embedding source 到路径
# os.makedirs(dep_path, exist_ok=True)
# _client = chromadb.PersistentClient(
#     path=dep_path,
#     settings=chromadb.Settings(
#         allow_reset=True,
#         anonymized_telemetry=False,
#         is_persistent=True
#     )
# )

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # 配置 embedding 参数
# embed_func_args = {
#     'source': EMBEDDING_SOURCE,
#     'model_name': LOCAL_EMBEDDING_MODEL if EMBEDDING_SOURCE == 'local' else TOGETHERAI_EMBEDDING_MODEL,
#     'together_client': client if EMBEDDING_SOURCE == 'togetherai' else None,
#     'batch_size': 64  # 可以根据需要调整
# }
# db_path = os.path.join(RAG_COLLECTION_BASE, KNOWLEDGE_TYPE, EMBEDDING_SOURCE) # 通过哪些数据构建
chroma_settings = chromadb.Settings(
    allow_reset=True,
    anonymized_telemetry=False,
    is_persistent=True,
    # Optional: sqlite_synchronous_mode="NORMAL" # 可选择调整同步模式以提高性能
)


def get_collection_hash(dep):
    """
    为dependency和version生成唯一的hash
    支持两种格式：
    1. dict[pkg, ver] - 标准格式
    2. dict[pkg, list[ver]] - appendSrcDep后的格式
    """
    # 使用dict_to_pkg_ver_tuples处理两种格式
    pkg_ver_tuples = dict_to_pkg_ver_tuples(dep)
    
    # 过滤掉None版本的依赖项，然后排序并生成字符串
    valid_deps = [(pkg, ver) for pkg, ver in pkg_ver_tuples if ver is not None]
    sorted_deps = sorted(valid_deps)
    dep_str = "_".join([f"{pkg}_{ver}" for pkg, ver in sorted_deps])
    return hashlib.md5(dep_str.encode()).hexdigest()[:60]

def get_collection_name(data, dataset, task):
    """获取 collection 的名称"""
    if dataset == "VersiCode":
        pkg, ver = data["dependency"], get_version(data["version"])
        return get_collection_hash({pkg:ver})
    elif dataset == "VersiBCB":
        dep = data["dependency"] if task == "VSCC" else data["target_dependency"]
        return get_collection_hash(dep)
    else:
        raise ValueError(f"不支持的数据集: {dataset}")

def get_query_from_data(data, task):
    """从数据中构建查询"""
    query = data["description"]
    if task == "VACE":
        query += data["origin_code"]
    return query

def get_or_create_collection(chroma_client, collection_name, embedding_function_instance):
    """获取或创建 collection"""
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function_instance,
    )
    return collection



def populate_collection(collection, collection_name, documents_to_add, embeddings_to_add):
    """批量添加文档到 collection"""
    if not documents_to_add:
        logging.warning(f"No documents to add to collection {collection_name}.")
        return
    
    BATCH_SIZE = 250
    total_docs = len(documents_to_add)
    total_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE
    
    logging.info(f"Adding {total_docs} documents to collection {collection_name} in {total_batches} batches.")
    
    # 使用 tqdm 显示进度
    for batch_idx in tqdm(range(total_batches), desc=f"Adding documents to {collection_name}", unit="batch"):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_docs)
        
        process_batch(collection, collection_name, documents_to_add, embeddings_to_add, start_idx, end_idx, batch_idx)

def process_batch(collection, collection_name, documents_to_add, embeddings_to_add, start_idx, end_idx, batch_idx):
    """处理单个批次的文档"""
    batch_documents = documents_to_add[start_idx:end_idx]
    batch_ids = [f"{collection_name}_{idx}" for idx in range(start_idx, end_idx)]
    
    # 处理空文档
    processed_batch_documents = [(doc if doc and doc.strip() else " ") for doc in batch_documents]

    try:
        if embeddings_to_add:
            batch_embeddings = prepare_embeddings(embeddings_to_add, start_idx, end_idx)
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
        logging.error(error_msg)
        print(error_msg)

def prepare_embeddings(embeddings_to_add, start_idx, end_idx):
    """准备 embedding 数据，确保格式正确"""
    batch_embeddings = embeddings_to_add[start_idx:end_idx]
    if isinstance(batch_embeddings, np.ndarray):
        return batch_embeddings.tolist()
    elif isinstance(batch_embeddings, list) and batch_embeddings and isinstance(batch_embeddings[0], np.ndarray):
        return [emb.tolist() for emb in batch_embeddings]
    return batch_embeddings

def query_collection(collection, query, n_results=RAG_DOCUMENT_NUM):
    """查询 collection"""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
    except Exception as e:
        logging.error(f"Error querying collection: {e}")
        return {"documents": [[]]}

def process_query_results(results, max_token_length):
    """处理查询结果，提取上下文"""
    retrieved_context = ""
    if results["documents"] and results["documents"][0]:
        # 过滤掉 None 值，只保留有效的文档
        valid_docs = [doc for doc in results["documents"][0] if doc is not None]
        if valid_docs:  # 确保还有有效文档
            retrieved_context = "\n".join(valid_docs)
    
    logging.info(f"Retrieved context length: {len(retrieved_context.split(' '))} words")
    retrieved_context = truncate_context(retrieved_context, max_token_length)
    return retrieved_context


# --------entry point--------- #
def get_FC_pred(data,max_token_length=FC_MAX_TOKEN_LENGTH,task="VSCC",dataset="VersiCode",model="mistralai/Mistral-7B-Instruct-v0.2",ban_deprecation=False):
    # if dataset == "VersiCode":
    #     context = getKnowledgeDocs(data,dataset=dataset,task=task)
    #     context = "\n".join(context)
    #     context = truncate_context(context, max_token_length)    
    #     # 根据prompt和data获取input
    # elif dataset == "VersiBCB":
    #     context = getKnowledgeDocs(data,dataset=dataset,task=task)
    #     context = truncate_BCB_context(context, max_token_length)
    # else:
    #     raise "Wrong dataset"
    context = ""
    # 获取对应的prompt
    input_for_api = format_prompt(data,context,dataset,task,ban_deprecation)
    response = together_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": input_for_api}],
    )
    return response.choices[0].message.content



def reset_collection(collection_name):
    """重置损坏的集合"""
    try:
        _client.delete_collection(collection_name)
        print(f"Deleted corrupted collection: {collection_name}")
    except Exception as e:
        print(f"Error deleting collection: {e}")
    
    try:
        return _client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=CustomEmbeddingFunction()
        )
    except Exception as e:
        print(f"Error creating new collection: {e}")
        return None

def load_data(dataset, task, Ban_Deprecation:bool):
    if dataset == "VersiCode":
        # with open("benchmark/data/VersiCode_Benchmark/blockcode_completion.json", "r") as f:
        #     datas = json.load(f)
        pass
    elif dataset == "VersiBCB":
        data_path = 'data/VersiBCB_Benchmark'
        data_name = f"{task.lower()}_datas{'_for_warning' if Ban_Deprecation else ''}.json"
        with open(os.path.join(data_path,data_name), "r") as f:
            datas = json.load(f)
    else:
        raise "Wrong dataset"
    return datas

def load_existing_results(output_path):
    """加载已存在的JSONL结果文件，返回已处理的ID集合和已有结果"""
    existing_results = []
    existing_ids = set()
    try:
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    existing_results.append(result)
                    existing_ids.add(result["id"])
                except json.JSONDecodeError:
                    continue
        print(f"Found {len(existing_ids)} existing results")
        return existing_results, existing_ids
    except FileNotFoundError:
        print("No existing results file found, starting fresh")
        return [], set()

def append_to_jsonl(file_path, data):
    """将单个结果追加到JSONL文件"""
    with open(file_path, 'a') as f:
        f.write(json.dumps(data) + '\n')

def getOutputPath(task, dataset, Ban_Deprecation, approach, model):
    output_path = 'output'
    model_name = model.split('/')[-1]
    embed_suffix = f"emb_{EMBEDDING_SOURCE}"
    
    if approach == 'RAG':
        output_name = f"{dataset.lower()}_{task.lower()}{'_BD' if Ban_Deprecation else ''}_{approach}_{KNOWLEDGE_TYPE}_{embed_suffix}_{model_name}.jsonl"
    else:
        output_name = f"{dataset.lower()}_{task.lower()}{'_BD' if Ban_Deprecation else ''}_{approach}_{model_name}.jsonl"  # 改为.jsonl后缀
    return os.path.join(output_path, output_name)

# --- Worker Setup ---

def init_worker(db_path, api_key_path, client_settings, embedding_args, worker_id=None, knowledge_type=None, model_name=None, model_type=None):
    """Initializer for each worker process."""
    global process_chroma_client, process_together_client, process_rag_retriever, embed_func_args
    global process_local_model, process_local_tokenizer  # Add global variables for model and tokenizer
    
    # 配置GPU（如果有多个）
    if worker_id is not None and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            gpu_id = worker_id % gpu_count
            torch.cuda.set_device(gpu_id)
            logging.info(f"Worker {os.getpid()} using GPU {gpu_id}")
    
    logging.info(f"Initializing worker {os.getpid()}...")
    
    # 初始化ChromaDB客户端
    process_chroma_client = chromadb.PersistentClient(
        path=db_path,
        settings=client_settings
    )
    
    # 初始化Together AI客户端
    with open(api_key_path, "r") as f:
        api_key = f.read().strip()
    process_together_client = Together(api_key=api_key)
    
    # 更新embedding参数
    worker_embed_args = embedding_args.copy()
    if worker_embed_args['source'] == 'togetherai':
        worker_embed_args['together_client'] = process_together_client
    
    # 初始化RAG检索器
    process_rag_retriever = RAGContextRetriever(
        process_chroma_client, 
        worker_embed_args,
        knowledge_type
    )
    
    # 初始化本地模型（如果需要）
    process_local_model = None
    process_local_tokenizer = None
    if model_type == "local" and model_name is not None:
        try:
            logging.info(f"Worker {os.getpid()} loading model {model_name}...")
            process_local_model = AutoModelForCausalLM.from_pretrained(model_name)
            process_local_tokenizer = AutoTokenizer.from_pretrained(model_name)
            logging.info(f"Worker {os.getpid()} model loaded successfully")
        except Exception as e:
            logging.error(f"Worker {os.getpid()} error loading model: {e}")
    
    logging.info(f"Worker {os.getpid()} initialized.")

def process_sample(item):
    """Function executed by each worker process."""
    # global process_chroma_client, process_together_client, process_rag_retriever
    global process_local_model, process_local_tokenizer  # Add references to global model variables
    i, data, current_dataset, current_task, current_ban_deprecation, current_model, model_type = item
    sample_id = data.get('id', f'index_{i}') # Get sample ID for logging
    
    try:
        # 确保客户端和检索器已初始化
        if process_rag_retriever is None:
            raise RuntimeError(f"Worker {os.getpid()} RAG retriever not initialized.")
            
        # 使用RAG检索器获取上下文
        context = process_rag_retriever.retrieve_context(
            data,
            current_dataset,
            current_task,
            RAG_MAX_TOKEN_LENGTH
        )

        # 构建提示
        input_for_api = format_prompt(data, context, current_dataset, current_task, current_ban_deprecation)
        
        # 调用API获取回答
        if model_type == "local":
            # 使用已加载的模型而不是重新加载
            if process_local_model is None or process_local_tokenizer is None:
                raise RuntimeError(f"Worker {os.getpid()} local model or tokenizer not initialized.")
            
            response = inference(process_local_model, process_local_tokenizer, input_for_api)
        else:
            response = process_together_client.chat.completions.create(
                model=current_model,
                messages=[{"role": "user", "content": input_for_api}],
            )
        
        return {"id": sample_id, "answer": response.choices[0].message.content}
    except Exception as e:
        tb_str = traceback.format_exc()
        logging.error(f"Error in worker {os.getpid()} processing sample {sample_id}: {e}\n{tb_str}")
        return {"id": sample_id, "error": str(e), "traceback": tb_str}

if __name__ == "__main__":



    multiprocessing.set_start_method('spawn', force=True) # Recommended for stability
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, default="VersiBCB")
    args.add_argument("--task", type=str, default="VACE") # VSCC or VACE
    args.add_argument("--Ban_Deprecation", type=lambda x: x.lower() == 'true', default=False) # 是否禁用deprecation
    args.add_argument("--approach", type=str, default="RAG") # RAG or FC
    args.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2") # 模型或者模型的路径
    args.add_argument("--num_processes", type=int, default=1) # 进程数量，对于local只支持使用1个进程
    args.add_argument("--selfdefined_OutputPath", type=str, default=None)
    args.add_argument("--knowledge_type", type=str, default="srccodes") # srccodes or docstring
    args.add_argument("--model_type", type=str, default="togetherai") # local or togetherai
    args = args.parse_args()
    dataset = args.dataset
    task = args.task
    Ban_Deprecation = args.Ban_Deprecation
    approach = args.approach
    model = args.model
    num_processes = args.num_processes
    selfdefined_OutputPath = args.selfdefined_OutputPath
    knowledge_type = args.knowledge_type
    model_type = args.model_type
    # 初始化 Chroma 客户端（持久化版）
    dep_path = os.path.join(RAG_COLLECTION_BASE, knowledge_type, EMBEDDING_SOURCE)  # 添加 embedding source 到路径
    os.makedirs(dep_path, exist_ok=True)
    global _client
    _client = chromadb.PersistentClient(
        path=dep_path,
        settings=chromadb.Settings(
            allow_reset=True,
            anonymized_telemetry=False,
            is_persistent=True
        )
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 配置 embedding 参数
    global embed_func_args
    embed_func_args = {
        'source': EMBEDDING_SOURCE,
        'model_name': LOCAL_EMBEDDING_MODEL if EMBEDDING_SOURCE == 'local' else TOGETHERAI_EMBEDDING_MODEL,
        'together_client': together_client if EMBEDDING_SOURCE == 'togetherai' else None,
        'batch_size': 64  # 可以根据需要调整
    }
    global db_path
    db_path = os.path.join(RAG_COLLECTION_BASE, knowledge_type, EMBEDDING_SOURCE) # 通过哪些数据构建


    # 任务配置（可以通过参数传递） process_sample用的


    # --- Prepare Shared Arguments for Workers ---
    # ChromaDB path and settings
    # db_path = os.path.join(RAG_COLLECTION_BASE, KNOWLEDGE_TYPE, EMBEDDING_SOURCE)
    # chroma_settings = chromadb.Settings(
    #     allow_reset=True,
    #     anonymized_telemetry=False,
    #     is_persistent=True,
    #     # Optional: sqlite_synchronous_mode="NORMAL" # 可选择调整同步模式以提高性能
    # )

    # API key path
    api_key_path = TOGETHER_API_KEY_PATH
    # Embedding args (ensure it's picklable or reconstructed in worker)
    embedding_args = embed_func_args.copy() # Pass a copy
    embedding_args.pop('together_client', None) # Remove non-picklable client

    # --- Load Data ---
    logging.info("Loading data...")
    datas = load_data(dataset, task, Ban_Deprecation)
    logging.info("Data loaded.")

    if not selfdefined_OutputPath:
        output_path = getOutputPath(task, dataset, Ban_Deprecation, approach, model)
    else:
        output_path = selfdefined_OutputPath
    _, existing_ids = load_existing_results(output_path)

    to_process = [(i, data, dataset, task, Ban_Deprecation, model, model_type) for i, data in enumerate(datas) 
                  if VSCC_LOW_BOUND <= i < VSCC_HIGH_BOUND and data.get("id", f'index_{i}') not in existing_ids]

    total_samples = len(datas)
    samples_to_process_count = len(to_process)
    logging.info(f"Total samples: {total_samples}")
    logging.info(f"Existing results found: {len(existing_ids)}")
    logging.info(f"Samples to process: {samples_to_process_count}")

    if samples_to_process_count == 0:
        logging.info("No new samples to process. Exiting.")
        exit()

    # --- Setup Multiprocessing Pool ---
    # 获取可用的GPU数量
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    # 配置进程数量：使用指定数量或根据GPU自动调整

    logging.info(f"Available GPUs: {gpu_count}")
    logging.info(f"Starting processing with {num_processes} worker processes...")
    
    processed_count = 0
    error_count = 0

    # 创建进程池，每个进程使用不同的worker_id
    pool_initializer_args = (db_path, api_key_path, chroma_settings, embedding_args, None, knowledge_type, model, model_type)
    
    # 使用进程初始化器
    if gpu_count > 0:
        # 如果有GPU，为每个进程分配GPU ID
        workers = []
        for worker_id in range(num_processes):
            worker = multiprocessing.Process(
                target=init_worker,
                args=(db_path, api_key_path, chroma_settings, embedding_args, worker_id, knowledge_type, model, model_type)
            )
            workers.append(worker)
            worker.start()
        
        # 在这种情况下需要自己管理任务分发
        # 这里简化处理，仍然使用Pool但添加worker_id
        pool = multiprocessing.Pool(
            processes=num_processes,
            initializer=init_worker,
            initargs=pool_initializer_args
        )
    else:
        # 如果没有GPU，使用标准Pool
        pool = multiprocessing.Pool(
            processes=num_processes,
            initializer=init_worker,
            initargs=pool_initializer_args
        )

    try:
        # 使用imap_unordered获取结果，提高I/O性能
        results_iterator = pool.imap_unordered(process_sample, to_process)
        
        # 处理结果，显示进度条
        for result in tqdm(results_iterator, total=samples_to_process_count, desc=f"Processing {approach} predictions"):
            if "answer" in result:
                append_to_jsonl(output_path, {"id": result["id"], "answer": result["answer"]})
                processed_count += 1
            elif "error" in result:
                error_count += 1
                # 错误已在worker中记录
                logging.warning(f"Sample {result['id']} failed. See logs for details.")
                # 可选：将错误写入单独的文件
            # 定期记录进度
            if (processed_count + error_count) % 10 == 0:
                logging.info(f"Progress: {processed_count} succeeded, {error_count} failed out of {samples_to_process_count}")

    finally:
        pool.close()
        pool.join()

    logging.info(f"\nFinished! Successfully processed: {processed_count}, Failed: {error_count}")

