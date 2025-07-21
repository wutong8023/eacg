'''
Description:
    获取MemoryLLM所需的context。专门造的连接件，使得RAG模块服务于MemoryLLM
'''
from transformers import AutoTokenizer
from utils.RAGutils.document_utils import  truncate_context, truncate_BCB_context
from benchmark.pred_other import getKnowledgeDocs,chroma_settings
from utils.docTokenDistribute import distribute_doc_tokens,truncate_doc
from utils.RAGutils.RAGRetriever import RAGContextRetriever
from together import Together
import chromadb
import os
import time
import logging
import datetime
from utils.RAGutils.config.default_config import RAG_STRING_TRUNCATE_TOKENIZER, QUERY_CACHE_DIR
from config.paths import API_KEY_PATH
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'context_timing_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def appendContextToData(data, tokenizer=AutoTokenizer.from_pretrained(RAG_STRING_TRUNCATE_TOKENIZER), 
                        max_token_length=2000, useRAG=True, corpus_type=None, task_type=None, 
                        max_dependency_num=None, append_srcDep=False, embedding_source="local", 
                        query_cache_dir=QUERY_CACHE_DIR, 
                        rag_document_num=10, rag_config=None):
    '''
    Description:
        当context为空时，将context添加到data中
    Args:
        data: dict,数据
        tokenizer: 分词器
        max_token_length: 最大token长度
        useRAG: 是否使用RAG
        corpus_type: 语料库类型，"docstring"或"srccodes"
        task_type: 任务类型，默认为"VACE"
        max_dependency_num: 最大依赖数量（已弃用，使用rag_config）
        append_srcDep: 是否添加源依赖（已弃用，使用rag_config）
        embedding_source: 嵌入源（已弃用，使用rag_config）
        query_cache_dir: 查询缓存目录（已弃用，使用rag_config）
        rag_document_num: RAG文档数量（已弃用，使用rag_config）
        rag_config: RAG配置对象，包含所有RAG相关配置
            QAContext_path: QA缓存路径
            useQAContext: 是否使用QA缓存,override useRAG
    Returns:
        tuple: (data: dict, lock_status: str) 
               - data: 修改后的数据
               - lock_status: "success", "locked", "error" 中的一个
    '''
    total_start_time = time.time()
    logger.info(f"Starting appendContextToData for data_id: {data.get('id', 'unknown')}")
    
    # # 如果没有提供rag_config，使用传入的参数创建一个临时配置（向后兼容）
    # if rag_config is None:
    #     from utils.RAGutils.rag_config import RAGConfig
    #     from benchmark.config.code.config import (
    #         LOCAL_EMBEDDING_MODEL, TOGETHERAI_EMBEDDING_MODEL,
    #         RAG_COLLECTION_BASE, DOCSTRING_EMBEDDING_BASE_PATH, 
    #         SRCCODE_EMBEDDING_BASE_PATH, DOCSTRING_CORPUS_PATH, 
    #         SRCCODE_CORPUS_PATH, TOGETHER_API_KEY_PATH
    #     )
        
    #     # 创建临时配置对象，包含所有必需参数
    #     class TempArgs:
    #         def __init__(self):
    #             # 必需参数
    #             self.local_embedding_model = LOCAL_EMBEDDING_MODEL
    #             self.togetherai_embedding_model = TOGETHERAI_EMBEDDING_MODEL
    #             self.rag_collection_base = RAG_COLLECTION_BASE
    #             self.docstring_embedding_base_path = DOCSTRING_EMBEDDING_BASE_PATH
    #             self.srccode_embedding_base_path = SRCCODE_EMBEDDING_BASE_PATH
    #             self.docstring_corpus_path = DOCSTRING_CORPUS_PATH
    #             self.srccode_corpus_path = SRCCODE_CORPUS_PATH
    #             # 可选参数
    #             self.embedding_source = embedding_source
    #             self.rag_document_num = rag_document_num
    #             self.max_dependency_num = max_dependency_num
    #             self.append_srcDep = append_srcDep
    #             self.query_cache_dir = query_cache_dir
    #             self.max_token_length = max_token_length
    #             self.together_api_key_path = TOGETHER_API_KEY_PATH
        
    #     temp_args = TempArgs()
    #     rag_config = RAGConfig.from_args(temp_args)
    
    # 确保rag_config是RAGConfig对象，而不是OmegaConf字典
    if not hasattr(rag_config, 'get_embedding_model'):
        logger.error(f"rag_config is not a proper RAGConfig object: {type(rag_config)}")
        # 如果rag_config不是正确的对象，尝试从其属性重新创建
        if hasattr(rag_config, 'local_embedding_model'):
            from utils.RAGutils.rag_config import RAGConfig
            rag_config = RAGConfig(
                local_embedding_model=rag_config.local_embedding_model,
                togetherai_embedding_model=rag_config.togetherai_embedding_model,
                rag_collection_base=rag_config.rag_collection_base,
                docstring_embedding_base_path=rag_config.docstring_embedding_base_path,
                srccode_embedding_base_path=rag_config.srccode_embedding_base_path,
                docstring_corpus_path=rag_config.docstring_corpus_path,
                srccode_corpus_path=rag_config.srccode_corpus_path,
                embedding_source=getattr(rag_config, 'embedding_source', 'local'),
                rag_document_num=getattr(rag_config, 'rag_document_num', 10),
                max_dependency_num=getattr(rag_config, 'max_dependency_num', None),
                append_srcDep=getattr(rag_config, 'append_srcDep', False),
                query_cache_dir=getattr(rag_config, 'query_cache_dir', query_cache_dir),
                max_token_length=getattr(rag_config, 'max_token_length', max_token_length),
                together_api_key_path=getattr(rag_config, 'together_api_key_path', f'{API_KEY_PATH}/together_api_key.txt')
            )
        else:
            logger.error("rag_config is not a valid RAGConfig object and cannot be reconstructed")
            data["context"] = ""
            return data, "error"
    
    # 设置默认task_type
    if task_type is None:
        task_type = "VACE"
        logger.info(f"未指定task_type，使用默认值: {task_type}")
    
    # 设置默认corpus_type
    if corpus_type is None:
        corpus_type = "docstring"
        logger.info(f"未指定corpus_type，使用默认值: {corpus_type}")
    
    # 初始化API客户端
    api_init_start = time.time()
    try:
        with open(rag_config.together_api_key_path, "r") as f:
            api_key = f.read()
        together_client = Together(api_key=api_key)
        embed_func_args = {
            'source': rag_config.embedding_source,
            'model_name': rag_config.get_embedding_model(),
            'together_client': together_client if rag_config.embedding_source == 'togetherai' else None,
            'batch_size': 64  # 可以根据需要调整
        }
        api_init_end = time.time()
        logger.info(f"API initialization took {api_init_end - api_init_start:.2f} seconds")
    except Exception as e:
        logger.error(f"API初始化失败: {e}")
        data["context"] = ""
        return data, "error"
    
    # 使用新的路径结构: collection_base/knowledge_type/embed_model_name/
    embed_model_name = rag_config.get_embedding_model().split('/')[-1]
    db_path = os.path.join(rag_config.rag_collection_base, corpus_type, embed_model_name)
    logger.info(f"DB path: {db_path}")
    
    if 'context' not in data or data["context"] == "":
        try:
            if rag_config.useQAContext:
                if rag_config.QAContext_path is None:
                    logger.error("QAContext_path is not provided")
                    data["context"] = ""
                    return data, "error"
                from utils.io_utils import loadJsonl
                QA_cache_context = loadJsonl(rag_config.QAContext_path)
                from utils.RAGutils.queryResultByInfer.contextLoad import loadQAContext
                context = loadQAContext(data,max_token_length,QA_cache_context)
                data["context"] = context if context is not None else ""
                total_end_time = time.time()
                logger.info(f"Total appendContextToData execution took {total_end_time - total_start_time:.2f} seconds (QAContext)")
                return data, "success"
            elif useRAG:
                rag_start = time.time()
                try:
                    # 计时：ChromaDB客户端初始化
                    chroma_init_start = time.time()
                    chroma_client = chromadb.PersistentClient(path=db_path, settings=chroma_settings)
                    chroma_init_end = time.time()
                    logger.info(f"ChromaDB client initialization took {chroma_init_end - chroma_init_start:.2f} seconds")
                    
                    # 计时：RAG检索器初始化
                    retriever_init_start = time.time()
                    # 确保传递正确的corpus_type参数
                    ragretriever = RAGContextRetriever(
                        chroma_client=chroma_client, 
                        embed_func_args=embed_func_args, 
                        corpus_type=corpus_type,
                        rag_collection_base=rag_config.rag_collection_base,
                        knowledge_type=corpus_type,
                        embedding_source=rag_config.embedding_source,
                        docstring_embedding_base_path=rag_config.docstring_embedding_base_path,
                        srccode_embedding_base_path=rag_config.srccode_embedding_base_path,
                        max_dependency_num=rag_config.max_dependency_num,
                        append_srcDep=rag_config.append_srcDep,
                        query_cache_dir=rag_config.query_cache_dir,
                        rag_document_num=rag_config.rag_document_num,
                        docstring_corpus_path=rag_config.docstring_corpus_path,
                        srccode_corpus_path=rag_config.srccode_corpus_path
                    )
                    retriever_init_end = time.time()
                    logger.info(f"RAG retriever initialization took {retriever_init_end - retriever_init_start:.2f} seconds")
                    
                    # 计时：上下文检索
                    context_retrieval_start = time.time()
                    context = ragretriever.retrieve_context(data, benchmark_type='VersiBCB', task=task_type, max_token_length=max_token_length)
                    context_retrieval_end = time.time()
                    logger.info(f"Context retrieval took {context_retrieval_end - context_retrieval_start:.2f} seconds")
                    
                    # 检查是否为锁冲突
                    if context is None:
                        logger.warning(f"Context retrieval returned None for data_id {data.get('id', 'unknown')} - collection building is locked")
                        data["context"] = ""
                        total_end_time = time.time()
                        logger.info(f"Total appendContextToData execution took {total_end_time - total_start_time:.2f} seconds (locked)")
                        return data, "locked"
                    
                    # 正常情况：设置context
                    data["context"] = context if context is not None else ""
                    
                    rag_end = time.time()
                    logger.info(f"Total RAG process took {rag_end - rag_start:.2f} seconds")
                    logger.info(f"Context length: {len(data['context'])} characters")
                    logger.info(f"Context获取完成，可进行预测")
                except Exception as e:
                    logger.error(f"Error retrieving RAG context: {e}")
                    data["context"] = ""
                    total_end_time = time.time()
                    logger.info(f"Total appendContextToData execution took {total_end_time - total_start_time:.2f} seconds (error)")
                    return data, "error"
            else:
                non_rag_start = time.time()
                
                # 计时：获取知识文档
                docs_loading_start = time.time()
                # 根据corpus_type确定corpus路径
                docstring_corpus_path = rag_config.docstring_corpus_path
                srccode_corpus_path = rag_config.srccode_corpus_path
                
                knowledge_docs = getKnowledgeDocs(
                    data, 
                    corpus_type=corpus_type, 
                    dataset='VersiBCB', 
                    task=task_type,
                    docstring_embedding_base_path=rag_config.docstring_embedding_base_path,
                    srccode_embedding_base_path=rag_config.srccode_embedding_base_path,
                    docstring_corpus_path=docstring_corpus_path,
                    srccode_corpus_path=srccode_corpus_path
                )
                docs_loading_end = time.time()
                logger.info(f"Knowledge docs loading took {docs_loading_end - docs_loading_start:.2f} seconds")
                logger.info(f"Knowledge_docs加载完成，准备获取{max_token_length}的context")
                
                # 计时：处理文档
                processing_start = time.time()
                if isinstance(knowledge_docs, dict):  # versiBCB
                    knowledge_docs = distribute_doc_tokens(knowledge_docs, max_token_length, tokenizer)
                    data["context"] = "\n".join(knowledge_docs[pack] for pack in knowledge_docs)
                else:  # versicode
                    knowledge_doc = truncate_doc("\n".join(knowledge_docs), max_token_length, tokenizer)
                    data["context"] = knowledge_doc
                processing_end = time.time()
                logger.info(f"Document processing took {processing_end - processing_start:.2f} seconds")
                
                non_rag_end = time.time()
                logger.info(f"Total non-RAG process took {non_rag_end - non_rag_start:.2f} seconds")
                logger.info(f"Context length: {len(data['context'])} characters")
        except Exception as e:
            logger.error(f"Error in appendContextToData: {e}")
            data["context"] = ""
            total_end_time = time.time()
            logger.info(f"Total appendContextToData execution took {total_end_time - total_start_time:.2f} seconds (error)")
            return data, "error"
    else:
        logger.info("Context already exists in data, skipping retrieval")
    
    total_end_time = time.time()
    logger.info(f"Total appendContextToData execution took {total_end_time - total_start_time:.2f} seconds")
    
    return data, "success"