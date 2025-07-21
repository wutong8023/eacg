import hashlib
import os
import json
import logging
import numpy as np
import time
from tqdm import tqdm
from utils.RAGutils.RAGEmbedding import CustomEmbeddingFunction, PrecomputedEmbeddingsManager
from utils.RAGutils.document_utils import get_version
from utils.RAGutils.query_dependency_filter import QueryDependencyFilter
import re
#TODO:将RAGRetriever创建collection过程调用collectinbuilder的方法
from utils.RAGutils.config.default_config import QUERY_CACHE_DIR
from utils.RAGutils.config.default_config import RAG_STRING_TRUNCATE_TOKENIZER
# Define a path for the JSON cache


def edit_distance(str1, str2):
    """
    计算两个字符串之间的编辑距离（Levenshtein距离）
    """
    if not str1:
        return len(str2) if str2 else 0
    if not str2:
        return len(str1)
    
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 初始化
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # 填充DP表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]

def normalize_api_path(path):
    """
    规范化API路径，应用等价规则
    
    Args:
        path: API路径字符串
        
    Returns:
        str: 规范化后的路径
    """
    if not path:
        return path
    
    # 等价规则映射
    equivalence_rules = {
        'scipy.fftpack': 'scipy.fft',
        'matplotlib.axes._axes': 'matplotlib.axes',
    }
    part_equal_rules = {
        'pylab':'pyplot'
    }
    
    # 应用等价规则
    normalized = path
    for old_prefix, new_prefix in equivalence_rules.items():
        if normalized.startswith(old_prefix):
            normalized = new_prefix + normalized[len(old_prefix):]
            break
    
    # 应用部分等价规则
    for old_part, new_part in part_equal_rules.items():
        normalized = normalized.replace(old_part, new_part)
    
    return normalized

def apply_corpus_mapping(corpus_path):
    """
    对corpus中的path/alias应用映射规则，用于扩展exact match
    
    Args:
        corpus_path: corpus中的API路径字符串
        
    Returns:
        str: 应用映射后的路径
    """
    if not corpus_path:
        return corpus_path
    
    # 等价规则映射（完整前缀转换）
    equivalence_rules = {
        'scipy.fftpack': 'scipy.fft',
        'matplotlib.axes._axes': 'matplotlib.axes',
    }
    
    # 部分等价规则（部分前缀直接替换）
    part_equal_rules = {
        'pylab': 'pyplot'
    }
    
    # 应用完整前缀等价规则
    mapped_path = corpus_path
    for old_prefix, new_prefix in equivalence_rules.items():
        mapped_path = mapped_path.replace(old_prefix, new_prefix)
    
    # 应用部分等价规则
    for old_part, new_part in part_equal_rules.items():
        mapped_path = mapped_path.replace(old_part, new_part)
    
    return mapped_path

def tokenize_for_matching(text, simple_tokenizer=True):
    """
    为匹配目的对文本进行分词
    
    Args:
        text: 输入文本
        simple_tokenizer: 是否使用简单分词器
        
    Returns:
        set: 分词结果集合
    """
    if not text:
        return set()
    
    if simple_tokenizer:
        # 简单分词：按非字母数字字符分割，转小写，过滤短词
        import re
        tokens = re.findall(r'[a-zA-Z0-9_]+', text.lower())
        return set(token for token in tokens if len(token) > 1)
    else:
        # 可以在这里添加更复杂的分词逻辑
        return set(text.lower().split())

def calculate_token_set_similarity(query_tokens, doc_tokens):
    """
    计算两个token集合的相似度
    
    Args:
        query_tokens: 查询token集合
        doc_tokens: 文档token集合
        
    Returns:
        float: 相似度分数 (0-1之间，越高越相似)
    """
    if not query_tokens or not doc_tokens:
        return 0.0
    
    intersection = query_tokens.intersection(doc_tokens)
    union = query_tokens.union(doc_tokens)
    
    if not union:
        return 0.0
    
    # Jaccard相似度
    jaccard = len(intersection) / len(union)
    
    # 给完全匹配的token更高权重，但不超过1.0
    if intersection == query_tokens or intersection == doc_tokens:
        # 使用加权平均而不是直接乘以1.5，确保结果不超过1.0
        bonus = 0.3  # 给完全匹配额外0.3分
        jaccard = min(1.0, jaccard + bonus)
    
    return jaccard

def get_document_match_score(doc_data, query_dict):
    """
    计算文档与查询的匹配分数
    
    Args:
        doc_data: 文档数据字典，包含path、doc、aliases字段
        query_dict: 查询字典，包含path、description字段
    
    Returns:
        tuple: (is_exact_match: bool, similarity_score: float)
    """
    if not isinstance(query_dict, dict) or 'path' not in query_dict:
        return False, 0.0
    
    query_path = query_dict['path']
    query_description = query_dict.get('description', '')
    
    if not query_path:
        return False, 0.0
    
    # 1. 尝试直接exact match (应用等价规则)
    normalized_query_path = normalize_api_path(query_path.lower())
    
    # 检查path字段的直接exact match
    if 'path' in doc_data and doc_data['path']:
        doc_path = doc_data['path']
        normalized_doc_path = normalize_api_path(doc_path.lower())
        if normalized_query_path == normalized_doc_path:
            return True, 1.0
    
    # 检查aliases字段的直接exact match
    if 'aliases' in doc_data and isinstance(doc_data['aliases'], list):
        for alias in doc_data['aliases']:
            if isinstance(alias, str):
                normalized_alias = normalize_api_path(alias.lower())
                if normalized_query_path == normalized_alias:
                    return True, 1.0
    
    # 2. 尝试映射转换后的exact match
    # 对corpus中的path/alias应用映射，然后再次尝试exact match
    
    # 检查path字段的映射后exact match
    if 'path' in doc_data and doc_data['path']:
        doc_path = doc_data['path']
        mapped_doc_path = apply_corpus_mapping(doc_path.lower())
        # 对mapped path也应用normalize规则
        normalized_mapped_doc_path = normalize_api_path(mapped_doc_path)
        if normalized_query_path == normalized_mapped_doc_path:
            return True, 1.0
    
    # 检查aliases字段的映射后exact match
    if 'aliases' in doc_data and isinstance(doc_data['aliases'], list):
        for alias in doc_data['aliases']:
            if isinstance(alias, str):
                mapped_alias = apply_corpus_mapping(alias.lower())
                # 对mapped alias也应用normalize规则
                normalized_mapped_alias = normalize_api_path(mapped_alias)
                if normalized_query_path == normalized_mapped_alias:
                    return True, 1.0
    
    # 3. 如果都没有exact match，使用token集合相似度
    # 构建查询的token集合
    query_text = query_path
    if query_description:
        # 只取description的第一个句子/部分
        desc_first_part = query_description.split('.')[0]
        query_text += " " + desc_first_part
    
    query_tokens = tokenize_for_matching(query_text)
    
    # 构建文档的token集合
    doc_text = ""
    if 'path' in doc_data and doc_data['path']:
        doc_text += doc_data['path']
    
    if 'doc' in doc_data and doc_data['doc']:
        # 只取doc的第一个句子/部分
        doc_first_part = doc_data['doc'].split('.')[0]
        doc_text += " " + doc_first_part
    
    doc_tokens = tokenize_for_matching(doc_text)
    
    # 计算相似度
    similarity = calculate_token_set_similarity(query_tokens, doc_tokens)
    
    return False, similarity

def getKnowledgeDocs(data, corpus_type="docstring", dataset="VersiCode", task="VSCC",docstring_embedding_base_path=None,srccode_embedding_base_path=None,docstring_corpus_path=None,srccode_corpus_path=None):
    '''
    Description:
        获取知识文档字符串
    Args:
        data: dict,数据,实际上只用到dependency部分信息，但是由于支持的多样性，所以保留了其他信息
        corpus_type: str,"docstring" or "srccodes"
    Returns:
        knowledge_doc: list,知识文档 if dataset=="versicode"
        knowledge_doc: dict,知识文档 if dataset=="versiBCB" dict[pack:list]。如果是sourcecode，list中每个元素代表一个文件的内容；
        如果是doc,目前list每个元素代表一个class or func or data
    '''
    global _embeddings_manager
    if corpus_type == "docstring":
        _embeddings_manager = PrecomputedEmbeddingsManager(docstring_embedding_base_path)
    elif corpus_type == "srccodes":
        _embeddings_manager = PrecomputedEmbeddingsManager(srccode_embedding_base_path)
    else:
        raise ValueError("Wrong corpus_type")
    if corpus_type == "docstring":
        corpus_base = docstring_corpus_path
    elif corpus_type == "srccodes":
        corpus_base = srccode_corpus_path
    else:
        raise ValueError("Wrong corpus_type")
    #TODO: 根据corpus_type选择corpus_base
    if dataset == "VersiCode":
        dependency = data["dependency"] if task == "VSCC" else data["target_dependency"]
        version = get_version(data["version"]) if task == "VSCC" else get_version(data["target_version"])
        
        corpus_path = os.path.join(corpus_base, dependency, version + ".jsonl")
        knowledge_docs = []
        try:
            with open(corpus_path, "r") as f:
                for line in f:
                    knowledge_docs.append(line)
        except Exception as e:
            print(f"Error reading corpus: {e}")
            print(f"Corpus path: {corpus_path}")
            return []
            
    elif dataset == "VersiBCB":
        dependency = data["dependency"] if task == "VSCC" else data["target_dependency"]
        knowledge_docs = {}
        for pack, version in dependency.items():
            if version is None:
                continue
                
            # 检查是否有预编码
            cached_data = _embeddings_manager.load_embeddings(pack, version)
            if cached_data is not None and 'documents' in cached_data:
                print(f"Using precomputed embeddings for {pack} {version}")
                knowledge_docs[pack] = cached_data['documents']
                continue
                
            all_corpus_path = os.path.join(corpus_base, pack, version + ".jsonl")
            knowledge_docs[pack] = []
            try:
                with open(all_corpus_path, "r") as f:
                    for line in f:
                        knowledge_docs[pack].append(line)
            except Exception as e:
                print(f"Error reading corpus: {e}")
                print(f"Corpus path: {all_corpus_path}")
                knowledge_docs[pack] = []
    else:
        raise ValueError("Wrong dataset")
    return knowledge_docs
def getEvolveKnowledgeDocs(datapath):
    '''
    Description:
        获取evolve的上下文
    Args:
        datapath: 数据路径,str
    Returns:
        data: 数据,list[str]
    '''
    with open(datapath,"r") as f:
        datas = json.load(f)
    contexts = [f"{data['api']}:{data['evolve_description']}" for data in datas]
    return contexts
class RAGContextRetriever:
    """
    RAG上下文检索器，负责管理ChromaDB集合的创建、文档添加和向量检索功能。
    采用面向对象方法，封装内部状态和操作，提高代码内聚性。
    初始化RAG上下文检索器
    
    Args:
        chroma_client: ChromaDB客户端实例（用于向后兼容，新版本会为每个collection创建独立客户端）
        embed_func_args: 嵌入函数的参数
        corpus_type: 语料库类型 ("docstring" 或 "srccodes")
        rag_collection_base: RAG collection基础路径
        knowledge_type: 知识类型
        embedding_source: 嵌入源
        docstring_embedding_base_path: docstring嵌入基础路径
        srccode_embedding_base_path: 源码嵌入基础路径
        max_dependency_num: 最大依赖数量
        append_srcDep: 是否添加源依赖
        query_cache_dir: 查询缓存目录
        rag_document_num: RAG文档数量
        docstring_corpus_path: docstring语料库路径
        srccode_corpus_path: 源码语料库路径
        generated_queries_file: 生成的查询文件路径
        use_generated_queries: 是否使用生成的查询
        fixed_docs_per_query: 每个query固定获取的文档数量，None表示使用progressive模式
        enable_dependency_filtering: 是否启用dependency过滤
        enable_query_dependency_filtering: 是否启用查询层面的dependency过滤
        query_filter_strict_mode: 查询过滤器的严格模式
    """
    def __init__(self, chroma_client, embed_func_args, corpus_type, 
                 rag_collection_base, knowledge_type, embedding_source,
                 docstring_embedding_base_path, srccode_embedding_base_path,
                 max_dependency_num=None, append_srcDep=False, 
                 query_cache_dir=QUERY_CACHE_DIR, 
                 rag_document_num=10,docstring_corpus_path=None,srccode_corpus_path=None,
                 generated_queries_file=None, use_generated_queries=False,
                 fixed_docs_per_query=None, enable_dependency_filtering=True,
                 enable_query_dependency_filtering=False, query_filter_strict_mode=True,
                 max_doc_tokens=None, doc_truncate_tokenizer=RAG_STRING_TRUNCATE_TOKENIZER,
                 api_name_str_match=False):
        """
        初始化RAG上下文检索器
        
        Args:
            chroma_client: ChromaDB客户端实例（用于向后兼容，新版本会为每个collection创建独立客户端）
            embed_func_args: 嵌入函数的参数
            corpus_type: 语料库类型 ("docstring" 或 "srccodes")
            rag_collection_base: RAG collection基础路径
            knowledge_type: 知识类型
            embedding_source: 嵌入源
            docstring_embedding_base_path: docstring嵌入基础路径
            srccode_embedding_base_path: 源码嵌入基础路径
            max_dependency_num: 最大依赖数量
            append_srcDep: 是否添加源依赖
            query_cache_dir: 查询缓存目录
            rag_document_num: RAG文档数量
            docstring_corpus_path: docstring语料库路径
            srccode_corpus_path: 源码语料库路径
            generated_queries_file: 生成的查询文件路径
            use_generated_queries: 是否使用生成的查询
            fixed_docs_per_query: 每个query固定获取的文档数量，None表示使用progressive模式
            enable_dependency_filtering: 是否启用dependency过滤
            enable_query_dependency_filtering: 是否启用查询层面的dependency过滤
            query_filter_strict_mode: 查询过滤器的严格模式
            max_doc_tokens: 单个文档doc字段的最大token数，如果设置则会在合并前对doc字段进行截断
            doc_truncate_tokenizer: 用于测量doc字段token数的分词器模型
            api_name_str_match: 是否启用API名称字符串匹配，仅对包含"path"字段的dict query有效
        """
        self.chroma_client = chroma_client  # 保留用于向后兼容
        self.embed_func_args = embed_func_args
        self.corpus_type = corpus_type
        self.rag_collection_base = rag_collection_base
        self.knowledge_type = knowledge_type
        self.embedding_source = embedding_source
        self.docstring_embedding_base_path = docstring_embedding_base_path
        self.srccode_embedding_base_path = srccode_embedding_base_path
        self.docstring_corpus_path = docstring_corpus_path
        self.srccode_corpus_path = srccode_corpus_path
        self.generated_queries_file = generated_queries_file
        self.use_generated_queries = use_generated_queries
        self.fixed_docs_per_query = fixed_docs_per_query
        self.enable_dependency_filtering = enable_dependency_filtering
        self.enable_query_dependency_filtering = enable_query_dependency_filtering
        self.query_filter_strict_mode = query_filter_strict_mode
        self.max_dependency_num = max_dependency_num
        self.append_srcDep = append_srcDep
        self.query_cache_dir = query_cache_dir
        self.rag_document_num = rag_document_num
        self.max_doc_tokens = max_doc_tokens
        self.doc_truncate_tokenizer = doc_truncate_tokenizer
        self.api_name_str_match = api_name_str_match
        
        # 设置logger - 移到这里以确保在_load_generated_queries之前初始化
        self.logger = logging.getLogger("RAGContextRetriever")
        
        # 初始化查询依赖过滤器
        if self.enable_query_dependency_filtering:
            self.query_dependency_filter = QueryDependencyFilter(
                strict_mode=self.query_filter_strict_mode,
                enable_alias_matching=True
            )
            self.logger.info("Query dependency filtering enabled")
        else:
            self.query_dependency_filter = None
            self.logger.info("Query dependency filtering disabled")
        
        # 加载生成的查询
        self.generated_queries = {}
        if self.use_generated_queries and self.generated_queries_file:
            self._load_generated_queries()
        
        # 确保缓存目录存在
        os.makedirs(self.query_cache_dir, exist_ok=True)
        
        # 初始化embeddings管理器
        if corpus_type == "docstring":
            self.embeddings_manager = PrecomputedEmbeddingsManager(docstring_embedding_base_path)
        elif corpus_type == "srccodes":
            self.embeddings_manager = PrecomputedEmbeddingsManager(srccode_embedding_base_path)
        else:
            raise ValueError("Wrong corpus_type")
        
        # 初始化embedding函数
        self.embedding_function_instance = CustomEmbeddingFunction(
            **self.embed_func_args
        )
        
        # 初始化文档截断用的tokenizer
        self.doc_tokenizer = None
        if self.max_doc_tokens:
            try:
                from transformers import AutoTokenizer
                self.doc_tokenizer = AutoTokenizer.from_pretrained(self.doc_truncate_tokenizer, trust_remote_code=True)
                self.logger.info(f"Initialized document truncation tokenizer: {self.doc_truncate_tokenizer}")
            except Exception as e:
                self.logger.error(f"Failed to load tokenizer {self.doc_truncate_tokenizer}: {e}")
                self.logger.warning("Document tokenization will be disabled")

    def _get_dependency_folder_name(self, dependencies):
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
        from utils.getDependencyUtils import dict_to_pkg_ver_tuples
        pkg_ver_tuples = dict_to_pkg_ver_tuples(dependencies)
        
        # 过滤掉None版本的依赖项，然后排序
        valid_deps = [(pkg, ver) for pkg, ver in pkg_ver_tuples if ver is not None]
        sorted_deps = sorted(valid_deps)
        return "_".join([f"{pkg}_{ver}" for pkg, ver in sorted_deps])

    def _get_or_create_collection_with_new_structure(self, data, benchmark_type, task):
        """
        使用新的存储结构获取或创建collection
        
        Args:
            data: 数据字典
            benchmark_type: 数据集类型
            task: 任务类型
            
        Returns:
            collection对象或None（如果被锁或构建失败）
        """
        sample_id = data.get('id', 'unknown')
        try:
            # 获取依赖项
            if benchmark_type == "VersiCode":
                pkg, ver = data["dependency"], get_version(data["version"])
                dependencies = {pkg: ver}
            elif benchmark_type == "VersiBCB":
                dependencies = data["dependency"] if task == "VSCC" else data["target_dependency"]
            else:
                raise ValueError(f"不支持的数据集: {benchmark_type}")
            
            # 应用依赖项过滤和组合逻辑
            if benchmark_type == "VersiBCB":
                target_dep = data["dependency"] if task == "VSCC" else data["target_dependency"]
                src_dep = data["origin_dependency"] if task == "VACE" else data["dependency"]
                from utils.getDependencyUtils import getSubsetDep, combineDep
                
                if self.max_dependency_num:
                    target_dep = getSubsetDep(target_dep, self.max_dependency_num)
                    src_dep = getSubsetDep(src_dep, self.max_dependency_num)
                if self.append_srcDep:
                    dependencies = combineDep(target_dep, src_dep)
                else:
                    dependencies = target_dep
            
            self.logger.info(f"Sample {sample_id}: Processing dependencies: {dependencies}")
            
            # 构建collection路径
            folder_name = self._get_dependency_folder_name(dependencies)
            collection_hash = self._get_collection_hash(dependencies)
            embed_model_name = self.embed_func_args['model_name'].split('/')[-1]
            collection_path = os.path.join(
                self.rag_collection_base, 
                self.knowledge_type, 
                embed_model_name, 
                folder_name, 
                collection_hash
            )
            
            self.logger.info(f"Sample {sample_id}: Collection path: {collection_path}")
            
            # 如果路径不存在，创建路径
            if not os.path.exists(collection_path):
                self.logger.info(f"Sample {sample_id}: Collection path does not exist, creating: {collection_path}")
                os.makedirs(collection_path, exist_ok=True)
            
            # 创建独立的ChromaDB客户端
            try:
                import chromadb
                client = chromadb.PersistentClient(
                    path=collection_path,
                    settings=chromadb.Settings(
                        allow_reset=True,
                        anonymized_telemetry=False,
                        is_persistent=True
                    )
                )
                self.logger.info(f"Sample {sample_id}: Successfully created ChromaDB client")
            except Exception as e:
                self.logger.error(f"Sample {sample_id}: Error creating ChromaDB client for path {collection_path}: {type(e).__name__}: {str(e)}")
                import traceback
                self.logger.error(f"Sample {sample_id}: ChromaDB client creation traceback:\n{traceback.format_exc()}")
                return None
            
            collection_name = self._get_collection_hash(dependencies)
            
            try:
                # 尝试获取已存在的collection
                collection = client.get_collection(name=collection_name)
                self.logger.info(f"Sample {sample_id}: Found existing collection: {collection_name} at {collection_path}")
                return collection
            except Exception as e:
                self.logger.info(f"Sample {sample_id}: Collection {collection_name} not found at {collection_path}, will create new one. Error: {type(e).__name__}: {str(e)}")
                
                # 使用带锁的CollectionBuilder构建新的collection
                self.logger.info(f"Sample {sample_id}: Building new collection using CollectionBuilder for dependencies: {dependencies}")
                
                try:
                    # 导入异常类
                    from utils.RAGutils.CollectionBuilder import CollectionBuilder, CollectionBuildingLockedException
                    
                    # 创建CollectionBuilder实例
                    builder = CollectionBuilder(
                        chroma_client=client,
                        embed_func_args=self.embed_func_args,
                        corpus_base=self.docstring_corpus_path if self.corpus_type == "docstring" else self.srccode_corpus_path,
                        collection_base=self.rag_collection_base,
                        batch_size=250,
                        verbose=True,
                        num_processes=1,
                        knowledge_type=self.corpus_type,
                        embedding_base_path=self.docstring_embedding_base_path if self.corpus_type == "docstring" else self.srccode_embedding_base_path
                    )
                    
                    self.logger.info(f"Sample {sample_id}: CollectionBuilder created successfully")
                    
                    # 使用带锁的构建方法，超时时间设为0（不等待）
                    built_collection_name = builder.build_collection_with_lock(
                        dependencies, 
                        collection_name, 
                        force_rebuild=False,
                        timeout=0  # 不等待，立即返回
                    )
                    
                    if built_collection_name:
                        # 获取构建好的collection
                        collection = client.get_collection(name=built_collection_name)
                        self.logger.info(f"Sample {sample_id}: Successfully built and retrieved collection: {built_collection_name}")
                        return collection
                    else:
                        self.logger.error(f"Sample {sample_id}: Failed to build collection for dependencies: {dependencies}")
                        return None
                        
                except CollectionBuildingLockedException as lock_error:
                    self.logger.warning(f"Sample {sample_id}: Collection building locked: {lock_error}")
                    # 返回特殊标记表示锁冲突
                    return "LOCKED"
                
                except Exception as build_error:
                    self.logger.error(f"Sample {sample_id}: Error building collection with CollectionBuilder: {type(build_error).__name__}: {str(build_error)}")
                    import traceback
                    self.logger.error(f"Sample {sample_id}: CollectionBuilder error traceback:\n{traceback.format_exc()}")
                    return None
        
        except Exception as e:
            import traceback
            self.logger.error(f"Sample {sample_id}: Unexpected error in _get_or_create_collection_with_new_structure: {type(e).__name__}: {str(e)}")
            self.logger.error(f"Sample {sample_id}: Full traceback:\n{traceback.format_exc()}")
            return None

    def _get_or_create_collection_with_dependencies(self, dependencies, sample_id):
        """
        基于dependencies直接获取或创建collection - 解耦模式
        
        Args:
            dependencies: 依赖字典 {pkg: version}
            sample_id: 样本ID用于日志记录
            
        Returns:
            collection对象或None（如果被锁或构建失败）
        """
        try:
            self.logger.info(f"Sample {sample_id}: Processing dependencies in decoupled mode: {dependencies}")
            
            # 应用依赖项过滤和组合逻辑
            target_dep = dependencies
            if self.max_dependency_num:
                from utils.getDependencyUtils import getSubsetDep
                target_dep = getSubsetDep(target_dep, self.max_dependency_num)
            
            self.logger.info(f"Sample {sample_id}: Final dependencies after filtering: {target_dep}")
            
            # 构建collection路径
            folder_name = self._get_dependency_folder_name(target_dep)
            collection_hash = self._get_collection_hash(target_dep)
            embed_model_name = self.embed_func_args['model_name'].split('/')[-1]
            collection_path = os.path.join(
                self.rag_collection_base, 
                self.knowledge_type, 
                embed_model_name, 
                folder_name, 
                collection_hash
            )
            
            self.logger.info(f"Sample {sample_id}: Collection path: {collection_path}")
            
            # 如果路径不存在，创建路径
            if not os.path.exists(collection_path):
                self.logger.info(f"Sample {sample_id}: Collection path does not exist, creating: {collection_path}")
                os.makedirs(collection_path, exist_ok=True)
            
            # 创建独立的ChromaDB客户端
            try:
                import chromadb
                client = chromadb.PersistentClient(
                    path=collection_path,
                    settings=chromadb.Settings(
                        allow_reset=True,
                        anonymized_telemetry=False,
                        is_persistent=True
                    )
                )
                self.logger.info(f"Sample {sample_id}: Successfully created ChromaDB client")
            except Exception as e:
                self.logger.error(f"Sample {sample_id}: Error creating ChromaDB client for path {collection_path}: {type(e).__name__}: {str(e)}")
                import traceback
                self.logger.error(f"Sample {sample_id}: ChromaDB client creation traceback:\n{traceback.format_exc()}")
                return None
            
            collection_name = self._get_collection_hash(target_dep)
            
            try:
                # 尝试获取已存在的collection
                collection = client.get_collection(name=collection_name)
                self.logger.info(f"Sample {sample_id}: Found existing collection: {collection_name} at {collection_path}")
                return collection
            except Exception as e:
                self.logger.info(f"Sample {sample_id}: Collection {collection_name} not found at {collection_path}, will create new one. Error: {type(e).__name__}: {str(e)}")
                
                # 使用带锁的CollectionBuilder构建新的collection
                self.logger.info(f"Sample {sample_id}: Building new collection using CollectionBuilder for dependencies: {target_dep}")
                
                try:
                    # 导入异常类
                    from utils.RAGutils.CollectionBuilder import CollectionBuilder, CollectionBuildingLockedException
                    
                    # 创建CollectionBuilder实例
                    builder = CollectionBuilder(
                        chroma_client=client,
                        embed_func_args=self.embed_func_args,
                        corpus_base=self.docstring_corpus_path if self.corpus_type == "docstring" else self.srccode_corpus_path,
                        collection_base=self.rag_collection_base,
                        batch_size=250,
                        verbose=True,
                        num_processes=1,
                        knowledge_type=self.corpus_type,
                        embedding_base_path=self.docstring_embedding_base_path if self.corpus_type == "docstring" else self.srccode_embedding_base_path
                    )
                    
                    self.logger.info(f"Sample {sample_id}: CollectionBuilder created successfully")
                    
                    # 使用带锁的构建方法，超时时间设为0（不等待）
                    built_collection_name = builder.build_collection_with_lock(
                        target_dep, 
                        collection_name, 
                        force_rebuild=False,
                        timeout=0  # 不等待，立即返回
                    )
                    
                    if built_collection_name:
                        # 获取构建好的collection
                        collection = client.get_collection(name=built_collection_name)
                        self.logger.info(f"Sample {sample_id}: Successfully built and retrieved collection: {built_collection_name}")
                        return collection
                    else:
                        self.logger.error(f"Sample {sample_id}: Failed to build collection for dependencies: {target_dep}")
                        return None
                        
                except CollectionBuildingLockedException as lock_error:
                    self.logger.warning(f"Sample {sample_id}: Collection building locked: {lock_error}")
                    # 返回特殊标记表示锁冲突
                    return "LOCKED"
                
                except Exception as build_error:
                    self.logger.error(f"Sample {sample_id}: Error building collection with CollectionBuilder: {type(build_error).__name__}: {str(build_error)}")
                    import traceback
                    self.logger.error(f"Sample {sample_id}: CollectionBuilder error traceback:\n{traceback.format_exc()}")
                    return None
        
        except Exception as e:
            import traceback
            self.logger.error(f"Sample {sample_id}: Unexpected error in _get_or_create_collection_with_dependencies: {type(e).__name__}: {str(e)}")
            self.logger.error(f"Sample {sample_id}: Full traceback:\n{traceback.format_exc()}")
            return None

    def retrieve_context(self, data=None, benchmark_type=None, task=None, max_token_length=2000, 
                         queries=None, dependencies=None, sample_id=None):
        """
        检索RAG上下文 - 支持直接传入queries以解耦数据格式依赖
        
        Args:
            data: 数据字典 (可选，当使用传统模式时)
            benchmark_type: 数据集类型，"VersiCode"或"VersiBCB" (可选，当使用传统模式时)
            task: 任务类型，"VSCC"或"VACE" (可选，当使用传统模式时)
            max_token_length: 返回上下文的最大token长度
            queries: 查询列表，可以是字符串列表或dict列表 (可选，优先使用此参数)
            dependencies: 依赖字典 {pkg: version} (可选，用于collection构建和过滤)
            sample_id: 样本ID (可选，用于日志记录)
            
        Returns:
            retrieved_context: str,检索到的上下文
            
        Note:
            支持两种使用模式：
            1. 传统模式：传入data, benchmark_type, task (向后兼容)
            2. 解耦模式：直接传入queries, dependencies
        """
        # 确定sample_id
        if sample_id is None:
            if data is not None:
                sample_id = data.get('id', 'unknown')
            else:
                sample_id = 'direct_query'
        
        try:
            total_start = time.time()
            
            # 1. 获取查询列表
            query_start = time.time()
            if queries is not None:
                # 解耦模式：直接使用传入的queries
                if isinstance(queries, str):
                    queries = [queries]  # 单个查询转换为列表
                elif not isinstance(queries, list):
                    raise ValueError("queries must be a string or list of strings/dicts")
                
                self.logger.info(f"Sample {sample_id}: Using provided queries: {len(queries)} queries")
            else:
                # 传统模式：从data中生成queries
                if data is None or task is None:
                    raise ValueError("Either provide 'queries' parameter or 'data' and 'task' parameters")
                queries = self._get_query_from_data(data, task)
                self.logger.info(f"Sample {sample_id}: Generated queries from data: {len(queries)} queries")
            
            query_end = time.time()
            self.logger.info(f"Sample {sample_id}: 1. 查询获取耗时: {query_end - query_start:.4f}秒, 获得{len(queries)}个查询")
            
            # 2. 获取或创建collection
            collection_start = time.time()
            if dependencies is not None:
                # 解耦模式：使用传入的dependencies构建collection
                collection = self._get_or_create_collection_with_dependencies(dependencies, sample_id)
            else:
                # 传统模式：使用data构建collection
                if data is None or benchmark_type is None or task is None:
                    raise ValueError("Either provide 'dependencies' parameter or 'data', 'benchmark_type', and 'task' parameters")
                collection = self._get_or_create_collection_with_new_structure(data, benchmark_type, task)
            
            if collection == "LOCKED":
                self.logger.warning(f"Sample {sample_id}: Collection is being built by another process, skipping context retrieval")
                total_end = time.time()
                self.logger.info(f"Sample {sample_id}: 总耗时 (失败): {total_end - total_start:.4f}秒")
                return None
            elif collection is None:
                self.logger.error(f"Sample {sample_id}: Failed to get or create collection")
                total_end = time.time()
                self.logger.info(f"Sample {sample_id}: 总耗时 (失败): {total_end - total_start:.4f}秒")
                return None
            
            self.logger.info(f"Sample {sample_id}: Successfully obtained collection")
            collection_end = time.time()
            self.logger.info(f"Sample {sample_id}: 2. Collection获取/创建耗时: {collection_end - collection_start:.4f}秒")

            # 3. 执行渐进式查询
            progressive_query_start = time.time()
            
            # 获取target_dependencies用于过滤
            target_dependencies = None
            if dependencies is not None:
                # 解耦模式：直接使用传入的dependencies
                target_dependencies = dependencies
                if self.max_dependency_num:
                    from utils.getDependencyUtils import getSubsetDep
                    target_dependencies = getSubsetDep(target_dependencies, self.max_dependency_num)
            elif data is not None and benchmark_type is not None and task is not None:
                # 传统模式：从data中提取dependencies
                if benchmark_type == "VersiBCB":
                    target_dependencies = data["dependency"] if task == "VSCC" else data["target_dependency"]
                    # 应用依赖项过滤和组合逻辑（与collection构建保持一致）
                    if self.max_dependency_num:
                        from utils.getDependencyUtils import getSubsetDep
                        target_dependencies = getSubsetDep(target_dependencies, self.max_dependency_num)
                elif benchmark_type == "VersiCode":
                    pkg, ver = data["dependency"], get_version(data["version"])
                    target_dependencies = {pkg: ver}
            
            # 根据模式选择检索策略
            if self.fixed_docs_per_query is not None:
                # 固定文档数量模式
                self.logger.info(f"Sample {sample_id}: Using fixed docs mode: {self.fixed_docs_per_query} docs per query")
                retrieved_context = self._query_collection_fixed_docs(collection, queries, max_token_length, target_dependencies)
            elif len(queries) > 1:
                # 渐进式检索模式（多查询）
                self.logger.info(f"Sample {sample_id}: Using progressive retrieval for {len(queries)} queries")
                retrieved_context = self._query_collection_progressive(collection, queries, max_token_length, target_dependencies)
            elif len(queries) == 1:
                # 单个查询，使用原始方法
                self.logger.info(f"Sample {sample_id}: Using single query retrieval")
                results = self._query_collection(collection, queries[0])
                retrieved_context = self._process_query_results(results, max_token_length, target_dependencies)
            else:
                raise ValueError("queries must be a string or list of strings/dicts")
            
            progressive_query_end = time.time()
            self.logger.info(f"Sample {sample_id}: 3. 渐进式查询耗时: {progressive_query_end - progressive_query_start:.4f}秒")
            
            # 4. 记录总耗时
            total_end = time.time()
            self.logger.info(f"Sample {sample_id}: 总耗时: {total_end - total_start:.4f}秒")
            
            return retrieved_context
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.logger.error(f"Sample {sample_id}: Error in retrieve_context: {type(e).__name__}: {str(e)}")
            self.logger.error(f"Sample {sample_id}: Full traceback:\n{error_details}")
            return None
    
    def _get_collection_hash(self, dep):
        """
        为dependency和version生成唯一的hash
        支持两种格式：
        1. dict[pkg, ver] - 标准格式
        2. dict[pkg, list[ver]] - appendSrcDep后的格式
        """
        # 使用dict_to_pkg_ver_tuples处理两种格式
        from utils.getDependencyUtils import dict_to_pkg_ver_tuples
        pkg_ver_tuples = dict_to_pkg_ver_tuples(dep)
        
        # 过滤掉None版本的依赖项，然后排序并生成字符串
        valid_deps = [(pkg, ver) for pkg, ver in pkg_ver_tuples if ver is not None]
        sorted_deps = sorted(valid_deps)
        dep_str = "_".join([f"{pkg}_{ver}" for pkg, ver in sorted_deps])
        return hashlib.md5(dep_str.encode()).hexdigest()[:60]

    def _get_collection_name(self, data, dataset, task):
        """获取collection的名称"""
        if dataset == "VersiCode":
            pkg, ver = data["dependency"], get_version(data["version"])
            return self._get_collection_hash({pkg:ver})
        elif dataset == "VersiBCB":
            dep = data["dependency"] if task == "VSCC" else data["target_dependency"]
            return self._get_collection_hash(dep)
        else:
            raise ValueError(f"不支持的数据集: {dataset}")

    def _get_query_from_data(self, data, task):
        """从数据中构建查询，返回查询列表或单个查询"""
        sample_id = data.get('id', None)
        
        # 如果启用了生成的查询且有对应的查询，返回查询列表
        if self.use_generated_queries and str(sample_id) in self.generated_queries:
            queries_info = self.generated_queries[str(sample_id)]  # 修复：使用str(sample_id)保持一致性
            queries = queries_info.get("queries", [])
            
            if queries:
                # 处理生成的查询（dict格式）
                if isinstance(queries, list) and queries and isinstance(queries[0], dict):
                    # 如果启用API名称匹配，直接返回dict格式的查询
                    if self.api_name_str_match:
                        # 应用查询层面的dependency过滤
                        if self.enable_query_dependency_filtering and self.query_dependency_filter:
                            target_dependencies = self._get_target_dependencies_for_filtering(data, task)
                            if target_dependencies:
                                filtered_queries, filter_stats = self.query_dependency_filter.filter_queries(
                                    queries, target_dependencies
                                )
                                
                                self.logger.info(f"Sample {sample_id}: Query filtering applied - "
                                               f"{filter_stats['kept']}/{filter_stats['total']} queries kept "
                                               f"({(1-filter_stats['filter_ratio'])*100:.1f}%)")
                                
                                if filtered_queries:
                                    self.logger.info(f"Using {len(filtered_queries)} filtered generated dict queries for sample {sample_id}")
                                    return filtered_queries
                                else:
                                    self.logger.warning(f"All generated queries filtered out for sample {sample_id}, falling back to default")
                            else:
                                self.logger.warning(f"Cannot determine target dependencies for sample {sample_id}, skipping query filtering")
                        
                        self.logger.info(f"Using {len(queries)} generated dict queries for sample {sample_id}")
                        return queries
                    
                    # 将dict格式的查询转换为字符串列表
                    query_strings = []
                    for query_dict in queries:
                        if isinstance(query_dict, dict):
                            # 从dict中提取查询字符串
                            if 'path' in query_dict:
                                query_str = query_dict['path']
                                if 'description' in query_dict and query_dict['description']:
                                    query_str += " " + query_dict['description']
                                query_strings.append(query_str)
                            elif 'description' in query_dict:
                                query_strings.append(query_dict['description'])
                        elif isinstance(query_dict, str):
                            query_strings.append(query_dict)
                    
                    if query_strings:
                        # 应用查询层面的dependency过滤
                        if self.enable_query_dependency_filtering and self.query_dependency_filter:
                            target_dependencies = self._get_target_dependencies_for_filtering(data, task)
                            if target_dependencies:
                                # 将字符串查询转换回dict格式进行过滤
                                dict_queries = []
                                for i, query_str in enumerate(query_strings):
                                    original_query_dict = queries[i] if i < len(queries) else {}
                                    dict_queries.append(original_query_dict)
                                
                                filtered_queries, filter_stats = self.query_dependency_filter.filter_queries(
                                    dict_queries, target_dependencies
                                )
                                
                                self.logger.info(f"Sample {sample_id}: Query filtering applied - "
                                               f"{filter_stats['kept']}/{filter_stats['total']} queries kept "
                                               f"({(1-filter_stats['filter_ratio'])*100:.1f}%)")
                                
                                # 转换过滤后的查询回字符串格式
                                filtered_query_strings = []
                                for query_dict in filtered_queries:
                                    if 'path' in query_dict:
                                        query_str = query_dict['path']
                                        if 'description' in query_dict and query_dict['description']:
                                            query_str += " " + query_dict['description']
                                        filtered_query_strings.append(query_str)
                                    elif 'description' in query_dict:
                                        filtered_query_strings.append(query_dict['description'])
                                
                                if filtered_query_strings:
                                    self.logger.info(f"Using {len(filtered_query_strings)} filtered generated queries for sample {sample_id}")
                                    return filtered_query_strings
                                else:
                                    self.logger.warning(f"All generated queries filtered out for sample {sample_id}, falling back to default")
                            else:
                                self.logger.warning(f"Cannot determine target dependencies for sample {sample_id}, skipping query filtering")
                        
                        self.logger.info(f"Using {len(query_strings)} generated queries for sample {sample_id}")
                        return query_strings
                elif isinstance(queries, list) and queries and isinstance(queries[0], str):
                    # 已经是字符串列表
                    self.logger.info(f"Using {len(queries)} generated string queries for sample {sample_id}")
                    return queries
                else:
                    self.logger.warning(f"Generated queries format not recognized for sample {sample_id}, falling back to default")
            else:
                self.logger.warning(f"No valid generated queries found for sample {sample_id}, falling back to default")
        
        # 回退到原始查询生成逻辑，返回单个查询
        if self.api_name_str_match:
            # API匹配模式下，如果没有生成的查询，则不能正常工作
            raise ValueError("API name string matching is enabled but no generated queries with 'path' field found. "
                           "Please provide --generated_queries_file with dict format queries containing 'path' field, "
                           "or disable --api_name_str_match.")
        
        query = data["description"]
        if task == "VACE":
            query += data["origin_code"]
        return [query]  # 包装为列表以保持一致性

    def _get_target_dependencies_for_filtering(self, data, task):
        """获取用于查询过滤的target_dependencies"""
        try:
            if task == "VSCC":
                target_dependencies = data.get("dependency", {})
            elif task == "VACE":
                target_dependencies = data.get("target_dependency", {})
            else:
                self.logger.warning(f"Unknown task type: {task}")
                return None
            
            # 应用依赖项过滤和组合逻辑（与collection构建保持一致）
            if target_dependencies:
                if self.max_dependency_num:
                    from utils.getDependencyUtils import getSubsetDep
                    target_dependencies = getSubsetDep(target_dependencies, self.max_dependency_num)
                
                # 注意：这里不应用append_srcDep，因为我们只想根据target_dependency过滤查询
                return target_dependencies
            
            return None
        except Exception as e:
            self.logger.error(f"Error getting target dependencies for filtering: {e}")
            return None

    def _get_or_create_collection(self, collection_name, embedding_function_instance):
        """获取或创建collection"""
        collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_function_instance,
        )
        return collection

    def _load_documents_for_collection(self, data, dataset, task):
        """加载文档和对应的编码"""
        documents_to_add = []
        embeddings_to_add = []
        
        if dataset == "VersiCode":
            documents = getKnowledgeDocs(data,self.corpus_type, dataset=dataset)
            if not documents:
                self.logger.warning("No documents found for VersiCode, cannot populate collection.")
                return [], []
            documents_to_add = documents

        elif dataset == "VersiBCB":
            raw_docs_map = getKnowledgeDocs(data,self.corpus_type, dataset=dataset, task=task)
            all_documents = []
            all_embeddings = []
            
            target_dep = data["dependency"] if task == "VSCC" else data["target_dependency"]
            src_dep = data["origin_dependency"] if task == "VACE" else data["dependency"]
            from utils.getDependencyUtils import getSubsetDep,combineDep,dict_to_pkg_ver_tuples
            if self.max_dependency_num:
                target_dep = getSubsetDep(target_dep,self.max_dependency_num)
                src_dep = getSubsetDep(src_dep,self.max_dependency_num)
            if self.append_srcDep:
                target_dep = combineDep(target_dep,src_dep)
            for pack, version in dict_to_pkg_ver_tuples(target_dep):
                if version is None:
                    continue
                
                docs_for_pack = raw_docs_map.get(pack, [])
                if not docs_for_pack:
                    self.logger.info(f"No raw documents loaded for {pack} {version}, skipping.")
                    continue

                # 尝试加载预编码
                cached_data = self.embeddings_manager.load_embeddings(pack, version)
                pack_embeddings = None
                
                if cached_data is not None and 'embeddings' in cached_data and len(cached_data['embeddings']) == len(docs_for_pack):
                    self.logger.info(f"Using cached embeddings for {pack} {version} for collection add...")
                    pack_embeddings = cached_data['embeddings']
                else:
                    self.logger.info(f"Computing embeddings for {pack} {version} for collection add...")
                    pack_embeddings = self.embeddings_manager.precompute_embeddings(
                        pack, version, docs_for_pack,
                        embedding_function_instance=self.embedding_function_instance
                    )

                if pack_embeddings is not None:
                    all_documents.extend(docs_for_pack)
                    all_embeddings.extend(pack_embeddings)
                else:
                    self.logger.warning(f"Failed to get embeddings for {pack} {version}, skipping for collection add.")

            if not all_documents:
                self.logger.warning("No valid documents with embeddings found for VersiBCB, cannot populate collection.")
                return [], []
                
            documents_to_add = all_documents
            embeddings_to_add = all_embeddings
        
        return documents_to_add, embeddings_to_add

    def _populate_collection(self, collection, collection_name, documents_to_add, embeddings_to_add):
        """批量添加文档到collection"""
        if not documents_to_add:
            self.logger.warning(f"No documents to add to collection {collection_name}.")
            return
        
        BATCH_SIZE = 250
        total_docs = len(documents_to_add)
        total_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE
        
        self.logger.info(f"Adding {total_docs} documents to collection {collection_name} in {total_batches} batches.")
        
        # 使用tqdm显示进度
        for batch_idx in tqdm(range(total_batches), desc=f"Adding documents to {collection_name}", unit="batch"):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, total_docs)
            
            self._process_batch(collection, collection_name, documents_to_add, embeddings_to_add, start_idx, end_idx, batch_idx)

    def _process_batch(self, collection, collection_name, documents_to_add, embeddings_to_add, start_idx, end_idx, batch_idx):
        """处理单个批次的文档"""
        batch_documents = documents_to_add[start_idx:end_idx]
        batch_ids = [f"{collection_name}_{idx}" for idx in range(start_idx, end_idx)]
        
        # 处理空文档
        processed_batch_documents = [(doc if doc and doc.strip() else " ") for doc in batch_documents]

        try:
            if embeddings_to_add:
                batch_embeddings = self._prepare_embeddings(embeddings_to_add, start_idx, end_idx)
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
            print(error_msg)

    def _prepare_embeddings(self, embeddings_to_add, start_idx, end_idx):
        """准备embedding数据，确保格式正确"""
        batch_embeddings = embeddings_to_add[start_idx:end_idx]
        if isinstance(batch_embeddings, np.ndarray):
            return batch_embeddings.tolist()
        elif isinstance(batch_embeddings, list) and batch_embeddings and isinstance(batch_embeddings[0], np.ndarray):
            return [emb.tolist() for emb in batch_embeddings]
        return batch_embeddings

    def _query_collection(self, collection, query, n_results=None):
        """查询collection"""
        if n_results is None:
            n_results = self.rag_document_num
            
        collection_name = collection.name
        # Create a hashable representation of embed_func_args for the cache key
        # Sort items to ensure consistent key regardless of dict order
        embed_model_config_str = "_".join([f"{k}_{v}" for k, v in sorted(self.embed_func_args.items()) if isinstance(v, (str, int, float, bool))])
        # Create a cache key string that can be used as a filename
        cache_key = hashlib.md5(f"{query}_{embed_model_config_str}_{collection_name}_{n_results}".encode()).hexdigest()
        cache_file = os.path.join(self.query_cache_dir, f"{cache_key}.json")
        
        # Check if cache file exists
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_results = json.load(f)
                self.logger.info(f"Cache hit for query in collection {collection_name}")
                return cached_results
            except Exception as e:
                self.logger.warning(f"Error loading cache file: {e}")
                # Continue with query if cache loading fails
        
        self.logger.info(f"Cache miss for query in collection {collection_name}. Performing query.")
        try:
            query_start = time.time()
            # Fix the "size" column issue by only including documents and distances
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "distances"]  # Only include what we need, which should avoid the "size" column issue
            )
            
            # Ensure results is JSON serializable
            serializable_results = self._make_serializable(results)
            
            # Store in JSON cache file
            try:
                with open(cache_file, 'w') as f:
                    json.dump(serializable_results, f, indent=2)
            except Exception as e:
                self.logger.warning(f"Error saving to cache file: {e}")
            
            query_end = time.time()
            
            # 记录查询详情
            result_count = len(results.get('documents', [[]])[0]) if results and 'documents' in results else 0
            self.logger.info(f"查询执行耗时: {query_end - query_start:.4f}秒, 返回结果数: {result_count}")
            
            return serializable_results
        except Exception as e:
            self.logger.error(f"Error querying collection {collection_name}: {e}")
            return None
    
    def _make_serializable(self, obj):
        """将对象转换为可JSON序列化的格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        else:
            return obj

    def _query_collection_with_api_matching(self, collection, query_dict, n_results=None):
        """
        基于(api_name_path,description)这一dict通过字符串匹配查询collection
        只接受包含"path"字段的dict类型query
        
        Args:
            collection: ChromaDB collection
            query_dict: dict，必须包含"path"字段
            n_results: 返回结果数量
            
        Returns:
            dict: 按编辑距离排序的查询结果
        """
        if n_results is None:
            n_results = self.rag_document_num
            
        # 验证query格式
        if not isinstance(query_dict, dict):
            raise ValueError("API name string matching requires query to be a dict type")
        
        if 'path' not in query_dict:
            raise ValueError("API name string matching requires query dict to contain 'path' field")
        
        query_path = query_dict['path']
        if not isinstance(query_path, str) or not query_path.strip():
            raise ValueError("Query 'path' field must be a non-empty string")
        
        self.logger.info(f"Using API name string matching for query path: {query_path}")
        
        # 获取collection中的所有文档
        collection_name = collection.name
        try:
            # 查询所有文档
            all_results = collection.get(include=["documents"])
            
            if not all_results or not all_results.get('documents'):
                self.logger.warning(f"No documents found in collection {collection_name}")
                return {"documents": [[]], "distances": [[]]}
            
            documents = all_results['documents']
            exact_matches = []
            similarity_scores = []
            
            # 为每个文档计算匹配分数
            for doc in documents:
                try:
                    # 尝试解析文档为JSON
                    doc_data = json.loads(doc)
                    is_exact, score = get_document_match_score(doc_data, query_dict)
                    
                    if is_exact:
                        # 精确匹配优先
                        exact_matches.append((doc, 1.0))
                    else:
                        # 相似度匹配
                        similarity_scores.append((doc, score))
                        
                except json.JSONDecodeError:
                    # 如果不是JSON格式，跳过
                    continue
                except Exception as e:
                    self.logger.warning(f"Error processing document for API matching: {e}")
                    continue
            
            # 优先返回精确匹配，然后是高相似度匹配
            all_matches = exact_matches.copy()
            
            if len(all_matches) < n_results:
                # 按相似度排序，取补充结果
                similarity_scores.sort(key=lambda x: x[1], reverse=True)
                needed = n_results - len(all_matches)
                all_matches.extend(similarity_scores[:needed])
            
            if not all_matches:
                self.logger.warning(f"No valid API matches found for path: {query_path}")
                return {"documents": [[]], "distances": [[]]}
            
            # 取前n_results个结果
            selected_documents = all_matches[:n_results]
            
            # 构造返回结果，保持与向量搜索一致的格式
            result_documents = [doc for doc, _ in selected_documents]
            result_scores = [score for _, score in selected_documents]
            
            # 转换为距离（1 - score，使得分数越高距离越小）
            result_distances = [1.0 - score for score in result_scores]
            
            exact_count = len(exact_matches)
            similarity_count = len([s for s in similarity_scores if s[1] > 0])
            
            self.logger.info(f"API string matching found {len(result_documents)} results: "
                           f"{exact_count} exact matches, {similarity_count} similarity matches")
            if result_scores:
                self.logger.info(f"Best match score: {result_scores[0]:.3f}")
            
            return {
                "documents": [result_documents],
                "distances": [result_distances]
            }
            
        except Exception as e:
            self.logger.error(f"Error in API name string matching for collection {collection_name}: {e}")
            return {"documents": [[]], "distances": [[]]}

    def _process_query_results(self, results, max_token_length, target_dependencies=None):
        """处理查询结果并返回格式化上下文"""
        if not results or not results.get('documents') or not results['documents'][0]:
            self.logger.warning("查询结果为空，返回空上下文")
            return ""
        
        process_start = time.time()
        
        # 提取文档列表，过滤None值
        documents = [doc for doc in results['documents'][0] if doc is not None]
        
        # 应用dependency过滤
        if target_dependencies:
            documents = self._filter_documents_by_dependency(documents, target_dependencies)
        
        # 记录初始文档数量和总长度
        initial_doc_count = len(documents)
        initial_context_length = sum(len(doc) for doc in documents)
        self.logger.info(f"初始查询结果: {initial_doc_count}个文档, 总长度: {initial_context_length}字符")
        
        # 合并文档
        context_build_start = time.time()
        retrieved_context = "\n".join(documents)
        context_build_end = time.time()
        self.logger.info(f"- 上下文构建耗时: {context_build_end - context_build_start:.4f}秒")
        
        # 使用原始的truncate_context函数进行截断
        trunc_start = time.time()
        from utils.RAGutils.document_utils import truncate_context
        self.logger.info(f"应用truncate_context进行截断，最大token长度: {max_token_length}")
        retrieved_context = truncate_context(retrieved_context, max_token_length)
        trunc_end = time.time()
        self.logger.info(f"- 上下文截断耗时: {trunc_end - trunc_start:.4f}秒")
        
        process_end = time.time()
        # final_token_length = len(self.tokenizer.encode(retrieved_context, add_special_tokens=True))
        # 记录处理结果
        self.logger.info(f"上下文处理总耗时: {process_end - process_start:.4f}秒")
        self.logger.info(f"处理后长度  character长度: {len(retrieved_context)}")
        
        return retrieved_context

    def _load_generated_queries(self):
        """加载生成的查询"""
        try:
            import json
            with open(self.generated_queries_file, 'r', encoding='utf-8') as f:
                self.generated_queries = json.load(f)
            self.logger.info(f"Loaded {len(self.generated_queries)} generated queries from {self.generated_queries_file}")
        except FileNotFoundError:
            self.logger.error(f"Generated queries file not found: {self.generated_queries_file}")
            self.generated_queries = {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing generated queries JSON file: {e}")
            self.generated_queries = {}
        except Exception as e:
            self.logger.error(f"Error loading generated queries: {e}")
            self.generated_queries = {}

    def _query_collection_progressive(self, collection, queries, max_token_length, target_dependencies=None):
        """
        对每个query分别检索，逐步增加检索数量直到达到token限制
        优化版本：第1轮使用精确匹配+少量embedding，第2轮开始直接检索rag_document个文档逐个叠加
        
        Args:
            collection: ChromaDB collection
            queries: 查询列表
            max_token_length: 最大token长度
            
        Returns:
            retrieved_context: str,检索到的上下文
        """
        if not queries:
            self.logger.warning("No queries provided for progressive retrieval")
            return ""
        
        self.logger.info(f"Starting optimized progressive retrieval with {len(queries)} queries, max_token_length: {max_token_length}")
        
        # 初始化tokenizer用于准确的token计算
        tokenizer = None
        try:
            from transformers import AutoTokenizer
            # 使用default tokenizer路径（从document_utils.py中看到的）
            tokenizer = AutoTokenizer.from_pretrained("/datanfs2/chenrongyi/models/Llama-3.1-8B", trust_remote_code=True)
            self.logger.info("Loaded tokenizer for accurate token counting")
        except Exception as e:
            self.logger.warning(f"Failed to load tokenizer: {e}, falling back to character-based estimation")
            tokenizer = None
        
        def count_tokens(text):
            """计算文本的token数量"""
            if tokenizer is not None:
                try:
                    return len(tokenizer.encode(text, add_special_tokens=False))
                except:
                    pass
            # 回退到字符估算
            return len(text) // 4
        
        all_documents = []
        current_token_count = 0
        
        # 第一轮：对每个query进行精确匹配 + 少量embedding匹配
        self.logger.info("Round 1: exact matching + minimal embedding search")
        for query_idx, query in enumerate(queries):
            if current_token_count >= max_token_length:
                break
            
            self.logger.info(f"  Query {query_idx + 1}/{len(queries)}: '{query[:50]}...'")
            
            # 精确匹配
            # 对于dict类型的query，提取path用于精确匹配
            exact_match_query = query
            if isinstance(query, dict) and 'path' in query:
                exact_match_query = query['path']
            elif isinstance(query, dict):
                exact_match_query = query.get('description', str(query))
            
            exact_matches = self._exact_match_documents(collection, exact_match_query)
            query_documents = exact_matches
            self.logger.info(f"    Found {len(exact_matches)} exact matches")
            
            # 如果精确匹配不足1个，补充1个搜索匹配
            if len(exact_matches) < 1:
                if self.api_name_str_match and isinstance(query, dict):
                    # 使用API名称字符串匹配
                    self.logger.info(f"    Need 1 more document, using API name string matching")
                    api_results = self._query_collection_with_api_matching(collection, query, n_results=3)
                    
                    if api_results and api_results.get('documents') and api_results['documents'][0]:
                        api_documents = [doc for doc in api_results['documents'][0] if doc is not None]
                        # 过滤掉已经在精确匹配中的文档
                        api_documents = [doc for doc in api_documents if doc not in exact_matches]
                        query_documents.extend(api_documents[:1])
                else:
                    # 使用embedding搜索
                    self.logger.info(f"    Need 1 more document, using embedding search")
                    # 对于API匹配模式，query可能是dict，需要转换为字符串
                    query_text = query if isinstance(query, str) else (query.get('description', '') if isinstance(query, dict) else str(query))
                    embedding_results = self._query_collection(collection, query_text, n_results=3)  # 多取一些以防重复
                    
                    if embedding_results and embedding_results.get('documents') and embedding_results['documents'][0]:
                        embedding_documents = [doc for doc in embedding_results['documents'][0] if doc is not None]
                        # 过滤掉已经在精确匹配中的文档
                        embedding_documents = [doc for doc in embedding_documents if doc not in exact_matches]
                        query_documents.extend(embedding_documents[:1])
            
            # 只添加新文档（避免重复）
            new_documents = []
            for doc in query_documents:
                if doc not in all_documents:
                    new_documents.append(doc)
            
            # 逐个添加文档，直到达到token限制
            for doc in new_documents:
                doc_tokens = count_tokens(doc)
                if current_token_count + doc_tokens <= max_token_length:
                    all_documents.append(doc)
                    current_token_count += doc_tokens
                    self.logger.info(f"    Added document with {doc_tokens} tokens")
                else:
                    self.logger.info(f"    Skipping document ({doc_tokens} tokens) due to token limit")
                    break
        
        self.logger.info(f"Round 1 completed: {len(all_documents)} documents, {current_token_count} tokens")
        
        # 第二轮开始：为每个query一次性检索rag_document个文档，然后逐个叠加
        if current_token_count < max_token_length:
            self.logger.info(f"Round 2+: retrieving {self.rag_document_num} documents per query and adding progressively")
            
            # 为每个query一次性检索rag_document个文档
            query_documents_map = {}
            for query_idx, query in enumerate(queries):
                self.logger.info(f"  Pre-retrieving {self.rag_document_num} documents for query {query_idx + 1}")
                
                if self.api_name_str_match and isinstance(query, dict):
                    # 使用API名称字符串匹配
                    results = self._query_collection_with_api_matching(collection, query, n_results=self.rag_document_num)
                else:
                    # 使用embedding搜索
                    query_text = query if isinstance(query, str) else (query.get('description', '') if isinstance(query, dict) else str(query))
                    results = self._query_collection(collection, query_text, n_results=self.rag_document_num)
                
                if results and results.get('documents') and results['documents'][0]:
                    query_documents = [doc for doc in results['documents'][0] if doc is not None]
                    # 过滤掉已经添加的文档
                    new_query_documents = [doc for doc in query_documents if doc not in all_documents]
                    query_documents_map[query_idx] = new_query_documents
                    self.logger.info(f"    Pre-retrieved {len(new_query_documents)} new documents for query {query_idx + 1}")
                else:
                    query_documents_map[query_idx] = []
            
            # 逐个叠加文档直到达到token限制
            doc_added = True
            round_num = 2
            while doc_added and current_token_count < max_token_length:
                doc_added = False
                self.logger.info(f"Round {round_num}: adding next document from each query")
                
                for query_idx, query in enumerate(queries):
                    if current_token_count >= max_token_length:
                        break
                    
                    available_docs = query_documents_map.get(query_idx, [])
                    if available_docs:
                        # 取第一个可用文档
                        doc = available_docs.pop(0)
                        doc_tokens = count_tokens(doc)
                        
                        if current_token_count + doc_tokens <= max_token_length:
                            all_documents.append(doc)
                            current_token_count += doc_tokens
                            doc_added = True
                            self.logger.info(f"    Added document from query {query_idx + 1} with {doc_tokens} tokens")
                        else:
                            self.logger.info(f"    Skipping document from query {query_idx + 1} ({doc_tokens} tokens) due to token limit")
                            # 将文档放回队列开头
                            available_docs.insert(0, doc)
                            break
                
                round_num += 1
                self.logger.info(f"Round {round_num - 1} completed: total {len(all_documents)} documents, {current_token_count} tokens")
                
                # 安全检查：避免无限循环
                if round_num > 100:
                    self.logger.warning("Reached maximum rounds limit, stopping progressive retrieval")
                    break
        
        # 构建最终上下文
        if all_documents:
            # 应用dependency过滤
            if target_dependencies:
                all_documents = self._filter_documents_by_dependency(all_documents, target_dependencies)
            
            # 应用文档处理（预截断和signature过滤）
            if self.max_doc_tokens:
                all_documents = self._process_individual_documents(all_documents)
                
            retrieved_context = "\n".join(all_documents)
            final_tokens = count_tokens(retrieved_context)
            self.logger.info(f"Progressive retrieval completed: {len(all_documents)} documents, {final_tokens} tokens, {len(retrieved_context)} characters")
            
            # 使用原始的truncate_context函数进行最终截断（以防万一）
            from utils.RAGutils.document_utils import truncate_context
            retrieved_context = truncate_context(retrieved_context, max_token_length)
            final_tokens_after_truncate = count_tokens(retrieved_context)
            self.logger.info(f"After final truncation: {final_tokens_after_truncate} tokens, {len(retrieved_context)} characters")
            
            return retrieved_context
        else:
            self.logger.warning("No documents retrieved from progressive retrieval")
            return ""

    def _exact_match_documents(self, collection, query):
        """
        在collection中进行精确的path/aliases字符串匹配
        
        Args:
            collection: ChromaDB collection
            query: 查询字符串
            
        Returns:
            exact_matches: 精确匹配的文档列表
        """
        exact_matches = []
        
        try:
            # 获取collection中的所有文档
            all_results = collection.get()
            
            if not all_results or not all_results.get('documents'):
                self.logger.warning("No documents found in collection for exact matching")
                return exact_matches
            
            all_documents = all_results['documents']
            
            # 遍历所有文档，查找精确匹配
            for doc in all_documents:
                if doc is None:
                    continue
                    
                try:
                    import json
                    # 尝试解析文档为JSON
                    doc_data = json.loads(doc)
                    
                    if isinstance(doc_data, dict):
                        # 检查path字段
                        doc_path = doc_data.get('path', '')
                        if doc_path and query in doc_path:
                            exact_matches.append(doc)
                            self.logger.debug(f"Exact match found in path: {doc_path}")
                            continue
                        
                        # 检查aliases字段
                        doc_aliases = doc_data.get('aliases', [])
                        if isinstance(doc_aliases, list):
                            for alias in doc_aliases:
                                if isinstance(alias, str) and query in alias:
                                    exact_matches.append(doc)
                                    self.logger.debug(f"Exact match found in alias: {alias}")
                                    break
                except (json.JSONDecodeError, AttributeError):
                    # 如果不是JSON格式，进行简单的字符串匹配
                    if query in doc:
                        exact_matches.append(doc)
                        self.logger.debug(f"Exact match found in document content")
                        
        except Exception as e:
            self.logger.error(f"Error during exact matching: {e}")
            
        self.logger.info(f"Found {len(exact_matches)} exact matches for query: '{query[:50]}...'")
        return exact_matches

    def _filter_documents_by_dependency(self, documents, target_dependencies):
        """
        过滤文档，只保留属于target_dependency的api
        
        Args:
            documents: 文档列表
            target_dependencies: 目标依赖字典 {pkg: version}
            
        Returns:
            filtered_documents: 过滤后的文档列表
        """
        if not self.enable_dependency_filtering or not target_dependencies:
            return documents
        
        filtered_documents = []
        target_packages = set(target_dependencies.keys())
        
        for doc in documents:
            if doc is None:
                continue
                
            try:
                import json
                # 尝试解析文档为JSON
                doc_data = json.loads(doc)
                
                if isinstance(doc_data, dict):
                    # 检查文档中的package信息
                    doc_package = None
                    
                    # 方法1: 检查path字段中的包名
                    doc_path = doc_data.get('path', '')
                    if doc_path:
                        # path通常格式为 "package.module.function"
                        path_parts = doc_path.split('.')
                        if path_parts:
                            # 取第一部分作为包名
                            potential_package = path_parts[0]
                            if potential_package in target_packages:
                                doc_package = potential_package
                    
                    # 方法2: 检查aliases字段中的包名
                    if doc_package is None:
                        doc_aliases = doc_data.get('aliases', [])
                        if isinstance(doc_aliases, list):
                            for alias in doc_aliases:
                                if isinstance(alias, str):
                                    alias_parts = alias.split('.')
                                    if alias_parts:
                                        potential_package = alias_parts[0]
                                        if potential_package in target_packages:
                                            doc_package = potential_package
                                            break
                    
                    # 方法3: 检查文档内容中是否包含包名
                    if doc_package is None:
                        doc_content = doc_data.get('docstring', '') or doc_data.get('content', '') or str(doc_data)
                        for package in target_packages:
                            if package in doc_content:
                                doc_package = package
                                break
                    
                    # 如果找到匹配的包名，保留该文档
                    if doc_package is not None:
                        filtered_documents.append(doc)
                        self.logger.debug(f"Kept document from package: {doc_package}")
                    else:
                        self.logger.debug(f"Filtered out document with path: {doc_path}")
                        
                else:
                    # 如果不是JSON格式，进行简单的字符串匹配
                    for package in target_packages:
                        if package in doc:
                            filtered_documents.append(doc)
                            self.logger.debug(f"Kept document containing package: {package}")
                            break
                            
            except (json.JSONDecodeError, AttributeError):
                # 如果解析失败，进行简单的字符串匹配
                for package in target_packages:
                    if package in doc:
                        filtered_documents.append(doc)
                        self.logger.debug(f"Kept document containing package: {package}")
                        break
        
        original_count = len(documents)
        filtered_count = len(filtered_documents)
        if original_count > 0:
            filter_ratio = filtered_count / original_count
            self.logger.info(f"Dependency filtering: kept {filtered_count}/{original_count} documents ({filter_ratio:.2%})")
        
        return filtered_documents

    def _process_individual_documents(self, documents):
        """
        处理单个文档的token截断
        
        Args:
            documents: 文档列表
            
        Returns:
            processed_documents: 处理后的文档列表
        """
        if not documents or not self.max_doc_tokens:
            return documents
        
        processed_documents = []
        truncated_docs_count = 0
        error_docs_count = 0
        
        for doc in documents:
            if doc is None:
                continue
            
            # 对文档的doc字段进行token截断
            try:
                processed_doc = self._truncate_document_safely(doc, self.max_doc_tokens)
                processed_documents.append(processed_doc)
                truncated_docs_count += 1
                self.logger.debug(f"Processed document for token truncation: {len(doc)} -> {len(processed_doc)} chars")
            except ValueError as e:
                self.logger.error(f"Failed to process document for truncation: {e}")
                # 对于不能解析的文档，跳过处理
                error_docs_count += 1
                continue
        
        original_count = len(documents)
        processed_count = len(processed_documents)
        
        self.logger.info(f"Document processing: {original_count} -> {processed_count} documents")
        self.logger.info(f"  Processed {truncated_docs_count} documents for token truncation (max {self.max_doc_tokens} tokens)")
        if error_docs_count > 0:
            self.logger.warning(f"  Skipped {error_docs_count} documents due to parsing errors")
        
        return processed_documents

    def _truncate_document_safely(self, document, max_tokens):
        """
        截断文档的doc字段到指定的token数
        
        Args:
            document: 要截断的文档字符串
            max_tokens: doc字段的最大token数
            
        Returns:
            truncated_document: 截断后的文档字符串
            
        Raises:
            ValueError: 如果文档不能解析为dict
        """
        if not self.doc_tokenizer:
            # 如果没有tokenizer，返回原文档
            return document
        
        try:
            import json
            # 必须能够解析为JSON dict，否则抛出错误
            doc_data = json.loads(document)
            
            if not isinstance(doc_data, dict):
                raise ValueError(f"Document is not a dict: {type(doc_data)}")
            
            # 检查是否有doc字段需要截断
            if 'doc' not in doc_data or not isinstance(doc_data['doc'], str):
                # 没有doc字段或doc字段不是字符串，直接返回
                return document
            
            doc_content = doc_data['doc']
            
            # 计算当前doc字段的token数
            doc_tokens = self.doc_tokenizer.encode(doc_content, add_special_tokens=False)
            current_token_count = len(doc_tokens)
            
            # 删除members字段（减少内容，不管是否需要截断都删除）
            truncated_data = doc_data.copy()
            if 'members' in truncated_data:
                del truncated_data['members']
            
            if current_token_count <= max_tokens:
                # 不需要截断doc字段，但已经删除了members
                if 'members' in doc_data:  # 只有原来有members才需要重新序列化
                    truncated_json = json.dumps(truncated_data, ensure_ascii=False)
                    return truncated_json
                else:
                    return document
            
            # 需要截断doc字段
            truncated_tokens = doc_tokens[:max_tokens]
            truncated_doc_content = self.doc_tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            
            # 更新doc字段
            truncated_data['doc'] = truncated_doc_content
            # 重新序列化
            truncated_json = json.dumps(truncated_data, ensure_ascii=False)
            
            return truncated_json
                
        except (json.JSONDecodeError, TypeError) as e:
            # 不能解析为JSON，抛出错误
            raise ValueError(f"Document cannot be parsed as JSON: {e}")
        except Exception as e:
            # 其他错误也抛出
            raise ValueError(f"Error processing document: {e}")

    def _query_collection_fixed_docs(self, collection, queries, max_token_length, target_dependencies=None):
        """
        固定文档数量模式下的检索
        
        Args:
            collection: ChromaDB collection
            queries: 查询列表
            max_token_length: 最大token长度
            target_dependencies: 目标依赖字典 {pkg: version}
            
        Returns:
            retrieved_context: str,检索到的上下文
        """
        if not queries:
            self.logger.warning("No queries provided for fixed docs retrieval")
            return ""
        
        self.logger.info(f"Starting fixed docs retrieval with {len(queries)} queries, max_token_length: {max_token_length}")
        
        all_documents = []
        current_token_count = 0
        
        for query_idx, query in enumerate(queries):
            self.logger.info(f"  Retrieving {self.fixed_docs_per_query} documents for query {query_idx + 1}")
            
            if self.api_name_str_match and isinstance(query, dict):
                # 使用API名称字符串匹配
                results = self._query_collection_with_api_matching(collection, query, n_results=self.fixed_docs_per_query)
            else:
                # 使用embedding搜索
                query_text = query if isinstance(query, str) else (query.get('description', '') if isinstance(query, dict) else str(query))
                results = self._query_collection(collection, query_text, n_results=self.fixed_docs_per_query)
            
            # 初始化query_documents为空列表，确保变量总是定义
            query_documents = []
            
            if results and results.get('documents') and results['documents'][0]:
                query_documents = [doc for doc in results['documents'][0] if doc is not None]
                # 过滤掉已经在检索中的文档
                new_query_documents = [doc for doc in query_documents if doc not in all_documents]
                all_documents.extend(new_query_documents)
                self.logger.info(f"    Retrieved {len(new_query_documents)} new documents for query {query_idx + 1}")
            else:
                self.logger.warning(f"No valid documents retrieved for query {query_idx + 1}")
            
            # 计算当前查询的token数（即使没有文档也不会出错）
            current_token_count += sum(len(doc) for doc in query_documents)
        
        self.logger.info(f"Retrieved {len(all_documents)} documents, {current_token_count} tokens")
        
        # 应用dependency过滤
        if target_dependencies:
            all_documents = self._filter_documents_by_dependency(all_documents, target_dependencies)
        
        # 应用文档处理（token截断）
        if self.max_doc_tokens:
            all_documents = self._process_individual_documents(all_documents)
        
        # 合并文档
        context_build_start = time.time()
        retrieved_context = "\n".join(all_documents)
        context_build_end = time.time()
        self.logger.info(f"- 上下文构建耗时: {context_build_end - context_build_start:.4f}秒")
        
        # 使用原始的truncate_context函数进行截断
        trunc_start = time.time()
        from utils.RAGutils.document_utils import truncate_context
        self.logger.info(f"应用truncate_context进行截断，最大token长度: {max_token_length}")
        retrieved_context = truncate_context(retrieved_context, max_token_length)
        trunc_end = time.time()
        self.logger.info(f"- 上下文截断耗时: {trunc_end - trunc_start:.4f}秒")
        
        return retrieved_context

        