from dataclasses import dataclass
from typing import Optional
from config.paths import QUERY_CACHE_DIR, API_KEY_PATH

@dataclass
class RAGConfig:
    """RAG配置参数的数据类，用于管理所有RAG相关的配置"""
    
    # 必需参数（没有默认值的参数）
    local_embedding_model: str
    togetherai_embedding_model: str
    rag_collection_base: str
    docstring_embedding_base_path: str
    srccode_embedding_base_path: str
    docstring_corpus_path: str
    srccode_corpus_path: str
    
    # 可选参数（有默认值的参数）
    embedding_source: str = "local"  # "local" or "togetherai"
    rag_document_num: int = 10
    max_dependency_num: Optional[int] = None
    append_srcDep: bool = False
    query_cache_dir: str = QUERY_CACHE_DIR
    max_token_length: int = 2000
    together_api_key_path: str = f"{API_KEY_PATH}/togetherai.txt"
    useQAContext: bool = False
    QAContext_path: str = None
    @classmethod
    def from_args(cls, args):
        """从命令行参数创建RAGConfig实例"""
        return cls(
            local_embedding_model=args.local_embedding_model,
            togetherai_embedding_model=args.togetherai_embedding_model,
            rag_collection_base=args.rag_collection_base,
            docstring_embedding_base_path=args.docstring_embedding_base_path,
            srccode_embedding_base_path=args.srccode_embedding_base_path,
            docstring_corpus_path=args.docstring_corpus_path,
            srccode_corpus_path=args.srccode_corpus_path,
            embedding_source=args.embedding_source,
            rag_document_num=args.rag_document_num,
            max_dependency_num=args.max_dependency_num,
            append_srcDep=args.append_srcDep,
            query_cache_dir=args.query_cache_dir,
            max_token_length=args.max_token_length,
            together_api_key_path=args.together_api_key_path,
            useQAContext=args.useQAContext,
            QAContext_path=args.QAContext_path
        )
    
    def get_embedding_model(self) -> str:
        """根据embedding_source返回对应的模型名称"""
        if self.embedding_source == "local":
            return self.local_embedding_model
        elif self.embedding_source == "togetherai":
            return self.togetherai_embedding_model
        else:
            raise ValueError(f"Unknown embedding_source: {self.embedding_source}")
    
    def get_corpus_path(self, corpus_type: str) -> str:
        """根据corpus_type返回对应的语料库路径"""
        if corpus_type == "docstring":
            return self.docstring_corpus_path
        elif corpus_type == "srccodes":
            return self.srccode_corpus_path
        else:
            raise ValueError(f"Unknown corpus_type: {corpus_type}")
    
    def get_embedding_base_path(self, corpus_type: str) -> str:
        """根据corpus_type返回对应的嵌入基础路径"""
        if corpus_type == "docstring":
            return self.docstring_embedding_base_path
        elif corpus_type == "srccodes":
            return self.srccode_embedding_base_path
        else:
            raise ValueError(f"Unknown corpus_type: {corpus_type}") 