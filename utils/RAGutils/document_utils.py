import os
import re
from transformers import AutoTokenizer,GPT2Tokenizer
from utils.RAGutils.config.default_config import RAG_STRING_TRUNCATE_TOKENIZER
# from benchmark.config.code.config import CORPUS_PATH

def get_version(text):
    text = re.sub(r"^>=", "", text)
    text = re.sub(r"^<=", "", text)
    text = re.sub(r"^==", "", text)
    return text

def truncate_context(context, max_token_length=31000, model_name="gpt2", tokenizer=None):
    """
    使用 HuggingFace tokenizer 截断文本到指定 token 长度
    Args:
        context: 需要截断的文本
        max_token_length: 最大token长度
        model_name: 模型名称，用于加载对应的 tokenizer
        tokenizer: 可选，直接传入的 tokenizer 实例
    Returns:
        str: 截断后的文本
    """
    try:
        # 如果没有传入 tokenizer，则根据 model_name 加载
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(RAG_STRING_TRUNCATE_TOKENIZER, trust_remote_code=True)
        
        # 使用传入的或新加载的 tokenizer 进行编码
        tokens = tokenizer.encode(context, add_special_tokens=False)
        
        # 截断到指定长度
        if len(tokens) > max_token_length:
            tokens = tokens[:max_token_length]
        
        # 解码回文本
        return tokenizer.decode(tokens, skip_special_tokens=True)
    
    except Exception as e:
        # 如果出现任何错误（如模型加载失败），使用简单的空白字符分割作为后备方案
        print(f"Warning: Tokenizer failed ({str(e)}), falling back to basic splitting")
        tokens = context.split()
        if len(tokens) > max_token_length:
            tokens = tokens[:max_token_length]
        return " ".join(tokens)

def truncate_BCB_context(context, max_token_length=31000, model_name="gpt2", tokenizer=None):
    '''
    Description:
        截断BCB的context,每个package分配max_token_length/len(context)的token
    Args:
        context: dict[pkg,list[str]],知识文档
        max_token_length: int,最大token长度
        model_name: str,模型名称，用于加载对应的tokenizer
        tokenizer: 可选，直接传入的tokenizer实例
    Returns:
        context: str,截断后的context
    '''
    try:
        # 如果没有传入 tokenizer，则根据 model_name 加载
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 计算每个pkg的token数量，依据行数分配token量
        pkg_tokens_count = {}
        pkg_tokens = {}
        pkg_distribute_tokens = {}
        
        # 对每个包的文档进行编码
        for pkg, docs in context.items():
            pkg_tokens[pkg] = [tokenizer.encode(doc, add_special_tokens=False) for doc in docs]
            pkg_tokens_count[pkg] = len(pkg_tokens[pkg])
        
        all_tokens_count = sum(pkg_tokens_count.values())
        
        # 计算每个包应分配的token数量
        for pkg in pkg_tokens_count:
            pkg_distribute_tokens[pkg] = int(pkg_tokens_count[pkg]/all_tokens_count*max_token_length)
        
        # 按比例截断每个包的tokens
        for pkg in pkg_tokens_count:
            pkg_tokens[pkg] = pkg_tokens[pkg][:pkg_distribute_tokens[pkg]]
        
        # 将token转换回文本并组合
        context_str = ""
        for pkg in pkg_tokens:
            decoded_docs = [tokenizer.decode(token, skip_special_tokens=True) for token in pkg_tokens[pkg]]
            context_str += f"{pkg}\n{'\n'.join(decoded_docs)}\n"
        
        return context_str
    
    except Exception as e:
        print(f"Warning: Tokenizer operations failed ({str(e)}), returning original context")
        return str(context)  # 返回原始文本作为后备方案

def get_datacollectionLocation(data):
    '''
    Description:
        Deprecated!!!!!获取datacollectionLocation
    Args:
        data: dict,数据
    Returns:
        file_path: str,文件路径
    '''
    base_dir = "chroma_data/library"
    library_name = data["dependency"]
    version = get_version(data["version"])
    file_path = os.path.join(base_dir, library_name, version + '.jsonl')
    return file_path