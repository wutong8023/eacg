def distribute_doc_tokens(meta_docs, max_token_length,tokenizer):
    '''
    Description:
        根据max_token_length的限额，按照一定标准，对于每个doc进行token分配
    Args:
        docs: dict[str,str] 每个元素代表一个package
        max_token_length: int,max_token_length
    Returns:
        doc_tokens: list[str] 
    '''
    if isinstance(meta_docs,dict):
        docs = list(meta_docs.values())
        doc_names = list(meta_docs.keys())
    else:
        raise "Wrong input"
    # 计算每个doc的行数，根据行数分配max_token_leng
    encoded_docs = [tokenizer.encode(doc) for doc in docs]
    doc_token_length = [len(encoded_docs[i]) for i in range(len(encoded_docs))]
    # 计算每个doc的token数，根据token数分配max_token_length
    length_portion = [doc_token_length[i] / sum(doc_token_length) for i in range(len(doc_token_length))]
    # 根据行数分配max_token_length
    doc_tokens = [int(length_portion[i] * max_token_length) for i in range(len(length_portion))]
    # 根据doc_tokens进行截断
    truncated_docs = [tokenizer.decode(encoded_docs[i][:doc_tokens[i]]) for i in range(len(encoded_docs))]


    return dict(zip(doc_names,truncated_docs))
def truncate_doc(doc,max_token_length,tokenizer):
    '''
    Description:
        根据max_token_length的限额，对于每个doc进行截断
    Args:
        doc: str,doc
        max_token_length: int,max_token_length
    Returns:
        truncated_doc: str,截断后的doc
    '''
    pos = find_truncation_position(doc, max_token_length, tokenizer)
    doc = doc[:pos*2]
    encoded_doc = tokenizer.encode(doc)
    return tokenizer.decode(encoded_doc[:max_token_length])
def find_truncation_position(doc, max_token_length, tokenizer, chunk_size=10000):
    '''
    Description:
        通过流式编码，快速定位 max_token_length 对应的字符位置。
    Args:
        doc: str, 输入文档
        max_token_length: int, 最大 token 长度
        tokenizer: 使用的 tokenizer
        chunk_size: int, 每次处理的字符块大小（默认 1000）
    Returns:
        position: int, 截断位置的字符索引
    '''
    current_length = 0
    position = 0

    # 逐步处理文档
    while position < len(doc) and current_length < max_token_length:
        # 获取当前字符块
        chunk = doc[position:position + chunk_size]

        # 将字符块分割为 token
        tokens = tokenizer.tokenize(chunk)

        # 更新当前 token 长度
        current_length += len(tokens)

        # 更新字符位置
        position += chunk_size

    # 返回截断位置
    return position