def loadQAContext(data,max_tokens,QA_cache):
    '''
    description:
        根据当前data项加载对应的QA_cache_context
    params:
        data: 当前data项（包括id字段）
        max_tokens: 最大tokens数 （这里弄成tokens，不太对）
        QA_cache: QA缓存上下文,list[dict]
    return:
        context
    '''
    #TODO:修正max_tokens
    data_id = data.get('id', None)
    qa_pairs = []
    if data_id is None:
        raise ValueError("data项中没有id字段")
    for item in QA_cache:
        if item.get('original_item_id', None) is not None:
            if str(item.get('original_item_id', None)) == str(data_id):
                qa_pairs.append((item["query"],item["answer"]))
    context = ""
    for query,answer in qa_pairs:
        context += f"query: {query}\nanswer: {answer}\n"
    context = context[:max_tokens]
    return context