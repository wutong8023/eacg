import json

def removeDuplicateQuery(queries):
    '''
    params:
        queries: list[dict] dict key:(query,target_api)
    return:
        list[dict] dict key:(query,target_api)
    '''
    # 使用集合来存储唯一的查询
    seen_queries = set()
    unique_queries = []
    
    for i, query in enumerate(queries):
        # 保留原有的过滤条件：索引>=3，是字典类型，且不包含'python'
        if (i >= 3 and 
            isinstance(query, dict) and 
            'python' not in query.get('query', '')):
            
            # 创建查询的唯一标识
            query_key = (query.get('query', ''), query.get('target_api', ''))
            
            # 如果这个查询还没有出现过，就添加到结果中
            if query_key not in seen_queries:
                seen_queries.add(query_key)
                unique_queries.append(query)
    
    return unique_queries

def tuple2dict(tuple_items):
    return [dict(item) for item in tuple_items]

with open("data/generated_queries/versibcb_vace_queries.json", "r") as f:
    data = json.load(f)
query_count = {}

for k,v in data.items():
    queries = v["queries"]
    queries_deduplicated = removeDuplicateQuery(queries)
    query_count[k] = len(queries_deduplicated)
    data[k]["queries"] = queries_deduplicated

print(query_count)
# 获取最大的v以及对应的k
query_count_sorted = sorted(query_count.items(), key=lambda x: x[1], reverse=True)
print(query_count_sorted[0])
print(data[query_count_sorted[0][0]]["queries"])
with open("data/generated_queries/versibcb_vace_queries_deduplicated.json", "w") as f:
    json.dump(data, f, indent=4)



