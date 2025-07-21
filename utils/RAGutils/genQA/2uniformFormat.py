import json
with open("data/temp/query_results_concurrent.jsonl","r") as f:
    query_results = [json.loads(line) for line in f]
# 格式为dict,key包括id,queries,其中queries为list,每个元素为dict,key包括query_content,target_api

# 需要转换为dict,key为id

with open("data/temp/query_results_concurrent_uniform.jsonl","w") as f:
    for query_result in query_results:
        f.write(json.dumps(query_result) + "\n")