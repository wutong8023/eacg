# 对于重复id的结果清除，仅保留一个
import json
exist_id = set()
result_path = 'data/temp/inference/versibcb_vscc_results.jsonl'
output_path = 'data/temp/inference/versibcb_vscc_results_filtered.jsonl'
filtered_results = []
with open(result_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        id = data['id']
        if id in exist_id:
            continue
        exist_id.add(id)
        filtered_results.append(data)

with open(output_path, 'w') as f:
    for data in filtered_results:
        f.write(json.dumps(data) + '\n')