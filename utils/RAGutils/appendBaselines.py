import json
def getAllids(jsonl_file_path):
    ids = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            ids.append(data["id"])
    return ids
def getJsonlDatas(jsonl_file_path):
    datas = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            datas.append(data)
    return datas
def appendBaselines(jsonl_file_path, baseline_jsonl_file_path,target_file_path):
    exist_ids = getAllids(jsonl_file_path)
    datas = getJsonlDatas(jsonl_file_path)
    baseline_datas = getJsonlDatas(baseline_jsonl_file_path)
    for baselinedata in baseline_datas:
        if baselinedata["id"] not in exist_ids:
            datas.append(baselinedata)
    with open(target_file_path, 'w', encoding='utf-8') as f:
        for data in datas:
            f.write(json.dumps(data) + "\n")

if __name__ == "__main__":
    jsonl_file_path = 'output/approach_eval/RAG/Llama-3.1-8B/versibcb_vace_review_docstring_emb_local_4000_Llama-3.1-8B_maxdep10.jsonl'
    baseline_jsonl_file_path = 'output/approach_eval/BASELINE/Llama-3.1-8B/versibcb_vace_Llama-3.1-8B_maxdep10.jsonl'
    target_file_path = 'output/approach_eval/RAG/Llama-3.1-8B/versibcb_vace_review_docstring_emb_local_4000_Llama-3.1-8B_maxdep10_with_baselines.jsonl'
    appendBaselines(jsonl_file_path, baseline_jsonl_file_path, target_file_path)