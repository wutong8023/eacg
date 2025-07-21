import json
with open("output/VersiBCB_Benchmark/codegemma-7b-it/codegemma-7b-it_udg_docstringSIFT_merged_lora_lora_pred_result.json", "r") as f:
    data = json.load(f)

with open("output/VersiBCB_Benchmark/codegemma-7b-it/codegemma-7b-it_udg_docstringSIFT_merged_lora_lora_pred_result.jsonl", "w") as f:
    for item in data:
        item["answer"] = item["answer"][0]
        f.write(json.dumps(item) + "\n")