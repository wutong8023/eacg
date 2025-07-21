import json
from utils.io_utils import loadJsonl
with open("data/generated_queries/versibcb_vace_queries_deduplicated.json", "r") as f:
    data = json.load(f)
seqid2bcbid = {}
bcb_ids = data.keys()
seq_ids = [str(i) for i in range(0,len(data))]

for bcb_id, seq_id in zip(bcb_ids, seq_ids):
    seqid2bcbid[seq_id] = int(bcb_id)

result = loadJsonl("data/temp/inference/modified_pipeline_results.jsonl")
for item in result:
    item["original_item_id"] = seqid2bcbid[item["original_item_id"]]
with open("data/temp/inference/modified_pipeline_results_fixed.jsonl", "w") as f:
    for item in result:
        f.write(json.dumps(item) + "\n")