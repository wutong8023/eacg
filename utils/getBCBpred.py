import json
from utils import writeJsonl
with open("data/VersiBCB_Benchmark/vace_datas.json",'r') as f:
    data = json.load(f)
output_data = []
for item in data:
    output_item = {}
    output_item = {"id":item["id"],"answer":item["target_code"]}
    output_data.append(output_item)
writeJsonl("data/VersiBCB_Benchmark/vace_datas_reference.jsonl",output_data)



# 


