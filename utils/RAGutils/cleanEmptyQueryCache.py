# 清理空query cache
# len(ids[0])==0
base_folder = "/datanfs2/chenrongyi/RAG_cache/.rag_query_cache_json"

import os
import json
to_remove = []
file_nums =0
for file in os.listdir(base_folder):
    if file.endswith(".json"):
        file_nums += 1
        with open(os.path.join(base_folder, file), "r") as f:
            data = json.load(f)
            if len(data["ids"][0]) == 0:
                to_remove.append(file)
print("file num:",file_nums)
print("to_remove:",len(to_remove))
choice = input("Do you want to remove the empty query cache? (y/n)")
if choice == "y":
    for file in to_remove:
        os.remove(os.path.join(base_folder, file))
else:
    print("No files removed")