import json
with open("data/EvolveItemsWithInterval_processed_clean.json", "r") as f:
    data = json.load(f)
# 统计package数量，给出dict
package_count = {}
for item in data:
    package = item["package"]
    if package not in package_count:
        package_count[package] = 0
    package_count[package] += 1
print(sorted(package_count.items(), key=lambda x: x[1], reverse=True))
# 统计package数量，给出dict ，key为package，value为evolve_type的种类数