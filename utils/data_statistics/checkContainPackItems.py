import json

with open("data/VersiBCB_Benchmark/vace_datas.json",'r') as f:
    data = json.load(f)

pkg_nums = {}
for item in data:
    for k,v in item["target_dependency"].items():
        if k in pkg_nums:
            pkg_nums[k] += 1
        else:
            pkg_nums[k] = 1

print(pkg_nums)
print(len(pkg_nums))