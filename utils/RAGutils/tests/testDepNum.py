import json
with open('benchmark/data/VersiBCB_Benchmark/vace_datas.json','r') as f:
    datas = json.load(f)

target_deps = []
for data in datas[247:]:
    target_deps.append(tuple(data['target_dependency'].items()))
print(len(set(target_deps)))

# 目前看来，python并没有使得dep变得少很多，所以还是加上了(但是后续可以考虑去除)