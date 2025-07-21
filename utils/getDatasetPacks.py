import json
import os
def getPackVersions(datas):
    packVersions = {}
    for data in datas:
        dependency = data["target_dependency"]
        for pack,version in dependency.items():
            if pack not in packVersions:
                packVersions[pack] = [version]
            else:
                packVersions[pack].append(version)

    # remove redundant versions
    packVersions = {pack:list(set(versions)) for pack,versions in packVersions.items()}
    return packVersions
def getDocLength(pkg,version):
    corpus_path = "/datanfs2/chenrongyi/data/docs"
    data_path = os.path.join(corpus_path,pkg,version+".jsonl")
    doc_full_length = 0
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            doc_length = len(data)
            doc_full_length += doc_length
    return doc_full_length
if __name__ == "__main__":
    with open("benchmark/data/VersiBCB_Benchmark/vace_datas_for_warning.json", "r") as f:
        datas = json.load(f)
    packVersions = getPackVersions(datas)
    print(packVersions)
    packVersionCount = {pack:len(versions) for pack,versions in packVersions.items()}
    print(packVersionCount)
    print(sum(packVersionCount.values()))
    for pack,versions in packVersions.items():
        for version in versions:
            doc_full_length = getDocLength(pack,version)
            print(pack,version,doc_full_length)
