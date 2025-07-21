import json
file_names = ["vace_datas.json","vscc_datas.json","vace_datas_for_warning.json","vscc_datas_for_warning.json"]
bench_base = 'data/VersiBCB_Benchmark/'
def getItemPkgVersion(item):
    dependency = item["dependency"] if "dependency" in item else item["target_dependency"]
    pkgVersions = {k:[v] for k,v in dependency.items()}
    return pkgVersions

def mergePkgVersions(src_dict,tar_dict):
    '''
    params:
        src_dict: 源字典，key为pkg，value为versions
        tar_dict: 目标字典，key为pkg，value为versions
    return:
        tar_dict: 目标字典，key为pkg，value为versions
    '''
    for pkg,versions in src_dict.items():
        if pkg in tar_dict:
            tar_dict[pkg].extend(versions)
        else:
            tar_dict[pkg] = versions
    return tar_dict
if __name__ == "__main__":
    pkgVersions = {}
    for file_name in file_names:
        with open(bench_base + file_name, "r",encoding="utf-8") as f:
            items = json.load(f)
            for item in items:
                itemPkgVersions = getItemPkgVersion(item)
                pkgVersions = mergePkgVersions(itemPkgVersions,pkgVersions)
    # remove Duplicate
    pkgVersions = {k:list(set(v)) for k,v in pkgVersions.items()}
    with open(bench_base + "pkgVersions.json", "w",encoding="utf-8") as f:
        json.dump(pkgVersions, f, indent=4)