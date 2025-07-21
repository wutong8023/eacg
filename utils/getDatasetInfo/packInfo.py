import re
def remove_prefix(text):
    # 删除prefix >= > <= < ==
    text = re.sub(r"^>=", "", text)
    text = re.sub(r"^<=", "", text)
    text = re.sub(r"^==", "", text)
    return text
def getVersionForPackRequirement(pack,version_requirement):
    """
    根据pack和version_requirement，获取对应的版本
    pack: 包名
    version_requirement: 版本要求 如>=2.0.0,<=3.0 ==2.0.0
    """
    version_requirement = remove_prefix(version_requirement)
    return version_requirement
def getAllPackRequirements(data,is_versiBCB=True):
    '''
    Description:
        获取所有包和版本要求
    Args:
        data: 数据
        is_versiBCB: 是否是versiBCB数据集
    Returns:
        packRequirements: dict[pack,list[version]],包和版本要求
    '''
    if not is_versiBCB:
        packRequirements = {}
        for item in data:
            pack = item["dependency"]
            version_requirement = getVersionForPackRequirement(pack,item["version"])
            if pack not in packRequirements:
                packRequirements[pack] = [version_requirement]
            else:
                packRequirements[pack].append(version_requirement)
    else:
        packRequirements = {}
        for item in data:
            # 获取dependency
            if "dependency" in item:
                dependency = item["dependency"]
            else:
                dependency = item["target_dependency"]
            # 将dependency中的包和版本要求加入packRequirements
            for pack,version in dependency.items():
                if pack not in packRequirements:
                    packRequirements[pack] = [version]
                else:
                    packRequirements[pack].append(version)
    # 消除重复项
    for pack in packRequirements:
        packRequirements[pack] = list(set(packRequirements[pack]))
    return packRequirements