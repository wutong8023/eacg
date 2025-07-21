import re
from benchmark.config.code.config import CORPUS_PATH
import os
def remove_prefix(text):
    # 删除prefix >= > <= < ==
    text = re.sub(r"^>=", "", text)
    text = re.sub(r"^<=", "", text)
    text = re.sub(r"^==", "", text)
    return text
def getBestMatchVersion(pkg,version):
    '''
    Description:
        获取最佳匹配版本
    Args:
        pkg: str,包名
        version: str,版本号，可能包含prefix e.g. >=1.24.3
    Returns:
        best_match_version: str,最佳匹配版本
    '''
    # 获取指定路径下所有版本
    version = remove_prefix(version)
    try:
        versions_file = os.listdir(f"{CORPUS_PATH}/{pkg}")
    except FileNotFoundError:
        return None
    # 获取所有版本
    versions = []
    for version_file in versions_file:
        cand_version = re.sub(r"\.jsonl$", "", version_file)
        versions.append(cand_version)
    # 获取最佳匹配版本
    best_match_version = None
    for cand_version in versions:
        if cand_version == version or cand_version[:len(version)] == version:
            return cand_version
    return None
if __name__ == "__main__":
    print(getBestMatchVersion("torch","2.0.0"))

