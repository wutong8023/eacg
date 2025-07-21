import json
import os
from benchmark.pred_other import _embeddings_manager, get_version
from utils.getDatasetInfo.packInfo import getAllPackRequirements
from tqdm import tqdm
def precompute_package_embeddings(package_versions):
    """
    预计算指定package版本的embeddings
    Args:
        package_versions: dict[pack,list[version]],包和版本要求
    """
    for pack,versions in tqdm(package_versions.items(),"Precomputing embeddings for packages"):
        for version in tqdm(versions,"Precomputing embeddings for versions"):
        
            # 检查是否已经预计算
            if _embeddings_manager.load_embeddings(pack, version) is not None:
                print(f"Embeddings for {pack} {version} already exist, skipping...")
                continue
            
            # 获取文档
            from utils.RAGutils.document_utils import getKnowledgeDocs
            data = {
                "dependency": {pack: version}
            }
            documents = getKnowledgeDocs(data, dataset="versiBCB")
            
            if documents and pack in documents and documents[pack]:
                print(f"Precomputing embeddings for {pack} {version}...")
                result = _embeddings_manager.precompute_embeddings(pack, version, documents[pack])
                if result is None:
                    print(f"Failed to precompute embeddings for {pack} {version}")
            else:
                print(f"No documents found for {pack} {version}")

if __name__ == "__main__":
    # 示例：预计算特定package版本的embeddings
    # package_versions = [
    #     {"package": "pandas", "version": "2.2.3"},
    #     # 添加更多需要预计算的package版本
    # ]
    with open("benchmark/data/VersiBCB_Benchmark/vace_datas_for_warning.json","r") as f:
        datas = json.load(f)
    package_versions = getAllPackRequirements(datas)
    precompute_package_embeddings(package_versions)