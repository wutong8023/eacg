import yaml
from pathlib import Path

# 加载 YAML 配置文件
config_path = Path(__file__).parent / "collectionBuild_config.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 导出变量
RAG_COLLECTION_BASE = config["RAG_COLLECTION_BASE"]
KNOWLEDGE_TYPE = config["KNOWLEDGE_TYPE"]
EMBEDDING_SOURCE = config["EMBEDDING_SOURCE"]
LOCAL_EMBEDDING_MODEL = config["LOCAL_EMBEDDING_MODEL"]
TOGETHERAI_EMBEDDING_MODEL = config["TOGETHERAI_EMBEDDING_MODEL"]
CORPUS_PATH = config["CORPUS_PATH"]
EMBEDDING_BASE_PATH = config["EMBEDDING_BASE_PATH"]