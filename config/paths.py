# Base paths
BASE_PATH = "/path/to/base"
DATA_PATH = f"{BASE_PATH}/data"
MODELS_PATH = f"{BASE_PATH}/models"
CACHE_PATH = f"{BASE_PATH}/cache"
ENV_PATH = f"{BASE_PATH}/env"

# Model paths
LLAMA_MODEL_PATH = f"{MODELS_PATH}/Llama-3.1-8B"
LLAMA_INSTRUCT_PATH = f"{MODELS_PATH}/Llama-3.1-8B-Instruct"
CODELLAMA_PATH = f"{MODELS_PATH}/CodeLlama-13b-Instruct-hf"
MISTRAL_PATH = f"{MODELS_PATH}/Mistral-7B-Instruct-v0.2"

# Data paths
DOCSTRING_PATH = f"{DATA_PATH}/docs"
SRCCODE_PATH = f"{DATA_PATH}/srccodes"
RAG_DATA_PATH = f"{DATA_PATH}/RAG"

# RAG specific paths
RAG_COLLECTION_BASE = f"{RAG_DATA_PATH}/chroma_data"
DOCSTRING_EMBEDDING_PATH = f"{RAG_DATA_PATH}/docs_embeddings"
SRCCODE_EMBEDDING_PATH = f"{RAG_DATA_PATH}/srccodes_embeddings"

# Cache paths
QUERY_CACHE_DIR = f"{CACHE_PATH}/.rag_query_cache_json"
API_KEY_PATH = f"{CACHE_PATH}/API_KEYSET"

# LoRA paths
LORA_BASE_PATH = f"{MODELS_PATH}/loraadaptors"
LORA_MERGED_PATH = f"{LORA_BASE_PATH}/merged_lora_model"

# Environment paths
CONDA_ENV_PATH = f"{ENV_PATH}/conda_env"

def get_model_path(model_name: str) -> str:
    """Get the full path for a model."""
    return f"{MODELS_PATH}/{model_name}"

def get_data_path(data_type: str) -> str:
    """Get the full path for a data type."""
    return f"{DATA_PATH}/{data_type}"

def get_cache_path(cache_type: str) -> str:
    """Get the full path for a cache type."""
    return f"{CACHE_PATH}/{cache_type}" 