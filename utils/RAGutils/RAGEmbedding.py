import os
import pickle
import time
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from tqdm import tqdm
from together import Together  # Import Together client

# Global variable for device selection (can be configured)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, source='local', model_name='all-MiniLM-L6-v2', together_client=None, batch_size=32):
        self.source = source
        self.model_name = model_name
        self.together_client = together_client
        self.batch_size = batch_size
        
        if self.source == 'local':
            self.model = SentenceTransformer(model_name).to(device)
        elif self.source == 'togetherai':
            if self.together_client is None:
                raise ValueError("Together client must be provided for 'togetherai' source.")
        else:
            raise ValueError(f"Unsupported embedding source: {source}")

    def _embed_local(self, texts: Documents) -> Embeddings:
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            with torch.no_grad():
                batch_embeddings = self.model.encode(
                    batch,
                    batch_size=self.batch_size, # Use internal batching of SentenceTransformer
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    device=device
                )
            embeddings.extend(batch_embeddings.cpu().numpy())
            del batch_embeddings # Explicitly delete tensor
        return embeddings

    def _embed_togetherai(self, texts: Documents) -> Embeddings:
        all_embeddings = []
        # Together AI API might handle batching implicitly or have rate limits
        # Process in smaller batches if necessary, or one by one for simplicity first
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings (Together AI)"):
             batch_texts = texts[i:i+self.batch_size]
             try:
                 res = self.together_client.embeddings.create(input=batch_texts, model=self.model_name)
                 batch_embeddings = [item.embedding for item in res.data]
                 all_embeddings.extend(batch_embeddings)
             except Exception as e:
                 print(f"Error getting embeddings from Together AI for batch {i//self.batch_size}: {e}")
                 # Add error handling: fill with zeros or skip? Let's add zeros for now
                 all_embeddings.extend([[0.0] * 768] * len(batch_texts)) # Assuming embedding dim 768, adjust if needed
                 time.sleep(1) # Basic retry delay

        return np.array(all_embeddings)

    def __call__(self, texts: Documents) -> Embeddings:
        # Replace Nones or empty strings with a placeholder to avoid errors
        processed_texts = [(text if text and text.strip() else " ") for text in texts]
        
        if self.source == 'local':
            return self._embed_local(processed_texts)
        elif self.source == 'togetherai':
            return self._embed_togetherai(processed_texts)
        else:
            # Should not happen due to __init__ check, but for safety:
            return []


class PrecomputedEmbeddingsManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def get_embedding_path(self, pack, version):
        """获取预计算embeddings的文件路径"""
        dir_path = os.path.join(self.base_dir, pack)
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_path, f"{version}_embeddings.pkl")

    def save_embeddings(self, pack, version, documents, embeddings):
        """保存embeddings到文件"""
        path = self.get_embedding_path(pack, version)
        with open(path, 'wb') as f:
            pickle.dump({'documents': documents, 'embeddings': embeddings}, f)
        print(f"Saved precomputed embeddings for {pack} {version} to {path}")

    def load_embeddings(self, pack, version):
        """加载预计算embeddings"""
        path = self.get_embedding_path(pack, version)
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                print(f"Loaded precomputed embeddings for {pack} {version} from {path}")
                return data
            except Exception as e:
                print(f"Error loading embeddings file {path}: {e}. Will recompute.")
                return None
        return None

    def precompute_embeddings(self, pack, version, documents,
                              embedding_function_instance: CustomEmbeddingFunction,
                              batch_size=256):
        """预计算文档的embeddings"""
        if not documents:
            print(f"No documents found for {pack} {version}, skipping...")
            return None

        # Use the provided embedding_function_instance
        embed_func = embedding_function_instance
        source = embed_func.source # Get source from the instance

        print(f"Precomputing embeddings for {pack} {version} using {source} source via provided instance...")

        # Replace Nones or empty strings
        processed_documents = [(doc if doc and doc.strip() else " ") for doc in documents]

        try:
            # Add progress bar for the overall process
            with tqdm(total=len(processed_documents), desc=f"Generating embeddings for {pack} {version}") as pbar:
                if source == 'local':
                    embeddings = []
                    # Use the batch_size from the embedding_function_instance if desired, or keep specific one here
                    # For now, using the batch_size argument passed to this method.
                    current_batch_size = batch_size
                    for i in range(0, len(processed_documents), current_batch_size):
                        batch = processed_documents[i:i + current_batch_size]
                        with torch.no_grad():
                            # Assuming embed_func.model.encode handles its own device placement if model is on GPU
                            batch_embeddings = embed_func.model.encode(
                                batch,
                                batch_size=current_batch_size, # Pass the batch size to encode
                                show_progress_bar=False,
                                convert_to_tensor=True
                                # device=device # Already handled by embed_func.model being on a device
                            )
                        # Ensure tensor is moved to CPU before converting to numpy
                        embeddings.extend(batch_embeddings.cpu().numpy())
                        del batch_embeddings # Explicitly delete tensor
                        pbar.update(len(batch))
                    embeddings = np.array(embeddings)
                else:  # togetherai
                    # The __call__ method of CustomEmbeddingFunction handles batching internally
                    embeddings = embed_func(processed_documents)
                    pbar.update(len(processed_documents))
        except Exception as e:
            print(f"Failed to generate embeddings for {pack} {version}: {e}")
            return None

        if embeddings is None or len(embeddings) == 0:
            print(f"No embeddings generated for {pack} {version}, skipping save...")
            return None
            
        # Ensure embeddings are numpy array
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        self.save_embeddings(pack, version, documents, embeddings)
        return embeddings
