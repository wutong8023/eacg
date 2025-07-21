#!/usr/bin/env python3
"""
Query-based Retrieval and Inference Script - Updated to support decoupled mode

This script performs context retrieval and inference based on query and target_api:
1. When target_api exists in corpus, directly retrieve the corresponding document
2. Otherwise, fallback to RAG retrieval to get the most relevant n documents
3. Perform inference based on the retrieved context

Usage:
    python query_based_retrieval_inference.py --query "How to sort array indices?" --target_api "numpy.argsort" --corpus_path "data/corpus" --dependencies '{"numpy": "1.16.6"}'
"""

import argparse
import json
import os
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Import from existing modules
from utils.RAGutils.RAGRetriever import RAGContextRetriever
from utils.RAGutils.RAGEmbedding import CustomEmbeddingFunction
from utils.loraTrain.loraTrainUtils import inference
from benchmark.config.code.config import (
    RAG_COLLECTION_BASE, LOCAL_EMBEDDING_MODEL, TOGETHERAI_EMBEDDING_MODEL,
    DOCSTRING_EMBEDDING_BASE_PATH, SRCCODE_EMBEDDING_BASE_PATH,
    DOCSTRING_CORPUS_PATH, SRCCODE_CORPUS_PATH
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QueryBasedRetriever:
    """
    Query-based retrieval system with exact API matching and RAG fallback
    Now supports both traditional and decoupled modes
    """
    
    def __init__(self, 
                 corpus_path: str,
                 dependencies: Optional[Dict[str, str]],
                 corpus_type: str = "docstring",
                 embedding_source: str = "local",
                 embedding_model: str = LOCAL_EMBEDDING_MODEL,
                 rag_collection_base: str = RAG_COLLECTION_BASE,
                 max_documents: int = 10,
                 max_tokens: int = 4000,
                 str_match: bool = True,
                 fixed_docs_per_query: int = 1,
                 jump_exact_match: bool = True):
        """
        Initialize the query-based retriever
        
        Args:
            corpus_path: Path to the corpus directory
            dependencies: Dictionary of package dependencies {package: version}
            corpus_type: Type of corpus ("docstring" or "srccodes")
            embedding_source: Source for embeddings ("local" or "togetherai")
            embedding_model: Embedding model to use
            rag_collection_base: Base path for RAG collections
            max_documents: Maximum number of documents to retrieve
            max_tokens: Maximum token length for context
            str_match: Enable string matching for API names
            fixed_docs_per_query: Enable fixed number of documents per query (required for str_match)
        """
        self.corpus_path = corpus_path
        self.dependencies = dependencies or {}
        self.corpus_type = corpus_type
        self.embedding_source = embedding_source
        self.embedding_model = embedding_model
        self.rag_collection_base = rag_collection_base
        self.max_documents = max_documents
        self.max_tokens = max_tokens
        self.str_match = str_match
        self.fixed_docs_per_query = fixed_docs_per_query
        # Initialize embedding function
        self.embed_func_args = {
            'source': embedding_source,
            'model_name': embedding_model,
            'together_client': None,
            'batch_size': 64
        }
        self.jump_exact_match = jump_exact_match
        # Initialize RAG retriever
        self._init_rag_retriever()
        
        logger.info(f"QueryBasedRetriever initialized with corpus_path: {corpus_path}")
        logger.info(f"Dependencies: {dependencies}")
        logger.info(f"Corpus type: {corpus_type}, Embedding source: {embedding_source}")
    
    def _init_rag_retriever(self):
        """Initialize RAG retriever"""
        try:
            # Create ChromaDB client
            chroma_client = chromadb.PersistentClient(
                path=os.path.join(self.rag_collection_base, "temp_query_retrieval"),
                settings=Settings(
                    allow_reset=True,
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
            
            # Initialize RAG retriever
            self.rag_retriever = RAGContextRetriever(
                chroma_client=chroma_client,
                embed_func_args=self.embed_func_args,
                corpus_type=self.corpus_type,
                rag_collection_base=self.rag_collection_base,
                knowledge_type=self.corpus_type,
                embedding_source=self.embedding_source,
                docstring_embedding_base_path=DOCSTRING_EMBEDDING_BASE_PATH,
                srccode_embedding_base_path=SRCCODE_EMBEDDING_BASE_PATH,
                docstring_corpus_path=DOCSTRING_CORPUS_PATH,
                srccode_corpus_path=SRCCODE_CORPUS_PATH,
                rag_document_num=self.max_documents,
                enable_dependency_filtering=True,
                api_name_str_match=self.str_match,
                fixed_docs_per_query=self.fixed_docs_per_query
            )
            
            logger.info("RAG retriever initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG retriever: {e}")
            raise
    
    def find_exact_api_document(self, target_api: str) -> Optional[str]:
        """
        Find exact document matching the target_api
        
        Args:
            target_api: Target API to search for (e.g., "numpy.argsort")
            
        Returns:
            Document content if found, None otherwise
        """
        if not target_api:
            return None
        
        logger.info(f"Searching for exact match of target_api: {target_api}")
        
        # Extract package name from target_api
        api_parts = target_api.split('.')
        if not api_parts:
            return None
        
        package_name = api_parts[0]
        
        # Check if package is in dependencies
        if package_name not in self.dependencies:
            logger.warning(f"Package {package_name} not found in dependencies")
            return None
        
        version = self.dependencies[package_name]
        
        # Construct corpus file path
        corpus_file_path = os.path.join(self.corpus_path, package_name, f"{version}.jsonl")
        
        if not os.path.exists(corpus_file_path):
            logger.warning(f"Corpus file not found: {corpus_file_path}")
            return None
        
        logger.info(f"Searching in corpus file: {corpus_file_path}")
        
        # Search for exact API match in corpus file
        try:
            with open(corpus_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        doc_data = json.loads(line)
                        
                        # Check if this document matches the target_api
                        if self._matches_target_api(doc_data, target_api):
                            logger.info(f"Found exact match for {target_api} in line {line_num}")
                            return line
                            
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON on line {line_num} in {corpus_file_path}")
                        continue
            
            logger.info(f"No exact match found for {target_api} in {corpus_file_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error reading corpus file {corpus_file_path}: {e}")
            return None
    
    def _matches_target_api(self, doc_data: Dict, target_api: str) -> bool:
        """
        Check if document matches the target API
        
        Args:
            doc_data: Document data dictionary
            target_api: Target API to match
            
        Returns:
            True if matches, False otherwise
        """
        if not isinstance(doc_data, dict):
            return False
        
        # Check path field
        doc_path = doc_data.get('path', '')
        if doc_path and doc_path == target_api:
            return True
        
        # Check aliases field
        doc_aliases = doc_data.get('aliases', [])
        if isinstance(doc_aliases, list):
            for alias in doc_aliases:
                if isinstance(alias, str) and alias == target_api:
                    return True
        
        # Check if target_api is contained in path or aliases (partial match)
        if doc_path and target_api in doc_path:
            return True
        
        for alias in doc_aliases:
            if isinstance(alias, str) and target_api in alias:
                return True
        
        return False
    
    def retrieve_context_with_rag(self, query: str, target_api: Optional[str] = None) -> str:
        """
        Retrieve context using RAG system with string matching support - now uses decoupled mode
        
        Args:
            query: Query string
            target_api: Optional target API for string matching
            
        Returns:
            Retrieved context
        """
        logger.info(f"Using RAG retrieval for query: {query[:100]}...")
        if target_api:
            logger.info(f"Target API for string matching: {target_api}")
        
        try:
            # Prepare queries based on whether target_api is provided
            if target_api:
                # Use string matching mode: create dict query with path and description
                queries = [{
                    "path": target_api,
                    "description": query
                }]
                logger.info(f"Using string matching mode with target_api: {target_api}")
            else:
                # Use regular string query mode
                queries = [query]
                logger.info("Using regular string query mode")
            
            # Use new decoupled mode - pass queries and dependencies directly
            context = self.rag_retriever.retrieve_context(
                queries=queries,  # Prepared queries (dict for string matching or string for regular)
                dependencies=self.dependencies,  # Direct dependencies
                max_token_length=self.max_tokens,
                sample_id=f"query_retrieval_{hash(query) % 10000}"
            )
            
            if context:
                logger.info(f"RAG retrieval successful, context length: {len(context)} characters")
                return context
            else:
                logger.warning("RAG retrieval returned empty context")
                return ""
                
        except Exception as e:
            logger.error(f"Error in RAG retrieval: {e}")
            return ""
    
    def retrieve_context(self, query: str, target_api: Optional[str] = None) -> Tuple[str, str]:
        """
        Retrieve context using exact API match or RAG fallback
        
        Args:
            query: Query string
            target_api: Optional target API for exact matching
            
        Returns:
            Tuple of (context, retrieval_method)
        """
        start_time = time.time()
        
        # Try exact API match first if target_api is provided and str_match is disabled
        if target_api and self.str_match:
            exact_doc = self.find_exact_api_document(target_api)
            if exact_doc:
                retrieval_time = time.time() - start_time
                logger.info(f"Exact API retrieval completed in {retrieval_time:.2f} seconds")
                if self.jump_exact_match:
                    return "", "exact_api_match"
                return exact_doc, "exact_api_match"
        
        # Use RAG retrieval (with string matching if enabled and target_api provided)
        rag_context = self.retrieve_context_with_rag(query, target_api if self.str_match else None)
        retrieval_time = time.time() - start_time
        
        # Determine retrieval method based on configuration
        if target_api and self.str_match:
            retrieval_method = "rag_string_match"
        elif target_api:
            retrieval_method = "rag_fallback"  # exact match failed, fallback to RAG
        else:
            retrieval_method = "rag_fallback"
            
        logger.info(f"RAG retrieval completed in {retrieval_time:.2f} seconds using {retrieval_method}")
        
        return rag_context, retrieval_method


class QueryBasedInference:
    """
    Inference system that uses retrieved context to answer queries
    """
    
    def __init__(self,
                 model_path: str,
                 inference_type: str = "local",
                 api_key: Optional[str] = None,
                 api_model_name: Optional[str] = None,
                 max_new_tokens: int = 512,
                 temperature: float = 0.2,
                 top_p: float = 0.95):
        """
        Initialize inference system
        
        Args:
            model_path: Path to the model or model name
            inference_type: Type of inference ("local", "huggingface", "togetherai")
            api_key: API key for remote inference
            api_model_name: Model name for API inference
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
        """
        self.model_path = model_path
        self.inference_type = inference_type
        self.api_key = api_key
        self.api_model_name = api_model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Initialize model and tokenizer for local inference
        self.model = None
        self.tokenizer = None
        
        if inference_type == "local":
            self._load_local_model()
    
    def _load_local_model(self):
        """Load local model and tokenizer"""
        try:
            logger.info(f"Loading local model: {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            logger.info("Local model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise
    
    def create_prompt(self, query: str, context: str, retrieval_method: str, extra_info: str = "") -> str:
        """
        Create inference prompt
        
        Args:
            query: User query
            context: Retrieved context
            retrieval_method: Method used for retrieval
            
        Returns:
            Formatted prompt
        """
        if retrieval_method == "exact_api_match":
            prompt = f"""Based on the following API documentation, please answer the user's query.
Extra Info: 
{extra_info}
API Documentation:
{context}

User Query: {query}

Please provide a clear and accurate answer based on the API documentation above.

Answer:"""
        elif retrieval_method == "rag_string_match":
            prompt = f"""Based on the following API documentation retrieved through string matching, please answer the user's query. make the answer as brief as possible to cover the query.
Extra Info: 
{extra_info}
API Documentation:
{context}

User Query: {query}

Please provide a clear and accurate answer based on the API documentation above.

Answer:"""
        else:
            prompt = f"""Based on the following context, please answer the user's query.
Extra Info: 
{extra_info}
Context:
{context}

User Query: {query}

Please provide a helpful answer based on the context above.

Answer:"""
        
        return prompt
    
    def generate_answer(self, query: str, context: str, retrieval_method: str, extra_info: str = "") -> str:
        """
        Generate answer based on query and context
        
        Args:
            query: User query
            context: Retrieved context
            retrieval_method: Method used for retrieval
            
        Returns:
            Generated answer
        """
        if not context.strip():
            return "I couldn't find relevant information to answer your query."
        
        prompt = self.create_prompt(query, context, retrieval_method, extra_info)
        
        logger.info(f"Generating answer with {retrieval_method} context")
        
        try:
            answer = inference(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                inference_type=self.inference_type,
                api_key=self.api_key,
                model_name=self.api_model_name,
                stop_tokens=["<|endoftext|>"]
            )
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return f"Error generating answer: {str(e)}"


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Query-based retrieval and inference")
    
    # Query arguments
    parser.add_argument("--query", type=str, required=True, help="Query string")
    parser.add_argument("--target_api", type=str, help="Target API for exact matching (optional)")
    
    # Corpus arguments
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to corpus directory")
    parser.add_argument("--dependencies", type=str, required=True, help="JSON string of dependencies")
    parser.add_argument("--corpus_type", type=str, default="docstring", choices=["docstring", "srccodes"], help="Type of corpus")
    
    # Retrieval arguments
    parser.add_argument("--embedding_source", type=str, default="local", choices=["local", "togetherai"], help="Embedding source")
    parser.add_argument("--embedding_model", type=str, default=LOCAL_EMBEDDING_MODEL, help="Embedding model")
    parser.add_argument("--max_documents", type=int, default=10, help="Maximum documents to retrieve")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Maximum token length for context")
    
    # Inference arguments
    parser.add_argument("--model_path", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="Model path or name")
    parser.add_argument("--inference_type", type=str, default="local", choices=["local", "huggingface", "togetherai"], help="Inference type")
    parser.add_argument("--api_key", type=str, help="API key for remote inference")
    parser.add_argument("--api_model_name", type=str, help="Model name for API inference")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    
    # Output arguments
    parser.add_argument("--output_file", type=str, help="Output file to save results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--enable_str_match", action="store_true", help="Enable string matching")
    parser.add_argument("--fixed_docs_per_query",type=int,default=1,help="Enable fixed number of documents per query (required for string matching)")
    parser.add_argument("--jump_exact_match",action="store_true",help="Jump exact match")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse dependencies
    try:
        dependencies = json.loads(args.dependencies)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid dependencies JSON: {e}")
        return
    
    logger.info("=== Query-based Retrieval and Inference (Decoupled Mode) ===")
    logger.info(f"Query: {args.query}")
    logger.info(f"Target API: {args.target_api}")
    logger.info(f"Dependencies: {dependencies}")
    
    # Initialize retriever
    try:
        retriever = QueryBasedRetriever(
            corpus_path=args.corpus_path,
            dependencies=dependencies,
            corpus_type=args.corpus_type,
            embedding_source=args.embedding_source,
            embedding_model=args.embedding_model,
            max_documents=args.max_documents,
            max_tokens=args.max_tokens,
            str_match =args.enable_str_match,
            fixed_docs_per_query=args.fixed_docs_per_query,
            jump_exact_match=args.jump_exact_match
        )
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {e}")
        return
    
    # Initialize inference system
    try:
        inference_system = QueryBasedInference(
            model_path=args.model_path,
            inference_type=args.inference_type,
            api_key=args.api_key,
            api_model_name=args.api_model_name,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
    except Exception as e:
        logger.error(f"Failed to initialize inference system: {e}")
        return
    
    # Retrieve context
    start_time = time.time()
    context, retrieval_method = retriever.retrieve_context(args.query, args.target_api)
    retrieval_time = time.time() - start_time
    
    logger.info(f"Context retrieved using: {retrieval_method}")
    logger.info(f"Context length: {len(context)} characters")
    logger.info(f"Retrieval time: {retrieval_time:.2f} seconds")
    
    # Generate answer
    start_time = time.time()
    answer = inference_system.generate_answer(args.query, context, retrieval_method)
    inference_time = time.time() - start_time
    
    logger.info(f"Inference time: {inference_time:.2f} seconds")
    
    # Prepare results
    results = {
        "query": args.query,
        "target_api": args.target_api,
        "dependencies": dependencies,
        "retrieval_method": retrieval_method,
        "context_length": len(context),
        "context": context[:500] + "..." if len(context) > 500 else context,  # Truncate for display
        "answer": answer,
        "retrieval_time": retrieval_time,
        "inference_time": inference_time,
        "total_time": retrieval_time + inference_time
    }
    
    # Output results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Query: {results['query']}")
    print(f"Target API: {results['target_api']}")
    print(f"Retrieval Method: {results['retrieval_method']}")
    print(f"Context Length: {results['context_length']} characters")
    print(f"Answer: {results['answer']}")
    print(f"Retrieval Time: {results['retrieval_time']:.2f}s")
    print(f"Inference Time: {results['inference_time']:.2f}s")
    print(f"Total Time: {results['total_time']:.2f}s")
    
    # Save to file if specified
    if args.output_file:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {args.output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


if __name__ == "__main__":
    main() 