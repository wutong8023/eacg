# Query-based Retrieval and Inference System

This system provides intelligent context retrieval and inference based on user queries and target APIs. It implements a two-tier strategy:

1. **Exact API Match**: When a target API is specified and exists in the corpus, directly retrieve the corresponding documentation
2. **RAG Fallback**: When exact match fails, use RAG (Retrieval-Augmented Generation) to find the most relevant documents

## Features

- **Intelligent Retrieval Strategy**: Prioritizes exact API matches over similarity search
- **Flexible Input Formats**: Supports both single queries and batch processing from JSON files
- **Multiple Corpus Types**: Works with both docstring and source code corpora
- **Multiple Inference Backends**: Supports local models and remote APIs (HuggingFace, TogetherAI)
- **Comprehensive Statistics**: Provides detailed performance metrics and success rates
- **Dependency-aware**: Filters documents based on package dependencies
- **Configurable Parameters**: Extensive configuration options for retrieval and inference

## System Architecture

```
Query + Target API
        ↓
   QueryBasedRetriever
    ↓               ↓
Exact API Match    RAG Fallback
    ↓               ↓
Retrieved Context ← ←
        ↓
 QueryBasedInference
        ↓
    Generated Answer
```

## Installation and Dependencies

Ensure you have the required dependencies installed:

```bash
# Core dependencies
pip install torch transformers sentence-transformers chromadb tqdm

# For specific embedding models and API access
pip install together accelerate
```

## Files Overview

### Core Components

1. **`query_based_retrieval_inference.py`** - Main single-query processing script
2. **`batch_query_inference.py`** - Batch processing script for multiple queries  
3. **`run_query_inference_example.sh`** - Example usage scenarios

### Supporting Files

- **Existing RAG infrastructure** from `utils/RAGutils/` and `benchmark/`
- **Configuration files** in `benchmark/config/code/`

## Usage

### Single Query Processing

```bash
python query_based_retrieval_inference.py \
    --query "What is the equivalent of numpy.argsort?" \
    --target_api "numpy.argsort" \
    --corpus_path "data/corpus" \
    --dependencies '{"numpy": "1.16.6", "matplotlib": "2.0.2"}' \
    --model_path "mistralai/Mistral-7B-Instruct-v0.2" \
    --output_file "results.json"
```

### Batch Processing

```bash
python batch_query_inference.py \
    --queries_file "data/generated_queries/versibcb_vace_queries_deduplicated.json" \
    --corpus_path "data/corpus" \
    --output_file "batch_results.jsonl" \
    --max_samples 100
```

### Key Parameters

#### Required Parameters
- `--query` or `--queries_file`: Query text or file containing queries
- `--corpus_path`: Path to the corpus directory
- `--dependencies`: JSON string of package dependencies (for single query)

#### Retrieval Parameters
- `--corpus_type`: Type of corpus ("docstring" or "srccodes")
- `--embedding_source`: Embedding source ("local" or "togetherai")
- `--max_documents`: Maximum documents to retrieve (default: 10)
- `--max_tokens`: Maximum token length for context (default: 4000)

#### Inference Parameters
- `--model_path`: Model path or name (default: "mistralai/Mistral-7B-Instruct-v0.2")
- `--inference_type`: Inference type ("local", "huggingface", "togetherai")
- `--max_new_tokens`: Maximum new tokens to generate (default: 512)
- `--temperature`: Sampling temperature (default: 0.2)

## Query Input Formats

The system supports multiple query input formats:

### 1. Simple Query Format
```json
{
    "id": "sample_1",
    "query": "How to sort array indices?",
    "target_api": "numpy.argsort",
    "target_dependency": {"numpy": "1.16.6"}
}
```

### 2. VersiBCB Format
```json
{
    "id": "sample_1",
    "description": "How to sort array indices?",
    "target_dependency": {"numpy": "1.16.6", "matplotlib": "2.0.2"},
    "queries": [
        {
            "query": "What is the equivalent of numpy.argsort?",
            "target_api": "numpy.argsort"
        }
    ]
}
```

### 3. Legacy Format
```json
{
    "id": "sample_1",
    "description": "How to sort array indices?",
    "dependency": {"numpy": "1.16.6"}
}
```

## Example Scenarios

### Scenario 1: Exact API Match

**Query**: "How to use numpy.argsort?"  
**Target API**: "numpy.argsort"  
**Result**: System finds exact documentation for `numpy.argsort` in the corpus

```bash
python query_based_retrieval_inference.py \
    --query "How to use numpy.argsort to get sorted indices?" \
    --target_api "numpy.argsort" \
    --corpus_path "data/corpus" \
    --dependencies '{"numpy": "1.16.6"}'
```

### Scenario 2: RAG Fallback

**Query**: "How to create scatter plots with colors?"  
**Target API**: None or not found  
**Result**: System uses RAG to retrieve relevant matplotlib documentation

```bash
python query_based_retrieval_inference.py \
    --query "How to create scatter plots with different colors?" \
    --corpus_path "data/corpus" \
    --dependencies '{"matplotlib": "2.0.2", "numpy": "1.16.6"}'
```

### Scenario 3: Batch Processing

Process multiple queries from a file:

```bash
python batch_query_inference.py \
    --queries_file "data/generated_queries/versibcb_vace_queries_deduplicated.json" \
    --corpus_path "data/corpus" \
    --output_file "batch_results.jsonl" \
    --start_index 0 \
    --max_samples 50 \
    --verbose
```

## Output Format

### Single Query Output
```json
{
    "query": "How to use numpy.argsort?",
    "target_api": "numpy.argsort", 
    "dependencies": {"numpy": "1.16.6"},
    "retrieval_method": "exact_api_match",
    "context_length": 1534,
    "answer": "The numpy.argsort function returns the indices that would sort an array...",
    "retrieval_time": 0.15,
    "inference_time": 2.34,
    "total_time": 2.49
}
```

### Batch Processing Output (JSONL)
Each line contains a result object with the same format as above, plus:
```json
{
    "id": "sample_123",
    "success": true,
    "processing_time": 2.67
}
```

## Performance Statistics

The batch processor provides comprehensive statistics:

```
BATCH PROCESSING STATISTICS
============================================================
Total queries processed: 100
Exact API matches: 67
RAG fallbacks: 31
Errors: 2
Success rate: 98.0%
Exact match rate: 67.0%
Average retrieval time: 0.23s
Average inference time: 2.15s
Results saved to: batch_results.jsonl
```

## Configuration

### Corpus Structure
```
data/corpus/
├── numpy/
│   └── 1.16.6.jsonl
├── matplotlib/
│   └── 2.0.2.jsonl
└── scipy/
    └── 1.4.1.jsonl
```

### Document Format in Corpus
Each line in the JSONL files should contain:
```json
{
    "path": "numpy.argsort",
    "doc": "Returns the indices that would sort an array...",
    "aliases": ["np.argsort"],
    "signature": "argsort(a, axis=-1, kind=None, order=None)",
    "type": "function"
}
```

## Advanced Usage

### Remote API Inference

For HuggingFace API:
```bash
python query_based_retrieval_inference.py \
    --query "How to compute eigenvalues?" \
    --target_api "numpy.linalg.eig" \
    --corpus_path "data/corpus" \
    --dependencies '{"numpy": "1.16.6"}' \
    --inference_type "huggingface" \
    --api_key "your_hf_token" \
    --api_model_name "mistralai/Mistral-7B-Instruct-v0.1"
```

For TogetherAI:
```bash
python query_based_retrieval_inference.py \
    --query "How to create subplots?" \
    --corpus_path "data/corpus" \
    --dependencies '{"matplotlib": "2.0.2"}' \
    --inference_type "togetherai" \
    --api_key "your_together_api_key" \
    --api_model_name "mistralai/Mistral-7B-Instruct-v0.1"
```

### Source Code Corpus

To use source code instead of documentation:
```bash
python query_based_retrieval_inference.py \
    --query "Show me matplotlib plotting examples" \
    --corpus_path "data/corpus" \
    --dependencies '{"matplotlib": "2.0.2"}' \
    --corpus_type "srccodes"
```

## Error Handling

The system handles various error scenarios:

1. **Missing Target API**: Falls back to RAG retrieval
2. **Empty Corpus**: Returns informative error message
3. **Invalid Dependencies**: Uses default dependencies with warning
4. **Model Loading Errors**: Detailed error reporting
5. **Inference Failures**: Graceful error handling with context

## Performance Optimization

### Tips for Better Performance

1. **Use Local Models**: Faster than API calls but requires GPU memory
2. **Batch Processing**: More efficient than individual queries
3. **Limit Token Length**: Reduce `max_tokens` for faster retrieval
4. **Cache Embeddings**: RAG system caches embeddings automatically
5. **Filter Dependencies**: Smaller dependency sets improve retrieval speed

### Memory Management

For large-scale processing:
- Use smaller models or API inference to reduce memory usage
- Process in batches with `--max_samples`
- Use `--start_index` for resumable processing

## Troubleshooting

### Common Issues

1. **"No queries loaded"**: Check JSON file format and path
2. **"Package not found in dependencies"**: Add required packages to dependencies
3. **"Model loading failed"**: Verify model path and available GPU memory
4. **"RAG retrieval failed"**: Check corpus path and ChromaDB permissions

### Debug Mode

Enable verbose logging for detailed information:
```bash
python query_based_retrieval_inference.py --verbose [other args]
```

## Integration with Existing Systems

This system is designed to work with the existing RAG infrastructure:

- **Compatible with** `benchmark/pred_rag.py` configurations
- **Uses existing** RAG retrieval and embedding systems
- **Supports** same corpus formats and dependency structures
- **Integrates with** existing evaluation pipelines

## Future Enhancements

Planned improvements:
- Support for multi-modal queries (code + text)
- Enhanced API matching with fuzzy string matching
- Integration with code execution for verification
- Support for cross-package API recommendations
- Real-time corpus updates and indexing 