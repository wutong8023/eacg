# Separated Context Generation and Multi-Worker Inference System

This system separates the context generation and inference processes for better scalability and resource utilization. It consists of two main phases:

1. **Context Generation Phase**: Generate contexts for all queries once
2. **Inference Phase**: Distribute inference across multiple workers with GPU allocation

## Architecture Overview

```
Queries → Context Generation → Contexts File → Multi-Worker Inference → Results
   ↓              ↓                  ↓                    ↓               ↓
JSON/JSONL    Single Process      JSONL File        Multiple GPUs    Final JSONL
```

## Key Benefits

- **Efficiency**: Context generation is done once and reused
- **Scalability**: Inference can be distributed across multiple GPUs
- **Flexibility**: Different inference configurations can use the same contexts
- **Resource Optimization**: Better GPU utilization and memory management
- **Fault Tolerance**: If inference fails, contexts don't need to be regenerated

## Components

### 1. Batch Context Generator (`batch_context_generator.py`)

Generates contexts for all queries in a single process. This phase doesn't require GPUs and focuses on retrieval operations.

**Features:**
- Supports exact API matching and RAG fallback
- String matching capabilities
- Configurable retrieval parameters
- Progress tracking and statistics

### 2. Multi-Worker Inference (`multi_worker_inference.py`)

Performs distributed inference on pre-generated contexts using multiple workers with GPU allocation.

**Features:**
- Automatic GPU allocation per worker
- Support for `torchrun` distributed execution
- Worker-specific result files with automatic merging
- Comprehensive error handling and statistics
- Support for both local and remote API inference

## Usage

### Phase 1: Context Generation

```bash
python utils/RAGutils/queryResultByInfer/batch_context_generator.py \
    --queries_file "data/queries.json" \
    --corpus_path "data/corpus" \
    --output_file "contexts.jsonl" \
    --corpus_type "docstring" \
    --embedding_source "local" \
    --max_documents 10 \
    --max_tokens 4000 \
    --enable_str_match \
    --max_samples 1000 \
    --verbose
```

### Phase 2: Multi-Worker Inference

#### Single Worker
```bash
python utils/RAGutils/queryResultByInfer/multi_worker_inference.py \
    --contexts_file "contexts.jsonl" \
    --output_file "results.jsonl" \
    --model_path "mistralai/Mistral-7B-Instruct-v0.2" \
    --inference_type "local" \
    --precision "fp16"
```

#### Multi-Worker with torchrun
```bash
torchrun --nproc_per_node=4 \
    utils/RAGutils/queryResultByInfer/multi_worker_inference.py \
    --contexts_file "contexts.jsonl" \
    --output_file "results.jsonl" \
    --model_path "mistralai/Mistral-7B-Instruct-v0.2" \
    --inference_type "local" \
    --precision "fp16"
```

## Command Line Arguments

### Context Generator Arguments

#### Required
- `--queries_file`: Path to queries JSON/JSONL file
- `--corpus_path`: Path to corpus directory
- `--output_file`: Output JSONL file for contexts

#### Optional
- `--corpus_type`: Type of corpus ("docstring" or "srccodes")
- `--embedding_source`: Embedding source ("local" or "togetherai")
- `--max_documents`: Maximum documents to retrieve (default: 10)
- `--max_tokens`: Maximum token length for context (default: 4000)
- `--enable_str_match`: Enable string matching
- `--start_index`: Starting index for processing
- `--max_samples`: Maximum number of samples to process
- `--verbose`: Enable verbose logging

### Multi-Worker Inference Arguments

#### Required
- `--contexts_file`: Path to contexts JSONL file
- `--output_file`: Output JSONL file for results

#### Model Configuration
- `--model_path`: Model path or name (default: "mistralai/Mistral-7B-Instruct-v0.2")
- `--inference_type`: Inference type ("local", "huggingface", "togetherai")
- `--precision`: Model precision ("fp32", "fp16", "bf16")

#### API Configuration (for remote inference)
- `--api_key`: API key for remote inference
- `--api_model_name`: Model name for API inference

#### Generation Parameters
- `--max_new_tokens`: Maximum new tokens to generate (default: 512)
- `--temperature`: Sampling temperature (default: 0.2)
- `--top_p`: Top-p sampling (default: 0.95)

#### Control
- `--verbose`: Enable verbose logging

## File Formats

### Input Queries Format
```json
{
    "id": "sample_1",
    "query": "How to sort array indices?",
    "target_api": "numpy.argsort",
    "target_dependency": {"numpy": "1.16.6"}
}
```

### Context File Format
```json
{
    "id": "sample_1",
    "query": "How to sort array indices?",
    "target_api": "numpy.argsort",
    "dependencies": {"numpy": "1.16.6"},
    "retrieval_method": "exact_api_match",
    "context": "numpy.argsort documentation...",
    "context_length": 1534,
    "retrieval_time": 0.15,
    "success": true
}
```

### Final Results Format
```json
{
    "id": "sample_1",
    "query": "How to sort array indices?",
    "target_api": "numpy.argsort",
    "dependencies": {"numpy": "1.16.6"},
    "retrieval_method": "exact_api_match",
    "context": "numpy.argsort documentation...",
    "context_length": 1534,
    "answer": "The numpy.argsort function returns...",
    "retrieval_time": 0.15,
    "inference_time": 2.34,
    "total_time": 2.49,
    "worker_rank": 0,
    "success": true
}
```

## Multi-Worker Setup

### Using torchrun

The system uses PyTorch's `torchrun` for distributed execution:

```bash
# 4 workers on 4 GPUs
torchrun --nproc_per_node=4 multi_worker_inference.py [args]

# 2 workers on 2 GPUs
torchrun --nproc_per_node=2 multi_worker_inference.py [args]

# 8 workers for API inference (no GPU needed)
torchrun --nproc_per_node=8 multi_worker_inference.py --inference_type huggingface [args]
```

### GPU Allocation

- Each worker is automatically assigned to a specific GPU
- `LOCAL_RANK` environment variable determines GPU assignment
- `CUDA_VISIBLE_DEVICES` is set per worker
- Models are moved to the assigned GPU automatically

### Worker Coordination

- Workers process disjoint subsets of contexts
- Results are saved to worker-specific files
- Rank 0 worker merges all results into final output
- Distributed barriers ensure synchronization

## Performance Considerations

### Context Generation
- Single-threaded but I/O intensive
- No GPU required
- Can be run on CPU-only machines
- Memory usage depends on corpus size and embedding model

### Multi-Worker Inference
- GPU memory usage per worker depends on model size
- Batch size is typically 1 for generation tasks
- Network overhead minimal (no parameter synchronization needed)
- Linear speedup expected with number of workers

## Example Workflows

### Complete Pipeline
```bash
# Step 1: Generate contexts
python batch_context_generator.py \
    --queries_file queries.json \
    --corpus_path corpus/ \
    --output_file contexts.jsonl

# Step 2: Run inference with 4 workers
torchrun --nproc_per_node=4 multi_worker_inference.py \
    --contexts_file contexts.jsonl \
    --output_file results.jsonl \
    --model_path model_name
```

### Reusing Contexts
```bash
# Generate contexts once
python batch_context_generator.py [args] --output_file contexts.jsonl

# Run multiple inference experiments
torchrun --nproc_per_node=4 multi_worker_inference.py \
    --contexts_file contexts.jsonl \
    --output_file results_temp02.jsonl \
    --temperature 0.2

torchrun --nproc_per_node=4 multi_worker_inference.py \
    --contexts_file contexts.jsonl \
    --output_file results_temp07.jsonl \
    --temperature 0.7
```

### Mixed Inference Types
```bash
# Local inference on GPUs
torchrun --nproc_per_node=4 multi_worker_inference.py \
    --contexts_file contexts.jsonl \
    --output_file results_local.jsonl \
    --inference_type local

# API inference (many workers, no GPU)
torchrun --nproc_per_node=16 multi_worker_inference.py \
    --contexts_file contexts.jsonl \
    --output_file results_api.jsonl \
    --inference_type huggingface \
    --api_key $HF_TOKEN
```

## Monitoring and Debugging

### Logging
- Each worker logs with rank identification
- Progress bars show per-worker progress
- Comprehensive error reporting with tracebacks
- Statistics printed per worker and overall

### Error Handling
- Individual sample failures don't stop processing
- Failed samples are marked in output with error messages
- Worker failures are isolated
- Partial results are preserved

### Statistics
```
WORKER 0 INFERENCE STATISTICS
============================================================
Total samples processed: 250
Successful inferences: 248
Errors: 2
Success rate: 99.2%
Average inference time: 2.15s
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce model precision (`--precision fp16`)
   - Use fewer workers per node
   - Use smaller model

2. **Worker Synchronization Issues**
   - Check that all workers have access to input files
   - Ensure consistent arguments across workers
   - Verify network connectivity for distributed setup

3. **Context File Corruption**
   - Validate JSONL format
   - Check for incomplete context generation
   - Regenerate contexts if needed

4. **Uneven Worker Load**
   - Workers process roughly equal numbers of samples
   - Some variation expected due to different inference times
   - Monitor individual worker progress

### Performance Optimization

1. **Context Generation**
   - Use local embeddings for faster retrieval
   - Adjust `max_tokens` based on model context window
   - Use SSD storage for corpus files

2. **Multi-Worker Inference**
   - Match number of workers to available GPUs
   - Use appropriate precision (fp16 recommended)
   - Monitor GPU utilization

3. **I/O Optimization**
   - Use fast storage for context and result files
   - Consider batch writing for large datasets
   - Monitor disk space usage

## Integration with Existing Systems

This separated system is designed to integrate with existing workflows:

- **Input Compatibility**: Supports same query formats as original system
- **Output Compatibility**: Produces same result format with additional metadata
- **Configuration**: Uses same corpus and model configurations
- **Monitoring**: Compatible with existing logging and monitoring systems

## Future Enhancements

Potential improvements to the system:

1. **Dynamic Load Balancing**: Redistribute work based on worker performance
2. **Checkpointing**: Resume inference from partial results
3. **Streaming**: Process contexts as they're generated
4. **Caching**: Cache inference results for repeated queries
5. **Monitoring Dashboard**: Real-time progress and performance monitoring 