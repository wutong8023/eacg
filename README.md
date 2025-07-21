# Data Conversion
cVersicode2Longbench.py converts Versicode and VersiBCB Benchmark from json format to the data format required by emllm, used for running memoryLLM.

# Usage
## Collection Building (Optional)
For both rag and memory approaches, you need to prepare corresponding collections first for retrieving relevant context.
```
bash scripts/build_collection.sh 
```

## LoRA Adaptor Training
scripts/train_lora.sh is the MoE training script used for training MoE models. Currently, you need to manually specify the corresponding benchmark data. The process involves training loraadaptors for all packs based on the item's dependency.

# Experiment Description
To support experiments on dependency combination numbers and src2tar dimensions,max_dependency_num is the maximum number of dependencies, and append_srcDep determines whether to add source dependencies.

## Experiment 1: Foundation Model Performance on VersiBCB (Can be skipped)
In mvBCBbuilder Project, not in this project

## TABLE VII: pass@1 (Exec@1) and wpass@1() on Code Completion (PACG)
Run the following scripts to produce corresponding results (only need to modify the parameters to specify task type, VACM or VSCG)
- scripts/approachExperiment/run_rag_multiworker.sh
    - For this experiment, just switch the method to baseline to produce baseline results
- scripts/approachExperiment/run_emllm.sh
- scripts/approachExperiment/run_lora_multiworker.sh
For final experiment result calculation, switch to mvBCBbuilder Project, not in this project.

### Detailed Parameter Description

#### LoRA Method Parameters (scripts/approachExperiment/run_lora_multiworker.sh)

**Basic Prediction Parameters:**
- `MAX_DEPENDENCY_NUM`: Maximum number of dependencies (default: 9)
- `TEMPERATURE`: Sampling temperature (default: 1e-5)
- `TOP_P`: Nucleus sampling parameter (default: 1.0)
- `PRECISION`: Model precision (default: "fp16", options: "fp32", "bf16", "int8")
- `MODEL_NAME`: Model path (default: "/datanfs2/chenrongyi/models/Llama-3.1-8B")
- `KNOWLEDGE_TYPE`: Knowledge type (default: "srccodes", options: "docstring")
- `ADAPTOR_BASE_PATH`: LoRA adaptor base path
- `BASE_OUTPUT_DIR`: Output directory


**Multi-worker Configuration:**
- `TASKS`: Task list (default: ("vace"), options: "vscc")
- `DEPRECATION_FLAGS`: Whether to disable deprecated features (default: (false))
- `-w, --world_size`: Number of GPUs
- `-n, --num_gpus_per_job`: GPUs per task

**Feature Combination Usage:**
1. **Dependency mask only**: Set `ENABLE_TASK2MASKPACKS=true`
2. **Mask + Task filtering**: Set `ENABLE_TASK2MASKPACKS=true` and `ONLY_TASK2MASKPACKS_IDS=true`
3. **Independent task filtering**: Set `ENABLE_TASK_ID_FILTER=true`
4. **Combined filtering**: Enable multiple filtering features, applied in order

#### RAG Method Parameters (scripts/approachExperiment/run_rag_multiworker.sh)

**Basic RAG Parameters:**
- `DATASET`: Dataset name (default: "VersiBCB")
- `APPROACH`: Method type (default: "RAG", options: "BASELINE")
- `MODEL`: Model path
- `KNOWLEDGE_TYPE`: Knowledge type (default: "docstring", options: "srccodes")
- `INFERENCE_TYPE`: Inference type (default: "local", options: "api")

**Retrieval Configuration:**
- `USE_GENERATED_QUERIES`: Use generated queries for context retrieval (default: true)
- `GENERATED_QUERIES_FILE`: Generated queries file path
- `FIXED_DOCS_PER_QUERY`: Number of documents per query (default: 1)
- `ENABLE_QUERY_FILTERING`: Enable query dependency filtering (default: true)
- `QUERY_FILTER_STRICT_MODE`: Strict filtering mode (default: true)
- `QACacheContext_path`: Query cache context file path (overrides GENERATED_QUERIES_FILE and USE_GENERATED_QUERIES, directly loads generated QApairs)

**Knowledge Compression:**
- `ENABLE_KNOWLEDGE_COMPRESSION`: Enable knowledge compression (default: false)
- `MAX_DOC_TOKENS`: Maximum document tokens (default: 200)
- `DOC_TRUNCATE_TOKENIZER`: Truncation tokenizer path

**Other Features:**
- `ENABLE_API_NAME_STR_MATCH`: Enable API name string matching (default: false)
- `SKIP_GENERATION`: Skip generation phase (default: false)
- `APPENDCONTEXT`: Whether to append context to output (default: true)

#### Base Model Parameters (scripts/approachExperiment/run_emllm.sh)

**Model Configuration:**
- `model`: Model type (default: "llama31")
- `benchmark`: Benchmark (default: "long-bench")
- `corpus_types`: Corpus types (default: ("docstring"), options: "srccodes")
- `allow_disk_offload`: Allow disk offload (default: False)

**Output Configuration:**
- `output_dir_path`: Output directory path
- `config_file`: Configuration file path

### Usage Examples

**1. Basic LoRA Prediction:**
```bash
# Run vace task with default configuration
./scripts/approachExperiment/run_lora_multiworker.sh -w 4 -n 1
```

**2. Enable Masking Function:**
```bash
# Modify parameters in run_lora_multiworker.sh
ENABLE_TASK2MASKPACKS=true
TASK2MASKPACKS_FILE="data/task2maskpacks.json"
```

**3. Combined Filtering:**
```bash
# Enable both masking and task id filtering
ENABLE_TASK2MASKPACKS=true
ONLY_TASK2MASKPACKS_IDS=true
ENABLE_TASK_ID_FILTER=true
TASK_ID_FILTER_FILE="data/temp/taskids_filter.json"
```

**4. RAG Method:**
```bash
# Run RAG prediction
./scripts/approachExperiment/run_rag_multiworker.sh -w 2 -n 1
```

### Notes

1. **File Format Requirements:**
   - task2maskpacks file: JSON object format, keys are string task_ids, values are package name arrays
   - task_id_filter file: JSON array format, containing integer task_id list

2. **Filtering Order:**
   - Processed sample filtering
   - task2maskpacks filtering (if ONLY_TASK2MASKPACKS_IDS enabled)
   - task_id_filter filtering (if ENABLE_TASK_ID_FILTER enabled)
   - dependency mask application (during worker processing phase)


## TABLE IX: Generalization performance on unseen (L, V) combinations
Based on TABLE VII, add SPECIFIED_BENCH_PATH parameter for RAG and LoRA to specify corresponding Bench, but this situation cannot support predicting multiple tasks simultaneously, may improve later
```bash
SPECIFIED_BENCH_PATH="data/VersiBCB_Benchmark/vscc_datas_for_general.json"
```
For Memory method, just change the corresponding Dataset
```bash
DATASET="VersiBCB_vscc_GEN" # VersiBCB_vscc_BD VersiBCB_vace_BD VersiBCB_vscc_GEN VersiBCB_vace_GEN
```