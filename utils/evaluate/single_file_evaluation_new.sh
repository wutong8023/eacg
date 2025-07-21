python utils/evaluate/single_file_evaluation_withnew.py \
    --prediction_file /datanfs2/chenrongyi/codes/EM-LLM-model/output/testgen/BASELINE/Llama-3.1-8B-Instruct/versibcb_vace_Llama-3.1-8B-Instruct_maxdep10.jsonl \
    --benchmark_dir data/VersiBCB_Benchmark \
    --benchmark_file vace_datas_for_general_eval.json \
    --ban_deprecation false \
    --clean_func py_wrapper_first \
    --num_processes 4 \
    --output_file data/evalResults/gen/baseline/llama3.1_8b_instruct_vace.json \
    --log_file logs/llama_evaluation.log
# python utils/evaluate/single_file_evaluation_withnew.py \
#     --prediction_file /datanfs2/chenrongyi/codes/EM-LLM-model/output/testgen/RAG/Llama-3.1-8B-Instruct/versibcb_vace_docstring_emb_local_4000_Llama-3.1-8B-Instruct_maxdep10.jsonl \
#     --benchmark_dir data/VersiBCB_Benchmark \
#     --benchmark_file vace_datas_for_general_eval.json \
#     --ban_deprecation false \
#     --clean_func py_wrapper_first \
#     --num_processes 4 \
#     --output_file data/evalResults/gen/rag/llama3.1_8b_instruct_vace.json \
#     --log_file logs/llama_evaluation.log
# python utils/evaluate/single_file_evaluation_withnew.py \
#     --prediction_file /datanfs2/chenrongyi/codes/EM-LLM-model/output/testgen/BASELINE/Llama-3.1-8B/versibcb_vace_Llama-3.1-8B_maxdep10.jsonl \
#     --benchmark_dir data/VersiBCB_Benchmark \
#     --benchmark_file vace_datas_for_general_eval.json \
#     --ban_deprecation false \
#     --clean_func basic \
#     --num_processes 4 \
#     --output_file data/evalResults/gen/baseline/llama3.1_8b_vace.json \
#     --log_file logs/llama_evaluation.log
