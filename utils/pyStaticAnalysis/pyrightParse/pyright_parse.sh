# python tests/pyStaticAnalysis/pyrightParse/pyright_parser.py \
#     output/approach_eval/BASELINE/Llama-3.1-8B-Instruct/versibcb_vscc_Llama-3.1-8B-Instruct_maxdep10.jsonl \
#     data/VersiBCB_Benchmark/vscc_datas_for_warning.json \
#     data/temp/combined_errors_vscc_depre_new.json \
#     --max-workers 4
python utils/pyStaticAnalysis/pyrightParse/pyright_parser.py \
    output/approach_eval/BASELINE/Llama-3.1-8B-Instruct/versibcb_vace_bd_Llama-3.1-8B-Instruct_maxdep10.jsonl \
    data/VersiBCB_Benchmark/vscc_datas_for_warning.json \
    data/temp/combined_errors_vscc_depre_new.json \
    --max-workers 4
