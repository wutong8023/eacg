# 使用所有选项
OUTPUT_DIR="logs/custom_evaluation_results/gpt-3.5-turbo"
MODEL_NAME="gpt-3.5-turbo"
python utils/eval.py \
    --data_path data/model_predictions/${MODEL_NAME}/vscc_datas_for_warning.json \
    --ban_deprecation True \
    --output_dir ${OUTPUT_DIR} \
    --num_processes 4 \ 
    --task_type vscc

# python utils/evaluate/evaluation.py \
#     --data_path data/model_predictions/${MODEL_NAME}/vace_datas_for_warning.json \
#     --ban_deprecation True \
#     --output_dir ${OUTPUT_DIR} \
#     --num_processes 6 \
#     --task_type vace
# python utils/evaluate/evaluation.py \
#     --data_path data/model_predictions/${MODEL_NAME}/vscc_datas.json \
#     --ban_deprecation False \
#     --output_dir ${OUTPUT_DIR} \
#     --num_processes 4 \
#     --task_type vscc
# python utils/evaluate/evaluation.py \
#     --data_path data/model_predictions/${MODEL_NAME}/vscc_datas_for_warning.json \
#     --ban_deprecation True \
#     --output_dir ${OUTPUT_DIR} \
#     --num_processes 4 \
#     --task_type vscc
