"""
pass@k Indicator to evaluate the ability of token level
"""

import json
import os
import math

model_name = 'Llama-3-70b-chat-hf'

# result_path = f'output/togetherai/Llama-3-8b-chat-hf/downstream_application_code_token_out.json'
result_path = 'output/Versicode_Benchmark/Llama-2-7b-chat-hf_base_pred_result.json'
# result_path = f'../../dataset/final_dataset/final_generate_token_result/{model_name}/docstring.json'
# result_path = f'../../dataset/final_dataset/final_generate_token_result/{model_name}/respository.json'
# result_path = f'../../dataset/final_dataset/final_generate_token_result/{model_name}/stackoverflow.json'
def clean_model_output(model_output:str):
    if "<start>" in model_output and "<end>" in model_output:
        start_index = model_output.find("<start>") + len("<start>")
        end_index = model_output.find("<end>")
        content = model_output[start_index:end_index].replace('```python', '').replace('```', '')
    else:
        content = "no_answer"
    return content
def clean_model_output_soft(model_output: str):
    '''
    清理model_output，找到1.<start>和<end>，或是2.< start>和< end> 或是3.< start >和< end >,获取其中间的部分，并去除空格
    '''
    # 定义所有可能的开始和结束标记对
    start_tokens = ['<start>', '< start>', '< start >']
    end_tokens = ['<end>', '< end>', '< end >']
    
    # 如果输入为空，直接返回空字符串
    if not model_output:
        return ""
    
    # 尝试所有可能的标记对
    for start_token, end_token in zip(start_tokens, end_tokens):
        # 查找开始标记
        start_idx = model_output.find(start_token)
        if start_idx == -1:
            continue
            
        # 从开始标记后查找结束标记
        start_idx += len(start_token)
        end_idx = model_output[start_idx:].find(end_token)
        if end_idx == -1:
            continue
            
        # 提取内容并清理
        content = model_output[start_idx:start_idx + end_idx]
        # 去除首尾空格
        content = content.strip()
        return content
    
    # 如果没有找到任何有效的标记对，返回原始字符串去除首尾空格
    return model_output.strip()
def compute_score_k(answer:str, model_output:list, k:int):

    c = 0
    n = len(model_output)
    for output in model_output:
        if answer == output:
            c += 1
    if n-c<k:
        return 1.0

    score = 1 - (math.comb(n - c, k))/(math.comb(n, k))

    return score
def getPass_n(data_list:list):
    pass_1 = 0
    pass_3 = 0
    pass_5 = 0
    correct_set = set()
    for d in data_list:
        for i in range(len(d['model_prediction'])):
            if d['model_prediction'][i] == d['solution']:
                if i==0:
                    pass_1 += 1
                    pass_3 += 1
                    pass_5 += 1
                elif i<=2:
                    pass_3 += 1
                    pass_5 += 1
                else:
                    pass_5 += 1
                correct_set.add(d['model_prediction'][i])
                break
    return pass_1,pass_3,pass_5,correct_set
if __name__ == "__main__":

    # with open(result_path, 'r', encoding='utf-8')as fr:
    #     lodict = json.load(fr)
    # data = lodict

    # data_list = data
    # score_list = []

    # for d in data_list:
    #     answer = d['solution']
    #     # for one sample, copy it 6 times,which means 6 times of the same answer stripped of external brackets
    #     model_output_list = [clean_model_output_soft(d['model_prediction'])]#change block or token or line
    #     # for 'x.y.z',keep last split 'z'
    #     model_output_list = [element.split('.')[-1] for element in model_output_list]
    #     model_output_list = model_output_list*6
    #     temp_score = compute_score_k(answer, model_output_list[:1], 1)
    #     score_list.append(temp_score)
        


    # final_score = sum(score_list)/len(score_list)

    # print(final_score)

    # # 计算正确率
    result_paths = [
        'output/Versicode_Benchmark/starcoder-lora-rank-64-20B-tokens/Llama-2-7b-hf_IFT10000_lora_lora_pred_result_2048infer_1024IFT.json',
        'output/Versicode_Benchmark/starcoder-lora-rank-64-20B-tokens/Llama-2-7b-hf_IFT10000_lora_lora_pred_result_1024infer_1024IFT.json',
        'output/Versicode_Benchmark/starcoder-lora-rank-64-20B-tokens/Llama-2-7b-hf_IFT10000_lora_lora_pred_result_512infer_1024IFT.json',
        'output/Versicode_Benchmark/starcoder-lora-rank-64-20B-tokens_numpyCPT/Llama-2-7b-hf_IFT10000_lora_lora_pred_result.json',
        'output/Versicode_Benchmark/starcoder-lora-rank-64-20B-tokens_SIFT/Llama-2-7b-hf_SIFTtest_lora_lora_pred_result_8epoch.json',
        'output/Versicode_Benchmark/starcoder-lora-rank-64-20B-tokens_SIFT/Llama-2-7b-hf_SIFTtest_lora_lora_pred_result_16epoch.json',
        'output/Versicode_Benchmark/starcoder-lora-rank-64-20B-tokens_SIFT/Llama-2-7b-hf_SIFTshort&1024_lora_lora_pred_result.json',

    ]
    for result_path in result_paths:
        with open(result_path, 'r', encoding='utf-8')as fr:
            lodict = json.load(fr)
        data_list = lodict

        correct_len = 0
        question_len = len(data_list)
        partial_correct_len = 0
        correct_set = set()
        for d in data_list:
            if d['model_prediction'] == d['solution']:
                correct_len += 1
                correct_set.add(d['model_prediction'])
            if d['model_prediction'] in d['solution'] and len(d['model_prediction']) > 0:
                partial_correct_len += 1
        print("correct_len",correct_len)
        # print(question_len)
        print("partial_correct_len",partial_correct_len)
        print("correct_set",correct_set)
    
    # 计算pass@n
    result_paths = [
        'output/Versicode_Benchmark/starcoder-lora-rank-64-20B-tokens_SIFTshort1024&IFT10000/Llama-2-7b-hf_checkpoint_20250505_204258_epoch6_lora_lora_pred_result.json',
        'output/Versicode_Benchmark/starcoder-lora-rank-64-20B-tokens_SIFTshort1024&IFT10000/Llama-2-7b-hf_checkpoint_20250505_204258_epoch8_lora_lora_pred_result.json',
        'output/Versicode_Benchmark/starcoder-lora-rank-64-20B-tokens_SIFTshort1024&IFT10000/Llama-2-7b-hf_checkpoint_20250505_204258_epoch10_lora_lora_pred_result.json',
        'output/Versicode_Benchmark/starcoder-lora-rank-64-20B-tokens_SIFTshort1024&IFT10000/Llama-2-7b-hf_checkpoint_20250505_204258_epoch12_lora_lora_pred_result.json',
        'output/Versicode_Benchmark/starcoder-lora-rank-64-20B-tokens_SIFTshort1024&IFT10000/Llama-2-7b-hf_checkpoint_20250505_204258_epoch14_lora_lora_pred_result.json',
        'output/Versicode_Benchmark/starcoder-lora-rank-64-20B-tokens_SIFTshort1024&IFT10000/Llama-2-7b-hf_checkpoint_20250505_204258_epoch16_lora_lora_pred_result.json',
        'output/Versicode_Benchmark/randomstart_SIFTshort1024/Llama-2-7b-hf_SIFTshort1024_lora_lora_pred_result.json',
        'output/Versicode_Benchmark/randomstart_SIFTshort1024/Llama-2-7b-hf_SIFTshort1024_IFT_lora_lora_pred_result.json',
        'output/Versicode_Benchmark/randomstart_SIFTshort1024/Llama-2-7b-hf_SIFTshort1024_Wqkvo_lora_lora_pred_result.json',
        'output/Versicode_Benchmark/randomstart_SIFTshort1024/Llama-2-7b-hf_SIFTshort1024_Wudg_lora_lora_pred_result.json'
    ]
    for result_path in result_paths:
        with open(result_path, 'r', encoding='utf-8')as fr:
            lodict = json.load(fr)
        data_list = lodict
        pass_1,pass_3,pass_5,correct_set = getPass_n(data_list)
        print("pass_1",pass_1)
        print("pass_3",pass_3)
        print("pass_5",pass_5)
        print("correct_set",correct_set)