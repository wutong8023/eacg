# from main.testPassBCB import passTaskTest
import json
from config.createTarcode_config import DEPRECATION_KEYWORDS
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import argparse
import os
from datetime import datetime
def normalize_indentation(code: str) -> str:
    """
    根据第一个非空行的缩进值，对字符串的所有行减去相同缩进。
    如果所有行均无缩进，则返回原字符串。
    """
    lines = code.splitlines(keepends=True)
    if not lines:
        return code

    # 找到第一个非空行（跳过开头的空行）
    first_non_empty_line = None
    for line in lines:
        if line.strip():  # 如果该行非空（包含非空白字符）
            first_non_empty_line = line
            break

    # 如果所有行都是空行，直接返回
    if first_non_empty_line is None:
        return code

    # 计算第一个非空行的缩进（空格或制表符的数量）
    indent_len = len(first_non_empty_line) - len(first_non_empty_line.lstrip())

    # 如果首行无缩进，直接返回
    if indent_len == 0:
        return code

    # 处理每一行：减去首行缩进值（但不能超过该行原有缩进）
    processed_lines = []
    for line in lines:
        if line.strip():  # 非空行
            # 检查该行前indent_len个字符是否全是空白
            if line[:indent_len].isspace():
                processed_line = line[indent_len:]
            else:
                processed_line = line  # 如果该行缩进不足，保持原样
        else:  # 空行
            processed_line = line
        processed_lines.append(processed_line)

    return ''.join(processed_lines)
def deprecateCheck(deprecate_string):
    deprecate_keyword = DEPRECATION_KEYWORDS
    if deprecate_string and any(keyword.lower() in deprecate_string.lower() for keyword in deprecate_keyword):
        deprecate=True
    else:
        deprecate=False
    return deprecate

def clean_model_output(output):
    if "<start>" in output:
        start_index = output.find("<start>") + len("<start>")
        if "<end>" in output:
            end_index = output.find("<end>")
            content = output[start_index:end_index].replace('```python', '').replace('```', '')
        else:
            content = output[start_index:].replace('```python', '').replace('```', '')
    else:
        # 目前不要求严格start和end
        content = output.replace('```python', '').replace('```', '')
    
    return content
def clean_model_output1(output):
    '''
        以更加宽松的条件，用于codegemma的场景，因为其输出中没有<end>，但是有</end>
    '''
    if "<start>" in output:
        start_index = output.find("<start>") + len("<start>")
        if "<end>" in output:
            end_index = output.find("<end>")
            content = output[start_index:end_index].replace('```python', '').replace('```', '')
        else:
            if "</end>" in output:
                end_index = output.find("</end>")
                content = output[start_index:end_index].replace('```python', '').replace('```', '')
            else:
                content = output[start_index:].replace('```python', '').replace('```', '')
    else:
        # 目前不要求严格start和end
        content = output.replace('```python', '').replace('```', '')
    
    return content    
def clean_model_output_lora(output):
    '''
        用于lora的场景，1是因为explanation部分，2是因为###new code 的多次重复问题
    '''
    if "<start>" in output:
        start_index = output.find("<start>") + len("<start>")
        if "<end>" in output:
            end_index = output.find("<end>")
            content = output[start_index:end_index].replace('```python', '').replace('```', '')
        else:
            content = output[start_index:].replace('```python', '').replace('```', '')
    else:
        # 目前不要求严格start和end
        content = output.replace('```python', '').replace('```', '')
    # refactor new code
    refactorLabel_index = content.find("### Refactored new code")
    if refactorLabel_index != -1:
        content = content[:refactorLabel_index]
    # explanation
    explanationLabel_index = content.find("### Explanation")
    if explanationLabel_index != -1:
        content = content[:explanationLabel_index]
    return content
def clean_model_output_review(output):
    '''
        用于review的场景，1是因为explanation部分，2是因为###new code 的多次重复问题
    '''
    # refactor new code
    refactorLabel_index = output.find("### new version code after revision")
    if refactorLabel_index != -1:
        output = output[:refactorLabel_index]
    if "<start>" in output:
        start_index = output.find("<start>") + len("<start>")
        if "<end>" in output:
            end_index = output.find("<end>")

            content = output[start_index:end_index].replace('```python', '').replace('```', '')
        else:
            content = output[start_index:].replace('```python', '').replace('```', '')
    else:
        content = output.replace('```python', '').replace('```', '')
def clean_model_output_errorfix(output):
    '''
        用于review的场景，会出现### 符号，后续可能是1.循环 2.test_code 3.test result
    '''
    # refactor new code
    refactorLabel_index = output.find("###")
    if refactorLabel_index != -1:
        output = output[:refactorLabel_index]
    if "<start>" in output:
        start_index = output.find("<start>") + len("<start>")
        if "<end>" in output:
            end_index = output.find("<end>")

            content = output[start_index:end_index].replace('```python', '').replace('```', '')
        else:
            content = output[start_index:].replace('```python', '').replace('```', '')
    else:
        content = output.replace('```python', '').replace('```', '')
    return content
def clean_model_output_py_wrapper_first(output):
    '''
        优先获取首个```python```包裹的代码，未获取到再获取<start>和<end>包裹的代码
    '''
    # 首先尝试查找```python```包裹的代码
    if "```python" in output:
        start_marker = "```python"
        end_marker = "```"
        
        start_index = output.find(start_marker) + len(start_marker)
        # 查找下一个```作为结束标记
        remaining_text = output[start_index:]
        end_index = remaining_text.find(end_marker)
        
        if end_index != -1:
            content = remaining_text[:end_index].strip()
            return content
        else:
            # 如果找不到结束标记，取到末尾
            content = remaining_text.strip()
            return content
    
    # 如果没有找到```python```，则使用<start>和<end>标记
    if "<start>" in output:
        start_index = output.find("<start>") + len("<start>")
        if "<end>" in output:
            end_index = output.find("<end>")
            content = output[start_index:end_index].replace('```python', '').replace('```', '')
        else:
            if "</end>" in output:
                end_index = output.find("</end>")
                content = output[start_index:end_index].replace('```python', '').replace('```', '')
            else:
                content = output[start_index:].replace('```python', '').replace('```', '')
    else:
        # 如果都没有找到标记，直接清理markdown代码块
        content = output.replace('```python', '').replace('```', '')
    
    return content


if __name__=='__main__':
    #TODO: results中加入pass@n对应的instance id，从而定位到其对应的taskid，从而方便统计task通过率；
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='Evaluate VSCC predictions')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the prediction data JSON file')
    parser.add_argument('--ban_deprecation', type=lambda x: x.lower() == 'true', default=False,
                       help='Whether to ban deprecated functions')
    parser.add_argument('--output_dir', type=str, default='logs/evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--num_processes', type=int, default=8,
                       help='Number of processes for parallel evaluation')
    parser.add_argument('--task_type',type=str,default='vscc',
                       help='task type')
    args = parser.parse_args()