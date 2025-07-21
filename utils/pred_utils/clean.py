def clean_model_output(output):
    '''
    用于最基本的场景，优先匹配start和end，然后对于python wrapper进行匹配
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
        constraint：实际上只用于Llama-3.1-8B,因为其指令输出的特殊性
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
def clean_model_output_lora_v1(content):
    '''
        用于lora的场景，1是因为explanation部分，2是因为###new code 的多次重复问题
        constraint：实际上只用于Llama-3.1-8B,因为其指令输出的特殊性
    '''
    # 先过滤重复部分
    refactorLabel_index = content.find("###")
    if refactorLabel_index != -1:
        content = content[:refactorLabel_index]
    # 获取在```python```包裹的代码
    if "```python" in content:
        start_marker = "```python"
        end_marker = "```"
        start_index = content.find(start_marker) + len(start_marker)
        remaining_text = content[start_index:]
        end_index = remaining_text.find(end_marker)
        if end_index != -1:
            content = remaining_text[:end_index]
        else:
            content = remaining_text
        return content
    else:
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
def clean_model_output_py_wrapper_first(output,remove_example_usage=True):
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
    if remove_example_usage:
        # 删除Example_usage之后的所有内容
        example_usage_index = content.find("Example usage")
        if example_usage_index != -1:
            content = content[:example_usage_index]
    return content
def clean_model_output_finalcode(output):
    '''
        用于finalcode的场景,找到<finalcode>和</finalcode>包裹的代码。然后再获取```python```包裹的代码进行处理
    '''
    # 获取<finalcode>和</finalcode>包裹的代码
    finalcode_start_index = output.find("<finalcode>") + len("<finalcode>")
    finalcode_end_index = output.find("</finalcode>")
    if finalcode_start_index != -1 and finalcode_end_index != -1:
        finalcode = output[finalcode_start_index:finalcode_end_index]
    else:
        finalcode = output
    # 获取```python```包裹的代码
    if "```python" in finalcode:
        start_marker = "```python"
        end_marker = "```"
        start_index = finalcode.find(start_marker) + len(start_marker)
        remaining_text = finalcode[start_index:]
        end_index = remaining_text.find(end_marker)
        if end_index != -1:
            finalcode = remaining_text[:end_index]
        else:
            finalcode = remaining_text
    return finalcode