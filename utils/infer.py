import torch
def inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 0.95
) -> str:
    """使用模型进行推理
    Args:
        model: 模型
        tokenizer: 分词器
        prompt: 输入提示文本
        max_new_tokens: 每个样本最多生成的token数
        temperature: 生成的温度系数
        top_p: 生成的top-p系数
    Returns:
        str: 生成的文本（不包含原始提示）
    """
    # 对输入进行标记化
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    # 生成文本
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    # 解码生成的文本，并移除原始提示部分
    generated_text = tokenizer.decode(
        outputs[0][input_length:], 
        skip_special_tokens=True
    )
    
    # Explicitly delete tensors to help free GPU memory
    del inputs
    del outputs
    
    return generated_text