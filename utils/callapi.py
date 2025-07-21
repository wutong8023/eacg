from openai import OpenAI
from config.apikey import CLOSEAI_APIKEY,DEEPSEEK_APIKEY
import aiohttp
import asyncio
from openai import AsyncOpenAI
import json
import requests
def callCloseAI(prompt,model_name):
    client = OpenAI(
        base_url='https://api.openai-proxy.org/v1',
        api_key=CLOSEAI_APIKEY,
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_name, # 如果是其他兼容模型，比如deepseek，直接这里改模型名即可，其他都不用动
    )
    return chat_completion.choices[0].message.content

def callDeepSeekAI(prompt,model_name):
    client = OpenAI(
        base_url='https://api.deepseek.com',
        api_key=DEEPSEEK_APIKEY,
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_name,
    )
    return chat_completion.choices[0].message.content

async def async_callDeepSeekAI(prompt, model_name):
    client = AsyncOpenAI(
        base_url='https://api.deepseek.com',
        api_key=DEEPSEEK_APIKEY,
    )
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_name,
        stream=False
    )
    return chat_completion.choices[0].message.content

async def async_callCloseAI(prompt, model_name):
    client = AsyncOpenAI(
        base_url='https://api.openai-proxy.org/v1',
        api_key=CLOSEAI_APIKEY,
    )
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_name,
    )
    return chat_completion.choices[0].message.content
import time
from together import Together

from huggingface_hub import InferenceClient
def _huggingface_api_inference(prompt, max_new_tokens, temperature, top_p, api_key, model_name, api_base_url):
    """HuggingFace API推理实现"""

    if not api_key or not model_name:
        raise ValueError("api_key and model_name are required for TogetherAI API")
    
    client = InferenceClient(
        provider="hf-inference",
        api_key=api_key,
    )
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=None,
                stream=False
            )
            
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message
            else:
                raise ValueError("No choices returned from TogetherAI API")
                
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"TogetherAI API request failed after {max_retries} attempts: {e}")
            time.sleep(2 ** attempt)  # 指数退避

def _togetherai_api_inference(prompt, max_new_tokens, temperature, top_p, api_key, model_name):
    """TogetherAI API推理实现"""
    if not api_key or not model_name:
        raise ValueError("api_key and model_name are required for TogetherAI API")
    
    client = Together(api_key=api_key)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=None,
                stream=False
            )
            
            if response.choices and len(response.choices) > 0:
                return response.choices[0].text
            else:
                raise ValueError("No choices returned from TogetherAI API")
                
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"TogetherAI API request failed after {max_retries} attempts: {e}")
            time.sleep(2 ** attempt)  # 指数退避
def _qdd_api_inference(prompt, max_new_tokens, temperature, top_p, api_key, model_name, api_base_url):
    '''
    钱多多的api推理
    '''
    import requests
    import json
    import time
    
    if not api_key or not model_name:
        raise ValueError("api_key and model_name are required for QDD API")
    
    # 使用提供的api_base_url，如果没有则使用默认值
    url = api_base_url if api_base_url else "https://api2.aigcbest.top/v1/chat/completions"
    
    max_retries = 1
    for attempt in range(max_retries):
        try:
            payload = json.dumps({
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False
            })
            
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(url, headers=headers, data=payload)
            response.raise_for_status()  # 检查HTTP错误
            
            response_data = response.json()
            
            if response_data.get("choices") and len(response_data["choices"]) > 0:
                return response_data["choices"][0]["message"]["content"]
            else:
                raise ValueError("No choices returned from QDD API")
                
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"QDD API request failed after {max_retries} attempts: {e}")
            time.sleep(2 ** attempt)  # 指数退避
def test_qdd():
    url = "https://api2.aigcbest.top/v1/chat/completions"

    payload = json.dumps({
    "model": "@cf/meta/llama-3.1-8b-instruct",
    "messages": [
        {
            "role": "user",
            "content": "Hello!",
            "temperature": 0.8,
        }
    ]
    })
    headers = {
    'Accept': 'application/json',
    'Authorization': 'Bearer YOUR_API_KEY_HERE',
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    dict_response = json.loads(response.text)
    print(dict_response['choices'][0]['message']['content'])
    return dict_response['choices'][0]['message']['content']

if __name__ == '__main__':
    # print(callCloseAI("Say hi", "gpt-4o-mini"))
    test_qdd()

