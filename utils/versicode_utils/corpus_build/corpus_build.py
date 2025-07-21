from openai import OpenAI
from tests.testAPIindoc import apiPartname2fullPaths
import json
import concurrent.futures
from tqdm import tqdm
import google.generativeai as genai

print(apiPartname2fullPaths)
with open('/home/chenrongyi/API_KEYSET/deepseek.txt','r') as f:
    api_key = f.read()
with open('/home/chenrongyi/API_KEYSET/closeai.txt','r') as f:
    closeai_key = f.read()
client = OpenAI(api_key=api_key,base_url='https://api.deepseek.com/')
genai.configure(
    api_key=closeai_key,
    transport="rest",
    client_options={"api_endpoint": "https://api.openai-proxy.org/google"},
)
# prompt_template0 = """
# 生成5段{{'description':,'code':,'masked_token':}}格式的以numpy中的{api_name}作为masked_token的代码。description用来描述代码的用途，code需要包含{api_name}的调用,可以有多次调用，{api_name}以<token_mask>进行替换。每个instance以<instance></instance>进行包裹。确保填充masked_token后代码完整可运行。所有例子应该为英文。
# """
# prompt_template0 = """
# 生成5段{{'description':,'code':}}格式的使用了numpy中的{api_name}的代码。description用来描述代码的用途，code需要包含{api_name}的调用,可以存在1次或多次对{api_name}的调用，每个instance以<instance></instance>进行包裹。所有instance都应该为全英文，description不应该泄露对于使用的api {api_name}的任何信息，而应指出代码实现的功能，从而便于我们构建FIM数据集。
# """
prompt_template = """
请生成5条使用了numpy api {api_name}的FIM数据，其中每条数据格式包括3个字段，description描述代码的功能，masked_code为将{api_name}转为<masked_token>的代码，masked_token为{api_name}。以json格式返回,确保使用全英文，包裹在```json  ```中。以下是一条FIM数据的示例
{{
            "description": "This code defines a test class called BaseTest that inherits from absltest.TestCase. It also includes a test method called test_simple_functionality, which asserts that the method called system_under_test returns the value 1.",
            "masked_code": "@absltest.<masked_token>(\"Shared functionality\")\nclass BaseTest(absltest.TestCase):\n    def test_simple_functionality(self):\n        self.assertEqual(self.system_under_test.method(), 1)",
            "masked_token": "skipThisClass",
}}

"""
def generate_code(api_name,api_provider='closeai'):
    prompt = prompt_template.format(api_name=api_name)
    try:
        if api_provider == 'deepseek':
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
        elif api_provider == 'closeai':
            model = genai.GenerativeModel("gemini-2.0-flash-001")
            response = model.generate_content(prompt)
        return {"api_name": api_name, "content": response.text, "success": True}
    except Exception as e:
        return {"api_name": api_name, "content": str(e), "success": False}

def getInstancesFromResponse(response:str):
    pass

def process_api(api_name):
    result = generate_code(api_name)
    return result

if __name__ == "__main__":
    output_file = 'output/Versicode_Benchmark/numpy_corpus.jsonl'
    
    # Get total number of APIs
    total_apis = len(apiPartname2fullPaths)
    results = []
    
    # Number of parallel workers
    max_workers = 8  # Adjust based on rate limits and system capabilities
    
    print(f"Starting processing of {total_apis} APIs with {max_workers} parallel workers")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and create a dictionary mapping futures to their api_names
        future_to_api = {executor.submit(process_api, api_name): api_name 
                         for api_name in apiPartname2fullPaths}
        
        # Process results as they complete with progress bar
        with open(output_file, 'w') as f:
            for future in tqdm(concurrent.futures.as_completed(future_to_api), total=total_apis, desc="Generating API code examples"):
                api_name = future_to_api[future]
                try:
                    result = future.result()
                    if result["success"]:
                        print(f"Successfully processed: {api_name}")
                        f.write(json.dumps({'api_name': api_name, 'result': result["content"]})+'\n')
                    else:
                        print(f"Failed to process {api_name}: {result['content']}")
                except Exception as exc:
                    print(f"Error processing {api_name}: {exc}")
    
    print(f"Completed processing {total_apis} APIs. Results saved to {output_file}")
