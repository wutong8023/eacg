from config.apikey import DEEPSEEK_APIKEY
import json
from utils.callapi import async_callDeepSeekAI
from tqdm import tqdm
import asyncio
import aiofiles

async def process_item(item, prompt_template, IFT_datas):
    if item["evolve_id"] in IFT_datas:
        return None
    
    try:
        response = await async_callDeepSeekAI(prompt_template.format(item=item), "deepseek-chat")
        response = response.split("```json")[1].split("```")[0]
        print(response)
        response = json.loads(response)
        return item["evolve_id"], response
    except Exception as e:
        print(f"Error processing item {item['api']}: {str(e)}")
        return None

async def constructSIFTdata():
    '''
    Description:
        为每个API演化构造10条IFT数据，用于模型训练
    '''
    prompt_template = """
    请根据提供的API演化信息，构造10条指令微调数据，用于模型训练，以使得模型能够有效应用演化信息进行代码生成和代码版本迁移任务。

    API演化信息:
    {item}
    请将生成的指令微调数据放置在```json```代码块中返回,方便用户直接使用json.loads(json_str)转换为json对象。
    确保返回的IFT数据都是英文形式，具体格式要求如下：
    ```json
    [
    {{
    "instruction": "...",
    "input": "...",
    "output": "..."
    }},
    {{
    "instruction": "...",
    "input": "...",
    "output": "..."
    }},
    ...
    ]
    ```
    """
    
    # 读取现有数据
    async with aiofiles.open("data/IFTdata.json", "r") as f:
        content = await f.read()
        IFT_datas = json.loads(content)
    
    ids = list(IFT_datas.keys())
    
    # 读取需要处理的数据
    async with aiofiles.open("data/EvolveItemsWithInterval_processed_clean.json", "r") as f:
        content = await f.read()
        data = json.loads(content)
    
    # 创建任务列表
    tasks = []
    for item in data:
        if item["evolve_id"] not in ids:
            task = process_item(item, prompt_template, IFT_datas)
            tasks.append(task)
    
    # 并发执行任务
    results = await asyncio.gather(*tasks)
    
    # 处理结果
    for result in results:
        if result is not None:
            id, response = result
            IFT_datas[id] = response
    
    # 保存更新后的数据
    async with aiofiles.open("data/IFTdata.json", "w") as f:
        await f.write(json.dumps(IFT_datas, indent=4))

if __name__ == "__main__":
    asyncio.run(constructSIFTdata())

# Error processing item pandas.DataFrame.isnull: Unterminated string starting at: line 26 column 19 (char 1891)
# Error processing item str.format: Unterminated string starting at: line 11 column 19 (char 674)
# Error processing item matplotlib.axes.Axes.bar: Unterminated string starting at: line 41 column 19 (char 2566)
# Error processing item scipy.stats.mode: Unterminated string starting at: line 21 column 19 (char 1334)
# Error processing item pandas.DataFrame.empty: Unterminated string starting at: line 21 column 19 (char 1494)
# Error processing item numpy.float: Invalid \escape: line 41 column 180 (char 2365)
# Error processing item numpy.random.rand: Unterminated string starting at: line 21 column 19 (char 1380)
# Error processing item pandas.DataFrame.from_dict: Unterminated string starting at: line 11 column 19 (char 637)
# Error processing item matplotlib.pyplot.ylabel: Unterminated string starting at: line 21 column 19 (char 1839)
# Error processing item flask_restful.Api.add_resource: Unterminated string starting at: line 26 column 19 (char 2226)
# Error processing item scipy.stats.zscore: Unterminated string starting at: line 21 column 19 (char 1781)
# Error processing item matplotlib.axes.Axes.bar: Unterminated string starting at: line 51 column 19 (char 5836)
