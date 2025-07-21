import json
from openai import OpenAI
from config.apikey import DEEPSEEK_APIKEY

def process_evolve_items():
    with open("data/all_evolve_items.json", "r") as f:
        data = json.load(f)
    
    # 过滤包含"可能" "may"的evolve
    evolve_items = [item for item in data if "可能" not in item["evolve_description"] and "may" not in item["evolve_description"]]
    
    # 调用API处理每个evolve item
    client = OpenAI(api_key=DEEPSEEK_APIKEY, base_url="https://api.deepseek.com")
    
    processed_items = []
    for item in evolve_items:
        prompt = f"""请根据提供的API版本演化信息进行专业分析。按照以下要求处理：
1. 首先验证原始信息的准确性，如存在错误则修正
2. 根据修正后的信息生成更专业、清晰的版本演化描述
3. 精确标注版本变更区间

输入信息:
{item}

处理要求:
- 如确认信息无误，则优化表述使其更专业
- 如发现错误，先修正错误再生成清晰描述
- 版本区间必须来自可靠来源或明确推断
- 不确定的内容保持空白

输出格式要求:
```json
{{
    "api": "完整的API路径，如Pandas.DataFrame.plot",
    "evolve_description": "清晰的技术变更说明（如：'从同步接口改为异步接口'）或空字符串,如:In pandas 0.25.3 and before, the `dataframe.plot` method for scatter plots requires x and y columns to be numeric types, whereas in later versions this requirement is relaxed. (starting from approximately pandas 1.0.0),The method became more flexible with:Automatic conversion of compatible string representations of numbers;Support for categorical data (plotted using their internal codes);Better handling of datetime types",
    "evolve_versionInterval": ["起始版本号", "结束版本号"] ，演化发生的大致区间
}}
"""

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes API evolution information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            print(response.choices[0].message.content)
            response_content = response.choices[0].message.content
            result = response_content.split("```json")[1].split("```")[0]
            result = json.loads(result)
            
            # 更新item信息
            item["evolve_description"] = result["evolve_description"]
            item["evolve_versionInterval"] = result["evolve_versionInterval"]
            item["api"] = result["api"]
            
            processed_items.append(item)
            
        except Exception as e:
            print(f"Error processing item {item['api']}: {str(e)}")
            continue
    
        # 保存处理后的结果
        with open("data/processed_evolve_items.json", "w") as f:
            json.dump(processed_items, f, indent=2)
def enhanceEvolveItemsWithSearch():
    with open("data/EvolveItemsWithInterval.json", "r") as f:
        data = json.load(f)
    from utils.callapi import callCloseAI
    prompt_template = """请作为API演化信息验证和处理专家，通过互联网搜索和查阅相关资料，对提供的API演化信息进行验证、修正和丰富。请严格按照以下要求处理：

    1.  **理解并验证输入信息:**
        * 您将接收一个包含API演化初步信息的JSON对象（字段包括 `api`, `evolve_description`, `evolve_version`, `evolve_type`, `evolve_versionInterval`, `package`, `evolve_id`, `equivalentAPI`）。
        * 请使用互联网搜索核实这些信息的准确性。重点核实 `api` 名称、`package` 名称、`evolve_description` 中描述的具体功能变化或引入/移除行为，以及 `evolve_versionInterval` 和 `evolve_version` 指示的版本信息。
        * 特别注意 `evolve_type` 字段。如果它是 "api_not_exist"，通常意味着在某个版本之前API不存在，并在指定版本或区间被引入。请根据查证结果修正 `evolve_type`（例如，修正为 "api_introduced"）。
        * 如果原始信息描述的 API 演化事件（如某个 API 的引入、某个行为变化、某个 API 的移除等）经查证实际上并未发生、信息完全错误且无法修正，或者描述的 API 根本不存在，则判断该演化信息无效。

    2.  **修正和丰富信息:**
        * 如果原始信息有误，请根据查证结果进行修正。
        * 根据验证和修正后的信息，重新生成一个**清晰、准确、简洁**的 `evolve_description`。描述应明确指出 API 发生了什么变化（例如，具体的功能、参数或行为如何改变，API 是何时引入或移除的等）。
        * 精确确定并填写 `evolve_versionInterval`。如果 API 是在某个特定版本被引入或发生了导致演化的变化，区间应为 `[版本号, 版本号]`（例如 `["1.0.0", "1.0.0"]`）。如果演化是一个发生在某个版本范围内的过程或行为在某个区间内不同，则使用 `[起始版本号, 结束版本号]`。请查找最准确的版本信息。
        * 根据查证结果修正 `evolve_type`。常见的演化类型可以包括 "api_introduced", "api_removed", "api_behavior_change", "api_parameter_change", "api_deprecated", 等。
        * 如果 API 是被移除或废弃的，请尝试查找并填写 `equivalentAPI`，指出推荐使用的等效 API。如果不存在等效 API，则留空。

    3.  **输出格式要求:**
        * 如果演化信息经查证**有效并已处理**，请返回一个JSON对象。该JSON对象应**保持与输入的 `item` JSON 对象相同的结构和所有字段**，但已将 `evolve_description`, `evolve_version`, `evolve_type`, `evolve_versionInterval`, `equivalentAPI` (如果适用) 等字段更新为验证和修正后的准确值。`api`, `package`, `evolve_id` 等字段应保持原始值。
        * 如果演化信息经查证**无效**（即原始描述的事件不存在或完全错误），请返回一个空的JSON对象：`{{}}`。
        * 请将最终的JSON对象完整地包含在 ```json``` 代码块中返回。

    以下是待处理的API演化信息 (JSON 格式):
    {item}
    """
    with open("data/EvolveItemsWithInterval_processed.json", "r") as f:
        results = json.load(f)
    parsed_ids = [item["evolve_id"] for item in results]
    inconsistent_ids = []
    for item in data:
        evolve_id = item["evolve_id"]
        if evolve_id in parsed_ids:
            continue
        response = callCloseAI(prompt_template.format(item=item), "gpt-4o-mini-search-preview")
        result = response.split("```json")[1].split("```")[0]
        try:
            result = json.loads(result)

            
        except Exception as e:
            print(f"Error processing item {item['api']}: {str(e)}")
            continue

        if "evolve_id" not in result or result["evolve_id"] != evolve_id:
            result["evolve_id"] = evolve_id
            inconsistent_ids.append(evolve_id)
        results.append(result)
        with open("data/EvolveItemsWithInterval_processed.json", "w") as f:
            json.dump(results, f, indent=2)
    print(f"inconsistent_ids: {inconsistent_ids}")
def removeEmptyEvolveItems():
    with open("data/EvolveItemsWithInterval_processed.json", "r") as f:
        data = json.load(f)
    data = [item for item in data if len(item)>1]
    with open("data/EvolveItemsWithInterval_processed_clean.json", "w") as f:
        json.dump(data, f, indent=2)
if __name__ == "__main__":
    # process_evolve_items()
    # enhanceEvolveItemsWithSearch()
    removeEmptyEvolveItems()

