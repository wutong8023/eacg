import json
exactmatch_info_template = "\nExtra_info:The {target_api} exist in target_dependency where {pack} version is{version}. "
ragmatch_info_template = "\nExtra_info:The {target_api} do not exist in target_dependency where {pack} version is {version}. "

def loadJsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def saveJsonl(data, file_path):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def extract_package_from_api(target_api):
    """从target_api中提取包名，例如从pandas.DataFrame.iterrows提取pandas"""
    parts = target_api.split('.')
    if len(parts) > 0:
        return parts[0]
    return None

def get_package_version(dependencies, package_name):
    """从dependencies中获取指定包的版本"""
    return dependencies.get(package_name, "unknown")

def fixData(data):
    '''
    如果解析后的path或aliases命中了target_api_path,则改为exact_api_match,并且将对应Extra Info加入到context中
    '''
    try:
        # 创建数据副本以避免修改原始数据
        fixed_data = data.copy()
        
        # 提取关键信息
        target_api = data.get('target_api', '')
        context_str = data.get('context', '')
        dependencies = data.get('dependencies', {})
        retrieval_method = data.get('retrieval_method', '')
        
        if not target_api or not context_str:
            return fixed_data
        
        # 解析context字段中的JSON
        try:
            context_obj = json.loads(context_str)
        except json.JSONDecodeError:
            # 如果context不是有效的JSON，直接返回原数据
            return fixed_data
        
        # 提取path和aliases
        context_path = context_obj.get('path', '')
        context_aliases = context_obj.get('aliases', [])
        
        # 检查是否匹配target_api
        is_exact_match = False
        
        # 检查path是否匹配
        if context_path == target_api:
            is_exact_match = True
        
        # 检查aliases是否有匹配的
        if not is_exact_match and context_aliases:
            for alias in context_aliases:
                if alias == target_api:
                    is_exact_match = True
                    break
        
        # 提取包名和版本信息
        package_name = extract_package_from_api(target_api)
        package_version = get_package_version(dependencies, package_name) if package_name else "unknown"
        
        # 根据匹配结果决定使用哪个模板
        if is_exact_match:
            # 精确匹配：改为exact_api_match，使用exactmatch模板
            fixed_data['retrieval_method'] = 'exact_api_match'
            extra_info = exactmatch_info_template.format(
                target_api=target_api,
                pack=package_name,
                version=f" {package_version}" if package_version != "unknown" else " unknown"
            )
        else:
            # 非精确匹配：保持原有retrieval_method，使用ragmatch模板
            extra_info = ragmatch_info_template.format(
                target_api=target_api,
                pack=package_name,
                version=package_version if package_version != "unknown" else "unknown"
            )
        
        # 将extra_info添加到context中
        # 如果context已经包含doc字段，将extra_info添加到doc的末尾
        fixed_data["extra_info"] = extra_info
        
        # # 更新context字段
        # fixed_data['context'] = json.dumps(context_obj, ensure_ascii=False)
        
        return fixed_data
        
    except Exception as e:
        print(f"Error processing data item {data.get('id', 'unknown')}: {e}")
        return data  # 出错时返回原数据

if __name__ == "__main__":
    datas = loadJsonl("data/temp/contexts/versibcb_vscc_contexts_strmatch.jsonl")
    output_datas = []
    
    print(f"Processing {len(datas)} data items...")
    
    for i, data in enumerate(datas):
        output_data = fixData(data)
        output_datas.append(output_data)
        
        # 打印一些处理结果的统计信息
        if i < 5:  # 打印前5个的详细信息
            print(f"\nItem {i+1}:")
            print(f"  ID: {data.get('id', 'unknown')}")
            print(f"  Target API: {data.get('target_api', 'unknown')}")
            print(f"  Original retrieval_method: {data.get('retrieval_method', 'unknown')}")
            print(f"  Fixed retrieval_method: {output_data.get('retrieval_method', 'unknown')}")
            
            # 检查是否添加了extra_info
            original_context = data.get('context', '')
            fixed_context = output_data.get('context', '')
            if len(fixed_context) > len(original_context):
                print(f"  Added extra info: Yes")
            else:
                print(f"  Added extra info: No")
    
    # 统计retrieval_method的变化
    exact_matches = sum(1 for item in output_datas if item.get('retrieval_method') == 'exact_api_match')
    print(f"\nSummary:")
    print(f"  Total items processed: {len(output_datas)}")
    print(f"  Items changed to exact_api_match: {exact_matches}")
    
    saveJsonl(output_datas, "data/temp/contexts/versibcb_vscc_contexts_strmatch_fix.jsonl")
    print(f"Results saved to: data/temp/contexts/versibcb_vscc_contexts_strmatch_fix.jsonl")