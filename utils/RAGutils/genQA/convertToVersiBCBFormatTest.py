import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

def load_vscc_data():
    """加载vscc数据以获取原始信息"""
    with open("data/VersiBCB_Benchmark/vscc_datas.json", "r") as f:
        return json.load(f)

def convert_test_data_to_versibcb_format(input_file, output_file):
    """
    将测试数据转换为versibcb格式进行验证
    """
    
    # 加载vscc原始数据
    vscc_datas = load_vscc_data()
    
    # 读取测试结果
    print(f"正在读取测试结果: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        test_results = [json.loads(line) for line in f if line.strip()]
    
    # 转换为versibcb格式
    versibcb_format = {}
    
    for result in test_results:
        id = result["id"]
        
        # 跳过失败的结果
        if result["status"] != "success":
            print(f"跳过ID {id}: 状态为 {result['status']}")
            continue
        
        # 获取原始vscc数据
        original_vscc_data = None
        for vscc_data in vscc_datas:
            if vscc_data["id"] == id:
                original_vscc_data = vscc_data
                break
        
        if not original_vscc_data:
            print(f"警告: 未找到ID {id} 对应的原始vscc数据")
            continue
        
        # 转换queries格式
        converted_queries = []
        for query in result.get("queries", []):
            converted_query = {
                "query": query.get("query_content", ""),
                "target_api": query.get("target_api_path", "")
            }
            converted_queries.append(converted_query)
        
        # 构建versibcb格式
        versibcb_entry = {
            "queries": converted_queries,
            "raw_response": f"Generated {len(converted_queries)} queries for ID {id}",
            "original_data": {
                "description": original_vscc_data.get("description", ""),
                "origin_dependency": original_vscc_data.get("origin_dependency", ""),
                "target_dependency": original_vscc_data.get("target_dependency", {})
            }
        }
        
        versibcb_format[str(id)] = versibcb_entry
        print(f"转换ID {id}: {len(converted_queries)} 个查询")
    
    # 保存结果
    print(f"正在保存到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(versibcb_format, f, indent=2, ensure_ascii=False)
    
    print(f"转换完成! 共处理 {len(versibcb_format)} 个ID")
    return versibcb_format

def show_converted_format(versibcb_format):
    """显示转换后的格式示例"""
    print("\n=== 转换格式示例 ===")
    
    for id_str, data in list(versibcb_format.items())[:2]:  # 只显示前2个
        print(f"\nID {id_str}:")
        print(f"  查询数量: {len(data['queries'])}")
        print(f"  原始数据描述: {data['original_data']['description'][:100]}...")
        print(f"  目标依赖: {data['original_data']['target_dependency']}")
        
        if data['queries']:
            print(f"  查询示例:")
            for i, query in enumerate(data['queries'][:2]):  # 只显示前2个查询
                print(f"    查询 {i+1}:")
                print(f"      内容: {query['query'][:80]}...")
                print(f"      目标API: {query['target_api']}")

def compare_with_original_format():
    """与原始versibcb格式进行比较"""
    print("\n=== 格式对比 ===")
    
    # 读取原始versibcb格式示例
    original_file = "data/generated_queries/versibcb_vscc_queries_with_code_review.json"
    if os.path.exists(original_file):
        with open(original_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        # 获取一个示例
        sample_id = list(original_data.keys())[0]
        sample_data = original_data[sample_id]
        
        print(f"原始格式示例 (ID {sample_id}):")
        print(f"  查询数量: {len(sample_data['queries'])}")
        print(f"  查询格式: {type(sample_data['queries'][0])}")
        if isinstance(sample_data['queries'][0], dict):
            print(f"  查询字段: {list(sample_data['queries'][0].keys())}")
        print(f"  原始数据字段: {list(sample_data['original_data'].keys())}")
    else:
        print("原始格式文件不存在，跳过对比")

if __name__ == "__main__":
    # 输入和输出文件路径
    input_file = "data/temp/query_results_test.jsonl"
    output_file = "data/temp/versibcb_format_test.json"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 测试文件不存在: {input_file}")
        print("请先运行测试版并发处理脚本")
        sys.exit(1)
    
    print("=== 开始测试格式转换 ===")
    
    # 执行转换
    versibcb_format = convert_test_data_to_versibcb_format(input_file, output_file)
    
    # 显示转换结果
    show_converted_format(versibcb_format)
    
    # 与原始格式对比
    compare_with_original_format()
    
    print(f"\n=== 测试转换完成 ===")
    print(f"测试结果: {output_file}")
    
    if versibcb_format:
        print("✅ 格式转换成功!")
        print("可以查看生成的文件验证格式是否正确")
    else:
        print("❌ 没有成功转换任何数据") 