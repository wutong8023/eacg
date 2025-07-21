import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

def load_vscc_data():
    """加载vscc数据以获取原始信息"""
    with open("data/VersiBCB_Benchmark/vscc_datas.json", "r") as f:
        return json.load(f)

def convert_concurrent_to_versibcb_format(input_file, output_file):
    """
    将concurrent结果转换为versibcb的统一格式
    
    Args:
        input_file: concurrent结果的JSONL文件路径
        output_file: 输出的versibcb格式JSON文件路径
    """
    
    # 加载vscc原始数据
    vscc_datas = load_vscc_data()
    
    # 读取concurrent结果
    print(f"正在读取concurrent结果: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        concurrent_results = [json.loads(line) for line in f if line.strip()]
    
    # 转换为versibcb格式
    versibcb_format = {}
    
    for result in concurrent_results:
        id = result["id"]
        
        # 跳过失败的结果
        if result["status"] != "success":
            print(f"跳过ID {id}: 状态为 {result['status']}")
            continue
        
        # 获取原始vscc数据
        original_vscc_data = None
        for vscc_data in vscc_datas:
            if str(vscc_data["id"]) == str(id):
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
                "target_dependency": original_vscc_data.get("dependency", {})
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

def analyze_conversion_results(versibcb_format):
    """分析转换结果"""
    print("\n=== 转换结果分析 ===")
    
    total_queries = 0
    id_count = len(versibcb_format)
    
    for id_str, data in versibcb_format.items():
        query_count = len(data.get("queries", []))
        total_queries += query_count
        print(f"ID {id_str}: {query_count} 个查询")
    
    print(f"\n总结:")
    print(f"总ID数: {id_count}")
    print(f"总查询数: {total_queries}")
    print(f"平均每个ID查询数: {total_queries/id_count:.2f}")

def create_sample_output(input_file, output_file, sample_size=5):
    """
    创建样本输出用于验证格式
    
    Args:
        input_file: concurrent结果的JSONL文件路径
        output_file: 输出的样本文件路径
        sample_size: 样本大小
    """
    
    # 加载vscc数据
    vscc_datas = load_vscc_data()
    
    # 读取concurrent结果
    with open(input_file, 'r', encoding='utf-8') as f:
        concurrent_results = [json.loads(line) for line in f if line.strip()]
    
    # 只取前几个成功的结果
    sample_results = []
    for result in concurrent_results:
        if result["status"] == "success" and len(sample_results) < sample_size:
            sample_results.append(result)
    
    # 转换为versibcb格式
    versibcb_format = {}
    
    for result in sample_results:
        id = result["id"]
        
        # 获取原始vscc数据
        original_vscc_data = None
        for vscc_data in vscc_datas:
            if vscc_data["id"] == id:
                original_vscc_data = vscc_data
                break
        
        if not original_vscc_data:
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
    
    # 保存样本
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(versibcb_format, f, indent=2, ensure_ascii=False)
    
    print(f"样本文件已保存到: {output_file}")
    return versibcb_format

if __name__ == "__main__":
    # 输入和输出文件路径
    input_file = "data/temp/query_results_concurrent.jsonl"
    output_file = "data/temp/versibcb_format_queries.json"
    sample_file = "data/temp/versibcb_format_queries_sample.json"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        print("请先运行并发处理脚本生成结果")
        sys.exit(1)
    
    print("=== 开始格式转换 ===")
    
    # 创建样本文件用于验证
    print("\n1. 创建样本文件...")
    sample_format = create_sample_output(input_file, sample_file, sample_size=3)
    
    # 执行完整转换
    print("\n2. 执行完整转换...")
    versibcb_format = convert_concurrent_to_versibcb_format(input_file, output_file)
    
    # 分析结果
    analyze_conversion_results(versibcb_format)
    
    print(f"\n=== 转换完成 ===")
    print(f"完整结果: {output_file}")
    print(f"样本文件: {sample_file}") 