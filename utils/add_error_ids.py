import json

def add_error_ids(input_file, output_file):
    """
    为每个error_info添加唯一的error_id
    """
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    error_id_counter = 1
    
    for item in data:
        if "error_infos" in item:
            for error_info in item["error_infos"]:
                error_info["error_id"] = f"error_{error_id_counter:04d}"
                error_id_counter += 1
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"已处理 {len(data)} 个数据项，添加了 {error_id_counter-1} 个error_id")
    print(f"结果已保存到 {output_file}")

if __name__ == "__main__":
    input_file = "data/temp/combined_errors_vscc.json"
    output_file = "data/temp/combined_errors_vscc_with_ids.json"
    add_error_ids(input_file, output_file) 