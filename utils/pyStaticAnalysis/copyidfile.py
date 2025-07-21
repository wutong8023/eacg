import json
def read_json_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data_item = json.loads(line)
            data.append(data_item)
    return data

# def write_json_file(data, file_path):
#     with open(file_path, 'w') as file:
#         json.dump(data, file)

def dumpCode2file(item, file_path):
    with open(file_path, 'w') as file:
        file.write(item['code'])

if __name__ == "__main__":
    id_choice = 6
    data = read_json_file('data/temp/pyright_test_results.jsonl')
    for item in data:
        if item['id'] == id_choice:
            dumpCode2file(item, 'data/temp/testcode.py')
            break