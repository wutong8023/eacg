import json
def loadJsonl(file_path):
    datas = []
    with open(file_path, 'r') as f:
        for line in f:
            datas.append(json.loads(line))
    return datas
def writeJsonl(file_path,datas):
    with open(file_path, 'w') as f:
        for data in datas:
            f.write(json.dumps(data)+'\n')