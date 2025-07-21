import json
import os
def getDataStatistic(all_data):
    for data_piece in all_data:
        library = data_piece['dependency']
        version = data_piece['version']
        if library not in library_data_count:
            library_data_count[library] = 0
        library_data_count[library] += 1
        if library not in library_version_count:
            library_version_count[library] = {}
        if version not in library_version_count[library]:
            library_version_count[library][version] = 0
        library_version_count[library][version] += 1
    return library_data_count,library_version_count
if __name__ == "__main__":
    with open('benchmark/data/Versicode_Benchmark/code_completion/downstream_application_code/downstream_application_code_token.json','r') as f:
        token_data = json.load(f)
    with open('benchmark/data/Versicode_Benchmark/code_completion/downstream_application_code/downstream_application_code_line.json','r') as f:
        line_data = json.load(f)
    library_data_count ={}
    library_version_count = {}
    all_data = token_data["data"]+line_data["data"] # line+token级别
    library_data_count,library_version_count = getDataStatistic(all_data)
    # token级别数量
    # line级别数量
    save_folder_base = 'utils/getDatasetInfo/versicode_info'
    with open(os.path.join(save_folder_base,'library_data_count.json'),'w') as f:
        json.dump(library_data_count,f)
    with open(os.path.join(save_folder_base,'library_version_count.json'),'w') as f:
        json.dump(library_version_count,f)
# torch numpy transformers  
# line+token   [('torch', 7588), ('numpy', 2365), ('transformers', 1778), ('jax', 590), ...]
# print(library_version_count['torch']) 
# token_level torch{'==0.3.1': 38, '==1.13.0': 10, '==1.1.0': 33, '==1.6.0': 20, '==1.9.0': 7, '>=1.7.0': 109, '==1.12.1+cu113': 5, '==2.0.1': 69, '==1.11.0+cu102': 1, '==1.12.1': 6, '==1.8.2': 4, '>=1.2.0': 5, '==1.10.0': 14, '>=1.9.1+cu111': 5, '==1.13.1': 8, '>=1.8.1': 26, '>=1.7': 2128, '>=1.0.1': 10, '>=1.5.0': 437, '>=1.9.0': 38, '>=1.7.1,<3': 6, '>=1.10.0': 544, '==1.4.0': 6, '>=2.1': 192, '>=1.7.1': 29, '==1.5.0+cu101': 6, '>=1.4.0': 13, '==2.0.0+cu117': 25}
