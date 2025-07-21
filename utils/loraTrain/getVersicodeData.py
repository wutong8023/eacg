# 构建generator，从对应的若干个文件中获取item
import json
import os
from utils.versicode_utils.test_token_generate_chunk import bulid_prompt as bulid_prompt_token
from utils.versicode_utils.test_token_generate_chunk import bulid_prompt as bulid_prompt_line
DOCSTRING_CORPUS_BASE = '/datanfs2/chenrongyi/data/docstring/'
VERSICODE_BENCH_PATH = 'benchmark/data/Versicode_Benchmark/'
def verifyAllPkgItemPattern(item):
    '''
        验证item是否符合要求，包括path,module,doc
    '''
    # print('path' in item)
    # print('module' in item)
    # print('doc' in item)
    return 'path' in item and 'module' in item and 'doc' in item
def itemConvertFormat(item,version):
    '''
        将item转换为符合要求的格式,加载到对应的数据栏目中,需要包括pkg,version,api_path,docstring
    '''
    item['pkg'] = item['module'].split('.')[0]
    item['version'] = version
    item['api_path'] = item['path']
    item['docstring'] = item['doc']
    del item['module']
    del item['path']
    del item['doc']
    return item
def getPkgDocstringItems(pkg,versions,generator_format=False):
    '''
        从对应的若干个文件中获取item,采用yield以实现lazy加载
        
        Args:
            pkg: 包名
            versions: 版本列表
            generator_format: 是否返回生成器(True)还是列表(False)
            
        Returns:
            如果generator_format=True，返回一个生成器对象
            如果generator_format=False，返回一个包含所有项目的列表
    '''
    # 当generator_format=False时，收集所有项目到列表中
    if not generator_format:
        items = []
        for version in versions:
            docstring_file = DOCSTRING_CORPUS_BASE + pkg + '/' + version + '.jsonl'
            try:
                with open(docstring_file,'r') as f:
                    for line in f:
                        try:
                            item = json.loads(line.strip())
                            if verifyAllPkgItemPattern(item):
                                item = itemConvertFormat(item,version)
                                items.append(item)
                        except json.JSONDecodeError as e:
                            print(f"JSON解析错误: {e}")
                            continue
            except Exception as e:
                print(f"处理文件错误: {e}")
                continue
        print(f"读取到 {len(items)} 条数据")
        return items
    
    # 当generator_format=True时，使用yield生成数据
    # for version in versions:
    #     docstring_file = DOCSTRING_CORPUS_BASE + pkg + '/' + version + '.jsonl'
    #     try:
    #         with open(docstring_file,'r') as f:
    #             for line in f:
    #                 try:
    #                     item = json.loads(line.strip())
    #                     if verifyAllPkgItemPattern(item):
    #                         item = itemConvertFormat(item,version)
    #                         yield item
    #                 except json.JSONDecodeError as e:
    #                     print(f"JSON解析错误: {e}")
    #                     continue
    #     except Exception as e:
    #         print(f"处理文件错误: {e}")
    #         continue
def getDataGeneratorLength(pkg,versions):
    '''
        获取数据生成器的长度
    '''
    i = 0
    for item in getPkgDocstringItems(pkg,versions,generator_format=True):
        i+=1
    print(f"数据生成器的长度为{i}")
    return i
# ------instructionfine tuning data load------
def getVersicodeBenchData(avoid_pkg_list,choice_file_list):
    '''
    Description:
        从对应的若干个文件中获取item,采用yield以实现lazy加载
    params:
        avoid_pkg_list: 需要避免的包列表
        choice_file_list: 需要加载的文件列表
    return:
        all_datas: 所有数据列表,list[dict]
    '''
    all_datas = []
    for file in choice_file_list:
        with open(os.path.join(VERSICODE_BENCH_PATH,file),'r') as f:
            datas = json.load(f)
        datas = datas['data']
        for data in datas:
            if data['dependency'] not in avoid_pkg_list:
                all_datas.append(data)
    return all_datas
def constructQAPairsFromBenchData(all_datas):
    '''
        从all_datas中构造QA对
    '''
    qa_pairs = []
    for data in all_datas:
        if data['granularity'] == 'token':
            prompt = bulid_prompt_token(data['dependency'] + data['version'],data['description'],data['masked_code'])
            qa_pairs.append((prompt,data['answer']))
        elif data['granularity'] == 'line':
            prompt = bulid_prompt_line(data['dependency'] + data['version'],data['description'],data['masked_code'])
            qa_pairs.append((prompt,data['masked_line']))
        else:
            raise ValueError(f"Granularity {data['granularity']} not supported")
        

    return qa_pairs
# ---main---
def testGetVersicodeBenchData():
    '''
        获取一条数据
    '''
    with open('data/versicodeRequiredVersion.json','r') as f:
        requiredPkgVersions = json.load(f)
    # print(getPkgDocstringItems('numpy',requiredPkgVersions['numpy']))
    i = 0
    for item in getPkgDocstringItems('numpy',requiredPkgVersions['numpy'],generator_format=True):
        print(item)
        i+=1
        if i > 2:
            break

def GetQAPairsFromBenchData(avoid_pkgs_list,choice_files_list,sample_nums=None):
    '''
        从Versicode_Benchmark中获取QA对,
        params:
            avoid_pkgs_list: 需要避免的包列表,list[str]
            choice_files_list: 需要加载的文件列表,list[list[str]]
            sample_nums: 需要采样的数量,list[int]
        
    '''
    all_datas = []
    for i,choice_files in enumerate(choice_files_list):
        datas = getVersicodeBenchData(avoid_pkgs_list[i],choice_files)
        datas = datas[:sample_nums[i]]
        all_datas.extend(datas)
    qa_pairs = constructQAPairsFromBenchData(all_datas)
    # if sample_num:
    #     qa_pairs = qa_pairs[:sample_num]
    return qa_pairs

def GetQAPairsFromFlatIFTData(filepath, sample_num=None):
    '''
    从flat_IFTdata.json中获取QA对
    params:
        filepath: 文件路径
        sample_num: 需要采样的数量
    returns:
        qa_pairs: QA对列表
    '''
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} examples from {filepath}")
        
        # If sample_num is specified, limit the number of examples
        if sample_num and sample_num < len(data):
            data = data[:sample_num]
            print(f"Sampled {sample_num} examples from the dataset")
        
        # Convert the data to the format expected by QADataset
        qa_pairs = []
        for item in data:
            # Format: (query, answer)
            # For flat_IFTdata format, we combine instruction and input as query
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            output = item.get('output', '')
            
            # Combine instruction and input as the query
            if input_text:
                query = f"{instruction}\n\n{input_text}"
            else:
                query = instruction
                
            qa_pairs.append((query, output))
            
        return qa_pairs
    except Exception as e:
        print(f"Error loading flat IFT data: {e}")
        return []

def testGetQAPairsFromBenchData():
    qa_pairs = GetQAPairsFromBenchData(['numpy'],['code_completion/downstream_application_code/downstream_application_code_token.json'])
    print(qa_pairs[0])
def testGetDataGeneratorLength():
    with open('data/versicodeRequiredVersion.json','r') as f:
        requiredPkgVersions = json.load(f)
    length = getDataGeneratorLength('numpy',requiredPkgVersions['numpy'])
    print(length)

def testGetQAPairsFromFlatIFTData():
    qa_pairs = GetQAPairsFromFlatIFTData('data/flat_IFTdata.json', sample_num=5)
    print(f"Loaded {len(qa_pairs)} QA pairs")
    print("First QA pair:")
    print("Query:", qa_pairs[0][0])
    print("Answer:", qa_pairs[0][1])

if __name__ == '__main__':
    # dataGenerator = getPkgDocstringItems('numpy',['1.17.4'])
    # testGetDataGeneratorLength()
    testGetQAPairsFromFlatIFTData()
    