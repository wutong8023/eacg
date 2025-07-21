'''
    由mvBCBbuilder复制而来
'''
import pickle
import os
import re
import json
import logging
def DeprecationStatistics():
    # 统计deprecate_log文件夹中，有多少个文件内容不为空
    count = 0
    for file in os.listdir("logs/deprecate_log"):
        pattern = r'^task_BigCodeBench_\d+_deprecate.log$'
        if re.match(pattern, file):
            with open(os.path.join("logs/deprecate_log", file), "r") as f:
                data = f.read()
            if len(data) > 10:
                count+=1
    print(count)

def getLibVersionCount():
    def appendLibVersions(libVersions,dep):
        for lib,version in dep.items():
            if lib not in libVersions:
                libVersions[lib] = []
            libVersions[lib].append(version)
        return libVersions
    vace_path= "data/output/VersiBCB_Benchmark/vace_datas.json"
    vscc_path= "data/output/VersiBCB_Benchmark/vscc_datas.json"
    BDvace_path= "data/output/VersiBCB_Benchmark/vace_datas_for_warning.json"
    BDvscc_path= "data/output/VersiBCB_Benchmark/vscc_datas_for_warning.json"
    id2metadata_path = "data/output/metadata.json"
    id2BDmetadata_path = "data/output/metadataForWarning.json"
    with open(vace_path,"r") as f:
        vace_datas = json.load(f)
    with open(vscc_path,"r") as f:
        vscc_datas = json.load(f)
    with open(BDvace_path,"r") as f:
        BDvace_datas = json.load(f)
    with open(BDvscc_path,"r") as f:
        BDvscc_datas = json.load(f)
    LibVersions = {}
    all_vscc_datas = vscc_datas +BDvscc_datas
    for data in all_vscc_datas:
        dep = data['dependency']
        appendLibVersions(LibVersions,dep)
    # remove redundancy
    for lib,versions in LibVersions.items():
        LibVersions[lib] = list(set(versions))
    # 计算lib数量和version总量
    lib_count = len(LibVersions)
    version_count = sum(len(versions) for versions in LibVersions.values())
    print(f"lib数量: {lib_count}, version总量: {version_count}")
    # 计算每个lib的version数量
    lib_version_count = {lib: len(versions) for lib, versions in LibVersions.items()}
    print(lib_version_count)
    # 计算metadatasize
    with open(id2metadata_path,"r") as f:
        id2metadata = json.load(f)
    with open(id2BDmetadata_path,"r") as f:
        id2BDmetadata = json.load(f)
    metadata_size = sum(len(metadata) for metadata in id2metadata.values())
    BDmetadata_size = sum(len(metadata) for metadata in id2BDmetadata.values())
    print(f"metadata_size: {metadata_size}, BDmetadata_size: {BDmetadata_size} sum：{metadata_size+BDmetadata_size}")
    # 计算instance总数
    print(f"vace_datas: {len(vace_datas)}, BDvace_datas: {len(BDvace_datas)}, vscc_datas: {len(vscc_datas)}, BDvscc_datas: {len(BDvscc_datas)}")
    print(f"instance总数: {len(vace_datas)+len(BDvace_datas)+len(vscc_datas)+len(BDvscc_datas)}")
    # 分别计算四个子任务的average input tokens num
    def getAverageInputTokensNum(metadata):
        '''
        将获取的每个metadata转换为str，按照空格数计算token数
        Args:
            metadata: dict or list, 包含需要计算token数的数据
        Returns:
            float: 平均token数
        '''
        total_tokens = 0
        count = 0
        
        def count_tokens(item):
            if isinstance(item, (str, int, float, bool)):
                # 将item转换为字符串并计算空格数（加1为实际token数）
                return len(str(item).split())
            elif isinstance(item, (list, tuple)):
                return sum(count_tokens(x) for x in item)
            elif isinstance(item, dict):
                # 对字典的key和value都计算token数
                return sum(count_tokens(k) + count_tokens(v) for k, v in item.items())
            else:
                return 0
        
        if isinstance(metadata, dict):
            # 如果是字典，遍历所有值
            for meta in metadata.values():
                tokens = count_tokens(meta)
                total_tokens += tokens
                count += 1
        elif isinstance(metadata, list):
            # 如果是列表，直接遍历
            for meta in metadata:
                tokens = count_tokens(meta)
                total_tokens += tokens
                count += 1
                
        return total_tokens / count if count > 0 else 0
    # 对四个任务分别计算
    vace_average_input_tokens_num = getAverageInputTokensNum(vace_datas)
    vscc_average_input_tokens_num = getAverageInputTokensNum(vscc_datas)
    BDvace_average_input_tokens_num = getAverageInputTokensNum(BDvace_datas)
    BDvscc_average_input_tokens_num = getAverageInputTokensNum(BDvscc_datas)
    print(f"vace_average_input_tokens_num: {vace_average_input_tokens_num}, vscc_average_input_tokens_num: {vscc_average_input_tokens_num}, BDvace_average_input_tokens_num: {BDvace_average_input_tokens_num}, BDvscc_average_input_tokens_num: {BDvscc_average_input_tokens_num}")
def getLibVersions():
    def appendLibVersions(libVersions,dep):
        for lib,version in dep.items():
            if lib not in libVersions:
                libVersions[lib] = []
            libVersions[lib].append(version)
        return libVersions
    vace_path= "data/output/VersiBCB_Benchmark/vace_datas.json"
    vscc_path= "data/output/VersiBCB_Benchmark/vscc_datas.json"
    BDvace_path= "data/output/VersiBCB_Benchmark/vace_datas_for_warning.json"
    BDvscc_path= "data/output/VersiBCB_Benchmark/vscc_datas_for_warning.json"
    id2metadata_path = "data/output/metadata.json"
    id2BDmetadata_path = "data/output/metadataForWarning.json"
    with open(vace_path,"r") as f:
        vace_datas = json.load(f)
    with open(vscc_path,"r") as f:
        vscc_datas = json.load(f)
    with open(BDvace_path,"r") as f:
        BDvace_datas = json.load(f)
    with open(BDvscc_path,"r") as f:
        BDvscc_datas = json.load(f)
    LibVersions = {}
    all_vscc_datas = vscc_datas +BDvscc_datas
    for data in all_vscc_datas:
        dep = data['dependency']
        appendLibVersions(LibVersions,dep)
    # remove redundancy
    for lib,versions in LibVersions.items():
        LibVersions[lib] = list(set(versions))
    return LibVersions
def checkExist(lib,version):
    packBase = '/mnt/d/codes/repos/'
    if os.path.exists(os.path.join(packBase,lib,version)):
        return True
    else:
        return False
def appendPack2Versions(pack2versions,lib,version):
    if lib not in pack2versions:
        pack2versions[lib] = []
    pack2versions[lib].append(version)
def appendTag2Pack(tag2pack,tag,pack):
    if tag not in tag2pack:
        tag2pack[tag] = set()
    tag2pack[tag].add(pack)
def getPackVersion2Tag(packVersions):
    """
    将包版本转换为对应的 tag
    
    Args:
        packVersions: dict, key为pack_name, value为list,list中为版本号
        
    Returns:
        dict, key为pack_name, value为dict,内层dict的key为version,value为对应的tag
    """
    # 读取 pack2tags.json 获取 tag 映射
    with open("data/pack2tags.json", "r") as f:
        packTags = json.load(f)
        
    # 对于每个包的每个版本,找到最匹配的 tag
    packVersion2Tags = {} # dict[pack_name][version] = tag
    for pack_name, versions in packVersions.items():
        packVersion2Tags[pack_name] = {}
        for version in versions:
            matching_tags = []
            if pack_name not in packTags:
                print(f"pack_name {pack_name} not in packTags")
                continue
                
            # 找到所有包含该版本号的 tag
            for tag in packTags[pack_name]:
                if version in tag:
                    matching_tags.append(tag)
                    
            # 如果有匹配的 tag,选择最短的一个(通常是最规范的格式)
            if matching_tags:
                packVersion2Tags[pack_name][version] = min(matching_tags, key=len)
                
    return packVersion2Tags

def getAlldata(data_range=['vace','vscc','BDvace','BDvscc']):
    vace_path= "benchmark/data/VersiBCB_Benchmark/vace_datas.json"
    vscc_path= "benchmark/data/VersiBCB_Benchmark/vscc_datas.json"
    BDvace_path= "benchmark/data/VersiBCB_Benchmark/vace_datas_for_warning.json"
    BDvscc_path= "benchmark/data/VersiBCB_Benchmark/vscc_datas_for_warning.json"
    vace_datas = []
    vscc_datas = []
    BDvace_datas = []
    BDvscc_datas = []
    if 'vace' in data_range:
        with open(vace_path,"r") as f:
            vace_datas = json.load(f)
    if 'vscc' in data_range:
        with open(vscc_path,"r") as f:
            vscc_datas = json.load(f)
    if 'BDvace' in data_range:
        with open(BDvace_path,"r") as f:
            BDvace_datas = json.load(f)
    if 'BDvscc' in data_range:
        with open(BDvscc_path,"r") as f:
            BDvscc_datas = json.load(f)
    data = vace_datas + vscc_datas + BDvace_datas + BDvscc_datas
    return data
def getTaskNum():
    data = getAlldata()
    task_ids = [data['taskid'] for data in data]
    task_ids = list(set(task_ids))
    print(f"task_ids: {len(task_ids)}")
def getDomainDistribution(data):
    '''
    获取每个domain的实例数量，并计算每个domain的占比(每个域的分布是根据每个任务出现的特定于域的库的频率计算的。例如，“Computation” 中的 “63%” 表示 BigCodeBench 中有 63% 的任务至少使用一个计算库。)
    '''
    DomainCount ={'Cryptography':0}
    # data = getAlldata()
    with open("utils/BCBanalysis/task2domain.json","r") as f:
        task2domain = json.load(f)
    taskids = [d['taskid'] for d in data]
    for taskid in taskids:
        domain = task2domain[taskid]
        for d in set(domain):
            if d not in DomainCount:
                DomainCount[d] = 0
            DomainCount[d] += 1
    domain_percentage = {k: v / len(taskids) for k, v in DomainCount.items()}
    print(sorted(DomainCount.items(),key=lambda x: x[1],reverse=True))
    print(sorted(domain_percentage.items(),key=lambda x: x[1],reverse=True))
    return DomainCount


def getID2BCBtaskid(data):
    '''
    获取vace等数据的id2BCBtaskid
    '''
    id2BCBtaskid = {}
    for d in data:
        id2BCBtaskid[d['id']] = d['taskid']
    return id2BCBtaskid
def calculatePassRate(id2domains,DomainCount,idPassInfo):
    '''
    Description:
        计算每个domain的pass rate
    Args:
        id2domains: dict[id,domains]
        DomainCount: dict[domain,count]
        idPassInfo: dict[id,pass_info]  pass_info = {'pass@1':bool,'pass@3':bool}
    Returns:
        dict[domain,pass_rate_dict]  pass_rate_dict = {'pass@1':pass_rate,'pass@3':pass_rate}
    '''
    domain_pass_rate = {}
    domain_pass_at_1 = {}
    domain_pass_at_3 = {}
    for id,pass_info in idPassInfo.items():
        domains = id2domains[int(id)]
        for d in set(domains):
            if d not in domain_pass_rate:
                domain_pass_rate[d] = {'pass@1':0,'pass@3':0}
            if pass_info['pass@1']:
                domain_pass_rate[d]['pass@1'] += 1
            if pass_info['pass@3']:
                domain_pass_rate[d]['pass@3'] += 1
    for domain in domain_pass_rate:
        domain_pass_rate[domain]['pass@1'] = domain_pass_rate[domain]['pass@1'] / DomainCount[domain]
        domain_pass_rate[domain]['pass@3'] = domain_pass_rate[domain]['pass@3'] / DomainCount[domain]
    return domain_pass_rate
def getDomainPassRate():
    '''
        获取每个domain的pass rate.首先获取指定domain存在的taskid数量，然后获取对应通过的taskid数量，相除
    '''
    data = getAlldata(['vace'])
    # 获取id2domains
    id2BCBtaskid = getID2BCBtaskid(data)
    with open("utils/BCBanalysis/task2domain.json","r") as f:
        task2domain = json.load(f)
    id2domains = {id:task2domain[BCBtaskid] for id,BCBtaskid in id2BCBtaskid.items()}
    # 获取在每个domain的pass rate
    DomainCount = getDomainDistribution(data)
    # domain_pass_at_1 = {}
    # domain_pass_at_3 = {}
    Model_domain_pr = {'vace':{'deepseek-chat':{},'Llama-3.3-70B-Instruct-Turbo-Free':{}},
                       'BDvace':{'deepseek-chat':{},'Llama-3.3-70B-Instruct-Turbo-Free':{}}}
    #读取对应pass信息，计算
    with open('logs/custom_evaluation_results/20250417/deepseek-chat_vace_datas_eval.json','r') as f:
        results = json.load(f)
    idPassInfo = results['results']['task_results']
    pass_rate = calculatePassRate(id2domains,DomainCount,idPassInfo)
    Model_domain_pr['vace']['deepseek-chat'] = pass_rate
    with open('logs/custom_evaluation_results/20250417/Llama-3.3-70B-Instruct-Turbo-Free_vace_datas_eval.json','r') as f:
        results = json.load(f)
    idPassInfo = results['results']['task_results']
    pass_rate = calculatePassRate(id2domains,DomainCount,idPassInfo)
    Model_domain_pr['vace']['Llama-3.3-70B-Instruct-Turbo-Free'] = pass_rate
    
    with open('data/output/domain_results.json','w') as f:
        json.dump(Model_domain_pr,f)
from utils.getDatasetPacks import getPackVersions
def load_and_aggregate_package_versions(benchmark_paths):
    """
    从多个benchmark文件中加载并汇总包版本信息
    
    Args:
        benchmark_paths: list of str, benchmark数据文件路径列表
        
    Returns:
        dict: 汇总的包版本信息 {pkg: [version1, version2, ...]}
    """
    aggregated_pack_versions = {}
    
    for benchmark_path in benchmark_paths:
        logging.info(f"正在处理benchmark文件: {benchmark_path}")
        try:
            with open(benchmark_path, "r") as f:
                datas = json.load(f)
            
            pack_versions = getPackVersions(datas)
            logging.info(f"从 {benchmark_path} 加载了 {len(pack_versions)} 个包")
            
            # 汇总包版本信息
            for pkg, versions in pack_versions.items():
                if pkg not in aggregated_pack_versions:
                    aggregated_pack_versions[pkg] = []
                aggregated_pack_versions[pkg].extend(versions)
                
        except FileNotFoundError:
            logging.error(f"Benchmark文件不存在: {benchmark_path}")
            continue
        except Exception as e:
            logging.error(f"处理benchmark文件时出错 {benchmark_path}: {e}")
            continue
    
    # 去重处理
    for pkg in aggregated_pack_versions:
        aggregated_pack_versions[pkg] = list(set(aggregated_pack_versions[pkg]))
    
    total_packages = len(aggregated_pack_versions)
    total_versions = sum(len(versions) for versions in aggregated_pack_versions.values())
    logging.info(f"汇总完成: 总计 {total_packages} 个包, {total_versions} 个包版本组合")
    
    # 打印详细统计信息
    logging.info("=== 包版本统计 ===")
    for pkg, versions in sorted(aggregated_pack_versions.items()):
        logging.info(f"  {pkg}: {len(versions)} 个版本 {versions}")
    
    return aggregated_pack_versions

def temp():
    LibVersions = getLibVersions()
    pack2versions = {}
    tag2pack = {}
    with open('data/pack2tagformat.json','r') as f:
        pack2tagformat = json.load(f)
    for lib,versions in LibVersions.items():
        for version in versions:
            if  not checkExist(lib,version) and lib!='python':
                appendPack2Versions(pack2versions,lib,version)
                appendTag2Pack(tag2pack,pack2tagformat[lib],lib)
    packVersion2Tags = getPackVersion2Tag(pack2versions)
    # with open('data/temp/pack2versions.json','w') as f:
    #     json.dump(pack2versions,f)
    # print(tag2pack)
    print(packVersion2Tags)
    with open('data/temp/packVersion2Tags.json','w') as f:
        json.dump(packVersion2Tags,f)
if __name__ == "__main__":
    getDomainPassRate()