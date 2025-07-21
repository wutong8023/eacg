from typing import Dict, List, Tuple, Union
def getSubsetDep(dependency,max_dependency_num=10):
    '''
    获取dict的子集,将python加入，但排除在外，不作为数量中的1个
    '''
    # return {k:v for i,(k,v) in enumerate(dependency.items()) if i < max_dependency_num}
    cnt = 0
    if max_dependency_num == 0:
        return {}
    output_dependency = {}
    for k,v in dependency.items():
        output_dependency[k] = v
        if k!="python":
            cnt += 1
        if cnt == max_dependency_num:
            break
    return output_dependency
def combineDep(dependency,src_dependency):
    '''
    将src_dependency中的依赖添加到dependency中
    params:
        dependency: dict[pack,version]
        src_dependency: dict[pack,version]
    return:
        dict[pack,[version]]
    '''
    for pack,src_version in src_dependency.items():
        if pack not in dependency:
            raise ValueError(f"pack {pack} not in target dependency")
            # dependency[pack] = [src_version]
        else:
            if isinstance(dependency[pack], list):
                dependency[pack].append(src_version)
            else:
                origin_ver = dependency[pack]
                dependency[pack] = [origin_ver, src_version]
    # 声明每个key的list没有重复元素
    dependency = {k:list(set(v)) if isinstance(v,list) else [v] for k,v in dependency.items()}
    return dependency
def dict_to_pkg_ver_tuples(pkg_dict: Dict[str, Union[str, List[str]]]) -> List[Tuple[str, str]]:
    """
    将字典 {pkg: version} 或 {pkg: [versions]} 转换为元组列表 [(pkg, ver), ...]
    
    参数:
        pkg_dict: 键为包名，值为版本（字符串）或版本列表的字典
    
    返回:
        元组列表，每个元组格式为 (pkg, ver)
    """
    result = []
    # 首先去重，防止重复的元素
    # pkg_dict = {k:list(set(v)) for k,v in pkg_dict.items() if v is not None}
    # 转换为tuple形式
    for pkg, ver in pkg_dict.items():
        if isinstance(ver, str):
            result.append((pkg, ver))
        elif isinstance(ver, list):
            result.extend((pkg, v) for v in ver)
    return result