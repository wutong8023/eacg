import json

# 在模块级别加载数据，以减少 I/O 操作
p2d_pip = None
p2d_conda = None
d2p_pip = None  # 新增：用于存储下载包到导入包的映射
d2p_conda=None
#pipdeptree import2download
i2d_pip = None
d2i_pip = None
def load_data(type='pip'):
    global p2d_pip, p2d_conda, d2p_pip, d2p_conda  # 更新全局变量声明
    if type=='pip':
        if p2d_pip is None:  # 仅在第一次调用时加载数据
            with open("data/import2download_pip.json", "r") as f:
                p2d_pip = json.load(f)
                # 创建反向映射
                d2p_pip = {v: k for k, v in p2d_pip.items()}
    else:
        if p2d_conda is None:  # 仅在第一次调用时加载数据
            with open("data/import2download_conda.json", "r") as f:
                p2d_conda = json.load(f)
                # 创建反向映射
                d2p_conda = {v: k for k, v in p2d_conda.items()}

def pack2download(pack: str,type='pip'):
    load_data(type)  # 确保数据已加载
    if type=='pip':
        return p2d_pip[pack]
    else:
        return p2d_conda[pack]

def download2pack(download: str, type='pip') -> str:
    """
    根据下载包名返回导入包名
    Args:
        download: 下载包名
        type: 包管理器类型，'pip' 或 'conda'
    Returns:
        对应的导入包名
    """
    load_data(type)  # 确保数据已加载
    if type == 'pip':
        return d2p_pip.get(download)
    else:
        return d2p_conda.get(download)

def load_pipdeptree_data():
    global i2d_pip,d2i_pip
    if p2d_pip is None:  # 仅在第一次调用时加载数据
        with open("data/pipdeptree_import2download.json", "r") as f:
            i2d_pip = json.load(f)
            # 创建反向映射
            d2i_pip = {v: k for k, v in i2d_pip.items()}    
def i2d(name,reverse=False):
    load_pipdeptree_data()
    if reverse:
        return d2i_pip.get(name)
    else:
        return i2d_pip.get(name)
if __name__ == "__main__":
    # print(pack2download("sklearn","conda"))
    # 测试新函数
    print(download2pack("scikit-learn"))