import re
import hashlib
import json

class pathConfigurator:
    def __init__(self):
        pass
    @classmethod
    def normalize_scientific_notation(cls,s):
        """
        将科学计数法字符串标准化（例如 "1e-06" → "1e-6"）
        
        参数:
            s (str): 输入的科学计数法字符串（如 "1.2e-03"、"5E+5"、"1e-06"）
            
        返回:
            str: 标准化后的字符串（如 "1.2e-3"、"5e5"、"1e-6"）
        """
        # 正则匹配科学计数法（兼容大小写e/E、正负号、整数/小数）
        match = re.fullmatch(
            r"(?P<coeff>[-+]?\d*\.?\d+)[eE](?P<exp>[-+]?\d+)", 
            s.strip()
        )
        if not match:
            return s  # 如果不是科学计数法，原样返回
        
        # 提取系数和指数
        coeff = match.group("coeff")
        exp = int(match.group("exp"))
        
        # 标准化：移除多余的零和小数点（如 "1.0e-6" → "1e-6"）
        if "." in coeff:
            coeff = coeff.rstrip("0").rstrip(".")  # 清理末尾的零和小数点
        
        # 合并结果（统一使用小写e，指数去掉前导零）
        return f"{coeff}e{exp}"
    def getPath(self,config,pkg,version,model_name=None,knowledge_type=None,pred_args=None):
        '''
            获取lora模型路径
        Args:
            config: dict, 配置信息
            pkg: str, 包名
            version: str, 版本号
            model_name: str, 模型名
            knowledge_type: str, 知识类型
            pred_args: dict, 预测参数
        '''
        #TODO:pred_args的合并，因为其实感觉也可以作为override加入config中，但是感觉还是有点麻烦
        # module,layer,r,alpha
        module_dict = {
            "up_proj":"up",
            "down_proj":"down",
            "gate_proj":"gate",
            "q_proj":"q",
            "k_proj":"k",
            "v_proj":"v",
            "out_proj":"o",
        }
        module = [module_dict[m] for m in config["target_modules"]]
        layer_interval = "-".join([str(config["target_layers"][0]),str(config["target_layers"][-1])])
        r = config["r"]
        alpha = config["alpha"]
        knowledge_type = config["knowledge_type"] if knowledge_type is None else knowledge_type
        learning_rate = self.normalize_scientific_notation(str(config["learning_rate"]))
        train_data_percentage = config["traindata_percentage"]
        epoch = config["num_epochs"]    
        module_str = "_".join(str(m) for m in module)
        precision = config["precision"]
        loraadaptor_save_path_base = config["loraadaptor_save_path_base"] if pred_args is None or not hasattr(pred_args, 'loraadaptor_save_path_base') or pred_args.loraadaptor_save_path_base is None else pred_args.loraadaptor_save_path_base
        if model_name:
            path = f"{loraadaptor_save_path_base}/{model_name}/{pkg}/{pkg}_{version}_{knowledge_type}_{module_str}_{layer_interval}_{r}_{alpha}_{learning_rate}_{epoch}_{train_data_percentage}_{precision}"
        else:
            path = f"{loraadaptor_save_path_base}/{pkg}/{pkg}_{version}_{knowledge_type}_{module_str}_{layer_interval}_{r}_{alpha}_{learning_rate}_{epoch}_{train_data_percentage}_{precision}"
        return path
    def getIFTPath(self, config, pkg, version, model_name=None, knowledge_type=None, pred_args=None, ift_suffix="IFT", ift_type=None):
        """
        获取IFT模型路径，基于原始LoRA模型路径，在其子目录下创建IFT文件夹
        
        Args:
            config: dict, 配置信息
            pkg: str, 包名
            version: str, 版本号
            model_name: str, 模型名
            knowledge_type: str, 知识类型
            pred_args: dict, 预测参数
            ift_suffix: str, IFT子文件夹名称，默认为"IFT"
            ift_type: str, IFT类型标识，用于区分不同策略训练的IFT模型
            
        Returns:
            str: IFT模型的完整路径
        """
        # 获取基础LoRA模型路径
        base_lora_path = self.getPath(config, pkg, version, model_name, knowledge_type, pred_args)
        
        # 根据是否有IFT类型决定文件夹名称
        if ift_type:
            ift_folder_name = f"{ift_suffix}_{ift_type}"
        else:
            ift_folder_name = ift_suffix
        
        # 在基础LoRA模型路径下创建IFT子文件夹
        ift_path = f"{base_lora_path}/{ift_folder_name}"
        
        return ift_path
    def getIFTPathWithDetailedConfig(self, config, pkg, version, ift_train_config, 
                                   model_name=None, knowledge_type=None, pred_args=None, 
                                   ift_type=None, data_strategy="same_minor_version", 
                                   data_source_hash=None):
        """
        获取带有详细配置信息的IFT模型路径，实现差异化命名策略
        
        Args:
            config: dict, 基础配置信息
            pkg: str, 包名
            version: str, 版本号
            ift_train_config: dict, IFT训练专用配置
            model_name: str, 模型名
            knowledge_type: str, 知识类型
            pred_args: dict, 预测参数
            ift_type: str, IFT类型标识
            data_strategy: str, 数据加载策略
            data_source_hash: str, 数据源哈希值（可选）
            
        Returns:
            tuple: (str, dict) - (IFT模型路径, 命名策略信息)
        """
        # 获取基础LoRA模型路径
        base_lora_path = self.getPath(config, pkg, version, model_name, knowledge_type, pred_args)
        
        # 构建IFT配置标识符
        ift_components = []
        
        # 1. 基础IFT标识
        ift_components.append("IFT")
        
        # 2. IFT类型（如果指定）
        if ift_type:
            ift_components.append(f"type_{ift_type}")
        
        # 3. 数据策略
        if data_strategy != "same_minor_version":  # 只在非默认策略时添加
            strategy_map = {
                "all_versions": "allver",
                "closest_n": "closestn",
                "same_minor_version": "sameminor"
            }
            strategy_short = strategy_map.get(data_strategy, data_strategy[:8])
            ift_components.append(f"data_{strategy_short}")
        
        # 4. 训练关键参数
        ift_lr = self.normalize_scientific_notation(str(ift_train_config.get("learning_rate", "1e-6")))
        ift_epochs = ift_train_config.get("num_epochs", 16)
        ift_batch_size = ift_train_config.get("batch_size", 1)
        target_batch_size = ift_train_config.get("target_batch_size", 2)
        
        # 只在非默认值时添加参数标识
        if ift_lr != "1e-6":
            ift_components.append(f"lr{ift_lr}")
        if ift_epochs != 16:
            ift_components.append(f"ep{ift_epochs}")
        if ift_batch_size != 1:
            ift_components.append(f"bs{ift_batch_size}")
        if target_batch_size != 2:
            ift_components.append(f"tbs{target_batch_size}")
        
        # 5. 数据源标识（如果提供）
        if data_source_hash:
            ift_components.append(f"ds{data_source_hash[:8]}")
        
        # 构建最终的IFT文件夹名称
        ift_folder_name = "_".join(ift_components)
        
        # 生成完整路径
        ift_path = f"{base_lora_path}/{ift_folder_name}"
        
        # 创建命名策略信息
        naming_strategy = {
            "base_lora_path": base_lora_path,
            "ift_folder_name": ift_folder_name,
            "ift_full_path": ift_path,
            "naming_components": {
                "ift_type": ift_type,
                "data_strategy": data_strategy,
                "learning_rate": ift_lr,
                "num_epochs": ift_epochs,
                "batch_size": ift_batch_size,
                "target_batch_size": target_batch_size,
                "data_source_hash": data_source_hash
            },
            "naming_rules": {
                "format": "IFT_[type_X]_[data_strategy]_[lr{lr}]_[ep{epochs}]_[bs{batch_size}]_[tbs{target_batch_size}]_[ds{data_hash}]",
                "description": "Components in [] are only included when different from defaults",
                "defaults": {
                    "data_strategy": "same_minor_version",
                    "learning_rate": "1e-6",
                    "num_epochs": 16,
                    "batch_size": 1,
                    "target_batch_size": 2
                }
            }
        }
        
        return ift_path, naming_strategy
    def getIFTPathWithCustomConfig(self, base_lora_path, ift_config=None, ift_suffix="IFT"):
        """
        基于已有的LoRA模型路径，使用自定义IFT配置生成IFT模型路径
        
        Args:
            base_lora_path: str, 基础LoRA模型路径
            ift_config: dict, IFT训练的特定配置（如不同的学习率、epoch等）
            ift_suffix: str, IFT子文件夹名称，默认为"IFT"
            
        Returns:
            str: IFT模型的完整路径
        """
        if ift_config is None:
            # 没有自定义配置，使用默认IFT文件夹
            return f"{base_lora_path}/{ift_suffix}"
        else:
            # 有自定义配置，生成包含配置信息的IFT路径
            ift_lr = self.normalize_scientific_notation(str(ift_config.get("learning_rate", "default")))
            ift_epochs = ift_config.get("num_epochs", "default")
            ift_batch_size = ift_config.get("batch_size", "default")
            
            ift_config_str = f"{ift_suffix}_lr{ift_lr}_ep{ift_epochs}_bs{ift_batch_size}"
            return f"{base_lora_path}/{ift_config_str}"
    def checkLoRAModelExists(self, config, pkg, version, model_name=None, knowledge_type=None, pred_args=None):
        """
        检查LoRA模型是否存在
        
        Args:
            config: dict, 配置信息
            pkg: str, 包名
            version: str, 版本号
            model_name: str, 模型名
            knowledge_type: str, 知识类型
            pred_args: dict, 预测参数
            
        Returns:
            tuple: (bool, str) - (是否存在, 模型路径)
        """
        import os
        
        lora_path = self.getPath(config, pkg, version, model_name, knowledge_type, pred_args)
        adapter_config_path = os.path.join(lora_path, "adapter_config.json")
        adapter_model_path = os.path.join(lora_path, "adapter_model.safetensors")
        
        # 检查两个关键文件是否都存在
        exists = os.path.exists(adapter_config_path) and os.path.exists(adapter_model_path)
        
        return exists, lora_path
    def checkIFTModelExists(self, config, pkg, version, model_name=None, knowledge_type=None, pred_args=None, ift_suffix="IFT", ift_type=None):
        """
        检查IFT模型是否存在
        
        Args:
            config: dict, 配置信息
            pkg: str, 包名
            version: str, 版本号
            model_name: str, 模型名
            knowledge_type: str, 知识类型
            pred_args: dict, 预测参数
            ift_suffix: str, IFT子文件夹名称
            ift_type: str, IFT类型标识，用于区分不同策略训练的IFT模型
            
        Returns:
            tuple: (bool, str) - (是否存在, IFT模型路径)
        """
        import os
        
        ift_path = self.getIFTPath(config, pkg, version, model_name, knowledge_type, pred_args, ift_suffix, ift_type)
        adapter_config_path = os.path.join(ift_path, "adapter_config.json")
        adapter_model_path = os.path.join(ift_path, "adapter_model.safetensors")
        
        # 检查两个关键文件是否都存在
        exists = os.path.exists(adapter_config_path) and os.path.exists(adapter_model_path)
        
        return exists, ift_path
    def getPredictionPath(self,config,benchmark,predict_task_name):
        module_dict = {
            "up_proj":"up",
            "down_proj":"down",
            "gate_proj":"gate",
            "q_proj":"q",
            "k_proj":"k",
            "v_proj":"v",
            "out_proj":"o",
        }
        modules_str = "".join(module_dict[m] for m in config["target_modules"])
        layer_interval = f"{config['target_layers'][0]}-{config['target_layers'][-1]}"
        r = config["r"]
        alpha = config["alpha"]
        learning_rate = config["learning_rate"]
        train_data_percentage = config["traindata_percentage"]
        precision = config["precision"]
        epoch = config["num_epochs"]            
        path = f"output/{benchmark}/{predict_task_name}_lora_pred_result_{modules_str}_{layer_interval}_{r}_{alpha}_{learning_rate}_{epoch}_{train_data_percentage}_{precision}.json"
        return path
    @classmethod
    def compute_data_source_hash(cls, data_sources, sample_num=None, data_max_length=None):
        """
        计算数据源配置的哈希值，用于IFT模型路径差异化
        
        Args:
            data_sources: dict, 数据源配置
            sample_num: int, 采样数量
            data_max_length: int, 数据最大长度
            
        Returns:
            str: 8位哈希值
        """
        # 创建一个包含关键数据源信息的字典
        hash_dict = {
            "flat_ift_paths": sorted(data_sources.get("flat_ift_paths", [])) if data_sources.get("flat_ift_paths") else [],
            "use_benchmark_data": data_sources.get("use_benchmark_data", False),
            "sample_num": sample_num,
            "data_max_length": data_max_length
        }
        
        # 如果有benchmark配置，也加入哈希计算
        if data_sources.get("benchmark_config"):
            benchmark_config = data_sources["benchmark_config"]
            hash_dict["benchmark_config"] = {
                "avoid_pkgs_list": sorted(benchmark_config.get("avoid_pkgs_list", [])),
                "choice_files_list": sorted([sorted(files) for files in benchmark_config.get("choice_files_list", [])]),
                "sample_nums": sorted(benchmark_config.get("sample_nums", []))
            }
        
        # 转换为JSON字符串并计算哈希
        hash_string = json.dumps(hash_dict, sort_keys=True)
        hash_obj = hashlib.md5(hash_string.encode('utf-8'))
        return hash_obj.hexdigest()[:8]