import os
import json
import time
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
from omegaconf import OmegaConf


class OutputManager:
    """
    统一的输出管理器，用于处理所有方法（MoE、RAG、Memory、LoRA）的输出路径和配置文件生成
    """
    
    def __init__(self, base_output_dir: str = "output/approach_eval"):
        self.base_output_dir = base_output_dir
    
    @staticmethod
    def _convert_value(value):
        """统一转换函数，处理OmegaConf对象和其他类型"""
        if OmegaConf.is_config(value):
            # 对于OmegaConf对象，转换为普通字典
            return OmegaConf.to_container(value, resolve=True)
        elif hasattr(value, '_content'):
            # OmegaConf对象的旧版本处理
            return value._content
        elif hasattr(value, '__dict__'):
            # 有__dict__的对象，递归转换
            return {k: OutputManager._convert_value(v) for k, v in value.__dict__.items()}
        elif hasattr(value, 'keys') and hasattr(value, '__getitem__'):
            # 类似字典的对象
            try:
                return {k: OutputManager._convert_value(value[k]) for k in value.keys()}
            except:
                return str(value)
        elif isinstance(value, dict):
            return {k: OutputManager._convert_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [OutputManager._convert_value(item) for item in value]
        elif isinstance(value, (str, int, float, bool, type(None))):
            return value
        else:
            # 其他类型尝试转换为字符串
            try:
                return str(value)
            except:
                return f"<non-serializable: {type(value).__name__}>"
    
    def get_approach_from_context(self, model_type: str = None, has_corpus_type: bool = False, 
                                 is_lora: bool = False) -> str:
        """
        根据上下文确定使用的方法
        
        Args:
            model_type: 模型类型
            has_corpus_type: 是否有语料库类型（表示使用RAG）
            is_lora: 是否是LoRA方法
            
        Returns:
            方法名称
        """
        if is_lora:
            return "LoRA"
        elif model_type == "em-llm":
            return "MEMORY"
        elif has_corpus_type:
            return "RAG"
        else:
            return "Memory"
    
    def generate_base_filename(self, dataset: str, model_name: str, approach: str, 
                             corpus_type: Optional[str] = None, 
                             embedding_source: Optional[str] = None,
                             max_tokens: Optional[int] = None,
                             max_dependency_num: Optional[int] = None,
                             append_srcDep: bool = False,
                             inference_type: str = "local",
                             memory_cardinfo: dict = None,
                             lora_cardinfo: dict = None,
                             useQAContext: bool = False,
                             **kwargs) -> str:
        """
        生成基础文件名
        
        Args:
            dataset: 数据集名称 对于memory而言，包括
            model_name: 模型名称
            approach: 方法名称
            corpus_type: 
            embedding_source: 嵌入源（RAG和MEMORY专用）
            max_retrieved_tokens: 最大检索token数（RAG和MEMORY专用）
            max_dependency_num: 最大依赖数
            append_srcDep: 是否添加源依赖
            inference_type: 推理类型
            memory_cardinfo: Memory卡信息
            useQAContext: 是否使用QA缓存,目前为memory专用
            **kwargs: 其他参数
            
        Returns:
            基础文件名
        """
        # 清理模型名称
        model_name = os.path.basename(model_name) if model_name else "unknown_model"
        
        # 构建基础文件名
        parts = [dataset.lower()]
        
        if  corpus_type:
            parts.append(corpus_type)
        if approach=='MEMORY' or approach=='RAG' or approach=='BASELINE':
            if embedding_source:
                parts.append(f"emb_{embedding_source}")
            if max_tokens:
                parts.append(str(max_tokens))

        # elif approach == "MoE":
        #     if max_dependency_num:
        #         parts.append(str(max_dependency_num))
        #     if append_srcDep:
        #         parts.append("appendsrcDep")
                
        parts.append(model_name)
        
        # 添加推理类型标识
        if inference_type != "local":
            parts.append(inference_type)
        # 对于memory，添加cardinfo
        if approach=='LoRA':
            if lora_cardinfo:
                for key, value in lora_cardinfo.items():
                    parts.append(f"{key}{value}")
        if approach=='MEMORY':
            if memory_cardinfo:
                for key, value in memory_cardinfo.items():
                    parts.append(f"{key}{value}")
        # 添加max_dependency标识
        if max_dependency_num:
            parts.append("maxdep"+str(max_dependency_num))
        if useQAContext:
            parts.append("QAContext")
        return "_".join(parts)
    
    def get_output_path_and_config(self, approach: str, base_filename: str, 
                                  rank: Optional[int] = None, 
                                  world_size: Optional[int] = None,
                                  newWhenExist=True,
                                  model_name_or_path=None,
                                  useQAContext=False) -> Tuple[str, str]:
        """
        获取输出路径和配置文件路径，自动处理文件名冲突
        
        Args:
            approach: 方法名称
            base_filename: 基础文件名
            rank: 进程排名（多进程时使用）
            world_size: 进程总数（多进程时使用）
            newWhenExist: 如果文件存在，是否创建新的文件
        Returns:
            (输出路径, 配置文件路径)
        """
        # 创建方法对应的输出目录
        if model_name_or_path=='/datanfs2/chenrongyi/hf/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/3ad372fc79158a2148299e3318516c786aeded6c':
            model_name = 'Mistral-7B-Instruct-v0.2'
        else:
            model_name = model_name_or_path.split("/")[-1]
        if model_name_or_path is not None:
            output_dir = os.path.join(self.base_output_dir, approach, model_name)
        else:
            output_dir = os.path.join(self.base_output_dir, approach)
        os.makedirs(output_dir, exist_ok=True)
        
        # 添加进程标识（如果是多进程）
        # if world_size and world_size > 1 and rank is not None:
        #     base_filename += f"_rank{rank}"
        
        # 检查文件是否存在，自动添加后缀
        suffix = 0

        while True:
            if suffix == 0:
                output_filename = f"{base_filename}.jsonl"
                config_filename = f"{base_filename}_config.json"

            else:
                output_filename = f"{base_filename}_{suffix}.jsonl"
                config_filename = f"{base_filename}_{suffix}_config.json"
            
            output_path = os.path.join(output_dir, output_filename)
            config_path = os.path.join(output_dir, config_filename)
            
            if not newWhenExist:
                return output_path, config_path 
            # 检查两个文件是否都不存在
            if not os.path.exists(output_path) and not os.path.exists(config_path):
                break
            suffix += 1

        return output_path, config_path
    
    def generate_config(self, approach: str, args: Any, 
                       rag_config: Optional[Any] = None,
                       additional_info: Optional[Dict] = None) -> Dict:
        """
        生成配置文件内容
        
        Args:
            approach: 方法名称
            args: 参数对象
            rag_config: RAG配置对象
            additional_info: 额外信息
            
        Returns:
            配置字典
        """
        config_data = {
            "experiment_info": {
                "approach": approach,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "processing_config": {},
            "paths": {}
        }
        
        # 添加通用实验信息
        for attr in ['datetime', 'dataset', 'datasets', 'task', 'Ban_Deprecation']:
            if hasattr(args, attr):
                value = getattr(args, attr)
                key = 'ban_deprecation' if attr == 'Ban_Deprecation' else attr
                config_data["experiment_info"][key] = self._convert_value(value)
            
        # 添加模型配置
        config_data["model_config"] = self._extract_model_config(args)
        
        # 添加生成配置
        config_data["generation_config"] = self._extract_generation_config(args)
        
        # 添加处理配置
        config_data["processing_config"] = self._extract_processing_config(args)
        
        # 根据不同方法添加特定配置
        if (approach=='BASELINE' or approach == "RAG") and rag_config:
            config_data["rag_config"] = self._extract_rag_config(args, rag_config)
        elif approach == "LoRA":
            config_data["lora_config"] = self._extract_lora_config(args)
        elif approach == "MEMORY":
            config_data["memory_config"] = self._extract_memory_config(args)
            
        # 添加额外信息
        if additional_info:
            config_data["additional_info"] = additional_info
            
        return config_data
    
    def save_config(self, config_path: str, config_data: Dict):
        """保存配置文件"""
        # 使用统一转换函数处理整个配置数据
        serializable_config = self._convert_value(config_data)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_config, f, indent=2, ensure_ascii=False)
        print(f"Configuration file saved to: {config_path}\n")
    
    def _extract_model_config(self, args) -> Dict:
        """提取模型配置"""
        config = {}
        
        # 处理不同的模型路径属性
        if hasattr(args, 'model'):
            model_attrs = ['path', 'type', 'tokenizer_path', 'use_hf_acc', 'allow_disk_offload', 
                          'vector_offload', 'disk_offload_threshold', 'vector_offload_threshold']
            
            for attr in model_attrs:
                if hasattr(args.model, attr):
                    key = 'model_path' if attr == 'path' else ('model_type' if attr == 'type' else attr)
                    config[key] = self._convert_value(getattr(args.model, attr))
                    
            # 处理特殊情况
            if hasattr(args.model, 'get'):
                if not config.get("tokenizer_path"):
                    config["tokenizer_path"] = self._convert_value(args.model.get("tokenizer_path"))
                config["attn_implementation"] = self._convert_value(args.model.get("attn_implementation", "sdpa"))
                
        # 直接从args中提取模型信息
        elif hasattr(args, 'model_name') or hasattr(args, 'model_path'):
            config["model_path"] = self._convert_value(getattr(args, 'model_path', getattr(args, 'model_name', None)))
            config["model_type"] = self._convert_value(getattr(args, 'model_type', None))
            
        # 添加推理类型和精度
        for attr in ['inference_type', 'precision']:
            if hasattr(args, attr):
                config[attr] = self._convert_value(getattr(args, attr))
            
        return config
    
    def _extract_generation_config(self, args) -> Dict:
        """提取生成配置"""
        config = {}
        
        gen_attrs = ['max_len', 'max_tokens', 'max_new_tokens', 'chunk_size', 
                    'conv_type', 'truncation', 'em_splitter', 'compute_ppl',
                    'temperature', 'top_p']
        
        for attr in gen_attrs:
            if hasattr(args, attr):
                config[attr] = self._convert_value(getattr(args, attr))
                
        return config
    
    def _extract_processing_config(self, args) -> Dict:
        """提取处理配置"""
        config = {}
        
        proc_attrs = ['rank', 'world_size', 'verbose', 'return_block_size',
                     'num_workers', 'max_concurrent_requests', 'skip_generation',
                     'overwrite_existing_results']
        
        for attr in proc_attrs:
            if hasattr(args, attr):
                config[attr] = self._convert_value(getattr(args, attr))
                
        return config
    
    def _extract_rag_config(self, args, rag_config) -> Dict:
        """提取RAG配置"""
        config = {}
        
        rag_attrs = ['corpus_type', 'knowledge_type', 'embedding_source',
                    'local_embedding_model', 'togetherai_embedding_model',
                    'rag_collection_base', 'docstring_embedding_base_path',
                    'srccode_embedding_base_path', 'docstring_corpus_path',
                    'srccode_corpus_path', 'rag_document_num', 'query_cache_dir',
                    'max_token_length', 'max_dependency_num', 'append_srcDep',
                    'together_api_key_path']
        
        for attr in rag_attrs:
            if hasattr(args, attr):
                config[attr] = self._convert_value(getattr(args, attr))
                
        return config
    
    def _extract_lora_config(self, args) -> Dict:
        """提取LoRA配置"""
        config = {}
        
        # 首先提取 args 中的基本属性
        lora_attrs = ['max_dependency_num', 'append_srcDep', 'lora_config_path',
                     'loraadaptor_save_path_base', 'dataset', 'task', 'temperature', 'top_p',
                     'knowledge_type', 'Ban_Deprecation']
        
        for attr in lora_attrs:
            if hasattr(args, attr):
                value = getattr(args, attr)
                # 处理特殊的属性名映射
                if attr == 'Ban_Deprecation':
                    config['ban_deprecation'] = self._convert_value(value)
                else:
                    config[attr] = self._convert_value(value)
        
        # 如果存在 lora_config OmegaConf 对象，只提取关键的LoRA训练参数
        if hasattr(args, 'lora_config'):
            lora_config_obj = getattr(args, 'lora_config')
            if lora_config_obj is not None:
                # 只提取核心的LoRA训练参数，避免元数据
                lora_training_params = ['r', 'alpha', 'learning_rate', 'batch_size', 'num_epochs',
                                      'target_modules', 'bias', 'dropout', 'fan_in_fan_out',
                                      'init_lora_weights', 'use_rslora', 'use_dora', 'layer_replication']
                
                for param in lora_training_params:
                    if hasattr(lora_config_obj, param):
                        config[param] = self._convert_value(getattr(lora_config_obj, param))
                    elif param in lora_config_obj:
                        config[param] = self._convert_value(lora_config_obj[param])
                
                # 处理提示模板信息
                prompt_attrs = ['versicode_vscc_prompt', 'versicode_vace_prompt', 
                              'versiBCB_vace_prompt', 'versiBCB_vscc_prompt']
                
                for prompt_attr in prompt_attrs:
                    if hasattr(lora_config_obj, prompt_attr):
                        # 只保存提示模板的前50个字符，避免配置文件过大
                        prompt_value = self._convert_value(getattr(lora_config_obj, prompt_attr))
                        if isinstance(prompt_value, str) and len(prompt_value) > 50:
                            config[prompt_attr + '_preview'] = prompt_value[:50] + "..."
                        else:
                            config[prompt_attr] = prompt_value
                            
        return config
    
    def _extract_memory_config(self, args) -> Dict:
        """提取Memory配置"""
        config = {}
        
        # 提取Memory特定配置
        if hasattr(args, 'model'):
            memory_attrs = ['disk_offload_dir', 'vector_offload_threshold',
                           'disk_offload_threshold', 'n_init', 'n_mem', 'n_local', 
                           'repr_topk', 'min_block_size', 'max_block_size']
            for attr in memory_attrs:
                if hasattr(args.model, attr):
                    config[attr] = self._convert_value(getattr(args.model, attr))
            
            # Add block_size (using max_block_size as block_size)
            if hasattr(args.model, 'max_block_size'):
                config['block_size'] = self._convert_value(getattr(args.model, 'max_block_size'))
        
        # Add chunk_size from args
        if hasattr(args, 'chunk_size'):
            config['chunk_size'] = self._convert_value(getattr(args, 'chunk_size'))
                    
        return config


# 便捷函数
def get_output_manager(base_output_dir: str = "output/approach_eval") -> OutputManager:
    """获取输出管理器实例"""
    return OutputManager(base_output_dir) 