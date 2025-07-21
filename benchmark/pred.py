# https://github.com/THUDM/LongBench/blob/main/pred.py
import os
from datasets import load_from_disk, load_dataset
import torch
import json
from tqdm import tqdm
import argparse
from omegaconf import OmegaConf
from em_llm.utils import patch_hf, GreedySearch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import sys
import time
import numpy as np
from datetime import datetime
import pprint as pp
from huggingface_hub import login
from pathlib import Path
from benchmark.config.code.config import VSCC_LOW_BOUND, VSCC_HIGH_BOUND
from utils.RAGutils.rag_config import RAGConfig
from utils.output_manager import OutputManager
import logging
logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

import argparse
import torch
from omegaconf import OmegaConf
from config.paths import (
    QUERY_CACHE_DIR,
    RAG_COLLECTION_BASE,
    DOCSTRING_EMBEDDING_PATH,
    SRCCODE_EMBEDDING_PATH,
    DOCSTRING_PATH,
    SRCCODE_PATH,
    API_KEY_PATH
)

# Load Hugging Face token from config or environment variable
def load_hf_token():
    # Try environment variable first
    if os.environ.get("HF_TOKEN"):
        return os.environ.get("HF_TOKEN")
    
    # Try config file
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "auth.yaml")
    if os.path.exists(config_path):
        try:
            auth_config = OmegaConf.load(config_path)
            if hasattr(auth_config, "hf_token") and auth_config.hf_token:
                return auth_config.hf_token
        except Exception as e:
            print(f"Warning: Failed to load auth config: {e}")
    
    # Try legacy path as fallback
    token_path = os.path.expanduser("~/tokens/hf.txt")
    if os.path.exists(token_path):
        with open(token_path, "r") as file:
            return file.read().strip()
    
    print("Warning: No Hugging Face token found. Some operations may fail.")
    return None

# Login to Hugging Face
# token = load_hf_token()
# if token:
#     login(token=token)

def print_dict(d, indent=0):
    for key, value in d.items():
        print('  ' * indent + str(key) + ':', end=' ')
        if isinstance(value, dict) or (str(value)[0] == '{' and str(value)[-1] == '}'):
            print()  # Move to the next line before printing the sub-dictionary
            print_dict(dict(value), indent + 1)
        else:
            print(value)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected. Got {v}, type: {type(v)}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_dir_path", required=True)
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--datetime", default=datetime.now().strftime('%Y-%m-%d %H:%M'))
    parser.add_argument("--corpus_type", type=str, default=None,choices=["docstring","srccodes"])
    parser.add_argument("--allow_disk_offload", type=str2bool, default=False)
    parser.add_argument("--max_dependency_num", type=int, default=None,help="最大依赖数量")
    parser.add_argument("--append_srcDep", action="store_true", help="是否添加源依赖")
    parser.add_argument("--newWhenExist", action="store_true", help="New when exist,如果文件存在，则创建新的文件")    
    parser.add_argument("--precision", type=str, default="fp16",help="precision of the model")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for nucleus sampling")
    parser.add_argument("--base_output_dir", type=str, default=None, help="Base output directory")
    parser.add_argument("--prompt_type", type=str, default="VSCC",  help="prompt type,详见dataset2prompt.py中")
    # RAG configuration parameters
    parser.add_argument("--embedding_source", type=str, default="local", choices=['local', 'togetherai'], help="嵌入源")
    parser.add_argument("--query_cache_dir", type=str, default="/datanfs4/chenrongyi/RAG_cache/.rag_query_cache_json", help="查询缓存目录")
    parser.add_argument("--rag_collection_base", type=str, default="/datanfs4/chenrongyi/data/RAG/chroma_data", help="RAG collection基础路径")
    parser.add_argument("--docstring_embedding_base_path", type=str, default="/datanfs4/chenrongyi/data/RAG/docs_embeddings/", help="文档字符串嵌入基础路径")
    parser.add_argument("--srccode_embedding_base_path", type=str, default="/datanfs4/chenrongyi/data/RAG/srccodes_embeddings/", help="源代码嵌入基础路径")
    parser.add_argument("--docstring_corpus_path", type=str, default="/datanfs4/chenrongyi/data/docs", help="文档字符串语料库路径")
    parser.add_argument("--srccode_corpus_path", type=str, default="/datanfs4/chenrongyi/data/srccodes", help="源代码语料库路径")
    parser.add_argument("--together_api_key_path", type=str, default="/datanfs4/chenrongyi/data/API_KEYSET/togetherai.txt", help="TogetherAI API密钥路径")
    parser.add_argument("--enable_extra_stopwords", type=str2bool, default=True, help="是否启用额外停止词")
    # 新增RAG相关参数2
    parser.add_argument("--useQAContext", type=str2bool, default=False, help="是否使用QA缓存")
    parser.add_argument("--QAContext_path", type=str, default=None, help="QA缓存路径")
    # error_info，用于args
    parser.add_argument("--enableCachedErrorInfo", type=str2bool, default=False, help="是否使用缓存的error_info")
    parser.add_argument("--errorinfos_filepath", type=str, default=None, help="error_info文件路径")
    # generated_target_code
    parser.add_argument("--generated_target_code_path", type=str, default=None, help="generated_target_code文件路径")
    # 是否跳过没有错误的样本
    parser.add_argument("--skip_no_error_samples", type=str2bool, default=False, help="是否跳过没有错误的样本")
    args, extra_args = parser.parse_known_args()
    conf = OmegaConf.load(args.config_path)
    cli_conf = OmegaConf.from_cli(extra_args)
    conf = OmegaConf.merge(conf, cli_conf)    
    conf.output_dir_path = args.output_dir_path
    conf.rank = args.rank
    conf.world_size = args.world_size
    conf.datetime = args.datetime
    conf.corpus_type = args.corpus_type
    conf.max_dependency_num = args.max_dependency_num
    conf.append_srcDep = args.append_srcDep
    conf.newWhenExist = args.newWhenExist
    conf.temperature = args.temperature
    conf.top_p = args.top_p
    conf.enableCachedErrorInfo = args.enableCachedErrorInfo
    conf.errorinfos_filepath = args.errorinfos_filepath
    conf.prompt_type = args.prompt_type
    conf.generated_target_code_path = args.generated_target_code_path
    conf.skip_no_error_samples = args.skip_no_error_samples
    # 将所有RAG相关参数添加到配置中
    conf.local_embedding_model = args.local_embedding_model
    conf.togetherai_embedding_model = args.togetherai_embedding_model
    conf.embedding_source = args.embedding_source
    conf.rag_collection_base = args.rag_collection_base
    conf.docstring_embedding_base_path = args.docstring_embedding_base_path
    conf.srccode_embedding_base_path = args.srccode_embedding_base_path
    conf.docstring_corpus_path = args.docstring_corpus_path
    conf.srccode_corpus_path = args.srccode_corpus_path
    conf.rag_document_num = args.rag_document_num
    conf.query_cache_dir = args.query_cache_dir
    conf.max_token_length = args.max_token_length
    conf.together_api_key_path = args.together_api_key_path
    conf.base_output_dir = args.base_output_dir
    # 创建RAG配置实例并添加到args对象中（避免OmegaConf处理）
    # args.rag_config = RAGConfig.from_args(args)
    # print(f"Embedding model: {args.rag_config.get_embedding_model()}")
    rag_config = RAGConfig.from_args(args)
    if torch.cuda.device_count() > 1:
        conf.model.use_hf_acc = True
    else:
        conf.model.use_hf_acc = False
    conf.model.allow_disk_offload = args.allow_disk_offload
    if args.allow_disk_offload:
        conf.model.world_size = args.world_size
    conf.model.disk_offload_dir = args.output_dir_path + f"/offload_data/{args.rank}"

    conf.model.tokenizer_path = conf.model.get("tokenizer_path", conf.model.path)
    conf.model.precision =args.precision
    conf.truncation = conf.get("truncation")
    conf.enable_extra_stopwords = args.enable_extra_stopwords

    datasets_str = args.datasets.strip().strip(",")
    datasets_list = datasets_str.split(",")
    conf.datasets = []
    for d in datasets_list:
        conf.datasets.append(d.strip())
    
    if args.rank is None or args.rank == 0:
        print_dict(dict(conf))
    return conf,rag_config

def get_model_and_tokenizer(model_config, llm_device_map="cuda", rank=0):
    def get_torch_dtype(precision):
        if precision == "fp16":
            return torch.float16
        elif precision == "bf16":
            return torch.bfloat16
        elif precision == "fp32":
            return torch.float32
        else:
            raise ValueError(f"Unsupported precision: {precision}")
        
    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_path, trust_remote_code=True)
    attn_impl = model_config.get("attn_implementation", "sdpa")
    print(f"Using attention type: {attn_impl}")
    
    if model_config.use_hf_acc:
        print(f'Model split across {torch.cuda.device_count()} GPUs')
        import warnings
        with init_empty_weights() and warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = AutoModelForCausalLM.from_pretrained(model_config.path
                                                        , torch_dtype=get_torch_dtype(model_config.precision)
                                                        , trust_remote_code=True
                                                        , attn_implementation=attn_impl)
        model = patch_hf(model, model_config.type, **model_config)
        model = load_checkpoint_and_dispatch(model, model_config.path
                                            , device_map="auto"
                                            , no_split_module_classes=["MistralDecoderLayer", "LlamaDecoderLayer", "Phi3DecoderLayer", "ContextManager"])   
        print(f'Worker {rank}: spreading model on {torch.cuda.device_count()} GPUs with device map: ', model.hf_device_map)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_config.path
                                                     , torch_dtype=get_torch_dtype(model_config.precision)
                                                     , trust_remote_code=True
                                                     , device_map=llm_device_map
                                                     , attn_implementation=attn_impl)
        model = patch_hf(model, model_config.type, **model_config)
    
    return model, tokenizer

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, conv_type, max_length, args):
    conv_type = conv_type.strip().lower()
    if conv_type == "vicuna":
        from fastchat.conversation import get_conv_template
        conv = get_conv_template("vicuna_v1.1")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif conv_type in ["mistral-inst", "qwen", "minicpm", "llama-3-inst", "phi-3-mini-inst"]:
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        raise NotImplementedError

    return prompt

def extend_passkey_context(context, passkey, max_len=512, tokenizer=None):
    # Given variables
    noise = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    passkey_sent = "The pass key is {}. Remember it. {} is the pass key."

    passkey_sentence = passkey_sent.format(passkey,passkey)
    if tokenizer is None:
        context_len = len(context.split())
        noise_len = len(noise.split())
        desired_len = int(max_len * 1000 * 0.79) # desired context length in terms of 1000s of tokens, with word to token ratio of 0.79 (for Mistral-7B)
    else:
        context_len = len(tokenizer(context))
        noise_len = len(tokenizer(noise))
        desired_len = max_len * 1000 # desired context length in terms of 1000s of tokens
    remaining_len = desired_len - context_len
    insertions = remaining_len // noise_len

    passkey_index = context.index(passkey_sentence)
    passkey_sentence += " "

    passkey_pos = passkey_index / len(context)
    pre_insertions = int(passkey_pos * insertions)
    post_insertions = insertions - pre_insertions

    pre_noise = noise * pre_insertions
    post_noise = noise * post_insertions

    new_context = context[:passkey_index] + pre_noise + passkey_sentence + \
                   post_noise + context[passkey_index + len(passkey_sentence):]
    if tokenizer is None:
        assert desired_len - noise_len <= len(new_context.split()) <= desired_len + noise_len, f"New context length: {len(new_context.split())} does not match the desired length: {desired_len}"
    else:
        new_context_len = len(tokenizer(new_context))
        assert desired_len - noise_len <= new_context_len <= desired_len + noise_len, f"New context length: {new_context_len} does not match the desired length: {desired_len}"
    assert str(passkey) in new_context, "Passkey not found in the new context."
    return new_context    

def load_infinite_bench(path, data_name, **kwargs) -> str:
    import re
    """
    Create prompt for a given example.

    Args:
        eg: example dict
        data_name: name of the dataset/task
    """
    print(f"read {data_name}.jsonl")
    fin = open(os.path.join(path, data_name.replace("__long", "") + ".jsonl"), "r")
    lines = fin.readlines()
    fin.close()
    data = [json.loads(line) for line in lines]
    def get_answer(inp: dict):
        if data_name in ["code_debug", "longbook_choice_eng"]:
            OPTIONS = "ABCD"
            if isinstance(inp["answer"], str):
                ret = [inp["answer"], OPTIONS[inp['options'].index(inp["answer"])]]
            elif isinstance(inp["answer"], list):
                if len(inp["answer"]) == 1:
                    ret = [inp["answer"][0], OPTIONS[inp['options'].index(inp["answer"][0])]]
                elif len(inp["answer"]) == 2 and inp["answer"][1] in ['A', 'B', 'C', 'D']:
                    ret = inp['answer']
                else:
                    raise ValueError
            else:
                raise ValueError
            return ret
        return inp["answer"]

    ret = []
    for eg in data:
        # ================= Code tasks
        if data_name == "code_run":
            find_result = re.findall(r"func_[0-9]+\(\-?[0-9]+\)", eg['input'])
            func_call = find_result[0]
            func = func_call.split("(")[0]
            instance = {"func": func, "func_call": func_call, "context": eg["context"]}
        elif data_name in ["code_debug", "code_debug_qa"]:
            # Load source code
            instance = {"context": eg["context"]}
            if data_name == "code_debug":
                instance.update({
                    "OPTION_A": eg["options"][0],
                    "OPTION_B": eg["options"][1],
                    "OPTION_C": eg["options"][2],
                    "OPTION_D": eg["options"][3]})
        # ================= Code tasks
        elif data_name == "longdialogue_qa_eng":
            instance = {"context": eg["context"]}
        # ==================== Long book tasks
        elif data_name in [
            "longbook_choice_eng",
            "longbook_qa_eng",
            "longbook_sum_eng",
            "longbook_qa_chn",
        ]:
            instance = {"context": eg["context"]}
            if data_name == "longbook_choice_eng":
                instance.update({
                    "question": eg["input"],
                    "OPTION_A": eg["options"][0],
                    "OPTION_B": eg["options"][1],
                    "OPTION_C": eg["options"][2],
                    "OPTION_D": eg["options"][3],
                })
            elif data_name in ["longbook_qa_eng", "longbook_qa_chn"]:
                instance.update({
                    "question": eg["input"],
                })
        elif data_name == "math_calc":
            instance = {"context": eg["context"]}
        elif data_name == "math_find":
            prompt = eg['input']
            context = eg['context']
            # Find "the * number" from the prompt
            find_result = re.findall(r"The .+ of", prompt)
            assert find_result, f"Cannot find the target number in {prompt}"
            target_number = find_result[0].lower()[:-3]
            # Replace the number with the answer
            prefix = f"What is {target_number} in the following list?"
            instance = {"prefix": prefix, "context": context, "input": prompt}
        elif data_name == "kv_retrieval":
            instance = {
                "context": eg["content"] if "content" in eg else eg["context"],
                "input": eg["input"],
                "key": eg["input"][6:44]
            }
            assert eg['input'][6] == '"'
            assert eg['input'][43] == '"'
        elif data_name == "passkey__long":
            instance = {
                "context": extend_passkey_context(eg["context"]
                                                    , eg['answer'][0]
                                                    , max_len=kwargs.get("extended_passkey", 1024)),
                "input": eg["input"],
            }
        else:
            instance = {
                "context": eg["content"] if "content" in eg else eg["context"],
                "input": eg["input"],
            }
        ans = get_answer(eg)
        instance["answers"] = ans if isinstance(ans, list) else [ans]
        instance["length"] = len(instance["context"].split())
        instance["all_classes"] = None

        ret.append(instance)
        # if len(ret) > 4:
        #     break
    return ret
def load_VersiBCB_data(dataset):
    if dataset == "VersiBCB_vace":
        with open("data/VersiBCB_Benchmark/vace_datas.json", "r") as f:
            data = json.load(f)
    elif dataset == "VersiBCB_vace_BD":
        with open("data/VersiBCB_Benchmark/vace_datas_for_warning.json", "r") as f:
            data = json.load(f)
    elif dataset == "VersiBCB_vscc":
        with open("data/VersiBCB_Benchmark/vscc_datas.json", "r") as f:
            data = json.load(f)
    elif dataset == "VersiBCB_vscc_BD":
        with open("data/VersiBCB_Benchmark/vscc_datas_for_warning.json", "r") as f:
            data = json.load(f)
    elif dataset == "VersiBCB_vace_GEN":
        with open("data/VersiBCB_Benchmark/vace_datas_for_general.json", "r") as f:
            data = json.load(f)
    elif dataset == "VersiBCB_vscc_GEN":
        with open("data/VersiBCB_Benchmark/vscc_datas_for_general.json", "r") as f:
            data = json.load(f) 
    return data
def load_data(dataset: str, **kwargs):
    if "VersiBCB" in dataset:
        data = load_VersiBCB_data(dataset)
        
    elif dataset.replace("__long", "") in set([
        "kv_retrieval", 
        "passkey", 
        "number_string", 
        "code_run", 
        "code_debug", 
        "longdialogue_qa_eng", 
        "longbook_qa_eng", 
        "longbook_sum_eng", 
        "longbook_choice_eng", 
        "longbook_qa_chn", 
        "math_find", 
        "math_calc"
    ]):
        path = "benchmark/data/infinite-bench"
        data = load_infinite_bench(path, dataset, **kwargs)

    elif "pg19" in dataset:
        if 'train' in dataset:
            split = 'train'
        elif 'val' in dataset:
            split = 'validation'
        elif 'test' in dataset:
            split = 'test'
        else:
            split = ''
        path = f"benchmark/data/pg19/{split}"
        data = load_from_disk(path)
        if split != '':
            dataset = f"pg19-{split}"
    else:
        try:
            data = load_from_disk(
                f"benchmark/data/longbench/{dataset}"
            )
        except:
            data = load_dataset('THUDM/LongBench', dataset, split='test', trust_remote_code=True)
            data.save_to_disk(os.path.join(f"benchmark/data/longbench", f"{dataset}"))

    print(f"Pred {dataset}")
    return data, dataset

def get_past_ids(out_path):
    """
    从输出文件中获取已经处理过的真实数据ID列表
    """
    past_ids = []
    past_worker_seq_ids = []
    
    if os.path.exists(out_path):
        with open(out_path, 'r') as file:
            for line in file:
                try:
                    entry = json.loads(line)
                    # 获取真实数据ID（主要用于判断是否已处理）
                    real_data_id = entry.get('id')
                    if real_data_id is not None:
                        past_ids.append(real_data_id)
                    
                    # 获取工作器序列ID（用于调试）
                    worker_seq_id = entry.get('worker_seq_id')
                    if worker_seq_id is not None:
                        past_worker_seq_ids.append(worker_seq_id)
                        
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse line in {out_path}: {line.strip()}")
                    continue
    
    print(f"Found {len(past_ids)} previously processed items in {out_path}")
    if past_worker_seq_ids:
        print(f"Worker seq ID range in existing file: {min(past_worker_seq_ids)} - {max(past_worker_seq_ids)}")
    
    return past_ids

def post_process(pred, conv_type, dataset):
    if conv_type == "qwen":
        pred = pred.split("<|im_end|>")[0]
    if "phi" in conv_type:
        pred = pred.strip()
    if "llama" in conv_type.lower():
        pred = pred.split("<|eot_id|>")[0]
    elif dataset == "samsum":
        pred = pred.split("\n")[0].strip()
    return pred

def get_pred(
    searcher:GreedySearch,
    tokenizer, 
    model_type,
    data, 
    max_length: int,
    max_gen: int, 
    prompt_format: str, 
    dataset, 
    conv_type: str, 
    gen_chunk_size = None, 
    truncation: str = None, 
    rank: int = None, 
    world_size: int = None,
    verbose: bool = False,
    out_path: str = None,
    return_block_size = False,
    args = None,
    rag_config = None,
    past_ids = [],
    task_type: str = None,
    extra_end_tokens_id_list = None,
    extra_stopwords = None,
):
    def get_worker_seq_id(cur):
        """生成工作器序列ID，用于多进程分工"""
        if world_size is None:
            return cur
        else:
            return world_size * cur + rank
        
    preds = []
    data = list(data)

    # pred运行区间
    # data = data[VSCC_LOW_BOUND:VSCC_HIGH_BOUND]

    # 首先过滤掉已经处理过的任务，确保任务均衡分配
    if past_ids:
        original_count = len(data)
        data = [json_obj for json_obj in data if json_obj.get("id") not in past_ids]
        filtered_count = len(data)
        if rank is None or rank == 0:
            print(f"Filtered out {original_count - filtered_count} already processed items. Remaining: {filtered_count}")
    
    # 多worker任务分配（在过滤后进行）
    if world_size is not None:
        data = data[rank::world_size]
        if rank is not None:
            print(f"Worker {rank}: Assigned {len(data)} tasks after balanced filtering")
    
    # 创建worker序列ID到真实数据ID的映射
    worker_seq_to_data_id = {}
    data_id_to_worker_seq = {}
    
    # 预处理：建立映射关系
    for cur, json_obj in enumerate(data):
        worker_seq_id = get_worker_seq_id(cur)
        real_data_id = json_obj.get("id")
        worker_seq_to_data_id[worker_seq_id] = real_data_id
        data_id_to_worker_seq[real_data_id] = worker_seq_id
    
    # 添加context
    from utils.context_utils import appendContextToData
    # 加入tqdm进度条    
    # data = [appendContextToData(json_obj,max_token_length=EMLLM_MAX_TOKEN_LENGTH) for json_obj in tqdm(data)]
    
    cur = 0
    total = len(data)
    skipped_locked_count = 0  # 添加跳过计数器

    if args.em_splitter == 'sentence':
        print("loading EN spacy")
        import spacy
        nlp = spacy.load('en_core_web_sm')
        nlp.max_length = 2147483647

    sent_len = []


    for json_obj in tqdm(data):
        worker_seq_id = get_worker_seq_id(cur)
        real_data_id = json_obj.get("id")

        # 对于json_obj,需要先进行替换,适配general场景,这边我们改用knowledge_doc替代
        if 'context' not in json_obj:
            json_obj['context'] = ""
        if 'true_target_dependency' in json_obj:
            json_obj['target_dependency'] = json_obj['true_target_dependency']
        elif 'true_dependency' in json_obj:
            json_obj['dependency'] = json_obj['true_dependency']

        # if rag_config.enableCachedQAContext:
        #     from utils.io_utils import loadJsonl
        #     if os.path.exists(rag_config.QAContextCachePath):
        #         QA_cache_context = loadJsonl(rag_config.QAContextCachePath)
        #     # 直接读取对应id的context (rag_config.QAContextCachePath) getIDContext应该在RAG有对应的函数，设置成一样的
        #     from utils.RAGutils.queryResultByInfer.contextLoad import loadQAContext
        #     json_obj["context"] = loadQAContext(json_obj,rag_config.max_doc_tokens,rag_config.QAContextCachePath)
        # else:
        # 实际下面一步只改了对应的json_obj["context"]
        json_obj, lock_status = appendContextToData(
            json_obj,
            tokenizer,
            max_token_length=args.max_token_length,
            useRAG=True,
            corpus_type=args.corpus_type,
            task_type=task_type,
            max_dependency_num=args.max_dependency_num,
            append_srcDep=args.append_srcDep,
            embedding_source=args.embedding_source,
            query_cache_dir=args.query_cache_dir,
            rag_document_num=args.rag_document_num,
            rag_config=rag_config
        )
        from benchmark.pred_rag import load_error_information
        if args.enableCachedErrorInfo:
            # 直接获取对应id的error_info (args.errorinfos_filepath) getIDErrorInfo应该在RAG有对应的函数，设置成一样的
            error_infos_dict = {}
            if args.errorinfos_filepath:
                logger.info(f"Worker {rank}: Loading error information from {args.errorinfos_filepath}")
                error_infos_dict = load_error_information(args.errorinfos_filepath)
                logger.info(f"Worker {rank}: Loaded error information for {len(error_infos_dict)} samples")
        json_obj["error_info"] = error_infos_dict.get(real_data_id, "")

        if args.generated_target_code_path:
            from benchmark.pred_rag import load_generated_target_code
            target_code_dict = load_generated_target_code(args.generated_target_code_path)
            # 确保选项，将dict的key转为str
            #TODO:和rag一样的临时补丁，等待修复
            target_code_dict = {int(k): v for k, v in target_code_dict.items()}
            json_obj["generated_target_code"] = target_code_dict.get(str(real_data_id), "")
            logging.info(f"Worker {rank}: Loaded generated target code for {real_data_id}")
        # 调用appendContextToData并处理锁状态（已过滤past_ids，无需重复检查）
        

        
        # 检查锁状态
        if lock_status == "locked":
            print(f"Skipping worker_seq_id={worker_seq_id}, real_data_id={real_data_id} due to collection building lock conflict")
            skipped_locked_count += 1
            cur += 1
            continue
        elif lock_status == "error":
            print(f"Skipping worker_seq_id={worker_seq_id}, real_data_id={real_data_id} due to context retrieval error")
            cur += 1
            continue

        # print(f"json_obj['context'] length: {len(json_obj['context'].split())}")

            
        long = len(json_obj['context'].split()) > args.model.disk_offload_threshold
        too_long = long and not (torch.cuda.device_count() >= 3 or (torch.cuda.device_count() >= 2 and args.model.allow_disk_offload) \
                    or (torch.cuda.device_count() == 1 and args.model.vector_offload))
        if too_long:
            print(f"Skipping worker_seq_id={worker_seq_id}, real_data_id={real_data_id} due to length: {len(json_obj['context'].split())}")
            cur += 1
            continue
        
        # 已过滤past_ids，只需检查长度限制
        try:
            prompt = prompt_format.format(**json_obj)
        except Exception as e:
            logger.info(f"Worker {rank}: json_obj: {json_obj.keys()}")
            logger.info(f"Worker {rank}: prompt_format: {prompt_format}")
            logger.error(f"Worker {rank}: Error formatting prompt for {real_data_id}: {e}")
            cur += 1
            return 


        extra_end_token_ids = []
        if conv_type == "llama-3-inst":
            extra_end_token_ids.append(tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0])
        elif conv_type == "phi-3-mini-inst":
            extra_end_token_ids.append(tokenizer.encode("<|end|>", add_special_tokens=False)[0])
        elif conv_type == "qwen":
            extra_end_token_ids.append(tokenizer.encode("<|im_end|>", add_special_tokens=False)[0])
        # elif conv_type == "codegemma":
        #     extra_end_token_ids.append(tokenizer.encode("<eos>", add_special_tokens=False)[0])
        # elif conv_type == "deepseekcoder":
        #     extra_end_token_ids.append(tokenizer.encode("<|EOT|>", add_special_tokens=False)[0])
        if dataset == "samsum":
            extra_end_token_ids.append(tokenizer.encode("\n", add_special_tokens=False)[-1])

        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p","VersiBCB_vace","VersiBCB_vace_BD","VersiBCB_vscc","VersiBCB_vscc_BD","VersiBCB_vace_GEN","VersiBCB_vscc_GEN"]: 
            # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, conv_type, max_length, args)

            if conv_type.strip().lower() in ['mistral-inst']:
                add_special_tokens = False
            else:
                add_special_tokens = True
        else:
            add_special_tokens = True

        if args.em_splitter == 'sentence':
            tokenized_prompt = []
            em_labels = []
            sentences = nlp(prompt).sents
            for sent in sentences:
                sent_inp_ids = tokenizer(sent.text, truncation=False, return_tensors="pt",
                                        add_special_tokens=add_special_tokens).input_ids[0]
                sent_len.append(len(sent_inp_ids))

                tokenized_prompt.append(sent_inp_ids)
                boundaries = torch.zeros(len(sent_inp_ids))
                boundaries[0] = 1
                em_labels.append(boundaries)
            tokenized_prompt = torch.hstack(tokenized_prompt)
            em_labels = torch.hstack(em_labels).bool()
        else:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids[0]
            em_labels = None

        if truncation is None:
            if len(tokenized_prompt) > max_length - max_gen:
                if verbose:
                    print(f"Length {len(tokenized_prompt)}. Skipped.")
                continue

        else:
            if truncation == "suffix":
                length = len(tokenized_prompt)
                if length > max_length - max_gen:
                    if verbose:
                        print("over length")
                    init_token_num = 128
                    prompt = tokenizer.decode(tokenized_prompt[:init_token_num].tolist() + tokenized_prompt[- (max_length - max_gen - init_token_num):].tolist())
                    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids[0]
            else:
                raise NotImplementedError

        offloading_args = {
            "vector_offload_threshold": args.model.vector_offload_threshold,
            "disk_offload_threshold": args.model.disk_offload_threshold
        }
        time1 = time.time()
        output = searcher.generate(
            input_ids=tokenized_prompt,
            em_labels=em_labels,
            max_length=max_gen,
            chunk_size=gen_chunk_size,
            extra_end_token_ids=extra_end_token_ids,
            temperature=args.temperature,
            extra_end_tokens_id_list=extra_end_tokens_id_list,  
            extra_stopwords=extra_stopwords,
            top_p=args.top_p,
            **offloading_args
        )
        time2 = time.time()

        pred = post_process(output["pred"], conv_type, dataset)
        if model_type == "em-llm" and return_block_size:
            block_sizes = [block.size for block in searcher.past_kv[0].global_blocks[0]]
            mean_block_size = np.mean(np.array(block_sizes))
        else:
            block_sizes = None
            mean_block_size = None

        # Extract model parameters for output
        model_params = {}
        if hasattr(args, 'model'):
            model_params = {
                "n_init": getattr(args.model, 'n_init', None),
                "n_mem": getattr(args.model, 'n_mem', None),
                "n_local": getattr(args.model, 'n_local', None),
                "repr_topk": getattr(args.model, 'repr_topk', None),
                "block_size": getattr(args.model, 'max_block_size', None),
                "chunk_size": getattr(args, 'chunk_size', None)
            }
        
        preds.append(
            {
                "id": real_data_id,  # 使用真实的数据ID
                "worker_seq_id": worker_seq_id,  # 添加工作器序列ID用于调试
                "pred": pred, 
                "answers": json_obj.get("answers"), 
                "all_classes": json_obj.get("all_classes"), 
                "length": json_obj.get("length"), 
                "token_length": len(tokenized_prompt) + max_gen, 
                "chunk_ppl": output.get("chunk_ppl"), 
                "total_ppl": output.get("total_ppl"),
                "block_sizes": block_sizes, 
                "mean_block_size": mean_block_size, 
                "generation_time": time2 - time1,
                "model_params": model_params,
                "temperature": args.temperature,
                "top_p": args.top_p
            }
        )

        if verbose:
            print(f"----------{cur}/{total}----------")
            print(f"Worker Seq ID: {worker_seq_id}, Real Data ID: {real_data_id}")
            print("Length: ", len(tokenized_prompt))
            print("Question:", prompt[-100:])
            print("Pred:", pred)
            print("Answer:", json_obj.get("answers"))
            print("")

        with open(out_path, "a+", encoding="utf-8") as f:
            json.dump(preds[-1], f, ensure_ascii=False)
            f.write('\n')

        searcher.clear()
        cur += 1

    if args.em_splitter == 'sentence':
        print(f"Avg sentence length: {sum(sent_len) / len(sent_len)}")
    
    # 输出统计信息
    print(f"Processing completed for worker {rank if rank is not None else 'single'}:")
    print(f"  Total items: {total}")
    print(f"  Successfully processed: {len(preds)}")
    print(f"  Skipped due to collection locks: {skipped_locked_count}")
    if skipped_locked_count > 0:
        print(f"  Note: {skipped_locked_count} items were skipped due to collection building conflicts.")
        print(f"        These items may be processed later when collections are available.")
    
    # 输出映射信息用于调试
    if verbose and world_size is not None:
        print(f"Worker {rank}: Processed {len(worker_seq_to_data_id)} items")
        print(f"Worker {rank}: Worker seq ID range: {min(worker_seq_to_data_id.keys()) if worker_seq_to_data_id else 'N/A'} - {max(worker_seq_to_data_id.keys()) if worker_seq_to_data_id else 'N/A'}")
    
    return preds

class DualLogger:
    def __init__(self, filename, mode='a', rank=0, world_size=1):
        self.terminal = sys.stdout
        self.log = open(filename, mode, buffering=1, encoding='utf-8')  # Line-buffering mode.
        self.rank = rank
        self.world_size = world_size

    def write(self, message):
        worker_message = f"Worker{self.rank}: " + str(message) if self.world_size > 1 else str(message) 
        self.terminal.write(worker_message)
        self.log.write(message)

    def flush(self):  # Needed for compatibility with flush operations
        self.terminal.flush()
        self.log.flush()

def main(args, rag_config):

    if args.enable_extra_stopwords:
        extra_stopwords = json.load(open("benchmark/config/extra_stopwords.json", "r"))
    else:
        extra_stopwords = None
    output_dir_path = args.output_dir_path
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path,exist_ok=True)

    if args.logging:
        log_dir_path = os.path.join(output_dir_path, f"logs/{args.datetime}/")
        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path,exist_ok=True)
        log_path = f'worker_{args.rank}.log'
        sys.stdout = sys.stderr = DualLogger(
            os.path.join(log_dir_path, log_path), 
            rank=args.rank, 
            world_size=args.world_size
        )

    if not args.model.use_hf_acc and args.model.allow_disk_offload:
        args.model.vector_offload = True
    else: 
        args.model.vector_offload = False
    model, tokenizer = get_model_and_tokenizer(args.model, rank=args.rank)
    extra_end_tokens_id_list = []
    if args.enable_extra_stopwords:
        extra_end_tokens_id_list = [tokenizer.encode(word, add_special_tokens=False) for word in extra_stopwords]
    searcher = GreedySearch(
        model, 
        tokenizer, 
        args.model.type, 
        em_splitter=args.em_splitter, 
        compute_ppl=args.compute_ppl,
    )

    datasets = args.datasets
    from benchmark.config.code.dataset2prompt import dataset2prompt
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2maxlen = json.load(open("benchmark/config/dataset2maxlen.json", "r"))

    multiprocessing = args.world_size is not None and args.world_size > 1
    if multiprocessing:
        assert args.rank in list(range(args.world_size))

    # 创建输出管理器
    output_manager = OutputManager(args.base_output_dir)

    # predict on each dataset, dataset2type仅用于生成指定id的query(不会对其他代码产生影响)
    dataset2tasktype = {'VersiBCB_vace':'VACE','VersiBCB_vscc':'VSCC','VersiBCB_vace_BD':'VACE','VersiBCB_vscc_BD':'VSCC',"VersiBCB_vace_GEN":"VACE","VersiBCB_vscc_GEN":"VSCC"}
    
    prompt_type = args.prompt_type
    logger.info(f"Worker {args.rank}: Prompt type: {prompt_type}")
    for i,dataset in enumerate(datasets):
        
        data, dataset = load_data(dataset, extended_passkey=args.get("extended_passkey", 1024))
        #TODO: 这边需要适配memory场景，这边用context替代knowledge_doc,实际上多了一步转换，又转换到context这一命名了，后面看看如何统一
        prompt_format = dataset2prompt["VersiBCB"][prompt_type].replace("{knowledge_doc}", "{context}")
        
        max_gen = dataset2maxlen[dataset]

        # 确定方法类型
        approach = output_manager.get_approach_from_context(
            model_type=args.model.type,
            has_corpus_type=hasattr(args, 'corpus_type') and args.corpus_type
        )
        
        # 生成基础文件名
        memory_cardinfo = {
            "n_init": args.model.n_init,
            "n_mem": args.model.n_mem,
            "n_local": args.model.n_local,
            "repr_topk": args.model.repr_topk,
            "block_size": args.model.max_block_size,
            "chunk_size": args.chunk_size,
        }
        base_filename = output_manager.generate_base_filename(
            dataset=dataset,
            model_name=args.model.path if hasattr(args.model, 'path') else "unknown_model",
            approach=approach,
            corpus_type=args.corpus_type if hasattr(args, 'corpus_type') else None,
            embedding_source=args.embedding_source if hasattr(args, 'embedding_source') else None,
            max_tokens=args.max_token_length if hasattr(args, 'max_token_length') else None,
            inference_type="local",
            memory_cardinfo=memory_cardinfo,
            max_dependency_num=args.max_dependency_num if hasattr(args, 'max_dependency_num') else None,
            useQAContext=rag_config.useQAContext if hasattr(rag_config, 'useQAContext') else False
        )
        
        # 获取输出路径和配置路径
        out_path, config_path = output_manager.get_output_path_and_config(
            approach=approach,
            base_filename=base_filename,
            rank=args.rank,
            world_size=args.world_size,
            newWhenExist=args.newWhenExist,
            model_name_or_path=args.model.path if hasattr(args.model, 'path') else None,

        )
        
        #TODO: 过滤对应的ids ,需考虑与rag创建统一的过滤函数
        if args.skip_no_error_samples:
            from benchmark.pred_rag import load_error_information
            error_infos_dict = load_error_information(args.errorinfos_filepath)
            error_infos_ids = list(error_infos_dict.keys())
            error_str_ids = [str(id) for id in error_infos_ids]
            data = [item for item in data if str(item["id"]) in error_str_ids]
            logger.info(f"Worker {args.rank}: only keep {len(data)} samples with error")
        past_ids = get_past_ids(out_path)

        get_pred(
            searcher=searcher,
            tokenizer=tokenizer,
            model_type=args.model.type,
            data=data,
            max_length=args.max_len,
            max_gen=max_gen,
            prompt_format=prompt_format,
            dataset=dataset,
            conv_type=args.conv_type,
            gen_chunk_size=args.chunk_size,
            truncation=args.truncation,
            rank=args.rank,
            world_size=args.world_size,
            verbose=args.verbose,
            out_path=out_path,
            return_block_size=args.return_block_size,
            args=args,
            rag_config=rag_config,
            past_ids=past_ids,
            task_type=dataset2tasktype.get(dataset), # 用于决定query类型，如果是VSCC就是只用description，如果是VACE就用description+srccode
            extra_end_tokens_id_list=extra_end_tokens_id_list,
            extra_stopwords=extra_stopwords
        )

        # 生成并保存配置文件
        config_data = output_manager.generate_config(
            approach=approach,
            args=args,
            rag_config=rag_config
        )
        output_manager.save_config(config_path, config_data)


if __name__ == '__main__':

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args, rag_config = parse_args()

# https://github.com/THUDM/LongBench/blob/main/pred.py
import os
from datasets import load_from_disk, load_dataset
import torch
import json
from tqdm import tqdm
import argparse
from omegaconf import OmegaConf
from em_llm.utils import patch_hf, GreedySearch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import sys
import time
import numpy as np
from datetime import datetime
import pprint as pp
from huggingface_hub import login
from pathlib import Path
from benchmark.config.code.config import VSCC_LOW_BOUND, VSCC_HIGH_BOUND
from utils.RAGutils.rag_config import RAGConfig
from utils.output_manager import OutputManager
import logging
logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Load Hugging Face token from config or environment variable
def load_hf_token():
    # Try environment variable first
    if os.environ.get("HF_TOKEN"):
        return os.environ.get("HF_TOKEN")
    
    # Try config file
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "auth.yaml")
    if os.path.exists(config_path):
        try:
            auth_config = OmegaConf.load(config_path)
            if hasattr(auth_config, "hf_token") and auth_config.hf_token:
                return auth_config.hf_token
        except Exception as e:
            print(f"Warning: Failed to load auth config: {e}")
    
    # Try legacy path as fallback
    token_path = os.path.expanduser("~/tokens/hf.txt")
    if os.path.exists(token_path):
        with open(token_path, "r") as file:
            return file.read().strip()
    
    print("Warning: No Hugging Face token found. Some operations may fail.")
    return None

# Login to Hugging Face
# token = load_hf_token()
# if token:
#     login(token=token)

def print_dict(d, indent=0):
    for key, value in d.items():
        print('  ' * indent + str(key) + ':', end=' ')
        if isinstance(value, dict) or (str(value)[0] == '{' and str(value)[-1] == '}'):
            print()  # Move to the next line before printing the sub-dictionary
            print_dict(dict(value), indent + 1)
        else:
            print(value)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected. Got {v}, type: {type(v)}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_dir_path", required=True)
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--datetime", default=datetime.now().strftime('%Y-%m-%d %H:%M'))
    parser.add_argument("--corpus_type", type=str, default=None,choices=["docstring","srccodes"])
    parser.add_argument("--allow_disk_offload", type=str2bool, default=False)
    parser.add_argument("--max_dependency_num", type=int, default=None,help="最大依赖数量")
    parser.add_argument("--append_srcDep", action="store_true", help="是否添加源依赖")
    parser.add_argument("--newWhenExist", action="store_true", help="New when exist,如果文件存在，则创建新的文件")    
    parser.add_argument("--precision", type=str, default="fp16",help="precision of the model")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for nucleus sampling")
    parser.add_argument("--base_output_dir", type=str, default=None, help="Base output directory")
    parser.add_argument("--prompt_type", type=str, default="VSCC",  help="prompt type,详见dataset2prompt.py中")
    # RAG configuration parameters
    parser.add_argument("--embedding_source", type=str, default="local", choices=['local', 'togetherai'], help="嵌入源")
    parser.add_argument("--query_cache_dir", type=str, default="/datanfs4/chenrongyi/RAG_cache/.rag_query_cache_json", help="查询缓存目录")
    parser.add_argument("--rag_collection_base", type=str, default="/datanfs4/chenrongyi/data/RAG/chroma_data", help="RAG collection基础路径")
    parser.add_argument("--docstring_embedding_base_path", type=str, default="/datanfs4/chenrongyi/data/RAG/docs_embeddings/", help="文档字符串嵌入基础路径")
    parser.add_argument("--srccode_embedding_base_path", type=str, default="/datanfs4/chenrongyi/data/RAG/srccodes_embeddings/", help="源代码嵌入基础路径")
    parser.add_argument("--docstring_corpus_path", type=str, default="/datanfs4/chenrongyi/data/docs", help="文档字符串语料库路径")
    parser.add_argument("--srccode_corpus_path", type=str, default="/datanfs4/chenrongyi/data/srccodes", help="源代码语料库路径")
    parser.add_argument("--together_api_key_path", type=str, default="/datanfs4/chenrongyi/data/API_KEYSET/togetherai.txt", help="TogetherAI API密钥路径")
    parser.add_argument("--enable_extra_stopwords", type=str2bool, default=True, help="是否启用额外停止词")
    # 新增RAG相关参数2
    parser.add_argument("--useQAContext", type=str2bool, default=False, help="是否使用QA缓存")
    parser.add_argument("--QAContext_path", type=str, default=None, help="QA缓存路径")
    # error_info，用于args
    parser.add_argument("--enableCachedErrorInfo", type=str2bool, default=False, help="是否使用缓存的error_info")
    parser.add_argument("--errorinfos_filepath", type=str, default=None, help="error_info文件路径")
    # generated_target_code
    parser.add_argument("--generated_target_code_path", type=str, default=None, help="generated_target_code文件路径")
    # 是否跳过没有错误的样本
    parser.add_argument("--skip_no_error_samples", type=str2bool, default=False, help="是否跳过没有错误的样本")
    args, extra_args = parser.parse_known_args()
    conf = OmegaConf.load(args.config_path)
    cli_conf = OmegaConf.from_cli(extra_args)
    conf = OmegaConf.merge(conf, cli_conf)    
    conf.output_dir_path = args.output_dir_path
    conf.rank = args.rank
    conf.world_size = args.world_size
    conf.datetime = args.datetime
    conf.corpus_type = args.corpus_type
    conf.max_dependency_num = args.max_dependency_num
    conf.append_srcDep = args.append_srcDep
    conf.newWhenExist = args.newWhenExist
    conf.temperature = args.temperature
    conf.top_p = args.top_p
    conf.enableCachedErrorInfo = args.enableCachedErrorInfo
    conf.errorinfos_filepath = args.errorinfos_filepath
    conf.prompt_type = args.prompt_type
    conf.generated_target_code_path = args.generated_target_code_path
    conf.skip_no_error_samples = args.skip_no_error_samples
    # 将所有RAG相关参数添加到配置中
    conf.local_embedding_model = args.local_embedding_model
    conf.togetherai_embedding_model = args.togetherai_embedding_model
    conf.embedding_source = args.embedding_source
    conf.rag_collection_base = args.rag_collection_base
    conf.docstring_embedding_base_path = args.docstring_embedding_base_path
    conf.srccode_embedding_base_path = args.srccode_embedding_base_path
    conf.docstring_corpus_path = args.docstring_corpus_path
    conf.srccode_corpus_path = args.srccode_corpus_path
    conf.rag_document_num = args.rag_document_num
    conf.query_cache_dir = args.query_cache_dir
    conf.max_token_length = args.max_token_length
    conf.together_api_key_path = args.together_api_key_path
    conf.base_output_dir = args.base_output_dir
    # 创建RAG配置实例并添加到args对象中（避免OmegaConf处理）
    # args.rag_config = RAGConfig.from_args(args)
    # print(f"Embedding model: {args.rag_config.get_embedding_model()}")
    rag_config = RAGConfig.from_args(args)
    if torch.cuda.device_count() > 1:
        conf.model.use_hf_acc = True
    else:
        conf.model.use_hf_acc = False
    conf.model.allow_disk_offload = args.allow_disk_offload
    if args.allow_disk_offload:
        conf.model.world_size = args.world_size
    conf.model.disk_offload_dir = args.output_dir_path + f"/offload_data/{args.rank}"

    conf.model.tokenizer_path = conf.model.get("tokenizer_path", conf.model.path)
    conf.model.precision =args.precision
    conf.truncation = conf.get("truncation")
    conf.enable_extra_stopwords = args.enable_extra_stopwords

    datasets_str = args.datasets.strip().strip(",")
    datasets_list = datasets_str.split(",")
    conf.datasets = []
    for d in datasets_list:
        conf.datasets.append(d.strip())
    
    if args.rank is None or args.rank == 0:
        print_dict(dict(conf))
    return conf,rag_config

def get_model_and_tokenizer(model_config, llm_device_map="cuda", rank=0):
    def get_torch_dtype(precision):
        if precision == "fp16":
            return torch.float16
        elif precision == "bf16":
            return torch.bfloat16
        elif precision == "fp32":
            return torch.float32
        else:
            raise ValueError(f"Unsupported precision: {precision}")
        
    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_path, trust_remote_code=True)
    attn_impl = model_config.get("attn_implementation", "sdpa")
    print(f"Using attention type: {attn_impl}")
    
    if model_config.use_hf_acc:
        print(f'Model split across {torch.cuda.device_count()} GPUs')
        import warnings
        with init_empty_weights() and warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = AutoModelForCausalLM.from_pretrained(model_config.path
                                                        , torch_dtype=get_torch_dtype(model_config.precision)
                                                        , trust_remote_code=True
                                                        , attn_implementation=attn_impl)
        model = patch_hf(model, model_config.type, **model_config)
        model = load_checkpoint_and_dispatch(model, model_config.path
                                            , device_map="auto"
                                            , no_split_module_classes=["MistralDecoderLayer", "LlamaDecoderLayer", "Phi3DecoderLayer", "ContextManager"])   
        print(f'Worker {rank}: spreading model on {torch.cuda.device_count()} GPUs with device map: ', model.hf_device_map)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_config.path
                                                     , torch_dtype=get_torch_dtype(model_config.precision)
                                                     , trust_remote_code=True
                                                     , device_map=llm_device_map
                                                     , attn_implementation=attn_impl)
        model = patch_hf(model, model_config.type, **model_config)
    
    return model, tokenizer

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, conv_type, max_length, args):
    conv_type = conv_type.strip().lower()
    if conv_type == "vicuna":
        from fastchat.conversation import get_conv_template
        conv = get_conv_template("vicuna_v1.1")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif conv_type in ["mistral-inst", "qwen", "minicpm", "llama-3-inst", "phi-3-mini-inst"]:
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        raise NotImplementedError

    return prompt

def extend_passkey_context(context, passkey, max_len=512, tokenizer=None):
    # Given variables
    noise = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    passkey_sent = "The pass key is {}. Remember it. {} is the pass key."

    passkey_sentence = passkey_sent.format(passkey,passkey)
    if tokenizer is None:
        context_len = len(context.split())
        noise_len = len(noise.split())
        desired_len = int(max_len * 1000 * 0.79) # desired context length in terms of 1000s of tokens, with word to token ratio of 0.79 (for Mistral-7B)
    else:
        context_len = len(tokenizer(context))
        noise_len = len(tokenizer(noise))
        desired_len = max_len * 1000 # desired context length in terms of 1000s of tokens
    remaining_len = desired_len - context_len
    insertions = remaining_len // noise_len

    passkey_index = context.index(passkey_sentence)
    passkey_sentence += " "

    passkey_pos = passkey_index / len(context)
    pre_insertions = int(passkey_pos * insertions)
    post_insertions = insertions - pre_insertions

    pre_noise = noise * pre_insertions
    post_noise = noise * post_insertions

    new_context = context[:passkey_index] + pre_noise + passkey_sentence + \
                   post_noise + context[passkey_index + len(passkey_sentence):]
    if tokenizer is None:
        assert desired_len - noise_len <= len(new_context.split()) <= desired_len + noise_len, f"New context length: {len(new_context.split())} does not match the desired length: {desired_len}"
    else:
        new_context_len = len(tokenizer(new_context))
        assert desired_len - noise_len <= new_context_len <= desired_len + noise_len, f"New context length: {new_context_len} does not match the desired length: {desired_len}"
    assert str(passkey) in new_context, "Passkey not found in the new context."
    return new_context    

def load_infinite_bench(path, data_name, **kwargs) -> str:
    import re
    """
    Create prompt for a given example.

    Args:
        eg: example dict
        data_name: name of the dataset/task
    """
    print(f"read {data_name}.jsonl")
    fin = open(os.path.join(path, data_name.replace("__long", "") + ".jsonl"), "r")
    lines = fin.readlines()
    fin.close()
    data = [json.loads(line) for line in lines]
    def get_answer(inp: dict):
        if data_name in ["code_debug", "longbook_choice_eng"]:
            OPTIONS = "ABCD"
            if isinstance(inp["answer"], str):
                ret = [inp["answer"], OPTIONS[inp['options'].index(inp["answer"])]]
            elif isinstance(inp["answer"], list):
                if len(inp["answer"]) == 1:
                    ret = [inp["answer"][0], OPTIONS[inp['options'].index(inp["answer"][0])]]
                elif len(inp["answer"]) == 2 and inp["answer"][1] in ['A', 'B', 'C', 'D']:
                    ret = inp['answer']
                else:
                    raise ValueError
            else:
                raise ValueError
            return ret
        return inp["answer"]

    ret = []
    for eg in data:
        # ================= Code tasks
        if data_name == "code_run":
            find_result = re.findall(r"func_[0-9]+\(\-?[0-9]+\)", eg['input'])
            func_call = find_result[0]
            func = func_call.split("(")[0]
            instance = {"func": func, "func_call": func_call, "context": eg["context"]}
        elif data_name in ["code_debug", "code_debug_qa"]:
            # Load source code
            instance = {"context": eg["context"]}
            if data_name == "code_debug":
                instance.update({
                    "OPTION_A": eg["options"][0],
                    "OPTION_B": eg["options"][1],
                    "OPTION_C": eg["options"][2],
                    "OPTION_D": eg["options"][3]})
        # ================= Code tasks
        elif data_name == "longdialogue_qa_eng":
            instance = {"context": eg["context"]}
        # ==================== Long book tasks
        elif data_name in [
            "longbook_choice_eng",
            "longbook_qa_eng",
            "longbook_sum_eng",
            "longbook_qa_chn",
        ]:
            instance = {"context": eg["context"]}
            if data_name == "longbook_choice_eng":
                instance.update({
                    "question": eg["input"],
                    "OPTION_A": eg["options"][0],
                    "OPTION_B": eg["options"][1],
                    "OPTION_C": eg["options"][2],
                    "OPTION_D": eg["options"][3],
                })
            elif data_name in ["longbook_qa_eng", "longbook_qa_chn"]:
                instance.update({
                    "question": eg["input"],
                })
        elif data_name == "math_calc":
            instance = {"context": eg["context"]}
        elif data_name == "math_find":
            prompt = eg['input']
            context = eg['context']
            # Find "the * number" from the prompt
            find_result = re.findall(r"The .+ of", prompt)
            assert find_result, f"Cannot find the target number in {prompt}"
            target_number = find_result[0].lower()[:-3]
            # Replace the number with the answer
            prefix = f"What is {target_number} in the following list?"
            instance = {"prefix": prefix, "context": context, "input": prompt}
        elif data_name == "kv_retrieval":
            instance = {
                "context": eg["content"] if "content" in eg else eg["context"],
                "input": eg["input"],
                "key": eg["input"][6:44]
            }
            assert eg['input'][6] == '"'
            assert eg['input'][43] == '"'
        elif data_name == "passkey__long":
            instance = {
                "context": extend_passkey_context(eg["context"]
                                                    , eg['answer'][0]
                                                    , max_len=kwargs.get("extended_passkey", 1024)),
                "input": eg["input"],
            }
        else:
            instance = {
                "context": eg["content"] if "content" in eg else eg["context"],
                "input": eg["input"],
            }
        ans = get_answer(eg)
        instance["answers"] = ans if isinstance(ans, list) else [ans]
        instance["length"] = len(instance["context"].split())
        instance["all_classes"] = None

        ret.append(instance)
        # if len(ret) > 4:
        #     break
    return ret
def load_VersiBCB_data(dataset):
    if dataset == "VersiBCB_vace":
        with open("data/VersiBCB_Benchmark/vace_datas.json", "r") as f:
            data = json.load(f)
    elif dataset == "VersiBCB_vace_BD":
        with open("data/VersiBCB_Benchmark/vace_datas_for_warning.json", "r") as f:
            data = json.load(f)
    elif dataset == "VersiBCB_vscc":
        with open("data/VersiBCB_Benchmark/vscc_datas.json", "r") as f:
            data = json.load(f)
    elif dataset == "VersiBCB_vscc_BD":
        with open("data/VersiBCB_Benchmark/vscc_datas_for_warning.json", "r") as f:
            data = json.load(f)
    elif dataset == "VersiBCB_vace_GEN":
        with open("data/VersiBCB_Benchmark/vace_datas_for_general.json", "r") as f:
            data = json.load(f)
    elif dataset == "VersiBCB_vscc_GEN":
        with open("data/VersiBCB_Benchmark/vscc_datas_for_general.json", "r") as f:
            data = json.load(f) 
    return data
def load_data(dataset: str, **kwargs):
    if "VersiBCB" in dataset:
        data = load_VersiBCB_data(dataset)
        
    elif dataset.replace("__long", "") in set([
        "kv_retrieval", 
        "passkey", 
        "number_string", 
        "code_run", 
        "code_debug", 
        "longdialogue_qa_eng", 
        "longbook_qa_eng", 
        "longbook_sum_eng", 
        "longbook_choice_eng", 
        "longbook_qa_chn", 
        "math_find", 
        "math_calc"
    ]):
        path = "benchmark/data/infinite-bench"
        data = load_infinite_bench(path, dataset, **kwargs)

    elif "pg19" in dataset:
        if 'train' in dataset:
            split = 'train'
        elif 'val' in dataset:
            split = 'validation'
        elif 'test' in dataset:
            split = 'test'
        else:
            split = ''
        path = f"benchmark/data/pg19/{split}"
        data = load_from_disk(path)
        if split != '':
            dataset = f"pg19-{split}"
    else:
        try:
            data = load_from_disk(
                f"benchmark/data/longbench/{dataset}"
            )
        except:
            data = load_dataset('THUDM/LongBench', dataset, split='test', trust_remote_code=True)
            data.save_to_disk(os.path.join(f"benchmark/data/longbench", f"{dataset}"))

    print(f"Pred {dataset}")
    return data, dataset

def get_past_ids(out_path):
    """
    从输出文件中获取已经处理过的真实数据ID列表
    """
    past_ids = []
    past_worker_seq_ids = []
    
    if os.path.exists(out_path):
        with open(out_path, 'r') as file:
            for line in file:
                try:
                    entry = json.loads(line)
                    # 获取真实数据ID（主要用于判断是否已处理）
                    real_data_id = entry.get('id')
                    if real_data_id is not None:
                        past_ids.append(real_data_id)
                    
                    # 获取工作器序列ID（用于调试）
                    worker_seq_id = entry.get('worker_seq_id')
                    if worker_seq_id is not None:
                        past_worker_seq_ids.append(worker_seq_id)
                        
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse line in {out_path}: {line.strip()}")
                    continue
    
    print(f"Found {len(past_ids)} previously processed items in {out_path}")
    if past_worker_seq_ids:
        print(f"Worker seq ID range in existing file: {min(past_worker_seq_ids)} - {max(past_worker_seq_ids)}")
    
    return past_ids

def post_process(pred, conv_type, dataset):
    if conv_type == "qwen":
        pred = pred.split("<|im_end|>")[0]
    if "phi" in conv_type:
        pred = pred.strip()
    if "llama" in conv_type.lower():
        pred = pred.split("<|eot_id|>")[0]
    elif dataset == "samsum":
        pred = pred.split("\n")[0].strip()
    return pred

def get_pred(
    searcher:GreedySearch,
    tokenizer, 
    model_type,
    data, 
    max_length: int,
    max_gen: int, 
    prompt_format: str, 
    dataset, 
    conv_type: str, 
    gen_chunk_size = None, 
    truncation: str = None, 
    rank: int = None, 
    world_size: int = None,
    verbose: bool = False,
    out_path: str = None,
    return_block_size = False,
    args = None,
    rag_config = None,
    past_ids = [],
    task_type: str = None,
    extra_end_tokens_id_list = None,
    extra_stopwords = None,
):
    def get_worker_seq_id(cur):
        """生成工作器序列ID，用于多进程分工"""
        if world_size is None:
            return cur
        else:
            return world_size * cur + rank
        
    preds = []
    data = list(data)

    # pred运行区间
    # data = data[VSCC_LOW_BOUND:VSCC_HIGH_BOUND]

    # 首先过滤掉已经处理过的任务，确保任务均衡分配
    if past_ids:
        original_count = len(data)
        data = [json_obj for json_obj in data if json_obj.get("id") not in past_ids]
        filtered_count = len(data)
        if rank is None or rank == 0:
            print(f"Filtered out {original_count - filtered_count} already processed items. Remaining: {filtered_count}")
    
    # 多worker任务分配（在过滤后进行）
    if world_size is not None:
        data = data[rank::world_size]
        if rank is not None:
            print(f"Worker {rank}: Assigned {len(data)} tasks after balanced filtering")
    
    # 创建worker序列ID到真实数据ID的映射
    worker_seq_to_data_id = {}
    data_id_to_worker_seq = {}
    
    # 预处理：建立映射关系
    for cur, json_obj in enumerate(data):
        worker_seq_id = get_worker_seq_id(cur)
        real_data_id = json_obj.get("id")
        worker_seq_to_data_id[worker_seq_id] = real_data_id
        data_id_to_worker_seq[real_data_id] = worker_seq_id
    
    # 添加context
    from utils.context_utils import appendContextToData
    # 加入tqdm进度条    
    # data = [appendContextToData(json_obj,max_token_length=EMLLM_MAX_TOKEN_LENGTH) for json_obj in tqdm(data)]
    
    cur = 0
    total = len(data)
    skipped_locked_count = 0  # 添加跳过计数器

    if args.em_splitter == 'sentence':
        print("loading EN spacy")
        import spacy
        nlp = spacy.load('en_core_web_sm')
        nlp.max_length = 2147483647

    sent_len = []


    for json_obj in tqdm(data):
        worker_seq_id = get_worker_seq_id(cur)
        real_data_id = json_obj.get("id")

        # 对于json_obj,需要先进行替换,适配general场景,这边我们改用knowledge_doc替代
        if 'context' not in json_obj:
            json_obj['context'] = ""
        if 'true_target_dependency' in json_obj:
            json_obj['target_dependency'] = json_obj['true_target_dependency']
        elif 'true_dependency' in json_obj:
            json_obj['dependency'] = json_obj['true_dependency']

        # if rag_config.enableCachedQAContext:
        #     from utils.io_utils import loadJsonl
        #     if os.path.exists(rag_config.QAContextCachePath):
        #         QA_cache_context = loadJsonl(rag_config.QAContextCachePath)
        #     # 直接读取对应id的context (rag_config.QAContextCachePath) getIDContext应该在RAG有对应的函数，设置成一样的
        #     from utils.RAGutils.queryResultByInfer.contextLoad import loadQAContext
        #     json_obj["context"] = loadQAContext(json_obj,rag_config.max_doc_tokens,rag_config.QAContextCachePath)
        # else:
        # 实际下面一步只改了对应的json_obj["context"]
        json_obj, lock_status = appendContextToData(
            json_obj,
            tokenizer,
            max_token_length=args.max_token_length,
            useRAG=True,
            corpus_type=args.corpus_type,
            task_type=task_type,
            max_dependency_num=args.max_dependency_num,
            append_srcDep=args.append_srcDep,
            embedding_source=args.embedding_source,
            query_cache_dir=args.query_cache_dir,
            rag_document_num=args.rag_document_num,
            rag_config=rag_config
        )
        from benchmark.pred_rag import load_error_information
        if args.enableCachedErrorInfo:
            # 直接获取对应id的error_info (args.errorinfos_filepath) getIDErrorInfo应该在RAG有对应的函数，设置成一样的
            error_infos_dict = {}
            if args.errorinfos_filepath:
                logger.info(f"Worker {rank}: Loading error information from {args.errorinfos_filepath}")
                error_infos_dict = load_error_information(args.errorinfos_filepath)
                logger.info(f"Worker {rank}: Loaded error information for {len(error_infos_dict)} samples")
        json_obj["error_info"] = error_infos_dict.get(real_data_id, "")

        if args.generated_target_code_path:
            from benchmark.pred_rag import load_generated_target_code
            target_code_dict = load_generated_target_code(args.generated_target_code_path)
            # 确保选项，将dict的key转为str
            #TODO:和rag一样的临时补丁，等待修复
            target_code_dict = {int(k): v for k, v in target_code_dict.items()}
            json_obj["generated_target_code"] = target_code_dict.get(str(real_data_id), "")
            logging.info(f"Worker {rank}: Loaded generated target code for {real_data_id}")
        # 调用appendContextToData并处理锁状态（已过滤past_ids，无需重复检查）
        

        
        # 检查锁状态
        if lock_status == "locked":
            print(f"Skipping worker_seq_id={worker_seq_id}, real_data_id={real_data_id} due to collection building lock conflict")
            skipped_locked_count += 1
            cur += 1
            continue
        elif lock_status == "error":
            print(f"Skipping worker_seq_id={worker_seq_id}, real_data_id={real_data_id} due to context retrieval error")
            cur += 1
            continue

        # print(f"json_obj['context'] length: {len(json_obj['context'].split())}")

            
        long = len(json_obj['context'].split()) > args.model.disk_offload_threshold
        too_long = long and not (torch.cuda.device_count() >= 3 or (torch.cuda.device_count() >= 2 and args.model.allow_disk_offload) \
                    or (torch.cuda.device_count() == 1 and args.model.vector_offload))
        if too_long:
            print(f"Skipping worker_seq_id={worker_seq_id}, real_data_id={real_data_id} due to length: {len(json_obj['context'].split())}")
            cur += 1
            continue
        
        # 已过滤past_ids，只需检查长度限制
        try:
            prompt = prompt_format.format(**json_obj)
        except Exception as e:
            logger.info(f"Worker {rank}: json_obj: {json_obj.keys()}")
            logger.info(f"Worker {rank}: prompt_format: {prompt_format}")
            logger.error(f"Worker {rank}: Error formatting prompt for {real_data_id}: {e}")
            cur += 1
            return 


        extra_end_token_ids = []
        if conv_type == "llama-3-inst":
            extra_end_token_ids.append(tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0])
        elif conv_type == "phi-3-mini-inst":
            extra_end_token_ids.append(tokenizer.encode("<|end|>", add_special_tokens=False)[0])
        elif conv_type == "qwen":
            extra_end_token_ids.append(tokenizer.encode("<|im_end|>", add_special_tokens=False)[0])
        # elif conv_type == "codegemma":
        #     extra_end_token_ids.append(tokenizer.encode("<eos>", add_special_tokens=False)[0])
        # elif conv_type == "deepseekcoder":
        #     extra_end_token_ids.append(tokenizer.encode("<|EOT|>", add_special_tokens=False)[0])
        if dataset == "samsum":
            extra_end_token_ids.append(tokenizer.encode("\n", add_special_tokens=False)[-1])

        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p","VersiBCB_vace","VersiBCB_vace_BD","VersiBCB_vscc","VersiBCB_vscc_BD","VersiBCB_vace_GEN","VersiBCB_vscc_GEN"]: 
            # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, conv_type, max_length, args)

            if conv_type.strip().lower() in ['mistral-inst']:
                add_special_tokens = False
            else:
                add_special_tokens = True
        else:
            add_special_tokens = True

        if args.em_splitter == 'sentence':
            tokenized_prompt = []
            em_labels = []
            sentences = nlp(prompt).sents
            for sent in sentences:
                sent_inp_ids = tokenizer(sent.text, truncation=False, return_tensors="pt",
                                        add_special_tokens=add_special_tokens).input_ids[0]
                sent_len.append(len(sent_inp_ids))

                tokenized_prompt.append(sent_inp_ids)
                boundaries = torch.zeros(len(sent_inp_ids))
                boundaries[0] = 1
                em_labels.append(boundaries)
            tokenized_prompt = torch.hstack(tokenized_prompt)
            em_labels = torch.hstack(em_labels).bool()
        else:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids[0]
            em_labels = None

        if truncation is None:
            if len(tokenized_prompt) > max_length - max_gen:
                if verbose:
                    print(f"Length {len(tokenized_prompt)}. Skipped.")
                continue

        else:
            if truncation == "suffix":
                length = len(tokenized_prompt)
                if length > max_length - max_gen:
                    if verbose:
                        print("over length")
                    init_token_num = 128
                    prompt = tokenizer.decode(tokenized_prompt[:init_token_num].tolist() + tokenized_prompt[- (max_length - max_gen - init_token_num):].tolist())
                    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids[0]
            else:
                raise NotImplementedError

        offloading_args = {
            "vector_offload_threshold": args.model.vector_offload_threshold,
            "disk_offload_threshold": args.model.disk_offload_threshold
        }
        time1 = time.time()
        output = searcher.generate(
            input_ids=tokenized_prompt,
            em_labels=em_labels,
            max_length=max_gen,
            chunk_size=gen_chunk_size,
            extra_end_token_ids=extra_end_token_ids,
            temperature=args.temperature,
            extra_end_tokens_id_list=extra_end_tokens_id_list,  
            extra_stopwords=extra_stopwords,
            top_p=args.top_p,
            **offloading_args
        )
        time2 = time.time()

        pred = post_process(output["pred"], conv_type, dataset)
        if model_type == "em-llm" and return_block_size:
            block_sizes = [block.size for block in searcher.past_kv[0].global_blocks[0]]
            mean_block_size = np.mean(np.array(block_sizes))
        else:
            block_sizes = None
            mean_block_size = None

        # Extract model parameters for output
        model_params = {}
        if hasattr(args, 'model'):
            model_params = {
                "n_init": getattr(args.model, 'n_init', None),
                "n_mem": getattr(args.model, 'n_mem', None),
                "n_local": getattr(args.model, 'n_local', None),
                "repr_topk": getattr(args.model, 'repr_topk', None),
                "block_size": getattr(args.model, 'max_block_size', None),
                "chunk_size": getattr(args, 'chunk_size', None)
            }
        
        preds.append(
            {
                "id": real_data_id,  # 使用真实的数据ID
                "worker_seq_id": worker_seq_id,  # 添加工作器序列ID用于调试
                "pred": pred, 
                "answers": json_obj.get("answers"), 
                "all_classes": json_obj.get("all_classes"), 
                "length": json_obj.get("length"), 
                "token_length": len(tokenized_prompt) + max_gen, 
                "chunk_ppl": output.get("chunk_ppl"), 
                "total_ppl": output.get("total_ppl"),
                "block_sizes": block_sizes, 
                "mean_block_size": mean_block_size, 
                "generation_time": time2 - time1,
                "model_params": model_params,
                "temperature": args.temperature,
                "top_p": args.top_p
            }
        )

        if verbose:
            print(f"----------{cur}/{total}----------")
            print(f"Worker Seq ID: {worker_seq_id}, Real Data ID: {real_data_id}")
            print("Length: ", len(tokenized_prompt))
            print("Question:", prompt[-100:])
            print("Pred:", pred)
            print("Answer:", json_obj.get("answers"))
            print("")

        with open(out_path, "a+", encoding="utf-8") as f:
            json.dump(preds[-1], f, ensure_ascii=False)
            f.write('\n')

        searcher.clear()
        cur += 1

    if args.em_splitter == 'sentence':
        print(f"Avg sentence length: {sum(sent_len) / len(sent_len)}")
    
    # 输出统计信息
    print(f"Processing completed for worker {rank if rank is not None else 'single'}:")
    print(f"  Total items: {total}")
    print(f"  Successfully processed: {len(preds)}")
    print(f"  Skipped due to collection locks: {skipped_locked_count}")
    if skipped_locked_count > 0:
        print(f"  Note: {skipped_locked_count} items were skipped due to collection building conflicts.")
        print(f"        These items may be processed later when collections are available.")
    
    # 输出映射信息用于调试
    if verbose and world_size is not None:
        print(f"Worker {rank}: Processed {len(worker_seq_to_data_id)} items")
        print(f"Worker {rank}: Worker seq ID range: {min(worker_seq_to_data_id.keys()) if worker_seq_to_data_id else 'N/A'} - {max(worker_seq_to_data_id.keys()) if worker_seq_to_data_id else 'N/A'}")
    
    return preds

class DualLogger:
    def __init__(self, filename, mode='a', rank=0, world_size=1):
        self.terminal = sys.stdout
        self.log = open(filename, mode, buffering=1, encoding='utf-8')  # Line-buffering mode.
        self.rank = rank
        self.world_size = world_size

    def write(self, message):
        worker_message = f"Worker{self.rank}: " + str(message) if self.world_size > 1 else str(message) 
        self.terminal.write(worker_message)
        self.log.write(message)

    def flush(self):  # Needed for compatibility with flush operations
        self.terminal.flush()
        self.log.flush()

def main(args, rag_config):

    if args.enable_extra_stopwords:
        extra_stopwords = json.load(open("benchmark/config/extra_stopwords.json", "r"))
    else:
        extra_stopwords = None
    output_dir_path = args.output_dir_path
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path,exist_ok=True)

    if args.logging:
        log_dir_path = os.path.join(output_dir_path, f"logs/{args.datetime}/")
        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path,exist_ok=True)
        log_path = f'worker_{args.rank}.log'
        sys.stdout = sys.stderr = DualLogger(
            os.path.join(log_dir_path, log_path), 
            rank=args.rank, 
            world_size=args.world_size
        )

    if not args.model.use_hf_acc and args.model.allow_disk_offload:
        args.model.vector_offload = True
    else: 
        args.model.vector_offload = False
    model, tokenizer = get_model_and_tokenizer(args.model, rank=args.rank)
    extra_end_tokens_id_list = []
    if args.enable_extra_stopwords:
        extra_end_tokens_id_list = [tokenizer.encode(word, add_special_tokens=False) for word in extra_stopwords]
    searcher = GreedySearch(
        model, 
        tokenizer, 
        args.model.type, 
        em_splitter=args.em_splitter, 
        compute_ppl=args.compute_ppl,
    )

    datasets = args.datasets
    from benchmark.config.code.dataset2prompt import dataset2prompt
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2maxlen = json.load(open("benchmark/config/dataset2maxlen.json", "r"))

    multiprocessing = args.world_size is not None and args.world_size > 1
    if multiprocessing:
        assert args.rank in list(range(args.world_size))

    # 创建输出管理器
    output_manager = OutputManager(args.base_output_dir)

    # predict on each dataset, dataset2type仅用于生成指定id的query(不会对其他代码产生影响)
    dataset2tasktype = {'VersiBCB_vace':'VACE','VersiBCB_vscc':'VSCC','VersiBCB_vace_BD':'VACE','VersiBCB_vscc_BD':'VSCC',"VersiBCB_vace_GEN":"VACE","VersiBCB_vscc_GEN":"VSCC"}
    
    prompt_type = args.prompt_type
    logger.info(f"Worker {args.rank}: Prompt type: {prompt_type}")
    for i,dataset in enumerate(datasets):
        
        data, dataset = load_data(dataset, extended_passkey=args.get("extended_passkey", 1024))
        #TODO: 这边需要适配memory场景，这边用context替代knowledge_doc,实际上多了一步转换，又转换到context这一命名了，后面看看如何统一
        prompt_format = dataset2prompt["VersiBCB"][prompt_type].replace("{knowledge_doc}", "{context}")
        
        max_gen = dataset2maxlen[dataset]

        # 确定方法类型
        approach = output_manager.get_approach_from_context(
            model_type=args.model.type,
            has_corpus_type=hasattr(args, 'corpus_type') and args.corpus_type
        )
        
        # 生成基础文件名
        memory_cardinfo = {
            "n_init": args.model.n_init,
            "n_mem": args.model.n_mem,
            "n_local": args.model.n_local,
            "repr_topk": args.model.repr_topk,
            "block_size": args.model.max_block_size,
            "chunk_size": args.chunk_size,
        }
        base_filename = output_manager.generate_base_filename(
            dataset=dataset,
            model_name=args.model.path if hasattr(args.model, 'path') else "unknown_model",
            approach=approach,
            corpus_type=args.corpus_type if hasattr(args, 'corpus_type') else None,
            embedding_source=args.embedding_source if hasattr(args, 'embedding_source') else None,
            max_tokens=args.max_token_length if hasattr(args, 'max_token_length') else None,
            inference_type="local",
            memory_cardinfo=memory_cardinfo,
            max_dependency_num=args.max_dependency_num if hasattr(args, 'max_dependency_num') else None,
            useQAContext=rag_config.useQAContext if hasattr(rag_config, 'useQAContext') else False
        )
        
        # 获取输出路径和配置路径
        out_path, config_path = output_manager.get_output_path_and_config(
            approach=approach,
            base_filename=base_filename,
            rank=args.rank,
            world_size=args.world_size,
            newWhenExist=args.newWhenExist,
            model_name_or_path=args.model.path if hasattr(args.model, 'path') else None,

        )
        
        #TODO: 过滤对应的ids ,需考虑与rag创建统一的过滤函数
        if args.skip_no_error_samples:
            from benchmark.pred_rag import load_error_information
            error_infos_dict = load_error_information(args.errorinfos_filepath)
            error_infos_ids = list(error_infos_dict.keys())
            error_str_ids = [str(id) for id in error_infos_ids]
            data = [item for item in data if str(item["id"]) in error_str_ids]
            logger.info(f"Worker {args.rank}: only keep {len(data)} samples with error")
        past_ids = get_past_ids(out_path)

        get_pred(
            searcher=searcher,
            tokenizer=tokenizer,
            model_type=args.model.type,
            data=data,
            max_length=args.max_len,
            max_gen=max_gen,
            prompt_format=prompt_format,
            dataset=dataset,
            conv_type=args.conv_type,
            gen_chunk_size=args.chunk_size,
            truncation=args.truncation,
            rank=args.rank,
            world_size=args.world_size,
            verbose=args.verbose,
            out_path=out_path,
            return_block_size=args.return_block_size,
            args=args,
            rag_config=rag_config,
            past_ids=past_ids,
            task_type=dataset2tasktype.get(dataset), # 用于决定query类型，如果是VSCC就是只用description，如果是VACE就用description+srccode
            extra_end_tokens_id_list=extra_end_tokens_id_list,
            extra_stopwords=extra_stopwords
        )

        # 生成并保存配置文件
        config_data = output_manager.generate_config(
            approach=approach,
            args=args,
            rag_config=rag_config
        )
        output_manager.save_config(config_path, config_data)


if __name__ == '__main__':

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args, rag_config = parse_args()

    main(args, rag_config)
    


