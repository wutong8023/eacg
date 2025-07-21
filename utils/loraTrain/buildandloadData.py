import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
# from benchmark.config.code.config import CORPUS_PATH
from tests.testLoadData import getCodeParrotData
from datasets import load_dataset
from collections import deque
# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Base", trust_remote_code=True)
# 自定义 Dataset
def optimize_sequences(inputids, block_size):
    # 按照长度排序，从短到长排序
    sorted_inputids = sorted(inputids, key=len)
    
    optimized_inputids = []
    input_deque = deque(sorted_inputids)
    
    while input_deque:
        # 取出最短的序列作为基础
        current = input_deque.popleft()
        
        if len(current) == block_size:
            # 已经达到block_size，直接加入
            optimized_inputids.append(current)
            continue
        
        # 创建一个新的列表来存储可能需要删除的索引
        indices_to_remove = []
        
        for i in range(len(input_deque)):
            if len(current) + len(input_deque[i]) <= block_size:
                # 合并序列
                current.extend(input_deque[i])
                indices_to_remove.append(i)
        
        # 删除已经合并过的序列（逆序删除避免影响索引）
        for index in reversed(indices_to_remove):
            del input_deque[index]
        
        # 添加到结果
        optimized_inputids.append(current)
    
    return optimized_inputids
class DocstringDataset(Dataset):
    '''
        用于将对应{'pkg':,'version':,'api_path':,'docstring':}的dict格式数据转换为用于训练序列的dataset
    '''
    def __init__(self,items,tokenizer,block_size=128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.items = items
        self.input_ids = self._tokenize_and_chunk(items)
    def _tokenize_and_chunk(self, items):
        inputids = [] # list of inputids,每一条数据是一个list
        # pkg_version_prefix = f"<{self.pkg} {self.version}>"
        # pkg_version_tokens = self.tokenizer.tokenize(pkg_version_prefix)
        # pkg_version_tokens_ids = self.tokenizer.convert_tokens_to_ids(pkg_version_tokens)
        for item in items:
            pkg = item['pkg']
            version = item['version']
            api_path = item['api_path']
            doc = item['docstring']

            api_name_prefix = f"below is part of the {api_path} docstring."
            api_tokens = self.tokenizer.tokenize(api_name_prefix)
            api_tokens_ids = self.tokenizer.convert_tokens_to_ids(api_tokens)

            pkg_version_prefix = f"<{pkg} {version}>"
            pkg_version_tokens = self.tokenizer.tokenize(pkg_version_prefix)
            pkg_version_tokens_ids = self.tokenizer.convert_tokens_to_ids(pkg_version_tokens)
            
            block_remain_tokens = self.block_size - len(pkg_version_tokens_ids) - len(api_tokens_ids)
            doc_tokens = self.tokenizer.tokenize(doc)
            doc_tokens_ids = self.tokenizer.convert_tokens_to_ids(doc_tokens)
            # 对于当前docstring，需要分块，每块大小为block_size，然后加入到inputids中
            for i in range(0,len(doc_tokens_ids),block_remain_tokens):
                chunk = doc_tokens_ids[i:i+block_remain_tokens]
                inputids.append(pkg_version_tokens_ids + api_tokens_ids + chunk)
        # print("before optimize",len(inputids))
        # optimized_inputids = optimize_sequences(inputids,self.block_size)
        # print("after optimize",len(optimized_inputids))
        return inputids
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        labels = input_ids[1:] + [self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id]
        if len(labels) < len(input_ids):
            labels.extend([self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id] * (len(input_ids) - len(labels)))
        elif len(labels) > len(input_ids):
            labels = labels[:len(input_ids)]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
class DocstringDataset1(Dataset):
    '''
        用于将对应{'pkg':,'version':,'api_path':,'docstring':}的dict格式数据转换为用于训练序列的dataset
        支持从预处理好的input_ids直接构造
    '''
    def __init__(self, items=None, tokenizer=None, block_size=128, pkg=None, version=None, input_ids=None):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.items = items
        self.pkg = pkg
        self.version = version
        
        if input_ids is not None:
            # 从预处理好的数据构造
            self.input_ids = input_ids
        elif items is not None:
            # 从原始数据进行分词和切块
            self.input_ids = self._tokenize_and_chunk(items)
        else:
            raise ValueError("必须提供 items 或 input_ids 其中之一")

    def _tokenize_and_chunk(self, items):
        inputids = [] # list of inputids,每一条数据是一个list
        # pkg_version_prefix = f"<{self.pkg} {self.version}>"
        # pkg_version_tokens = self.tokenizer.tokenize(pkg_version_prefix)
        # pkg_version_tokens_ids = self.tokenizer.convert_tokens_to_ids(pkg_version_tokens)
        for item in items:
            pkg = self.pkg
            version = self.version
            api_path = item['path']
            doc = item['doc']

            api_name_prefix = f"below is part of the {api_path} docstring."
            api_tokens = self.tokenizer.tokenize(api_name_prefix)
            api_tokens_ids = self.tokenizer.convert_tokens_to_ids(api_tokens)

            pkg_version_prefix = f"<{pkg} {version}>"
            pkg_version_tokens = self.tokenizer.tokenize(pkg_version_prefix)
            pkg_version_tokens_ids = self.tokenizer.convert_tokens_to_ids(pkg_version_tokens)
            
            block_remain_tokens = self.block_size - len(pkg_version_tokens_ids) - len(api_tokens_ids)
            doc_tokens = self.tokenizer.tokenize(doc)
            doc_tokens_ids = self.tokenizer.convert_tokens_to_ids(doc_tokens)
            # 对于当前docstring，需要分块，每块大小为block_size，然后加入到inputids中
            for i in range(0,len(doc_tokens_ids),block_remain_tokens):
                chunk = doc_tokens_ids[i:i+block_remain_tokens]
                inputids.append(pkg_version_tokens_ids + api_tokens_ids + chunk)
        # print("before optimize",len(inputids))
        # optimized_inputids = optimize_sequences(inputids,self.block_size)
        # print("after optimize",len(optimized_inputids))
        return inputids
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        labels = input_ids[1:] + [self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id]
        if len(labels) < len(input_ids):
            labels.extend([self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id] * (len(input_ids) - len(labels)))
        elif len(labels) > len(input_ids):
            labels = labels[:len(input_ids)]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
class SrccodeDataset(Dataset):
    '''
        用于将对应{'library_name':,'version_num':,'file_path':,'source_code':}的dict格式数据转换为用于训练序列的dataset
    '''
    def __init__(self,items,tokenizer,block_size=128,pkg=None,version=None):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.items = items
        self.pkg = pkg
        self.version = version
        self.input_ids = self._tokenize_and_chunk(items)

    def _tokenize_and_chunk(self, items):
        inputids = [] # list of inputids,每一条数据是一个list
        # pkg_version_prefix = f"<{self.pkg} {self.version}>"
        # pkg_version_tokens = self.tokenizer.tokenize(pkg_version_prefix)
        # pkg_version_tokens_ids = self.tokenizer.convert_tokens_to_ids(pkg_version_tokens)
        for item in items:
            pkg = self.pkg
            version = self.version
            file_path = item['file_path']
            source_code = item['source_code']

            api_name_prefix = f"below is part of the {file_path} docstring."
            api_tokens = self.tokenizer.tokenize(api_name_prefix)
            api_tokens_ids = self.tokenizer.convert_tokens_to_ids(api_tokens)

            pkg_version_prefix = f"<{pkg} {version}>"
            pkg_version_tokens = self.tokenizer.tokenize(pkg_version_prefix)
            pkg_version_tokens_ids = self.tokenizer.convert_tokens_to_ids(pkg_version_tokens)
            
            block_remain_tokens = self.block_size - len(pkg_version_tokens_ids) - len(api_tokens_ids)
            source_code_tokens = self.tokenizer.tokenize(source_code)
            source_code_tokens_ids = self.tokenizer.convert_tokens_to_ids(source_code_tokens)
            # 对于当前docstring，需要分块，每块大小为block_size，然后加入到inputids中
            for i in range(0,len(source_code_tokens_ids),block_remain_tokens):
                chunk = source_code_tokens_ids[i:i+block_remain_tokens]
                inputids.append(pkg_version_tokens_ids + api_tokens_ids + chunk)
        # print("before optimize",len(inputids))
        # optimized_inputids = optimize_sequences(inputids,self.block_size)
        # print("after optimize",len(optimized_inputids))
        return inputids
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        labels = input_ids[1:] + [self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id]
        if len(labels) < len(input_ids):
            labels.extend([self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id] * (len(input_ids) - len(labels)))
        elif len(labels) > len(input_ids):
            labels = labels[:len(input_ids)]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=128,pkg=None,version=None):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.pkg = pkg
        self.version = version
        self.input_ids = self._tokenize_and_chunk(texts)

    def _tokenize_and_chunk(self, texts):
        textid2inputids = [] # list of inputids,每一条数据是一个list
        # pkg_version_prefix = f"<{self.pkg} {self.version}>"
        # pkg_version_tokens = self.tokenizer.tokenize(pkg_version_prefix)
        # pkg_version_tokens_ids = self.tokenizer.convert_tokens_to_ids(pkg_version_tokens)
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            textid2inputids.append(self.tokenizer.convert_tokens_to_ids(tokens))
        # 按 block_size 分块
        text_chunks = []
        for textid2inputid in textid2inputids:
            chunks = [
                textid2inputid[i : i + self.block_size]
                for i in range(0, len(textid2inputid), self.block_size)
            ]
            text_chunks.extend(chunks)
        return text_chunks

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        labels = input_ids[1:] + [self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id]
        if len(labels) < len(input_ids):
            labels.extend([self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id] * (len(input_ids) - len(labels)))
        elif len(labels) > len(input_ids):
            labels = labels[:len(input_ids)]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
def encode_with_left_truncation(tokenizer, full_text, max_length, only_tokenize=False):
    '''
    description:
        对输入文本进行左截断，并返回tokenized的input_ids或直接返回tokenized的tokens
    Args:
        tokenizer: 分词器
        full_text: 输入文本
        max_length: 最大长度
        only_tokenize: 是否只返回tokenized的tokens
    Returns:
        encoded_input: tokenized的input_ids或tokenized的tokens
    '''
    # 使用原生tokenizer函数，但手动截断tokens
    # 先使用tokenize获取tokens
    tokens = tokenizer.tokenize(full_text)
    
    # 如果超过最大长度，从左侧截断
    if len(tokens) > max_length:
        truncated_tokens = tokens[-max_length:]
    else:
        truncated_tokens = tokens
    
    if only_tokenize:
        return truncated_tokens
    else:
        # 将tokens连接回文本
        truncated_text = tokenizer.convert_tokens_to_string(truncated_tokens)
        
        # 使用原生tokenizer函数处理截断后的文本，这样返回的是BatchEncoding对象
        # BatchEncoding对象有.to(device)方法
        return tokenizer(
            truncated_text,
            truncation=False,  # 已经截断过了
            padding=False,
            return_tensors="pt"
        )
class QADataset(Dataset):
    def __init__(self, qa_pairs, tokenizer, max_length=1024, query_prefix="", answer_prefix=""):
        """
        为因果语言模型(Causal LM)设计的QA数据集
        
        Args:
            qa_pairs: 问答对列表，每个元素为 (query_str, answer_str) 的元组
            tokenizer: 用于文本编码的tokenizer
            max_length: 序列最大长度
            query_prefix: 问题前缀
            answer_prefix: 回答前缀
        """
        self.qa_pairs = qa_pairs  # list of tuple[query_str, answer_str]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.query_prefix = query_prefix
        self.answer_prefix = answer_prefix
        
        # 确保tokenizer有正确的padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        """获取单个样本并进行处理"""
        query, answer = self.qa_pairs[idx]
        
        # 添加前缀并创建完整文本
        query_text = f"{self.query_prefix}{query}"
        answer_text = f"{self.answer_prefix}{answer}"
        full_text = f"{query_text}{answer_text}"
        

        # 编码完整文本，从左侧进行裁剪，保留右侧max_length的tokens
        # encoding = self.tokenizer(
        #     full_text,
        #     truncation=True,
        #     max_length=self.max_length,
        #     padding=False,
        #     return_tensors="pt",
        #     truncation_side='left'  # 指定从左侧进行裁剪
        # )
        encoding = encode_with_left_truncation(self.tokenizer,full_text,self.max_length)
        # 提取input_ids并压缩维度
        input_ids = encoding["input_ids"].squeeze()
        
        # 对于因果LM，标签就是输入序列（每个标记预测下一个标记）
        # 但我们仍然可以使用-100屏蔽问题部分的损失计算
        # 找到问题和答案部分的分界点
        query_encoding = self.tokenizer(
            query_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        query_length = query_encoding["input_ids"].size(1)
        
        # 创建标签: 问题部分用-100，答案部分用实际token
        # 注意: 答案标签需要偏移以对应next token prediction
        labels = torch.full_like(input_ids, -100)
        
        # 标记答案部分: 从查询结束位置开始，但标签向左偏移一个位置
        if query_length < len(input_ids):
            # 将答案部分的input_ids复制到labels，但偏移一位
            # 前query_length-1个位置保持为-100
            labels[query_length-1:-1] = input_ids[query_length:]
            # 最后一个标签对应EOS或补充
            if query_length < len(input_ids):
                labels[-1] = self.tokenizer.eos_token_id
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "query": query,
            "answer": answer
        }
    
    @staticmethod
    def collate_fn(batch,tokenizer):
        """
        数据批处理函数
        
        Args:
            batch: 样本列表
            
        Returns:
            处理后的批次数据
        """
        # 提取所有输入和标签
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        # 获取最大长度
        max_len = max(len(ids) for ids in input_ids)
        
        # 填充到相同长度
        padded_input_ids = []
        padded_labels = []
        attention_mask = []
        
        for ids, lab in zip(input_ids, labels):
            # 计算填充长度
            padding_len = max_len - len(ids)
            
            # 填充输入ID (使用pad_token_id填充)
            pad_token_id = ids.new_ones(padding_len) * ids.new_tensor([tokenizer.pad_token_id])  # 通常pad_token_id为0
            padded_ids = torch.cat([ids, pad_token_id])
            padded_input_ids.append(padded_ids)
            
            # 填充标签 (用-100填充，PyTorch会忽略这些位置的损失)
            padded_lab = torch.cat([
                lab, 
                torch.full((padding_len,), -100, dtype=torch.long)
            ])
            padded_labels.append(padded_lab)
            
            # 创建注意力掩码
            mask = torch.cat([
                torch.ones(len(ids)), 
                torch.zeros(padding_len)
            ])
            attention_mask.append(mask)
        
        # 堆叠为批次
        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(padded_labels),
        }

# --- New FIMDataset Class ---
class FIMDataset(Dataset):
    def __init__(self, texts, tokenizer):
        """
        Dataset specifically for Fill-in-the-Middle (FIM) tasks formatted for CodeGemma.

        Args:
            texts (list[str]): List of text sequences already formatted with FIM tokens.
            tokenizer: The tokenizer instance.
        """
        self.tokenizer = tokenizer
        # Get special token IDs once
        self.fim_middle_id = self.tokenizer.convert_tokens_to_ids("<|fim_middle|>")
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        # Pre-tokenize sequences
        self.sequences = self._tokenize_texts(texts)

    def _tokenize_texts(self, texts):
        tokenized_sequences = []
        print(f"Tokenizing {len(texts)} FIM sequences...")
        # Determine max length, considering model capabilities
        max_length = 2048
        print(f"Using max_length: {max_length}")
        for text in texts:
            # Tokenize the entire text sequence for FIM, handle truncation
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=False, # Usually False for pre-formatted FIM strings
                truncation=True,
                max_length=max_length
            )
            tokenized_sequences.append(tokens)
        print("Tokenization complete.")
        return tokenized_sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_ids = self.sequences[idx]
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)

        # Initialize labels with -100 (masked tokens)
        labels = torch.full_like(input_ids_tensor, -100)

        try:
            # Find the index of the <fim_middle> token
            # Use torch.where instead of nonzero for clarity and direct index access
            fim_middle_indices = torch.where(input_ids_tensor == self.fim_middle_id)[0]

            if fim_middle_indices.numel() > 0:
                fim_middle_idx = fim_middle_indices[0].item() # Get the first occurrence

                # Set labels for tokens *after* <fim_middle>
                # Labels should be the next tokens after each position
                target_len = len(input_ids) - (fim_middle_idx + 1)
                if target_len > 0:
                    # Copy the tokens after fim_middle to labels
                    labels[fim_middle_idx:fim_middle_idx+target_len] = input_ids_tensor[fim_middle_idx+1:]
                    
                    # Ensure the last position predicts EOS token
                    if not input_ids_tensor[-1] == self.eos_token_id:
                        # Add EOS token to input_ids if it doesn't end with one
                        input_ids_tensor = torch.cat([
                            input_ids_tensor, 
                            torch.tensor([self.eos_token_id], dtype=torch.long)
                        ])
                        # Extend labels accordingly
                        labels = torch.cat([
                            labels,
                            torch.tensor([self.eos_token_id], dtype=torch.long)
                        ])

            else:
                # If <fim_middle> is not found, log a warning. Labels remain -100.
                print(f"Warning: <fim_middle> token ID {self.fim_middle_id} not found in sequence index {idx}. All labels masked.")

        except Exception as e: # Catch potential errors during tensor operations
            print(f"Error processing sequence index {idx}: {e}. All labels masked.")
            # Ensure labels remain fully masked in case of error
            labels = torch.full_like(input_ids_tensor, -100)

        return {
            "input_ids": input_ids_tensor,
            "labels": labels,
        }

# --- End of FIMDataset Class ---

# --- New FIMEvalDataset Class specifically for evaluation/prediction ---
class FIMEvalDataset(Dataset):
    def __init__(self, texts, tokenizer):
        """
        Dataset specifically for Fill-in-the-Middle (FIM) inference tasks.
        Unlike FIMDataset, this only keeps the prefix and suffix parts (before <|fim_middle|>)
        for prediction.

        Args:
            texts (list[str]): List of text sequences already formatted with FIM tokens.
            tokenizer: The tokenizer instance.
        """
        self.tokenizer = tokenizer
        self.fim_middle_id = self.tokenizer.convert_tokens_to_ids("<|fim_middle|>")
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        
        # Parse and store original texts for reference
        self.original_texts = texts
        
        # Pre-tokenize sequences, but only keep parts before <|fim_middle|>
        self.sequences, self.original_prefixes, self.gold_solutions = self._tokenize_texts(texts)

    def _tokenize_texts(self, texts):
        tokenized_sequences = []
        original_prefixes = []  # Store original prefix+suffix text
        gold_solutions = []     # Store original solutions
        
        print(f"Tokenizing {len(texts)} FIM sequences for evaluation...")
        max_length = 2048
        
        for text in texts:
            # Find the position of <|fim_middle|> in the original text
            middle_pos = text.find("<|fim_middle|>")
            
            if middle_pos != -1:
                # Split text into prefix+suffix and solution
                prefix_suffix = text[:middle_pos + len("<|fim_middle|>")]  # Include the token
                solution = text[middle_pos + len("<|fim_middle|>"):]
                
                # Tokenize only the prefix+suffix part
                tokens = self.tokenizer.encode(
                    prefix_suffix,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_length
                )
                
                # Ensure input_ids ends with the fim_middle_id token, not with an EOS token for evaluation
                # Since we want the model to continue generating after the fim_middle token
                if len(tokens) > 0 and tokens[-1] == self.eos_token_id and tokens[-2] == self.fim_middle_id:
                    tokens = tokens[:-1]  # Remove EOS if it follows fim_middle
                
                tokenized_sequences.append(tokens)
                original_prefixes.append(prefix_suffix)
                gold_solutions.append(solution)
            else:
                print(f"Warning: <|fim_middle|> not found in text, skipping: {text[:100]}...")
                continue
                
        print(f"Tokenization complete. Processed {len(tokenized_sequences)} valid sequences.")
        return tokenized_sequences, original_prefixes, gold_solutions

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_ids = self.sequences[idx]
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        
        return {
            "input_ids": input_ids_tensor,
            "original_prefix": self.original_prefixes[idx],
            "gold_solution": self.gold_solutions[idx]
        }

# --- End of FIMEvalDataset Class ---

def getDataset(corpus_path,tokenizer,origin_qa=True):
    '''
    Description:
        由原始数据源构建dataset
    Args:
        corpus_path: str, 原始数据源路径,对应的数据存储dict的key包括pkg,version,api,docstring
    Returns:
        dataset: Dataset, 构建的dataset
    '''
    query_prompt_template = "please generate a description for {api} of {pkg} {version} in docstring format"

    qa_pairs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if origin_qa:
                qa_pairs.append((data["query"],data["answer"]))
            else:
                query_prompt = query_prompt_template.format(api=data["api"],pkg=data["pkg"],version=data["version"])
                qa_pairs.append((query_prompt,data["docstring"]))
    # 根据qa_pairs构建dataset
    dataset = QADataset(qa_pairs, tokenizer,max_length=1024)
    return dataset
def getCodeParrotDataset(tokenizer,number):
    '''
    Description:
        由原始数据源构建dataset
    '''
    code_parrot_data = getCodeParrotData(number)
    code_parrot_content = [data["content"] for data in code_parrot_data]
    dataset = TextDataset(code_parrot_content,block_size=512,tokenizer=tokenizer)
    return dataset

# --- Added SantaCoder FIM Dataset Loading (Modified to use FIMDataset) ---
# Define the special tokens used by CodeGemma for FIM tasks
FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"
FILE_SEPARATOR = "<|file_separator|>"

def transform_santacoder_to_codegemma_fim(example, add_file_name=False):
    """
    Transforms a single example from the SantaCoder dataset format
    to the CodeGemma Fill-in-the-Middle (FIM) fine-tuning format.
    """
    filename = example.get('name', '')
    prompt = example.get('prompt', '')
    suffix = example.get('suffix', '')
    middle = example.get('canonical_solution', '') # Use canonical_solution for the middle part

    # Construct the FIM formatted string
    if add_file_name:
        # Add filename and ensure file_separator is at the very end
        fim_text = f"{filename}\n{FIM_PREFIX}{prompt}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}"
    else:
        fim_text = f"{FIM_PREFIX}{prompt}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}"

    return {"codegemma_fim_text": fim_text}

def getSantaCoderFIMDataset(tokenizer, parquet_file, add_file_name=False, num_proc=4, for_eval=False):
    """
    Loads the SantaCoder dataset, transforms it to CodeGemma FIM format,
    and returns an FIMDataset or FIMEvalDataset based on the for_eval parameter.
    
    Args:
        tokenizer: The tokenizer to use
        parquet_file: Path to the SantaCoder parquet file
        add_file_name: Whether to include file names in the FIM text
        num_proc: Number of processes for parallel mapping
        for_eval: If True, returns FIMEvalDataset for inference (only prefix+suffix parts)
    """
    try:
        print(f"Loading SantaCoder dataset from {parquet_file}...")
        santa_dataset = load_dataset("parquet", data_files=parquet_file, split="train")
        print(f"Successfully loaded {len(santa_dataset)} examples.")

        print("Transforming dataset to CodeGemma FIM format...")
        codegemma_dataset = santa_dataset.map(
            lambda x: transform_santacoder_to_codegemma_fim(x, add_file_name=add_file_name),
            num_proc=num_proc,
            remove_columns=santa_dataset.column_names # Keep only the transformed text
        )
        print("Transformation complete.")

        # Extract the transformed text
        fim_texts = [item['codegemma_fim_text'] for item in codegemma_dataset]

        # Create appropriate dataset based on for_eval flag
        print(f"Creating {'FIMEvalDataset' if for_eval else 'FIMDataset'}...")
        if for_eval:
            fim_dataset = FIMEvalDataset(fim_texts, tokenizer)
        else:
            fim_dataset = FIMDataset(fim_texts, tokenizer)
            
        print(f"Dataset created with {len(fim_dataset)} samples.")
        return fim_dataset

    except Exception as e:
        print(f"Error processing SantaCoder dataset: {e}")
        return None

# --- End of Added SantaCoder FIM Code ---

def collate_fn(batch, tokenizer):
    """
    Generic collate function that handles padding for input_ids and labels.
    Assumes batch items are dictionaries with 'input_ids' and 'labels'.
    For causal language models, padding should be applied to the left side.
    Labels are padded with -100, inputs with tokenizer.pad_token_id.
    """
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    label_pad_token_id = -100  # Standard ignore index

    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Get lengths for left-side padding
    max_length = max(len(ids) for ids in input_ids)
    
    # Manually pad on the left side
    padded_input_ids = []
    padded_labels = []
    attention_mask = []
    
    for ids, lab in zip(input_ids, labels):
        padding_len = max_length - len(ids)
        
        # Left padding for inputs (for causal LM)
        input_padding = torch.full((padding_len,), pad_token_id, dtype=torch.long)
        padded_ids = torch.cat([input_padding, ids])
        padded_input_ids.append(padded_ids)
        
        # Left padding for labels (also with -100)
        label_padding = torch.full((padding_len,), label_pad_token_id, dtype=torch.long)
        padded_lab = torch.cat([label_padding, lab])
        padded_labels.append(padded_lab)
        
        # Attention mask (0 for padding, 1 for content)
        mask = torch.cat([
            torch.zeros(padding_len, dtype=torch.long),
            torch.ones(len(ids), dtype=torch.long)
        ])
        attention_mask.append(mask)
    
    # Stack tensors
    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(padded_labels)
    }

class LazyDocstringDataset(Dataset):
    '''
        用于惰性加载数据的数据集，适用于大规模数据
    '''
    def __init__(self, data_generator, tokenizer, block_size=128, cache_size=1000):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data_generator = data_generator
        self.cache_size = cache_size  # 缓存大小
        self.cache = []  # 数据缓存
        self.cache_indexes = {}  # 缓存索引映射
        self.total_items = 0
        self.processed_items = 0
        
        # 预处理一部分数据以确定总长度和缓存一部分结果
        self._preload_cache()
    
    def _preload_cache(self):
        """预先加载部分数据到缓存"""
        batch = []
        print("预加载数据缓存...")
        
        # 尝试加载指定数量的数据到缓存
        for i, item in enumerate(self.data_generator):
            if i < self.cache_size:
                self.processed_items += 1
                processed_item = self._process_item(item)
                for chunk in processed_item:
                    self.cache.append(chunk)
                    self.cache_indexes[self.total_items] = len(self.cache) - 1
                    self.total_items += 1
            else:
                # 记录还有数据可用
                self.more_data = True
                break
            
            if i % 100 == 0:
                print(f"已预加载 {i+1} 条数据，产生 {self.total_items} 个训练样本")
        
        print(f"缓存预加载完成，共处理 {self.processed_items} 条原始数据，产生 {self.total_items} 个训练样本")
    
    def _process_item(self, item):
        """处理单个数据项"""
        result = []
        pkg = item['pkg']
        version = item['version']
        api_path = item['api_path']
        doc = item['docstring']
        
        api_name_prefix = f"below is part of the {api_path} docstring."
        api_tokens = self.tokenizer.tokenize(api_name_prefix)
        api_tokens_ids = self.tokenizer.convert_tokens_to_ids(api_tokens)
        
        pkg_version_prefix = f"<{pkg} {version}>"
        pkg_version_tokens = self.tokenizer.tokenize(pkg_version_prefix)
        pkg_version_tokens_ids = self.tokenizer.convert_tokens_to_ids(pkg_version_tokens)
        
        block_remain_tokens = self.block_size - len(pkg_version_tokens_ids) - len(api_tokens_ids)
        doc_tokens = self.tokenizer.tokenize(doc)
        doc_tokens_ids = self.tokenizer.convert_tokens_to_ids(doc_tokens)
        
        # 对于当前docstring，需要分块，每块大小为block_size，然后加入到inputids中
        for i in range(0, len(doc_tokens_ids), block_remain_tokens):
            chunk = doc_tokens_ids[i:i+block_remain_tokens]
            result.append(pkg_version_tokens_ids + api_tokens_ids + chunk)
        
        return result
    
    def _load_more_if_needed(self, idx):
        """如果需要，加载更多数据"""
        if idx in self.cache_indexes:
            return
            
        # 继续处理数据直到找到目标索引
        try:
            while True:
                item = next(self.data_generator)
                self.processed_items += 1
                processed_item = self._process_item(item)
                
                for chunk in processed_item:
                    self.cache.append(chunk)
                    self.cache_indexes[self.total_items] = len(self.cache) - 1
                    self.total_items += 1
                    
                    if self.total_items - 1 >= idx:
                        return
                        
                if self.processed_items % 100 == 0:
                    print(f"已处理 {self.processed_items} 条原始数据，产生 {self.total_items} 个训练样本")
                    
        except StopIteration:
            # 数据已全部处理完毕
            print(f"所有数据已处理完毕：{self.processed_items} 条原始数据，{self.total_items} 个训练样本")
    
    def __len__(self):
        return self.total_items
    
    def __getitem__(self, idx):
        if idx >= self.total_items:
            raise IndexError(f"索引 {idx} 超出范围 (0-{self.total_items-1})")
            
        # 如果索引不在缓存中，加载更多数据
        self._load_more_if_needed(idx)
        
        # 获取缓存的索引
        cache_idx = self.cache_indexes.get(idx)
        if cache_idx is None:
            raise ValueError(f"无法找到索引 {idx} 的缓存数据")
            
        input_ids = self.cache[cache_idx]
        labels = input_ids[1:] + [self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id]
        
        if len(labels) < len(input_ids):
            labels.extend([self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id] * (len(input_ids) - len(labels)))
        elif len(labels) > len(input_ids):
            labels = labels[:len(input_ids)]
            
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long)
        }

if __name__ == "__main__":
    # 加载文档
    files_info = []
    with open("/mnt/d/codes/Corpus/downstream_application_code/version_corpus/accelerate/0.15.0.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line_data = json.loads(line)
            files_info.append(str(line_data))
            # print(str(line_data))

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Base", trust_remote_code=True)

    # 创建 Dataset 和 DataLoader
    dataset = TextDataset(files_info, tokenizer, block_size=128)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 打印示例
    for batch in dataloader:
        print(batch)
        break