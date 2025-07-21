# 统计各个长度的分布 0-256 256-512 512-768 768-1024 1024-1e9
import json
from transformers import AutoTokenizer
from utils.loraTrain.buildandloadData import optimize_sequences
numpy_version = '1.19.5'
CORPUS_PATH='/datanfs2/chenrongyi/data/docstring/numpy/'
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

# 统计各个长度的分布 0-256 256-512 512-768 768-1024 1024-2048 2048-4096 4096-1e9
length_dict = {'0-256': 0, '256-512': 0, '512-768': 0, '768-1024': 0, '1024-2048': 0,'2048-4096': 0,'4096-1e9': 0}
tokenized_docs_length = []
tokenized_docs = []
with open(f'{CORPUS_PATH}/{numpy_version}.jsonl', 'r') as f:
    for line in f:
        doc = json.loads(line)
        if 'doc' not in doc:
            continue
        tokenized_doc = tokenizer.tokenize(doc['doc'])
        if len(tokenized_doc) > 1024:
            for i in range(0,len(tokenized_doc),1024):
                tokenized_docs.append(tokenized_doc[i:i+1024])
        else:
            tokenized_docs.append(tokenized_doc)
        
        tokenized_docs_length.append(len(tokenized_doc))
print(len(tokenized_docs))
tokenized_docs = optimize_sequences(tokenized_docs,1024)
print(len(tokenized_docs))
# 统计各个长度在不同区间的分布
for length in tokenized_docs_length:
    if length < 256:
        length_dict['0-256'] += 1
    elif length < 512:
        length_dict['256-512'] += 1
    elif length < 768:
        length_dict['512-768'] += 1
    elif length < 1024:
        length_dict['768-1024'] += 1
    elif length < 2048:
        length_dict['1024-2048'] += 1
    elif length < 4096:
        length_dict['2048-4096'] += 1
    else:
        length_dict['4096-1e9'] += 1

print(length_dict)
