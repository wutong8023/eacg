'''
 在buildandloadData的基础上，进行data的加载
'''
from utils.loraTrain.buildandloadData import getDataset,getCodeParrotDataset,getSantaCoderFIMDataset
def load_dataset(config,tokenizer):
    '''
        用于codegemma的若干持续预训练的dataset组合
    '''
    datasets_to_concat = []
    # 1. 加载 QA dataset
    qa_corpus_path = f"/datanfs2/chenrongyi/data/docstring/allEvolveRelated_Docstring.jsonl"
    # print(f"Loading QA dataset from {qa_corpus_path}...")
    # qa_dataset = getDataset(qa_corpus_path,tokenizer,origin_qa=False)
    # if qa_dataset:
    #     datasets_to_concat.append(qa_dataset)
    #     print(f"Loaded QA dataset with {len(qa_dataset)} samples.")

    # 2. 加载 CodeParrot dataset
    code_parrot_num = config.get('codeParrotMixNumber', 0)
    if code_parrot_num > 0:
        print(f"Loading CodeParrot dataset with {code_parrot_num} samples...")
        code_parrot_dataset = getCodeParrotDataset(tokenizer, code_parrot_num)
        if code_parrot_dataset:
            datasets_to_concat.append(code_parrot_dataset)
            print(f"Loaded CodeParrot dataset with {len(code_parrot_dataset)} samples.")

    # 3. 加载 SantaCoder FIM dataset
    santa_parquet_path = config.get('santa_dataset_path', None) # Add path to your config or hardcode
    # Example path if not in config:
    if not santa_parquet_path:
        santa_parquet_path = "/datanfs2/chenrongyi/data/santacoder-fim-task/data/train-00000-of-00001-b6ec1fdd66018baa.parquet"
        print(f"Using default SantaCoder path: {santa_parquet_path}")

    # santa_coder_block_size = config.get('santaCoderBlockSize', 512)
    santa_coder_add_filename = config.get('santaCoderAddFilename', False)
    if santa_parquet_path:
        print("Loading SantaCoder FIM dataset...")
        santa_dataset = getSantaCoderFIMDataset(
            tokenizer,
            parquet_file=santa_parquet_path,
            add_file_name=santa_coder_add_filename,
            num_proc=config.get('numProcDataLoad', 4)
        )
        if santa_dataset:
            datasets_to_concat.append(santa_dataset)
            print(f"Loaded SantaCoder FIM dataset with {len(santa_dataset)} samples.")

    if not datasets_to_concat:
        raise ValueError("No datasets were loaded. Please check paths and configurations.")

    # 合并所有数据集
    print(f"Concatenating {len(datasets_to_concat)} datasets...")
    from torch.utils.data import ConcatDataset
    final_dataset = ConcatDataset(datasets_to_concat)
    print(f"Total samples in final dataset: {len(final_dataset)}")
    return final_dataset