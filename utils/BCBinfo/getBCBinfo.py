from datasets import load_from_disk
def load_bcb_dataset(local_folder="../data/BigCodeBench"):
    ds = load_from_disk(local_folder)
    return ds["v0.1.2"]
def getBCBinfoByID(taskid):
    id = taskid.split('/')[1]
    id = int(id)
    ds = load_bcb_dataset()
    return ds[id]
if __name__ == "__main__":
    print(getBCBinfoByID("v0.1.2/1").keys())# dict_keys(['task_id', 'complete_prompt', 'instruct_prompt', 'canonical_solution', 'code_prompt', 'test', 'entry_point', 'doc_struct', 'libs'])
    print('complete_prompt:')
    print(getBCBinfoByID("v0.1.2/1")["complete_prompt"])
    # print(getBCBinfoByID("v0.1.2/1")["instruct_prompt"])
    # print(getBCBinfoByID("v0.1.2/1")["canonical_solution"])
    print('code_prompt:')
    print(getBCBinfoByID("v0.1.2/1")["code_prompt"])
    # print(getBCBinfoByID("v0.1.2/1")["test"])
    # print(getBCBinfoByID("v0.1.2/1")["entry_point"])