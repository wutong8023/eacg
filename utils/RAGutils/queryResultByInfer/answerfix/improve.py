from utils.io_utils import loadJsonl,writeJsonl
def temp_fix(input_path,output_path):
    data = loadJsonl(input_path)
    exactmatch_info_template = "\nExtra_info:The {target_api} exist in target_dependency where {pack} version is{version}. "
    ragmatch_info_template = "\nExtra_info:The {target_api} do not exist in target_dependency where {pack} version is {version}. "
    for item in data:
        api = item["target_api"] if "target_api" in item else ""
        pack = api.split(".")[0]
        if "dependencies" in item:
            if pack in item["dependencies"]:
                packVersion = item["dependencies"][pack]
                if item["retrieval_method"] == "exact_api_match":
                    extra_info = exactmatch_info_template.format(target_api=api, pack=pack, version=packVersion)
                else:
                    extra_info = ragmatch_info_template.format(target_api=api, pack=pack, version=packVersion)
                # item["extra_info"] = extra_info
                item["answer"] = item["answer"]+extra_info
    writeJsonl(output_path,data)

def appendContextExtraInfo2Answer(context_info_path,answer_info_path,output_path):
    context_info = loadJsonl(context_info_path)
    answer_info = loadJsonl(answer_info_path)
    cnt = 0
    for answer_item in answer_info:
        for context_item in context_info:
            if answer_item["id"] == context_item["id"]:
                if "extra_info" in context_item:
                    answer_item["answer"] = answer_item["answer"]+context_item["extra_info"] 
                    cnt += 1
    print("answer_info length:",len(answer_info))
    print("context_info length:",len(context_info))
    print(f"Appended {cnt} context extra info to answer")
    writeJsonl(output_path,answer_info)
if __name__ == "__main__":
    # context_info_path = "data/temp/contexts/versibcb_vscc_contexts_strmatch_fix.jsonl"
    # answer_info_path = "data/temp/inference/versibcb_vscc_results_extrainfo_hidden.jsonl"
    # output_path = "data/temp/inference/versibcb_vscc_results_extrainfo_explicit.jsonl"
    # appendContextExtraInfo2Answer(context_info_path,answer_info_path,output_path)
    input_path = "data/temp/inference/versibcb_vscc_results_code_review.jsonl"
    output_path = "data/temp/inference/versibcb_vscc_results_code_review_extrainfo.jsonl"
    temp_fix(input_path,output_path)


