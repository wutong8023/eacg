from utils.BCBinfo.getBCBinfo import getBCBinfoByID
from utils.versicode_utils.test_line_generate_chunk import bulid_prompt as bulid_prompt_line
from utils.versicode_utils.test_token_generate_chunk import bulid_prompt as bulid_prompt_token
versiBCB_vace_prompt_basic="You are now a professional Python programming engineer. I will provide you with a code snippet and a description of its functionality, including the dependencies and versions used in the code. Then, I will provide the same dependencies but with a specified new version. Your task is to refactor the code using the methods provided by the specified new version and return the refactored code.\nImportant Notes:\n\n. 1.If you encounter ambiguities or uncertainties due to missing external knowledge, clearly state any assumptions you are making to proceed with the refactoring.\n2. Your goal is to produce functional and optimized code that aligns with the new version of the dependencies. 3.Please only return the refactored code.\n\n\n\n### Functionality description of the code\n{description}\n### origin_dependency\n{origin_dependency}\n### Old version code\n{origin_code}\n### target Dependency\n{target_dependency}\n\n### Refactored new code\n"

BD_string = "Note, it is not allowed to use deprecated APIs"
functionality_description_string = "### Functionality description of the code\n{description}\n"
versiBCB_vace_prompt="You are now a professional Python programming engineer. please edit the code to make it compatible with the new version of the dependencies, while keeping the original functionality.\n\n\n{functionality_description_string}\n### origin_dependency\n{origin_dependency}\n### Old version code\n{origin_code}\n### target Dependency\n{target_dependency}\n\n{BD_string}\n### Refactored new code\n{code_start}"
def get_prompt(dataset,task,description,origin_dependency,origin_code,target_dependency,code_start,enable_description=True,enable_BD=True):
    if dataset == 'VersiBCB' or dataset == 'versiBCB':
        if task=='vace':
            prompt_template = versiBCB_vace_prompt
            prompt = prompt_template.format(functionality_description_string=functionality_description_string.format(description=description) if enable_description else '',
                                            origin_dependency=origin_dependency,
                                            origin_code=origin_code,
                                            target_dependency=target_dependency,
                                            BD_string=BD_string if enable_BD else '',
                                            code_start=code_start)
        else:
            raise ValueError(f"Task {task} not supported for dataset {dataset}")
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    return prompt

def formatInput(data,config,instruct=False):

    if config["dataset"] == 'versicode':
        # input_prompt = (config.get("versicode_vscc_prompt") 
        #                 if config["task"] == 'vscc' 
        #                 else config.get("versicode_vace_prompt"))
        
        if config["task"] == 'vscc':
            if config['granularity'] == 'token':
                input = bulid_prompt_token(
                    version=data["dependency"] + data["version"],
                    description=data["description"],
                    masked_code=data["masked_code"]
                )
            elif config['granularity'] == 'line':
                input = bulid_prompt_line(
                    version=data["dependency"] + data["version"],
                    description=data["description"],
                    masked_code=data["masked_code"]
                )
            else:
                raise ValueError(f"Granularity {config['granularity']} not supported")
        else:
            input = input_prompt.format(
                description=data["description"],
                dependency=data["dependency"]
            )
    elif config["dataset"] == "versiBCB":

        enable_description = config["enable_description"]
        code_start = config["code_start"]
        description_instruct_format = config["description_instruct_format"]
        #获取对应的instruct description(if needed)
        BCB_data_info = getBCBinfoByID(data["taskid"])
        description = BCB_data_info["instruct_prompt"] if description_instruct_format else BCB_data_info['description']
        code_start = BCB_data_info["code_prompt"] if code_start else ''

        if instruct:
            input_prompt = config.get("versiBCB_vace_prompt_instruct") if config["task"] == "vace" else config.get("versiBCB_vscc_prompt_instruct")
        else:
            input_prompt = config.get("versiBCB_vace_prompt") if config["task"] == "vace" else config.get("versiBCB_vscc_prompt")
        if config["task"] == "vscc":
            input = input_prompt.format(
                description=data["description"],
                dependency=data["dependency"]
            )
        else:

            # input = get_prompt(
            #     config["dataset"],
            #     config["task"],
            #     description,
            #     data["origin_dependency"],
            #     data["origin_code"],
            #     data["target_dependency"],
            #     code_start=code_start,
            #     enable_description=enable_description,
            #     enable_BD=config["removeDeprecationData"])
            input = input_prompt.format(
                description=data["description"],
                origin_dependency=data["origin_dependency"],
                origin_code=data["origin_code"],
                target_dependency=data["target_dependency"]
            )
    else:
        raise ValueError(f"数据集不存在: {config['dataset']}")
    return input
if __name__ == "__main__":
    print(get_prompt('VersiBCB','vace','','this is description','','',enable_description=True,enable_BD=True))