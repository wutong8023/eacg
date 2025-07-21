import json
from benchmark.config.code.config_lora import load_config
LORA_CONFIG_PATH = 'benchmark/config/code/config_lora.yaml'
config = load_config(LORA_CONFIG_PATH)


input_prompt = config.get("versiBCB_vace_prompt")

datas = json.load(open("data/VersiBCB_Benchmark/model_predictions/codegemma-7b-it/vace_datas.json"))

output_datas = []
for data in datas:
    for pred in data["model_output"]:
        input = input_prompt.format(
            description=data["description"],
            origin_dependency=data["origin_dependency"],
            origin_code=data["origin_code"],
            target_dependency=data["target_dependency"]
        )
        output = pred
        output_datas.append({
            "instruction": "",
            "input": input,
            "output": output
        })

json.dump(output_datas, open("data/VersiBCBIFT/selfinducedDistribution.json", "w"), indent=4)
