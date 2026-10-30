import os
import json
import pprint

eval_path = "evaluation_results/outputs"

dataset_name = "gsm8k"
seed = 42

baselines = {

    "Projection":"qwen2.5-math-1.5b-distillm_v2_orth_proj-50000",
    "PCA + Orth + Projection":"qwen2.5-math-1.5b-distillm_v2_orth_proj-50000",
    "Random + NoOrth":"qwen2.5-math-1.5b-distillm_v2_NOorth_random-50000",
    "Random + Orth":"qwen2.5-math-1.5b-distillm_v2_orth_random-50000",

    "PCA + Orth (proposed)": "qwen2.5-math-1.5b-distillm_v2_orth-50000",
}

results = {}
for k,value in baselines.items():
    folder_path = os.path.join(eval_path, value)
    if not os.path.isdir(folder_path):
        print(f"Folder {folder_path} does not exist.")
        continue
    else:
        file_name = f"{dataset_name}_seed{seed}.json"
        result_file = os.path.join(folder_path, file_name)

        with open(result_file, "r") as f:
            json_content = json.load(f)
            acc = json_content["summary"]["accuracy"]
            results[k] = acc

print("Evaluation Results:")
pprint.pprint(results, sort_dicts=False)