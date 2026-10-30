import os
import json
eval_path = "/root/username/distillm-2/evaluation_results/outputs_math"

dataset_name = "gsm8k"
seed = 42

baselines = {
    "Teacher": "Qwen2.5-Math-7B-Instruct",
    "Student": "Qwen2.5-Math-1.5B",
    "Student SFT": "qwen2.5-math-1.5b-sft-50k",

    "GKD":"qwen2.5-math-1.5b-gkd-5000",
    "Distillm-v1":"qwen2.5-math-1.5b-distillm_v1-5000",
    "Distillm-v2":"qwen2.5-math-1.5b-distillm_v2-5000-accumilation8",
    
    #"0": "qwen2.5-math-1.5b-distillm_v2_orth-50000-hidden_loss_weight-0",
    #"0.01":"qwen2.5-math-1.5b-distillm_v2_orth-50000-hidden_loss_weight-0.01",
    "0.1 (old best)": "qwen2.5-math-1.5b-distillm_v2_orth-50000(uncentered-best)",
    #"0.2":"qwen2.5-math-1.5b-distillm_v2_orth-50000-hidden_loss_weight-0.2",

    "lift0.1":"qwen2.5-math-1.5b-distillm_v2_lift-50000-hidden_loss_weight-0.1(random_init)",
    "lift0.2":"qwen2.5-math-1.5b-distillm_v2_lift-50000-hidden_loss_weight-0.2(random_init)",
    "lift0.2_orthinit":"qwen2.5-math-1.5b-distillm_v2_lift-50000-hidden_loss_weight-0.2(orth_init)",

    #"3layer0.1":"qwen2.5-math-1.5b-distillm_v2_orth_proj_3lyrs-50000"

    "0.2(1_lift_1orth)":"qwen2.5-math-1.5b-distillm_v2_lift_orth-50000-hidden_loss_weight-0.2(1_lift_1orth)",
    "0.2(0.2_lift_0.05orth)":"qwen2.5-math-1.5b-distillm_v2_lift_orth-50000-hidden_loss_weight-0.2(0.2_lift_0.05orth)",
    "0.2(0.2_lift_0.01orth)":"qwen2.5-math-1.5b-distillm_v2_lift_orth-50000-hidden_loss_weight-0.2",

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

print("Evaluation Results:",results)