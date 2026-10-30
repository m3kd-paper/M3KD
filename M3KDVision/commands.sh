# without the config, i have deliberately removed: --config_file accelerate_configs/deepspeed_zero3_single_gpu.yaml
# This means Accelerate will launch a normal single-GPU process rather than initializing ZeRO-3 before setup_chat_format().
# My A6000 has ~48 GB, while the 1.5B model is only ~3 GB in BF16, so this is a very reasonable configuration.
accelerate launch \
  --num_processes=1 \
  src/run_sft.py \
  training_configs/qwen2.5-1.5b-sft.yaml




accelerate launch \
  --num_processes=1 \
  src/run_sft.py \
  training_configs/qwen2.5-1.5b-sft-smoke.yaml


#Then create a tiny generation dataset
#We don't want to download/generate 256K examples just to prove the pipeline works.
#Create 100 prompts from the already-available UltraChat data:

python - <<'PY'
from datasets import load_dataset

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

samples = []
for x in ds.select(range(100)):
    user_messages = [m["content"] for m in x["messages"] if m["role"] == "user"]
    if user_messages:
        samples.append({"prompt": user_messages[0]})

smoke = ds.select([]).remove_columns(ds.column_names)

from datasets import Dataset
smoke = Dataset.from_list(samples)
smoke.save_to_disk("smoke_prompts")

print(smoke)
PY



#3. Download the teacher
#Now we're ready for the teacher:

mkdir -p ckpts
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-7B-Instruct"
out = "ckpts/Qwen2.5-7B-Instruct"

print("Downloading tokenizer...")
AutoTokenizer.from_pretrained(model_id).save_pretrained(out)

print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="bfloat16",
)

print("Saving teacher...")
model.save_pretrained(out, safe_serialization=True)

print("Done.")
PY


#4. First test: load the 7B teacher through vLLM
python - <<'PY'
from vllm import LLM

model = LLM(
    model="ckpts/Qwen2.5-7B-Instruct",
    dtype="bfloat16",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.80,
)

print("Teacher loaded successfully.")
PY



#finally, run the smoke test with vLLM and the 1.5B student model
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes=1 \
    src/run_distillm.py \
    training_configs/qwen2.5-1.5b-distillm2-smoke.yaml


                         ┌── 1.5B Qwen
                         │
                         ▼
                    SFT on UltraChat
                         │
                         ▼
                  SFT 1.5B Student
                         │
                         │ generate
                         ▼
                    Student outputs
                         │
                         │
Same prompts ────────────┼──────────────► 7B Teacher
                         │                    │
                         │                    │ generate
                         │                    ▼
                         │              Teacher outputs
                         │                    │
                         └────────┬───────────┘
                                  ▼
                         Reformat / pair
                                  │
                                  ▼
                    ┌──────────────────────┐
                    │ prompt               │
                    │ chosen = teacher     │
                    │ rejected = student   │
                    └──────────────────────┘
                                  │
                                  ▼
                         DistiLLM-2 training
                                  │
                                  ▼
                         Distilled 1.5B model



python - <<'PY'
import json

files = {
    "STUDENT 1.5B": "smoke_generation_student/output_42.json",
    "TEACHER 7B": "smoke_generation_teacher/output_42.json",
    "DISTILLED 1.5B": "smoke_generation_distilled/output_42.json",
}

data = {}
for name, path in files.items():
    with open(path) as f:
        data[name] = json.load(f)

for i in range(len(data["STUDENT 1.5B"])):
    print("\n" + "=" * 100)
    print(f"EXAMPLE {i+1}")
    print("=" * 100)

    print("\nPROMPT:")
    print(data["STUDENT 1.5B"][i]["prompt"])

    for name in files:
        print(f"\n--- {name} ---")
        print(data[name][i]["generated_text"])
PY


python eval/build_evaluation.py \
    --data-path1 smoke_generation_student/output_42.json \
    --data-path2 smoke_generation_distilled/output_42.json \
    --pairwise \
    --output-file smoke_student_vs_distilled \
    --judge gpt-4o


python eval/build_evaluation.py \
    --data-path1 smoke_generation_distilled/output_42.json \
    --data-path2 smoke_generation_student/output_42.json \
    --pairwise \
    --output-file smoke_distilled_vs_student \
    --judge openai/gpt-oss-20b