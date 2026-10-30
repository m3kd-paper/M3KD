
import argparse
import torch

from transformers import Qwen2VLForConditionalGeneration
from peft import PeftModel


parser = argparse.ArgumentParser()
parser.add_argument("--adapter_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
args = parser.parse_args()

BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

print(f"Loading base model: {BASE_MODEL}")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

print(f"Loading LoRA adapter: {args.adapter_path}")

model = PeftModel.from_pretrained(
    model,
    args.adapter_path,
)

print("Merging LoRA weights...")
model = model.merge_and_unload()

print(f"Saving merged model to: {args.output_path}")

model.save_pretrained(
    args.output_path,
    safe_serialization=True,
)

print("Merge completed successfully.")
