
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import argparse

parser = argparse.ArgumentParser(description="Merge LoRA adapter into the base model.")
parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Path to the base model.")
parser.add_argument("--adapter_path", type=str, default="/root/username/distillm-2/outputs/qwen2-vl-2b-distillm_v2-8000", help="Path to the trained LoRA adapter.")
parser.add_argument("--output_path", type=str, default="/root/username/distillm-2/outputs/qwen2-vl-2b-distillm_v2-8000-merged", help="Path to save the merged model.")

args = parser.parse_args()
#base_model = "Qwen/Qwen2-VL-2B-Instruct"
#adapter_path = (
#    "/root/username/distillm-2/outputs/"
#    "qwen2-vl-2b-distillm_v2-8000"
#)
#output_path = (
#    "/root/username/distillm-2/outputs/"
#    "qwen2-vl-2b-distillm_v2-8000-merged"
#)

# Load the original base model.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    args.base_model,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)

# Load and merge the trained LoRA adapter.
model = PeftModel.from_pretrained(model, args.adapter_path)
model = model.merge_and_unload()

# Save a standalone model for vLLM.
model.save_pretrained(
    args.output_path,
    safe_serialization=True,
    max_shard_size="4GB",
)

processor = AutoProcessor.from_pretrained(
    args.base_model,
    trust_remote_code=True,
)
processor.save_pretrained(args.output_path)

print(f"Merged model saved to: {args.output_path}")
