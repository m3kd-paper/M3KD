import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import argparse

parser = argparse.ArgumentParser(
    description="Merge LoRA adapter into the base model."
)
parser.add_argument(
    "--base_model",
    type=str,
    default="Qwen/Qwen3-VL-2B-Instruct",
    help="Path to the base model.",
)
parser.add_argument(
    "--adapter_path",
    type=str,
    required=True,
    help="Path to the trained LoRA adapter.",
)
parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help="Path to save the merged model.",
)

args = parser.parse_args()

# Load the original base model.
model = Qwen3VLForConditionalGeneration.from_pretrained(
    args.base_model,
    torch_dtype=torch.float16,
    device_map="cpu",
)

# Load and merge the trained LoRA adapter.
model = PeftModel.from_pretrained(model, args.adapter_path)
model = model.merge_and_unload()

# Save the standalone merged model.
model.save_pretrained(
    args.output_path,
    safe_serialization=True,
    max_shard_size="4GB",
)

# Save the matching processor.
processor = AutoProcessor.from_pretrained(args.base_model)
processor.save_pretrained(args.output_path)

print(f"Merged model saved to: {args.output_path}")
