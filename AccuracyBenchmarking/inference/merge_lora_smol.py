import argparse
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

parser = argparse.ArgumentParser(
    description="Merge a SmolVLM LoRA adapter into its base model."
)
parser.add_argument(
    "--base_model",
    type=str,
    required=True,
    help="Exact SmolVLM base checkpoint used during training.",
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
    help="Output directory for the merged model.",
)
args = parser.parse_args()

# Load the original SmolVLM base model.
model = AutoModelForImageTextToText.from_pretrained(
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
