"""
DriveBench batch inference using SmolVLM / SmolVLM2 with Ray Data.

Example:
    python inference/smolvlm.py \
        --model HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
        --data data/data/drivebench-test-simple.json \
        --output res/smolvlm2-500m/biterror \
        --system_prompt prompt.txt \
        --num_processes 1 \
        --num_images_per_prompt 6 \
        --corruption BitError
"""

import os
import argparse
import warnings
from typing import Any, Dict

import numpy as np
import ray
import torch
from PIL import Image
from packaging.version import Version
from transformers import AutoModelForImageTextToText, AutoProcessor

from utils import replace_system_prompt, load_json_files, save_json


assert Version(ray.__version__) >= Version("2.22.0"), (
    "Ray version must be at least 2.22.0"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="DriveBench SmolVLM batch inference"
    )

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--system_prompt", type=str, required=True)

    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--num_images_per_prompt", type=int, default=6)
    parser.add_argument("--corruption", type=str, default="")

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=512)

    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Resize images to this square size; use 0 to preserve original resolution.",
    )

    return parser.parse_args()


class SmolVLMPredictor:
    def __init__(
        self,
        model_name,
        system_prompt,
        sampling_params,
        num_images_per_prompt,
        max_model_len,
        tensor_parallel_size,
        corruption,
        image_size,
    ):
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

        self.system_prompt = system_prompt
        self.sampling_params = sampling_params
        self.corruption = corruption
        self.num_images_per_prompt = num_images_per_prompt
        self.image_size = image_size

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        questions = batch["question"]
        filenames = batch["image_path"]

        prompts = []
        preds = []

        for question, sample_filenames in zip(questions, filenames):
            image_paths = list(sample_filenames.values())
            image_paths = image_paths[: self.num_images_per_prompt]
            if len(image_paths) > self.num_images_per_prompt:
                raise ValueError(
                    f"Found {len(image_paths)} images, but "
                    f"--num_images_per_prompt is "
                    f"{self.num_images_per_prompt}."
                )

            system_prompt_text = replace_system_prompt(
                self.system_prompt,
                image_paths,
            )

            images = []

            for filename in image_paths:
                if self.corruption == "NoImage":
                    img = Image.fromarray(
                        np.zeros((224, 224, 3), dtype=np.uint8)
                    )
                else:
                    img_path = filename

                    if self.corruption:
                        img_path = img_path.replace(
                            "nuscenes/samples",
                            f"data/corruption/{self.corruption}",
                        )

                    try:
                        with Image.open(img_path) as source:
                            img = source.convert("RGB")

                        if self.image_size > 0:
                            img = img.resize(
                                (self.image_size, self.image_size),
                                Image.Resampling.BICUBIC,
                            )

                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to load image: {img_path}"
                        ) from exc

                images.append(img)

            # Build messages format for SmolVLM
            user_content = [{"type": "image"} for _ in images]
            user_content.append({"type": "text", "text": question})

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt_text}],
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ]

            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)

            inputs = self.processor(
                text=prompt,
                images=images,
                return_tensors="pt",
            ).to(self.model.device, dtype=torch.bfloat16)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.sampling_params.get("max_tokens", 512),
                    temperature=self.sampling_params.get("temperature", 0.2),
                    top_p=self.sampling_params.get("top_p", 0.2),
                )

            generated_texts = self.processor.batch_decode(
                generated_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            preds.append(generated_texts[0] if generated_texts else "")

        batch["prompts"] = prompts
        batch["pred"] = preds

        return batch


def main():
    args = parse_arguments()

    if args.num_processes < 1:
        raise ValueError("--num_processes must be at least 1.")

    with open(args.system_prompt, "r", encoding="utf-8") as file:
        system_prompt = file.read()

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }

    tensor_parallel_size = 1

    ds = ray.data.read_json(args.data)

    ds = ds.map_batches(
        SmolVLMPredictor,
        fn_constructor_args=(
            args.model,
            system_prompt,
            sampling_params,
            args.num_images_per_prompt,
            args.max_model_len,
            tensor_parallel_size,
            args.corruption,
            args.image_size,
        ),
        concurrency=args.num_processes,
        batch_size=1,
        num_gpus=1,
    )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ds.write_json(args.output)

    results = load_json_files(args.output)
    save_json(results, args.output + ".json")

    if os.path.isdir(args.output):
        import shutil
        shutil.rmtree(args.output)

    print(f"Inference completed: {args.output}.json")


if __name__ == "__main__":
    main()