
"""
DriveBench batch inference using Qwen2-VL and vLLM with Ray Data.

Example:
    python inference/qwen2vl.py \
        --model Qwen/Qwen2-VL-2B-Instruct \
        --data data/data/drivebench-test-simple.json \
        --output res/qwen2-vl-2b/biterror \
        --system_prompt prompt.txt \
        --num_processes 1 \
        --max_model_len 4096 \
        --num_images_per_prompt 6 \
        --corruption BitError
"""

import os
import argparse
import warnings
from typing import Any, Dict

import numpy as np
import ray
from PIL import Image
from packaging.version import Version
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

from utils import replace_system_prompt, load_json_files, save_json


assert Version(ray.__version__) >= Version("2.22.0"), (
    "Ray version must be at least 2.22.0"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="DriveBench Qwen2-VL batch inference"
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


class QwenPredictor:
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
        
        if "qwen3" in model_name.lower() or "intern" in model_name.lower():
            self.llm = LLM(
                model=model_name,
                trust_remote_code=True,
                max_model_len=max_model_len,
                limit_mm_per_prompt={"image": num_images_per_prompt},
                tensor_parallel_size=tensor_parallel_size,
                max_num_seqs=1,
                gpu_memory_utilization=0.90,
            )            
        else:
            self.llm = LLM(
                model=model_name,
                trust_remote_code=True,
                max_model_len=max_model_len,
                limit_mm_per_prompt={"image": num_images_per_prompt},
                tensor_parallel_size=tensor_parallel_size,
                rope_scaling={
                    "type": "mrope",
                    "mrope_section": [16, 24, 24],
                },
                max_num_seqs=1,
                gpu_memory_utilization=0.90,
            )


        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self.system_prompt = system_prompt
        self.sampling_params = sampling_params
        self.corruption = corruption
        self.num_images_per_prompt = num_images_per_prompt
        self.image_size = image_size

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        questions = batch["question"]
        filenames = batch["image_path"]

        prompts = []
        multimodal_data = []

        for question, sample_filenames in zip(questions, filenames):
            image_paths = list(sample_filenames.values())
            image_paths = image_paths[: self.num_images_per_prompt]
            if len(image_paths) > self.num_images_per_prompt:
                raise ValueError(
                    f"Found {len(image_paths)} images, but "
                    f"--num_images_per_prompt is "
                    f"{self.num_images_per_prompt}."
                )

            system_prompt = replace_system_prompt(
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

            # Build a native Qwen chat prompt. The processor inserts
            # the appropriate image-token representation.
            user_content = [
                {"type": "image"} for _ in images
            ]
            user_content.append(
                {
                    "type": "text",
                    "text": question,
                }
            )

            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
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
            multimodal_data.append({"image": images})

        batch_inputs = [
            {
                "prompt": prompt,
                "multi_modal_data": mm_data,
            }
            for prompt, mm_data in zip(prompts, multimodal_data)
        ]

        outputs = self.llm.generate(
            batch_inputs,
            self.sampling_params,
            use_tqdm=False,
        )

        batch["prompts"] = prompts

        if outputs is None:
            warnings.warn("vLLM returned no outputs.")
            batch["pred"] = [None] * len(prompts)
        else:
            batch["pred"] = [
                result.outputs[0].text if result.outputs else ""
                for result in outputs
            ]

        return batch


def main():
    args = parse_arguments()

    if args.num_processes < 1:
        raise ValueError("--num_processes must be at least 1.")

    with open(args.system_prompt, "r", encoding="utf-8") as file:
        system_prompt = file.read()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # Each Ray worker runs one vLLM instance on one GPU.
    tensor_parallel_size = 1

    ds = ray.data.read_json(args.data)

    ds = ds.map_batches(
        QwenPredictor,
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

    # Ray writes JSON parts into this directory.
    ds.write_json(args.output)

    # Preserve the output convention used by the original script.
    results = load_json_files(args.output)
    save_json(results, args.output + ".json")

    # Remove the temporary Ray output directory.
    if os.path.isdir(args.output):
        import shutil
        shutil.rmtree(args.output)

    print(f"Inference completed: {args.output}.json")


if __name__ == "__main__":
    main()
