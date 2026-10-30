
import argparse
import json
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
import numpy as np
from PIL import Image

def normalize_image(image):
    """Convert source images to RGB before passing them to Qwen2-VL."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            # Grayscale image: convert HxW to RGB.
            image = Image.fromarray(image)
        elif image.ndim == 3:
            # Convert channel-first images to channel-last if necessary.
            if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
                image = np.transpose(image, (1, 2, 0))
            image = Image.fromarray(image)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        return image.convert("RGB")

    raise TypeError(f"Unsupported image type: {type(image)}")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--max_model_len", type=int, default=4096)
    args = parser.parse_args()

    dataset = load_from_disk(args.data_dir)
    required = {"example_id", "images", "question"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")

    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"generations_seed{args.seed}.jsonl"

    # Avoid accidentally mixing results from previous runs.
    if output_file.exists():
        raise FileExistsError(
            f"{output_file} already exists. Use another output directory "
            "or deliberately remove the existing file."
        )

    print(f"Loading processor for {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)

    print(f"Loading vLLM model {args.model}")
    
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=1,
        max_num_seqs=1,
         rope_scaling={
                "type": "mrope",
                "mrope_section": [16, 24, 24],
            },
        gpu_memory_utilization=0.90,
        seed=args.seed,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    total = len(dataset)
    print(f"Generating responses for {total} examples")

    with output_file.open("w", encoding="utf-8") as f:
        for start in range(0, total, args.batch_size):
            end = min(start + args.batch_size, total)
            requests = []
            metadata = []

            for idx in range(start, end):
                row = dataset[idx]
                image = normalize_image(row["images"])

                if image is None:
                    raise ValueError(
                        f"Missing image for {row['example_id']}"
                    )

                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": row["question"]},
                    ],
                }]

                prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                requests.append({
                    "prompt": prompt,
                    "multi_modal_data": {"image": image},
                })
                metadata.append(row)

            outputs = llm.generate(requests, sampling)

            if len(outputs) != len(metadata):
                raise RuntimeError(
                    f"Output count mismatch: {len(outputs)} outputs "
                    f"for {len(metadata)} inputs"
                )

            for row, result in zip(metadata, outputs):
                generated_text = result.outputs[0].text
                record = {
                    "example_id": row["example_id"],
                    "source_id": row["source_id"],
                    "question": row["question"],
                    "generated_text": generated_text,
                    "model": args.model,
                    "seed": args.seed,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            f.flush()
            print(f"Completed {end}/{total}")

    print(f"Saved generations to {output_file}")


if __name__ == "__main__":
    main()
