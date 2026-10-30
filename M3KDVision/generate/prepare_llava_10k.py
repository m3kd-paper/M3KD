
# prepare_llava_50k.py

import argparse
import random
from pathlib import Path

from datasets import Dataset, load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Ethlake/llava-665k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default="data/llava_10k")
    parser.add_argument("--num_examples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.num_examples <= 0:
        raise ValueError("--num_examples must be positive")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists. Choose another path or "
            "remove it deliberately before rerunning."
        )

    print(f"Streaming {args.dataset}, split={args.split}...")
    stream = load_dataset(
        args.dataset,
        split=args.split,
        streaming=True,
    )

    rng = random.Random(args.seed)
    selected = []
    seen_source_ids = set()
    rows_scanned = 0

    for row in stream:
        rows_scanned += 1

        source_id = str(row.get("id", ""))
        image = row.get("images")
        turns = row.get("texts")

        if not source_id or image is None or not turns:
            continue

        if source_id in seen_source_ids:
            continue

        # Support the expected list of {user, assistant} dictionaries.
        if not isinstance(turns, list):
            continue

        valid_turns = [
            (turn_idx, turn)
            for turn_idx, turn in enumerate(turns)
            if isinstance(turn, dict)
            and isinstance(turn.get("user"), str)
            and turn["user"].strip()
            and isinstance(turn.get("assistant"), str)
            and turn["assistant"].strip()
        ]

        if not valid_turns:
            continue

        # Select one QA turn per image.
        turn_idx, turn = rng.choice(valid_turns)

        selected.append({
            "example_id": f"llava665k:{source_id}:turn{turn_idx}",
            "source_id": source_id,
            "subset": str(row.get("subset", "")),
            "images": image,
            "question": turn["user"].strip(),
            "reference_answer": turn["assistant"].strip(),
            "source_turn_index": turn_idx,
        })

        seen_source_ids.add(source_id)

        if len(selected) >= args.num_examples:
            break

        if rows_scanned % 1000 == 0:
            print(
                f"Scanned {rows_scanned:,} rows; "
                f"selected {len(selected):,} eligible examples."
            )

    print(f"Rows scanned: {rows_scanned:,}")
    print(f"Eligible examples selected: {len(selected):,}")

    if not selected:
        raise RuntimeError(
            "No valid image-question-answer examples found. "
            "Inspect the source dataset schema and turn format."
        )

    if len(selected) < args.num_examples:
        print(
            f"WARNING: requested {args.num_examples:,}, "
            f"but only found {len(selected):,} eligible examples."
        )

    # Dataset.from_list retains image objects as values. The datasets
    # library will infer features; verify the resulting image feature below.
    dataset = Dataset.from_list(selected)

    if "images" not in dataset.column_names:
        raise RuntimeError("The output dataset has no images column.")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))

    # Reload and verify the saved artifact.
    saved = Dataset.load_from_disk(str(output_dir))

    required = {
        "example_id",
        "source_id",
        "images",
        "question",
        "reference_answer",
    }
    missing = required - set(saved.column_names)
    if missing:
        raise RuntimeError(f"Saved dataset is missing columns: {missing}")

    ids = saved["example_id"]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate example_id values detected.")

    print(f"\nSaved to: {output_dir}")
    print(f"Rows: {len(saved):,}")
    print(f"Columns: {saved.column_names}")
    print(f"Image feature: {saved.features['images']}")
    print(f"First example ID: {saved[0]['example_id']}")
    print(f"First question: {saved[0]['question'][:200]}")
    print("Dataset saved and basic integrity checks passed.")


if __name__ == "__main__":
    main()
