
#!/usr/bin/env python3

"""Build deterministic train/evaluation VLM preference pairs."""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build aligned teacher/student VLM preference pairs."
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/llava_10k"),
    )
    parser.add_argument(
        "--student-file",
        type=Path,
        default=Path("datasets/full_student/generations_seed42.jsonl"),
    )
    parser.add_argument(
        "--teacher-file",
        type=Path,
        default=Path("datasets/full_teacher/generations_seed42.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/preference_pairs"),
        help="Root directory containing train/ and test/ datasets.",
    )
    parser.add_argument(
        "--total-size",
        type=int,
        default=8000,
        help="Expected number of teacher-generated examples.",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=500,
        help="Number of examples reserved for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory if it exists.",
    )

    return parser.parse_args()


def load_source_dataset(source_dir):
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Source dataset does not exist: {source_dir}"
        )

    loaded = load_from_disk(str(source_dir))

    if isinstance(loaded, DatasetDict):
        if "train" in loaded:
            dataset = loaded["train"]
        else:
            first_split = next(iter(loaded), None)
            if first_split is None:
                raise ValueError("The source DatasetDict is empty.")
            dataset = loaded[first_split]
            print(f"Using source split: {first_split}", file=sys.stderr)
    elif isinstance(loaded, Dataset):
        dataset = loaded
    else:
        raise TypeError(
            f"Unsupported source type: {type(loaded).__name__}"
        )

    required = {
        "example_id",
        "source_id",
        "images",
        "question",
        "reference_answer",
    }
    missing = required - set(dataset.column_names)

    if missing:
        raise ValueError(
            f"Source dataset missing columns: {sorted(missing)}"
        )

    source_by_id = {}

    for row in dataset:
        example_id = row["example_id"]

        if not example_id:
            raise ValueError("Source contains an empty example_id.")

        if example_id in source_by_id:
            raise ValueError(
                f"Duplicate source example_id: {example_id!r}"
            )

        source_by_id[example_id] = row

    print(f"Source examples: {len(source_by_id):,}")
    return source_by_id


def load_generations(path, model_label):
    if not path.is_file():
        raise FileNotFoundError(
            f"{model_label} generation file not found: {path}"
        )

    generations = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc

            example_id = record.get("example_id")
            generated_text = record.get("generated_text")

            if not example_id:
                raise ValueError(
                    f"Missing example_id in {path}, line {line_number}."
                )

            if example_id in generations:
                raise ValueError(
                    f"Duplicate {model_label} ID {example_id!r} "
                    f"in {path}, line {line_number}."
                )

            if not isinstance(generated_text, str):
                raise ValueError(
                    f"Invalid generated_text for {example_id!r} "
                    f"in {path}, line {line_number}."
                )

            generations[example_id] = record

    print(f"{model_label} generations: {len(generations):,}")
    return generations


def validate_question(record, source_row, model_label, example_id):
    question = record.get("question")

    if question is not None and question != source_row["question"]:
        raise ValueError(
            f"{model_label} question mismatch for {example_id!r}."
        )


def build_pairs(source_by_id, student, teacher, total_size):
    """Build pairs using teacher IDs as the authoritative population."""
    teacher_ids = set(teacher)
    # only use the total size (which can be smaller than the number of teacher generations) for the final dataset
    teacher_ids = set(sorted(teacher_ids)[:total_size])
    
    if len(teacher_ids) != total_size:
        raise ValueError(
            f"Expected {total_size:,} teacher generations, "
            f"found {len(teacher_ids):,}."
        )

    unknown_ids = teacher_ids - set(source_by_id)

    if unknown_ids:
        raise ValueError(
            f"{len(unknown_ids):,} teacher IDs are absent from the "
            f"source dataset. Examples: {sorted(unknown_ids)[:10]}"
        )

    missing_student = teacher_ids - set(student)

    if missing_student:
        raise ValueError(
            f"{len(missing_student):,} teacher examples lack student "
            f"responses. Examples: {sorted(missing_student)[:20]}"
        )

    pairs = []

    for example_id in sorted(teacher_ids):
        source_row = source_by_id[example_id]
        student_record = student[example_id]
        teacher_record = teacher[example_id]

        validate_question(
            student_record, source_row, "Student", example_id
        )
        validate_question(
            teacher_record, source_row, "Teacher", example_id
        )

        chosen = teacher_record["generated_text"].strip()
        rejected = student_record["generated_text"].strip()

        if not chosen:
            raise ValueError(
                f"Empty teacher response for {example_id!r}."
            )

        if not rejected:
            raise ValueError(
                f"Empty student response for {example_id!r}."
            )

        pairs.append(
            {
                "example_id": example_id,
                "source_id": source_row["source_id"],
                "images": source_row["images"],
                "question": source_row["question"],
                "reference_answer": source_row["reference_answer"],
                "chosen": chosen,
                "rejected": rejected,
                "chosen_model": teacher_record.get("model", "teacher"),
                "rejected_model": student_record.get("model", "student"),
            }
        )

    assert len(pairs) == total_size
    return pairs


def split_pairs(pairs, eval_size, seed):
    """Create reproducible, disjoint splits at the example level."""
    if not 0 < eval_size < len(pairs):
        raise ValueError(
            "--eval-size must be between 1 and total_size - 1."
        )

    # Shuffle IDs rather than dataset rows so split membership is
    # reproducible and independent of JSONL line ordering.
    ids = sorted(pair["example_id"] for pair in pairs)
    random.Random(seed).shuffle(ids)

    eval_ids = set(ids[:eval_size])
    train_ids = set(ids[eval_size:])

    pair_by_id = {pair["example_id"]: pair for pair in pairs}

    train_pairs = [
        pair_by_id[example_id] for example_id in sorted(train_ids)
    ]
    eval_pairs = [
        pair_by_id[example_id] for example_id in sorted(eval_ids)
    ]

    assert not train_ids.intersection(eval_ids)
    assert len(train_pairs) + len(eval_pairs) == len(pairs)

    return train_pairs, eval_pairs, train_ids, eval_ids


def main():
    args = parse_args()

    if args.total_size <= 1:
        raise ValueError("--total-size must be greater than one.")

    if not 0 < args.eval_size < args.total_size:
        raise ValueError(
            "--eval-size must be between 1 and total_size - 1."
        )

    source_by_id = load_source_dataset(args.source_dir)
    student = load_generations(args.student_file, "Student")
    teacher = load_generations(args.teacher_file, "Teacher")

    pairs = build_pairs(
        source_by_id=source_by_id,
        student=student,
        teacher=teacher,
        total_size=args.total_size,
    )

    train_pairs, eval_pairs, train_ids, eval_ids = split_pairs(
        pairs=pairs,
        eval_size=args.eval_size,
        seed=args.seed,
    )

    output_dir = args.output_dir

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory exists: {output_dir}\n"
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(train_pairs),
            "test": Dataset.from_list(eval_pairs),
        }
    )

    # Creates output_dir/train and output_dir/test, matching the
    # repository's dataset loader convention.
    dataset_dict.save_to_disk(str(output_dir))

    manifest_path = output_dir.parent / (
        output_dir.name + "_manifest.json"
    )

    manifest = {
        "total_size": len(pairs),
        "train_size": len(train_pairs),
        "eval_size": len(eval_pairs),
        "seed": args.seed,
        "source_dir": str(args.source_dir),
        "student_file": str(args.student_file),
        "teacher_file": str(args.teacher_file),
        "alignment_key": "example_id",
        "split_method": (
            "Shuffle sorted teacher example IDs using Python "
            "random.Random(seed); first eval_size IDs go to test."
        ),
        "chosen": "teacher generated_text",
        "rejected": "student generated_text",
        "train_example_ids": sorted(train_ids),
        "eval_example_ids": sorted(eval_ids),
    }

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print("\nPreference-pair datasets created successfully.")
    print(f"Total pairs:    {len(pairs):,}")
    print(f"Training pairs: {len(train_pairs):,}")
    print(f"Evaluation:     {len(eval_pairs):,}")
    print(f"Seed:           {args.seed}")
    print(f"Dataset root:   {output_dir}")
    print(f"Train path:     {output_dir / 'train'}")
    print(f"Test path:      {output_dir / 'test'}")
    print(f"Manifest:       {manifest_path}")
    print(
        "\nPreference labels are provisional: teacher outputs are "
        "chosen and student outputs are rejected, without automatic "
        "verification that the teacher answer is better."
    )


if __name__ == "__main__":
    main()
