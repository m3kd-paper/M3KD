#!/usr/bin/env bash
set -eo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen3vl

# Enable unset-variable checking after Conda hooks have run.
set -u

export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

unset CC
unset CXX
hash -r


# Models
#STUDENT_MODEL="Qwen/Qwen2-VL-2B-Instruct"
#TEACHER_MODEL="Qwen/Qwen2-VL-7B-Instruct"
STUDENT_MODEL="Qwen/Qwen3-VL-2B-Instruct"
TEACHER_MODEL="Qwen/Qwen3-VL-8B-Instruct"

# Canonical dataset prepared once and reused by both models
DATASET_DIR="data/llava_10k"

NUM_SAMPLES=8000
SEED=42
MAX_NEW_TOKENS=512
TEMPERATURE=0.8
TOP_P=0.95

###############################################################################
# Stage 1: Stream and prepare the dataset
###############################################################################

python generate/prepare_llava_10k.py \
    --num_examples "${NUM_SAMPLES}" \
    --seed "${SEED}" \
    --output_dir "${DATASET_DIR}"

###############################################################################
# Stage 2: Generate student responses
# Implement this script to read DATASET_DIR and preserve example_id.
###############################################################################
#
#python generate/generate_vllm_vlm_single_gpu_qwen3.py \
#    --data_dir data/llava_10k \
#    --model ${STUDENT_MODEL} \
#    --output_dir datasets/full_student_qwen3 \
#    --seed 42 \
#    --max_tokens 128 \
#    --batch_size 2 \
#    --limit ${NUM_SAMPLES}
#
################################################################################
## Stage 3: Generate teacher responses using the same examples
################################################################################
#
#python generate/generate_vllm_vlm_single_gpu_qwen3.py \
#    --data_dir data/llava_10k \
#    --model ${TEACHER_MODEL} \
#    --output_dir datasets/full_teacher_qwen3 \
#    --seed 42 \
#    --max_tokens 128 \
#    --batch_size 2 \
#    --limit ${NUM_SAMPLES}

###############################################################################
# Stage 4: Join generations by example_id and create preference pairs
###############################################################################

python3 generate/vlm_build_preference_pairs.py \
    --total-size ${NUM_SAMPLES} \
    --eval-size 500 \
    --seed 42 \
    --student-file datasets/full_student_qwen3/generations_seed42.jsonl \
    --teacher-file datasets/full_teacher_qwen3/generations_seed42.jsonl \
    --output-dir data/preference_pairs_qwen3 \
    --overwrite


#python3 generate/vlm_build_preference_pairs.py \
#    --total-size 10 \
#    --eval-size 1 \
#    --seed 42 \
#    --student-file datasets/full_student_qwen3/generations_seed42.jsonl \
#    --teacher-file datasets/full_teacher_qwen3/generations_seed42.jsonl \
#    --output-dir data/preference_pairs_qwen3_10 \
#    --overwrite

###############################################################################
# Stage 5: Validate counts, IDs, images, and split integrity
###############################################################################

#python - <<'PY'
#from datasets import load_from_disk
#
#path = "data/preference_pairs/train"
#dataset = load_from_disk(path)
#
#print("Image feature schema:")
#print(dataset.features["images"])
#
#print("\nFirst 5 image values:")
#for i in range(min(5, len(dataset))):
#    value = dataset[i]["images"]
#    print(f"\nExample {i}:")
#    print("Type:", type(value))
#    print("Value:", repr(value)[:1000])
#
#print("\nImage value types across the dataset:")
#from collections import Counter
#print(Counter(type(x).__name__ for x in dataset["images"]))
#
#print("\nNull image values:", sum(x is None for x in dataset["images"]))
#PY


python - <<'PY'
from datasets import load_from_disk

path = "data/preference_pairs_qwen3/train"

dataset = load_from_disk(path)

print(dataset)
print(f"Train examples: {len(dataset)}")

assert len(dataset) == 7500, (
    f"Expected 7500 preference pairs, got {len(dataset)}"
)

required_columns = {
    "example_id",
    "source_id",
    "images",
    "question",
    "reference_answer",
    "chosen",
    "rejected",
    "chosen_model",
    "rejected_model",
}

# print first 5 rows of the dataset
print("\nFirst 5 examples:")
for i in range(min(5, len(dataset))):
    print(f"\nExample {i}:")
    for col in required_columns:
        print(f"{col}: {dataset[i][col]}")

missing_columns = required_columns - set(dataset.column_names)
assert not missing_columns, (
    f"Missing required columns: {sorted(missing_columns)}"
)

example_ids = dataset["example_id"]
assert len(example_ids) == len(set(example_ids)), (
    "Duplicate example_id values found in the training dataset"
)

assert all(
    isinstance(x, str) and x.strip()
    for x in dataset["question"]
), "Empty or invalid questions found"

assert all(
    isinstance(x, str) and x.strip()
    for x in dataset["chosen"]
), "Empty teacher responses found"

assert all(
    isinstance(x, str) and x.strip()
    for x in dataset["rejected"]
), "Empty student responses found"

from PIL import Image

assert all(
    isinstance(x, Image.Image)
    for x in dataset["images"]
), "Missing or invalid decoded images found"

assert all(
    x.width > 0 and x.height > 0
    for x in dataset["images"]
), "Invalid image dimensions found"

from collections import Counter

image_modes = Counter(x.mode for x in dataset["images"])
print("Image modes:", image_modes)

grayscale_count = sum(
    x.mode != "RGB" for x in dataset["images"]
)
print(f"Images not in RGB mode: {grayscale_count}")

print("")
print("SUCCESS: 8,000 VLM teacher/student preference pairs validated.")
print("")
print("Dataset:")
print(f"  {path}")
print("")
print("Teacher = chosen")
print("Student = rejected")
print("Alignment = example_id")
print("Images, questions, and reference answers = preserved")
PY


echo "Stage 1 complete. Dataset saved to ${DATASET_DIR}."
