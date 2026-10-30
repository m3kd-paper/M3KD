#!/usr/bin/env bash
set -eo pipefail

###############################################################################
# DistiLLM-2 REALISTIC SMOKE TEST
#
# Purpose:
#   Test whether DistiLLM-2 can improve the existing Qwen 1.5B SFT model
#   using a meaningful amount of teacher-generated preference data.
#
# IMPORTANT:
#   - We DO NOT retrain the student SFT model.
#   - We reuse the existing SFT model.
#   - The teacher is Qwen2.5-7B-Instruct.
#   - Training data and evaluation data are disjoint.
#
# Pipeline:
#
#   Existing 1.5B SFT
#          |
#          +--------------------------+
#          |                          |
#          v                          v
#   Student responses          Teacher responses
#          |                          |
#          +------------+-------------+
#                       |
#                       v
#               Preference dataset
#                       |
#                       v
#                DistiLLM-2
#                       |
#                       v
#                Merge LoRA
#                       |
#                       v
#              Distilled model
#                       |
#             +---------+---------+
#             |                   |
#             v                   v
#          Student            Distilled
#             |                   |
#             +---------+---------+
#                       |
#                       v
#                  Pairwise judge
#                       |
#                       v
#             Symmetric weighted WR
#
###############################################################################


###############################################################################
# 0. MOVE TO REPOSITORY + ACTIVATE ENVIRONMENT
###############################################################################

cd ~/username/distillm-2
#source ~/miniconda3/etc/profile.d/conda.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate distillm2

###############################################################################
# 1. CUDA ENVIRONMENT
###############################################################################

export CUDA_VISIBLE_DEVICES=0

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"


###############################################################################
# 2. EXPERIMENT SETTINGS
#
# These are the ONLY values you should normally need to change.
###############################################################################

# Existing SFT student.
STUDENT_MODEL="outputs/qwen2.5-1.5b-sft"

# Frozen 7B teacher.
TEACHER_MODEL="ckpts/Qwen2.5-7B-Instruct"

# Number of preference-training examples.
NUM_TRAIN=2000

# Number of held-out evaluation examples.
NUM_EVAL=500

# DistiLLM optimization steps.
#
# Start with 500 if you want a conservative smoke test.
# Use 1000 if runtime allows.
DISTILL_STEPS=1000

# Generation settings.
MAX_NEW_TOKENS=256
TEMPERATURE=0.8
TOP_P=0.95
SEED=42


###############################################################################
# 3. OUTPUT DIRECTORIES
###############################################################################

ROOT="smoke_distillm"

PROMPTS_TRAIN="${ROOT}/prompts_train"
PROMPTS_EVAL="${ROOT}/prompts_eval"

STUDENT_GEN="${ROOT}/generation_student"
TEACHER_GEN="${ROOT}/generation_teacher"
DISTILLED_GEN="${ROOT}/generation_distilled"

PREFERENCE_DATA="${ROOT}/preference_dataset"

DISTILL_OUTPUT="outputs/qwen2.5-1.5b-distillm2-smoke-${DISTILL_STEPS}step-train-${NUM_TRAIN}-eval-${NUM_EVAL}-seed-${SEED}"

EVAL_STUDENT_VS_DISTILLED="smoke_distillm_student_vs_distilled_${DISTILL_STEPS}step-train-${NUM_TRAIN}-eval-${NUM_EVAL}-seed-${SEED}"
EVAL_DISTILLED_VS_STUDENT="smoke_distillm_distilled_vs_student_${DISTILL_STEPS}step-train-${NUM_TRAIN}-eval-${NUM_EVAL}-seed-${SEED}"


###############################################################################
# 4. DISPLAY CONFIGURATION
###############################################################################

echo
echo "======================================================================"
echo " DistiLLM-2 REALISTIC SMOKE TEST"
echo "======================================================================"
echo
echo "Student model:        $STUDENT_MODEL"
echo "Teacher model:        $TEACHER_MODEL"
echo
echo "Training examples:    $NUM_TRAIN"
echo "Evaluation examples:  $NUM_EVAL"
echo "Distill steps:        $DISTILL_STEPS"
echo
echo "GPU:                  $CUDA_VISIBLE_DEVICES"
echo "Seed:                 $SEED"
echo
echo "======================================================================"
echo


###############################################################################
# 5. CHECK THAT REQUIRED MODELS EXIST
###############################################################################

echo "[1/10] Checking models..."

if [[ ! -d "$STUDENT_MODEL" ]]; then
    echo "ERROR: Student model not found:"
    echo "       $STUDENT_MODEL"
    exit 1
fi

if [[ ! -d "$TEACHER_MODEL" ]]; then
    echo "ERROR: Teacher model not found:"
    echo "       $TEACHER_MODEL"
    exit 1
fi

echo "Student: OK"
echo "Teacher: OK"


###############################################################################
# 6. CLEAN ONLY THIS EXPERIMENT
###############################################################################

echo
echo "[2/10] Cleaning previous smoke-test outputs..."

rm -rf "$ROOT"
rm -rf "$DISTILL_OUTPUT"

rm -f "results/inputs/${EVAL_STUDENT_VS_DISTILLED}.jsonl"
rm -f "results/inputs/${EVAL_DISTILLED_VS_STUDENT}.jsonl"

rm -f "results/outputs/${EVAL_STUDENT_VS_DISTILLED}.jsonl"
rm -f "results/outputs/${EVAL_DISTILLED_VS_STUDENT}.jsonl"

mkdir -p "$ROOT"


###############################################################################
# 7. CREATE DISJOINT TRAINING + EVALUATION PROMPTS
###############################################################################

echo
echo "[3/10] Creating prompt datasets..."

python - <<PY
from datasets import load_dataset

NUM_TRAIN = $NUM_TRAIN
NUM_EVAL = $NUM_EVAL

train_out = "$PROMPTS_TRAIN"
eval_out = "$PROMPTS_EVAL"

# Use test_gen so this is separate from the SFT training split.
dataset = load_dataset(
    "HuggingFaceH4/ultrachat_200k",
    split="test_gen"
)

required = NUM_TRAIN + NUM_EVAL

if len(dataset) < required:
    raise RuntimeError(
        f"Need {required} examples but dataset contains {len(dataset)}"
    )

# Deterministic shuffle.
dataset = dataset.shuffle(seed=$SEED)

train_dataset = dataset.select(range(NUM_TRAIN))
eval_dataset = dataset.select(
    range(NUM_TRAIN, NUM_TRAIN + NUM_EVAL)
)

train_dataset.save_to_disk(train_out)
eval_dataset.save_to_disk(eval_out)

print(f"Training prompts:   {len(train_dataset)}")
print(f"Evaluation prompts: {len(eval_dataset)}")
print(f"Saved training to:   {train_out}")
print(f"Saved evaluation to: {eval_out}")
PY


###############################################################################
# 8. GENERATE STUDENT RESPONSES
###############################################################################

echo
echo "[4/10] Generating $NUM_TRAIN STUDENT responses..."
echo

python generate/generate_vllm_single_gpu.py \
    --data_dir "$PROMPTS_TRAIN" \
    --model "$STUDENT_MODEL" \
    --output_dir "$STUDENT_GEN" \
    --max_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --seed "$SEED"


###############################################################################
# 9. GENERATE TEACHER RESPONSES
###############################################################################

echo
echo "[5/10] Generating $NUM_TRAIN TEACHER responses..."
echo

python generate/generate_vllm_single_gpu.py \
    --data_dir "$PROMPTS_TRAIN" \
    --model "$TEACHER_MODEL" \
    --output_dir "$TEACHER_GEN" \
    --max_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --seed "$SEED"


###############################################################################
# 10. BUILD PREFERENCE DATASET
###############################################################################

echo
echo "[6/10] Building preference dataset..."
echo

python generate/reformat_smoke.py \
    --teacher_file "$TEACHER_GEN/output_${SEED}.json" \
    --student_file "$STUDENT_GEN/output_${SEED}.json" \
    --output_dir "$PREFERENCE_DATA"


###############################################################################
# 11. TRAIN DISTILLM-2
###############################################################################

echo
echo "[7/10] Training DistiLLM-2..."
echo
echo "Distillation steps: $DISTILL_STEPS"
echo

python - <<PY
from pathlib import Path
import yaml

# Start from the known-working smoke configuration.
source = Path("training_configs/qwen2.5-1.5b-distillm2-smoke.yaml")

cfg = yaml.safe_load(source.read_text())

# Existing SFT student.
cfg["model_name_or_path"] = "$STUDENT_MODEL"

# Frozen teacher.
cfg["ref_model_name_or_path"] = "$TEACHER_MODEL"

# Preference dataset generated above.
cfg["dataset_mixer"] = {
    "$PREFERENCE_DATA": 1.0
}

cfg["output_dir"] = "$DISTILL_OUTPUT"
cfg["run_name"] = "qwen2.5-1.5b-distillm2-smoke"

# Main smoke-test budget.
cfg["max_steps"] = $DISTILL_STEPS
cfg["num_train_epochs"] = 1

# Keep the known-working memory configuration.
cfg["per_device_train_batch_size"] = 1
cfg["gradient_accumulation_steps"] = 1

cfg["save_strategy"] = "steps"
cfg["save_steps"] = $DISTILL_STEPS
cfg["save_total_limit"] = 1

output = Path("/tmp/distillm2-realistic-smoke.yaml")
output.write_text(yaml.safe_dump(cfg, sort_keys=False))

print(f"Generated config: {output}")
PY

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes=1 \
    src/run_distillm.py \
    /tmp/distillm2-realistic-smoke.yaml


###############################################################################
# 12. MERGE THE DistiLLM LoRA
###############################################################################

echo
echo "[8/10] Merging DistiLLM-2 adapter..."
echo

python utils/merging.py \
    --base-model-name "$STUDENT_MODEL" \
    --lora-model-name "$DISTILL_OUTPUT"

DISTILLED_MODEL="${DISTILL_OUTPUT}/merged"


###############################################################################
# 13. GENERATE HELD-OUT STUDENT + DISTILLED RESPONSES
###############################################################################

echo
echo "[9/10] Generating held-out evaluation responses..."
echo

echo "Generating STUDENT evaluation responses..."

python generate/generate_vllm_single_gpu.py \
    --data_dir "$PROMPTS_EVAL" \
    --model "$STUDENT_MODEL" \
    --output_dir "${ROOT}/eval_student" \
    --max_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --seed "$SEED"

echo
echo "Generating DISTILLED evaluation responses..."

python generate/generate_vllm_single_gpu.py \
    --data_dir "$PROMPTS_EVAL" \
    --model "$DISTILLED_MODEL" \
    --output_dir "$DISTILLED_GEN" \
    --max_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --seed "$SEED"


###############################################################################
# 14. BUILD BOTH EVALUATION DIRECTIONS
###############################################################################

echo
echo "[10/10] Building pairwise evaluation requests..."
echo

# Direction 1:
# A = Student
# B = Distilled

python eval/build_evaluation.py \
    --data-path1 "${ROOT}/eval_student/output_${SEED}.json" \
    --data-path2 "$DISTILLED_GEN/output_${SEED}.json" \
    --pairwise \
    --output-file "$EVAL_STUDENT_VS_DISTILLED" \
    --judge gpt-4o


# Direction 2:
# A = Distilled
# B = Student

python eval/build_evaluation.py \
    --data-path1 "$DISTILLED_GEN/output_${SEED}.json" \
    --data-path2 "${ROOT}/eval_student/output_${SEED}.json" \
    --pairwise \
    --output-file "$EVAL_DISTILLED_VS_STUDENT" \
    --judge gpt-4o


###############################################################################
# 15. SHOW WHAT REMAINS
###############################################################################

echo
echo
echo "======================================================================"
echo " TRAINING + GENERATION + EVALUATION REQUESTS COMPLETE"
echo "======================================================================"
echo
echo "Preference dataset:"
echo "  $PREFERENCE_DATA"
echo
echo "Distilled model:"
echo "  $DISTILLED_MODEL"
echo
echo "Evaluation requests:"
echo "  results/inputs/${EVAL_STUDENT_VS_DISTILLED}.jsonl"
echo "  results/inputs/${EVAL_DISTILLED_VS_STUDENT}.jsonl"
echo
echo "======================================================================"
echo
echo "NEXT STEP:"
echo
echo "Run the repository's API processor on BOTH evaluation request files."
echo
echo "Then the judge outputs should be:"
echo
echo "  results/outputs/${EVAL_STUDENT_VS_DISTILLED}.jsonl"
echo "  results/outputs/${EVAL_DISTILLED_VS_STUDENT}.jsonl"
echo
echo "Finally run:"
echo
echo "python eval/grading.py \\"
echo "    --input1 results/outputs/${EVAL_STUDENT_VS_DISTILLED}.jsonl \\"
echo "    --input2 results/outputs/${EVAL_DISTILLED_VS_STUDENT}.jsonl \\"
echo "    --pairwise"
echo
echo "======================================================================"