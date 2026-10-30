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

echo "Environment: $CONDA_PREFIX"
echo "Python: $(which python)"
echo "Accelerate: $(which accelerate)"

# ---------------------------------------------------------------------------
# Model & Dataset Setup
# ---------------------------------------------------------------------------
STUDENT_MODEL="Qwen/Qwen3-VL-2B-Instruct"
TEACHER_MODEL="Qwen/Qwen3-VL-8B-Instruct"
DATASET="data/preference_pairs_qwen3"  # Adjust if you use a dedicated SFT json/parquet
OUTPUT_DIR="outputs/qwen3-vl-2b-sft"
RUN_NAME="qwen3-vl-2b-sft"
CONFIG="/tmp/${RUN_NAME}.yaml"

mkdir -p "$OUTPUT_DIR"

cat > "$CONFIG" <<EOF
model_name_or_path: $STUDENT_MODEL
ref_model_name_or_path: $TEACHER_MODEL

torch_dtype: bfloat16
attn_implementation: sdpa

use_peft: true
lora_r: 16
lora_alpha: 128
lora_dropout: 0.05
lora_target_modules:
- q_proj
- k_proj
- v_proj
- o_proj
- gate_proj
- up_proj
- down_proj

dataset_mixer:
  $DATASET: 1.0

dataset_splits:
- train
- test

do_eval: true
preprocessing_num_workers: 1
bf16: true

gradient_accumulation_steps: 8
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false

learning_rate: 5.0e-5
log_level: info
logging_steps: 10
logging_strategy: steps
report_to: tensorboard
lr_scheduler_type: cosine

num_train_epochs: 1
optim: adamw_torch

output_dir: $OUTPUT_DIR
run_name: $RUN_NAME

per_device_train_batch_size: 1
per_device_eval_batch_size: 1

push_to_hub: false
save_strategy: steps
save_steps: 500
save_total_limit: 1
seed: 42
warmup_ratio: 0.1
EOF

cat "$CONFIG"
cp "$CONFIG" "$OUTPUT_DIR/config.yaml"

# Run SFT training via standard accelerate + transformers SFTTrainer script
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes=1 \
    src/run_sft_qwen3.py \
    "$CONFIG"

echo "SFT Training Completed: $RUN_NAME"