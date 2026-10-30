###############################################################################
# DistiLLM-2 Vision Distillation
###############################################################################

cd ~/username/distillm-2

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
python -c "import sys, transformers; print('Python:', sys.executable); print('Transformers:', transformers.__version__)"

if [ "$(which python)" != "$CONDA_PREFIX/bin/python" ]; then
    echo "ERROR: Wrong Python executable"
    exit 1
fi

if [ "$(which accelerate)" != "$CONDA_PREFIX/bin/accelerate" ]; then
    echo "Accelerate: $(which accelerate)"
    echo "Expected: $CONDA_PREFIX/bin/accelerate"
    echo "ERROR: Wrong Accelerate executable"
    exit 1
fi

nvcc --version
python -c "import torch; print('PyTorch:', torch.__version__); print('PyTorch CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

STUDENT_MODEL="Qwen/Qwen3-VL-2B-Instruct"
TEACHER_MODEL="Qwen/Qwen3-VL-8B-Instruct"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# The loader appends the split name, so this resolves to:
# data/preference_pairs/train
DATASET="data/preference_pairs_qwen3"

NUM_TRAIN=8000
SEED=42

# ---------------------------------------------------------------------------
# M3KD: Monte Carlo sample-count ablation
# ---------------------------------------------------------------------------

LOSS_TYPE="Macro_Micro"
USE_REJECTED=true
USE_REJECTED_WEIGHT_CURRICULUM=true
MACRO_MIX=true
HIDDEN_LOSS_WEIGHT=0.0
INITIAL_BETA=0.5

# Include 8 as the baseline, then test smaller sample counts.
MICRO_SAMPLE_COUNTS=(8)

for NUM_MICRO_SAMPLES in "${MICRO_SAMPLE_COUNTS[@]}"; do

    echo "=============================================="
    echo "Loss:               $LOSS_TYPE"
    echo "Micro samples:      $NUM_MICRO_SAMPLES"
    echo "Use rejected:       $USE_REJECTED"
    echo "initial beta:       $INITIAL_BETA"
    echo "USE REJECTED_WEIGHT_CURRICULUM: $USE_REJECTED_WEIGHT_CURRICULUM"
    echo "=============================================="

    # Unique run name and output directory for each ablation.
    RUN_NAME="qwen3-vl-2b-m3kd-micro${NUM_MICRO_SAMPLES}-${NUM_TRAIN}-REJECTEDCURRICULUM-${USE_REJECTED_WEIGHT_CURRICULUM}-MACROMIX-${MACRO_MIX}"
    OUTPUT_DIR="outputs/${RUN_NAME}"
    CONFIG="/tmp/${RUN_NAME}.yaml"

    # Separate TensorBoard logs for every run.
    export TENSORBOARD_LOGGING_DIR="$PWD/$OUTPUT_DIR/runs"
    mkdir -p "$TENSORBOARD_LOGGING_DIR"
    mkdir -p "$OUTPUT_DIR"

    echo "Output directory: $OUTPUT_DIR"
    echo "TensorBoard logs: $TENSORBOARD_LOGGING_DIR"

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

max_length: 1024
max_prompt_length: 512

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

seed: $SEED
warmup_steps: 0.1

loss_type: $LOSS_TYPE
hidden_loss_weight: $HIDDEN_LOSS_WEIGHT

use_rejected: $USE_REJECTED
initial_beta: $INITIAL_BETA
num_micro_samples: $NUM_MICRO_SAMPLES
use_rejected_weight_curriculum: $USE_REJECTED_WEIGHT_CURRICULUM
use_macro_mixing: $MACRO_MIX
EOF

    # Preserve the exact configuration used for each experiment.
    cp "$CONFIG" "$OUTPUT_DIR/config.yaml"

    echo "Configuration: $OUTPUT_DIR/config.yaml"

    # Run sequentially on GPU 0.
    CUDA_VISIBLE_DEVICES=0 accelerate launch \
        --num_processes=1 \
        src/run_distivlm_qwen3.py \
        "$CONFIG"

    echo "Completed: $RUN_NAME"

done

echo "All M3KD sample-count ablations completed."