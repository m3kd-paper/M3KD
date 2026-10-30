###############################################################################
# DistiLLM-2 Vision Distillation
###############################################################################

cd ~/username/distillm-2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlm-distill

# Keep the active VLM environment first in PATH.
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME="$HOME/miniconda3/envs/distillm2"
export PATH="$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
hash -r

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

# Verify the active environment and launcher.
echo "Environment: $CONDA_PREFIX"
echo "Python: $(which python)"
echo "Accelerate: $(which accelerate)"
python -c "import sys, transformers; print('Python:', sys.executable); print('Transformers:', transformers.__version__)"

if [ "$(which python)" != "$CONDA_PREFIX/bin/python" ]; then
    echo "ERROR: Wrong Python executable"
    exit 1
fi

if [ "$(which accelerate)" != "$CONDA_PREFIX/bin/accelerate" ]; then
    echo "ERROR: Wrong Accelerate executable"
    exit 1
fi

nvcc --version
python -c "import torch; print('PyTorch:', torch.__version__); print('PyTorch CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

STUDENT_MODEL="Qwen/Qwen2-VL-2B-Instruct"
TEACHER_MODEL="Qwen/Qwen2-VL-7B-Instruct"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# The loader appends the split name, so this resolves to:
# data/preference_pairs/train
DATASET="data/preference_pairs"

NUM_TRAIN=8000
SEED=42

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

USE_REJECTED=true
USE_REJECTED_WEIGHT_CURRICULUM=true
MACRO_MIX=true
HIDDEN_LOSS_WEIGHT=0.0
INITIAL_BETA=0.5
NUM_MICRO_SAMPLES=1
LOSSES=(kd)

for loss in "${LOSSES[@]}"; do
    echo "================================"
    echo "Running training with loss: $loss"
    echo "================================"
    LOSS_TYPE="$loss"
    HIDDEN_LOSS_WEIGHT=0.0

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

    OUTPUT_DIR="outputs/qwen2-vl-2b-${LOSS_TYPE}-${NUM_TRAIN}"
    RUN_NAME="qwen2-vl-2b-${LOSS_TYPE}-${NUM_TRAIN}"
    CONFIG="/tmp/${RUN_NAME}.yaml"

    export TENSORBOARD_LOGGING_DIR="$PWD/$OUTPUT_DIR/runs"
    mkdir -p "$TENSORBOARD_LOGGING_DIR"
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
warmup_ratio: 0.1

loss_type: $LOSS_TYPE
hidden_loss_weight: $HIDDEN_LOSS_WEIGHT

use_rejected: $USE_REJECTED
initial_beta: $INITIAL_BETA
num_micro_samples: $NUM_MICRO_SAMPLES
use_rejected_weight_curriculum: $USE_REJECTED_WEIGHT_CURRICULUM
use_macro_mixing: $MACRO_MIX
EOF

    cat "$CONFIG"
    # save the config to the output directory for reference
    mkdir -p "$OUTPUT_DIR"
    cp "$CONFIG" "$OUTPUT_DIR/config.yaml"


    # Run the training script
    CUDA_VISIBLE_DEVICES=0 accelerate launch \
        --num_processes=1 \
        src/run_distivlm.py \
        "$CONFIG"
    echo "Completed: $RUN_NAME"

done
echo "All experiments completed."




