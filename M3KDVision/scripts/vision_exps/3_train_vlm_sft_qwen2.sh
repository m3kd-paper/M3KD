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
# Model & Dataset Setup
# ---------------------------------------------------------------------------

STUDENT_MODEL="Qwen/Qwen2-VL-2B-Instruct"
DATASET="data/preference_pairs"

OUTPUT_DIR="outputs/qwen2-vl-2b-sft"
RUN_NAME="qwen2-vl-2b-sft"
CONFIG="/tmp/${RUN_NAME}.yaml"

mkdir -p "$OUTPUT_DIR"

cat > "$CONFIG" <<EOF
model_name_or_path: $STUDENT_MODEL

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

do_eval: false
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

remove_unused_columns: false

push_to_hub: false
save_strategy: steps
save_steps: 500
save_total_limit: 1
seed: 42
warmup_ratio: 0.1
EOF

cat "$CONFIG"
cp "$CONFIG" "$OUTPUT_DIR/config.yaml"

# ---------------------------------------------------------------------------
# Run Qwen2-VL SFT
# ---------------------------------------------------------------------------

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes=1 \
    src/run_sft_qwen2.py \
    "$CONFIG"

echo "SFT Training Completed: $RUN_NAME"
