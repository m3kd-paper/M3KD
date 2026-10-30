#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define your python evaluation script name (adjust if named differently)
EVAL_SCRIPT="inference/qwen_vl_simple.py"


# --- Model 1: Qwen2-VL-7B-Instruct ---
echo "============================================================"
echo "[Step 1] Qwen2-VL-7B-Instruct"
echo "============================================================"
python3 "$EVAL_SCRIPT" \
  --model "/home/username/Desktop/username/(MLsys)_DriveBench/Models/Qwen2/Teacher_7B/Qwen2-VL-7B-Instruct-Q4_K_M.gguf" \
  --mmproj "/home/username/Desktop/username/(MLsys)_DriveBench/Models/Qwen2/Teacher_7B/mmproj-Qwen2-VL-7B-Instruct-f16.gguf" \
  --output-prefix "res_new_FULL_Efficiency/Qwen2-VL-7B-Instruct-gguf/h256"

# --- Model 2: Qwen2 Distilled M3KD ---
echo "============================================================"
echo "[Step 2] Qwen2 Distilled M3KD"
echo "============================================================"
python3 "$EVAL_SCRIPT" \
  --model "/home/username/Desktop/username/(MLsys)_DriveBench/Models/Qwen2/Distilled_M3KD/qwen2-vl-2b-m3kd-micro4-8000-REJECTEDCURRICULUM-true_merged-Q4_K_M.gguf" \
  --mmproj "/home/username/Desktop/username/(MLsys)_DriveBench/Models/Qwen2/Distilled_M3KD/mmproj-qwen2-vl-2b-m3kd-micro4-8000-REJECTEDCURRICULUM-true_merged-f16.gguf" \
  --output-prefix "res_new_FULL_Efficiency/Qwen2-VL-2b-m3kd-gguf/h256"

# --- Model 3: Qwen3 Teacher 8B ---
echo "============================================================"
echo "[Step 3] Qwen3 Teacher 8B"
echo "============================================================"
python3 "$EVAL_SCRIPT" \
  --model "/home/username/Desktop/username/(MLsys)_DriveBench/Models/Qwen3/Teacher_8B/Qwen3VL-8B-Instruct-Q4_K_M.gguf" \
  --mmproj "/home/username/Desktop/username/(MLsys)_DriveBench/Models/Qwen3/Teacher_8B/mmproj-Qwen3VL-8B-Instruct-F16.gguf" \
  --output-prefix "res_new_FULL_Efficiency/Qwen3VL-8B-Instruct-gguf/h256"

# --- Model 4: Qwen3 Distilled M3KD ---
echo "============================================================"
echo "[Step 4] Qwen3 Distilled M3KD"
echo "============================================================"
python3 "$EVAL_SCRIPT" \
  --model "/home/username/Desktop/username/(MLsys)_DriveBench/Models/Qwen3/Distilled_M3KD/qwen3-vl-2b-m3kd-micro4-8000-REJECTEDCURRICULUM-true_merged-Q4_K_M.gguf" \
  --mmproj "/home/username/Desktop/username/(MLsys)_DriveBench/Models/Qwen3/Distilled_M3KD/mmproj-qwen3-vl-2b-m3kd-micro4-8000-REJECTEDCURRICULUM-true_merged-f16.gguf" \
  --output-prefix "res_new_FULL_Efficiency/Qwen3-VL-m3kd-gguf/h256"

echo "All models evaluated successfully!"