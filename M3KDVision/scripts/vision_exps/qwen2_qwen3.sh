#!/bin/bash


echo "==========================="
echo "[Step 1] Running ABLATION Qwen3"
echo "==========================="

./scripts/vision_exps/3_train_vlm_distilation.sh


echo "==========================="
echo "[Step 2] Running ABLATION Qwen2"
echo "==========================="

./scripts/vision_exps/3_train_vlm_distilation_qwen3.sh