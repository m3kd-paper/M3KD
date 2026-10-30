#!/bin/bash


echo "==========================="
echo "[Step 1] Running Qwen2"
echo "==========================="

./script/llava1.5-7b.sh


echo "==========================="
echo "[Step 2] Running Qwen3"
echo "==========================="

./script/qwen3.sh