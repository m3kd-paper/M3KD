#!/usr/bin/env bash
set -Eeuo pipefail

###############################################################################
# LoRA Merge and GGUF Conversion
#
# Steps:
#   1. Clone/update llama.cpp
#   2. Merge LoRA adapters into Hugging Face models
#   3. Convert merged models and vision projectors to GGUF
###############################################################################

# Configuration
LLAMA_CPP_DIR="$HOME/username/llama.cpp"
OUTPUT_DIR="$HOME/Desktop/username/(MLsys)_DriveBench/Models/Qwen3/Distilled_M3KD"
REMOTE_DIR="/root/username/distillm-2/outputs/"
# Add your LoRA adapter directories here.
MODELS_TO_MERGE=(
    "qwen3-vl-2b-m3kd-micro4-8000-REJECTEDCURRICULUM-true_merged"
)

###############################################################################
# STEP 0: Clone or update llama.cpp
###############################################################################

#echo "---------------------------"
#echo "[STEP-0] Preparing llama.cpp"
#echo "---------------------------"
#
#if [[ -d "$LLAMA_CPP_DIR/.git" ]]; then
#    git -C "$LLAMA_CPP_DIR" pull --ff-only
#else
#    mkdir -p "$(dirname "$LLAMA_CPP_DIR")"
#    git clone https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
#fi

#cd "$LLAMA_CPP_DIR"
## Configure the build with CMake
#cmake -B build
#
## Compile the binaries in Release mode
#cmake --build build --config Release

###############################################################################
# STEP 1: copy LoRA adapters from remote server
###############################################################################
#echo "---------------------------"
#echo "[STEP-1] copying LoRA adapters from remote server"
#echo "---------------------------"

#for ADAPTER_PATH in "${MODELS_TO_MERGE[@]}"; do
#    echo "Copying adapter: $ADAPTER_PATH"
#    remote_path="/root/username/distillm-2/outputs/qwen3-vl-2b-m3kd-micro4-8000-REJECTEDCURRICULUM-true_merged"
#    #scp -r root@username:"$ADAPTER_PATH" "/home/username/Desktop/username/(MLsys)_DriveBench/Models/Qwen2/Distilled_M3KD"
#    rsync -avz --progress root@username:"$remote_path" "$OUTPUT_DIR"
#done


################################################################################
## STEP 2: Convert merged models to GGUF
################################################################################
echo "---------------------------"
echo "[STEP-2] Converting models to GGUF"
echo "---------------------------"

for MERGED_PATH in "${MODELS_TO_MERGE[@]}"; do
    MODEL_NAME="$(basename "$MERGED_PATH")"

    TEMP_F16_GGUF="$OUTPUT_DIR/${MODEL_NAME}-f16.gguf"
    MODEL_GGUF="$OUTPUT_DIR/${MODEL_NAME}-Q4_K_M.gguf"
    MMPROJ_GGUF="$OUTPUT_DIR/mmproj-${MODEL_NAME}-f16.gguf"
    MERGED_PATH="$OUTPUT_DIR/$MODEL_NAME"
    echo "Converting model to f16: $MERGED_PATH"

    # 1. Convert the main language model to f16 first.
    python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
        "$MERGED_PATH" \
        --outfile "$TEMP_F16_GGUF" \
        --outtype f16

    # 2. Quantize f16 down to Q4_K_M using llama-quantize.
    echo "Quantizing to Q4_K_M..."
    "$LLAMA_CPP_DIR/build/bin/llama-quantize" "$TEMP_F16_GGUF" "$MODEL_GGUF" Q4_K_M

    # Clean up the intermediate f16 file to save disk space.
    rm "$TEMP_F16_GGUF"

    # 3. Convert the vision projector for multimodal inference.
    python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
        "$MERGED_PATH" \
        --outfile "$MMPROJ_GGUF" \
        --outtype f16 \
        --mmproj

    echo "Model GGUF: $MODEL_GGUF"
    echo "Vision GGUF: $MMPROJ_GGUF"
done

echo "---------------------------"
echo "[DONE] GGUF conversion complete"
echo "---------------------------"

