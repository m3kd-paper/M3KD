#!/usr/bin/env bash

GPU=1

source "$(conda info --base)/etc/profile.d/conda.sh"

# Switch environments before enabling nounset.
conda activate vlm-distill

set -Eeuo pipefail

export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME="$HOME/miniconda3/envs/distillm2"
export PATH="$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
hash -r

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"




# Define corresponding output names and corruption values in order.
corruptions=(
    #""              # clean
    #"NoImage"       # noimage
    #"BitError"      # biterror
    #"CameraCrash"   # cameracrash
    #"Fog"           # fog
    "H256ABRCompression"  # h256
    "LowLight"      # lowlight
    "Rain"          # rain
    #"Snow"          # snow
    #"Brightness"    # bright
    #"ColorQuant"    # colorquant
    "FrameLost"     # framelost
    #"LensObstacleCorruption"  # lens
    "MotionBlur"    # motion
    #"Saturate"      # saturate
    #"ZoomBlur"      # zoom
    "WaterSplashCorruption"   # water
)

outputs=(
   # "clean"
   # "noimage"
    #"biterror"
    #"cameracrash"
    #"fog"
    "h256"
        #"lowlight"
    "rain"
    #"snow"
    #"bright"
    #"colorquant"
    #"framelost"
    #"lens"
    "motion"
    "saturate"
    #"zoom"
    "water"
)


models_to_merge=(
    #"/root/username/distillm-2/outputs/qwen2-vl-2b-m3kd-micro1-8000-REJECTEDCURRICULUM-true"
    #"/root/username/distillm-2/outputs/qwen2-vl-2b-m3kd-micro4-8000-REJECTEDCURRICULUM-true"
    #"/root/username/distillm-2/outputs/qwen2-vl-2b-m3kd-micro1-8000-REJECTEDCURRICULUM-true-MACROMIX-true"
    #"/root/username/distillm-2/outputs/qwen2-vl-2b-m3kd-micro4-8000-REJECTEDCURRICULUM-true-MACROMIX-true"
    #"/root/username/distillm-2/outputs/qwen2-vl-2b-m3kd-micro8-8000-REJECTEDCURRICULUM-true-MACROMIX-false"
    "/root/username/distillm-2/outputs/qwen2-vl-2b-sft"
    #"/root/username/distillm-2/outputs/qwen2-vl-2b-kd-8000"
)

models_for_inference=(
    #"/root/username/distillm-2/outputs/qwen2-vl-2b-m3kd-micro1-8000-REJECTEDCURRICULUM-true_merged"
    #"/root/username/distillm-2/outputs/qwen2-vl-2b-m3kd-micro4-8000-REJECTEDCURRICULUM-true_merged"
    #"/root/username/distillm-2/outputs/qwen2-vl-2b-m3kd-micro1-8000-REJECTEDCURRICULUM-true-MACROMIX-true_merged"
    #"/root/username/distillm-2/outputs/qwen2-vl-2b-m3kd-micro4-8000-REJECTEDCURRICULUM-true-MACROMIX-true_merged"
    #"/root/username/distillm-2/outputs/qwen2-vl-2b-m3kd-micro8-8000-REJECTEDCURRICULUM-true-MACROMIX-false_merged"


    "/root/username/distillm-2/outputs/qwen2-vl-2b-sft_merged"
)


    
base_models=(
    #"Qwen/Qwen2-VL-2B-Instruct"
    #"Qwen/Qwen2-VL-7B-Instruct"
)




echo "---------------------------"
echo "[STEP-1] Merge, run inference, and clean up"
echo "---------------------------"

for model in "${models_to_merge[@]}"; do
    merged_model="${model}_merged"

    echo "Merging LoRA weights for model: $model"
    python inference/merge_lora.py \
        --adapter_path "$model" \
        --output_path "$merged_model" || exit 1

    echo "Running inference for model: $merged_model"
    for i in "${!outputs[@]}"; do
        python inference/qwen2vl.py \
            --model "$merged_model" \
            --data data/data/drivebench-test-perception-100-scenes.json \
            --output "res_100scenes/$(basename "$model")/${outputs[i]}" \
            --system_prompt prompt.txt \
            --num_processes "${GPU}" \
            --max_model_len 8192 \
            --num_images_per_prompt 4 \
            --corruption "${corruptions[i]}" || exit 1
    done

    echo "Removing merged checkpoint: $merged_model"
    rm -rf -- "$merged_model"
done


for model in "${base_models[@]}"; do
    echo "Running inference for base model: $model"
    for i in "${!outputs[@]}"; do
        python inference/qwen2vl.py \
            --model "$model" \
            --data data/data/drivebench-test-perception-100-scenes.json \
            --output "res_100scenes/$(basename "$model")/${outputs[i]}" \
            --system_prompt prompt.txt \
            --num_processes "${GPU}" \
            --max_model_len 8192 \
            --num_images_per_prompt 4 \
            --corruption "${corruptions[i]}" || exit 1
    done
done



echo "---------------------------"
echo "All models processed"
echo "---------------------------"



echo "---------------------------"
echo "[STEP-3] Evaluating results"
echo "---------------------------"
for model in "${models_to_merge[@]}"; do
    model="$(echo "$model" | sed 's/\\//g')"
    for file in res_100scenes/$(basename "$model")/*.json; do
        name=$(basename "$file" .json)
        echo "Evaluating: $name"
        mkdir -p "results_Qwen2_ablations_100Scenes/$(basename "$model")"
        python evaluate/eval.py "$file" \
            > "results_Qwen2_ablations_100Scenes/$(basename "$model")/${name}.txt" \
            2>&1
    done
done


for model in "${base_models[@]}"; do
#split by \
    model="$(echo "$model" | sed 's/\\//g')"
    for file in res_100scenes/$(basename "$model")/*.json; do
        name=$(basename "$file" .json)
        echo "Evaluating: $name"
        mkdir -p "results_Qwen2_ablations_100Scenes/$(basename "$model")"
        python evaluate/eval.py "$file" \
            > "results_Qwen2_ablations_100Scenes/$(basename "$model")/${name}.txt" \
            2>&1
    done
done