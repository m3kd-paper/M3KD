set -Eeo pipefail
GPU=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen3vl

set -u
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

unset CC
unset CXX
hash -r

# Define corresponding output names and corruption values in order.
corruptions=(
    #""              # clean
    #"NoImage"       # noimage
    "BitError"      # biterror
    "CameraCrash"   # cameracrash
    "Fog"           # fog
    "H256ABRCompression"  # h256
    "LowLight"      # lowlight
    "Rain"          # rain
    "Snow"          # snow
    "Brightness"    # bright
    "ColorQuant"    # colorquant
    "FrameLost"     # framelost
    "LensObstacleCorruption"  # lens
    "MotionBlur"    # motion
    "Saturate"      # saturate
    "ZoomBlur"      # zoom
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
    #"/root/username/distillm-2/outputs/qwen3-vl-2b-distillm_v2-8000"
    #"/root/username/distillm-2/outputs/qwen3-vl-2b-speculative_kd-8000"
    #"/root/username/distillm-2/outputs/qwen3-vl-2b-HPD_Original-8000"
    #"/root/username/distillm-2/outputs/qwen3-vl-2b-gkd-8000"
    "/root/username/distillm-2/outputs/qwen3-vl-2b-sft"
    #"/root/username/distillm-2/outputs/qwen3-vl-2b-m3kd-micro1-8000-REJECTEDCURRICULUM-true-MACROMIX-true"
    #"/root/username/distillm-2/outputs/qwen3-vl-2b-m3kd-micro4-8000-REJECTEDCURRICULUM-true"
    #"/root/username/distillm-2/outputs/qwen3-vl-2b-m3kd-micro8-8000-REJECTEDCURRICULUM-true-MACROMIX-true"
)

models_for_inference=(
    #"/root/username/distillm-2/outputs/qwen3-vl-2b-distillm_v2-8000_merged"
    #"/root/username/distillm-2/outputs/qwen3-vl-2b-speculative_kd-8000_merged"
    #"/root/username/distillm-2/outputs/qwen3-vl-2b-HPD_Original-8000_merged"
    #"/root/username/distillm-2/outputs/qwen3-vl-2b-gkd-8000_merged"
    #"/root/username/distillm-2/outputs/qwen3-vl-2b-m3kd-micro1-8000-REJECTEDCURRICULUM-true-MACROMIX-true_merged"
    #"/root/username/distillm-2/outputs/qwen3-vl-2b-m3kd-micro4-8000-REJECTEDCURRICULUM-true_merged"
    #"/root/username/distillm-2/outputs/qwen3-vl-2b-m3kd-micro8-8000-REJECTEDCURRICULUM-true-MACROMIX-true_merged"
    "/root/username/distillm-2/outputs/qwen3-vl-2b-sft_merged"
)

base_models=(
    "Qwen/Qwen3-VL-2B-Instruct"
    "Qwen/Qwen3-VL-8B-Instruct"
)

echo "---------------------------"
echo "[STEP-1] Merge, run inference, and clean up"
echo "---------------------------"

for model in "${models_to_merge[@]}"; do
    merged_model="${model}_merged"

    echo "Merging LoRA weights for model: $model"
    python inference/merge_lora_qwen3.py \
        --adapter_path "$model" \
        --output_path "$merged_model" \
        --base_model "Qwen/Qwen3-VL-2B-Instruct" || exit 1

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
        mkdir -p "results_Qwen3_ablations_100Scenes/$(basename "$model")"
        python evaluate/eval.py "$file" \
            > "results_Qwen3_ablations_100Scenes/$(basename "$model")/${name}.txt" \
            2>&1
    done
done


for model in "${base_models[@]}"; do
#split by \
    model="$(echo "$model" | sed 's/\\//g')"
    for file in res_100scenes/$(basename "$model")/*.json; do
        name=$(basename "$file" .json)
        echo "Evaluating: $name"
        mkdir -p "results_Qwen3_ablations_100Scenes/$(basename "$model")"
        python evaluate/eval.py "$file" \
            > "results_Qwen3_ablations_100Scenes/$(basename "$model")/${name}.txt" \
            2>&1
    done
done