
#cd "/home/username/Desktop/username/(MLsys)_DriveBench/Models"
#
## Language model: Q4_K_M (~4.7 GB)
#curl -fL --retry 3 -o Qwen2-VL-7B-Instruct-Q4_K_M.gguf \
#  "https://huggingface.co/ggml-org/Qwen2-VL-7B-Instruct-GGUF/resolve/main/Qwen2-VL-7B-Instruct-Q4_K_M.gguf"
#
## Vision projector: F16 (~1.35 GB)
#curl -fL --retry 3 -o mmproj-Qwen2-VL-7B-Instruct-f16.gguf \
#  "https://huggingface.co/ggml-org/Qwen2-VL-7B-Instruct-GGUF/resolve/main/mmproj-Qwen2-VL-7B-Instruct-f16.gguf"





  #/
  #/mmproj-Qwen3VL-8B-Instruct-F16.gguf


#cd "/home/username/Desktop/username/(MLsys)_DriveBench/Models"
#
#
#curl -fL --retry 3 -o Qwen2-VL-2B-Instruct-Q4_K_M.gguf \
#"https://huggingface.co/ggml-org/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-Q4_K_M.gguf"


# Vision projector: F16 (~990 MB)
#curl -fL --retry 3 -o mmproj-Qwen2-VL-2B-Instruct-f16.gguf \
#  "https://huggingface.co/ggml-org/Qwen2-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen2-VL-2B-Instruct-f16.gguf"


#-----------------------------------------------------------#
echo "============================================================"
echo "[Step 2] Downloading Qwen3VL-8B-Instruct GGUF model and projector"
echo "============================================================"

cd "/home/username/Desktop/username/(MLsys)_DriveBench/Models"
mkdir -p Qwen3
mkdir -p Qwen3/Teacher_8B

# Language model: Q4_K_M
curl -fL --retry 3 --retry-all-errors \
  -o Qwen3VL-8B-Instruct-Q4_K_M.gguf \
  "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen3VL-8B-Instruct-Q4_K_M.gguf"

# Vision projector: F16
curl -fL --retry 3 --retry-all-errors \
  -o mmproj-Qwen3VL-8B-Instruct-F16.gguf \
  "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-8B-Instruct-F16.gguf"