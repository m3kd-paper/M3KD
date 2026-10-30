# M3KD: Macro-Micro Monte Carlo Knowledge Distillation

This repository contains the implementation and experimental pipelines for **M3KD**, a knowledge distillation framework for efficient Vision-Language Models (VLMs) targeting traffic understanding.

M3KD combines vocabulary-level distribution matching with student-sampled token-level supervision through a shared training curriculum. The repository provides scripts for model training, accuracy evaluation, and inference efficiency benchmarking.

## Repository Structure

```text
M3KD/
├── M3KDVision/
│   └── scripts/
│       └── vision_exps/
│           ├── 2_generate_KD_datasets.sh
│           ├── 2_generate_KD_datasets_qwen3.sh
│           ├── 3_train_vlm_distilation.sh
│           ├── 3_train_vlm_distilation_ablation.sh
│           ├── 3_train_vlm_distilation_qwen3.sh
│           ├── 3_train_vlm_distilation_qwen3_ablation.sh
│           ├── 3_train_vlm_sft.sh
│           ├── 3_train_vlm_sft_qwen2.sh
│           └── qwen2_qwen3.sh
│
├── AccuracyBenchmarking/
│   └── script/
│       ├── qwen2.sh
│       ├── qwen3.sh
│       └── qwen2_qwen3.sh
│
├── EfficiencyBenchmarking/
│   └── script/
│       └── benchmark_gguf/
│           ├── 0_copy_mode.sh
│           ├── 0_download_model.sh
│           ├── 1_run_all.sh
│           └── plot.py
│
└── README.md
```

## 1. M3KDVision: Model Training

`M3KDVision/` contains the training pipelines for the vision-language models and knowledge distillation experiments.

Scripts are located in:

```text
M3KDVision/scripts/vision_exps/
```

| Script | Description |
|---|---|
| `2_generate_KD_datasets.sh` | Generate distillation datasets for Qwen2-VL. |
| `2_generate_KD_datasets_qwen3.sh` | Generate distillation datasets for Qwen3-VL. |
| `3_train_vlm_distilation.sh` | Train Qwen2-VL using knowledge distillation. |
| `3_train_vlm_distilation_ablation.sh` | Run Qwen2-VL distillation ablation experiments. |
| `3_train_vlm_distilation_qwen3.sh` | Train Qwen3-VL using knowledge distillation. |
| `3_train_vlm_distilation_qwen3_ablation.sh` | Run Qwen3-VL distillation ablation experiments. |
| `3_train_vlm_sft.sh` | Run supervised fine-tuning experiments. |
| `3_train_vlm_sft_qwen2.sh` | Run Qwen2-VL supervised fine-tuning. |
| `qwen2_qwen3.sh` | Helper script for Qwen2/Qwen3 experiments. |

The training pipelines cover:
- Knowledge distillation using M3KD and baseline methods.
- Distillation dataset generation.
- Supervised fine-tuning (SFT) baselines.
- Ablation experiments across Qwen2-VL and Qwen3-VL model families.

## 2. AccuracyBenchmarking: Model Evaluation

`AccuracyBenchmarking/` contains scripts for evaluating trained models on traffic-understanding tasks using DriveBench.

Scripts are located in:

```text
AccuracyBenchmarking/script/
```

| Script | Description |
|---|---|
| `qwen2.sh` | Run accuracy benchmarking for Qwen2-VL models. |
| `qwen3.sh` | Run accuracy benchmarking for Qwen3-VL models. |
| `qwen2_qwen3.sh` | Helper script for Qwen2-VL and Qwen3-VL benchmarking. |

These experiments evaluate model performance, including multiple-choice question (MCQ) accuracy and text-generation metrics, under different image corruption conditions.

## 3. EfficiencyBenchmarking: Inference Efficiency

`EfficiencyBenchmarking/` contains the inference benchmarking pipeline for comparing compact distilled students against their larger teacher models.

The current implementation uses **GGUF models with llama.cpp**.

Scripts are located in:

```text
EfficiencyBenchmarking/script/benchmark_gguf/
```

| Script | Description |
|---|---|
| `0_copy_mode.sh` | Copy trained model adapters and related files for benchmarking. |
| `0_download_model.sh` | Download the required model files. |
| `1_run_all.sh` | Execute the inference benchmarking pipeline. |
| `plot.py` | Generate plots from the collected benchmarking results. |

The efficiency evaluation considers:
- End-to-end inference latency.
- Peak GPU memory consumption.
- Estimated GPU energy per example.
- Generation throughput (tokens/s).

The benchmark compares 2B distilled students against their corresponding larger teacher models using a resource-constrained GPU configuration.

## Experimental Models

The repository includes experiments involving the following model families:

- **Qwen2-VL:** 7B teacher and 2B student.
- **Qwen3-VL:** 8B teacher and 2B student.

## Notes

- Training and accuracy benchmarking scripts are organized by model family.
- Efficiency benchmarking uses GGUF model conversion and the `llama.cpp` inference runtime.
- Update model paths, dataset locations, output directories, and environment-specific settings in the scripts before execution.
- Required dependencies and hardware configurations may differ between training and inference benchmarking.

## Citation

If you use this repository in your research, please cite the associated M3KD paper.

*Citation details will be added upon publication.*
```