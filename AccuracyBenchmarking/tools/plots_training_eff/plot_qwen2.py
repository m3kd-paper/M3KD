#!/usr/bin/env python3
"""
Publication-style training-cost plots for Qwen2-VL and Qwen3-VL experiments.

Expected layout:
  EXPERIMENT_ROOT/
    qwen2/
      Qwen2-VL-2B-Instruct/train_results.json
      Qwen2-VL-7B-Instruct/train_results.json
      qwen2-vl-2b-kd-8000/train_results.json
      ...
    qwen3/
      Qwen3-VL-2B-Instruct/train_results.json
      Qwen3-VL-8B-Instruct/train_results.json
      qwen3-vl-2b-kd-8000/train_results.json
      ...

The script recursively searches each model folder for train_results.json.
Edit MODEL_DISPLAY_MAP_QWEN2 / MODEL_DISPLAY_MAP_QWEN3 to match the actual
folder names. Peak VRAM is plotted only when a recognized memory field exists.

Run:
  python plot_training_cost_matplotlib.py --root /path/to/EXPERIMENT_ROOT

Optional:
  python plot_training_cost_matplotlib.py --root ... --results-file all_results.json
"""

import argparse
import json
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


# Keys are folder names relative to ROOT/<family>/.
MODEL_DISPLAY_MAP_QWEN2 = {
    "Qwen2-VL-2B-Instruct": "Qwen2-2B",
    "Qwen2-VL-7B-Instruct": "Qwen2-7B",
    "qwen2-vl-2b-kd-8000": "+KD",
    "qwen2-vl-2b-speculative_kd-8000": "+SpeculativeKD",
    "qwen2-vl-2b-distillm_v2-8000": "+DistillM_v2",
    #"qwen2-vl-2b-HPD_Original-8000": "+HPD",
    #"qwen2-vl-2b-gkd-8000": "+GKD",
    "qwen2-vl-2b-m3kd-micro4-8000-REJECTEDCURRICULUM-true": "+M3KD",
}

# IMPORTANT: Update these keys to your actual Qwen3 output folder names.
MODEL_DISPLAY_MAP_QWEN3 = {
    "Qwen3-VL-2B-Instruct": "Qwen3-2B",
    "Qwen3-VL-8B-Instruct": "Qwen3-8B",
    "qwen3-vl-2b-kd-8000": "+KD",
    "qwen3-vl-2b-speculative_kd-8000": "+SpeculativeKD",
    "qwen3-vl-2b-distillm_v2-8000": "+DistillM_v2",
    #"qwen3-vl-2b-HPD_Original-8000": "+HPD",
    #"qwen3-vl-2b-gkd-8000": "+GKD",
    "qwen3-vl-2b-m3kd-micro4-8000-REJECTEDCURRICULUM-true": "+M3KD",
}

FAMILIES = {
    "qwen2": {
        "subdir": "qwen2",
        "display": "Qwen2-VL",
        "models": MODEL_DISPLAY_MAP_QWEN2,
    },
    "qwen3": {
        "subdir": "qwen3",
        "display": "Qwen3-VL",
        "models": MODEL_DISPLAY_MAP_QWEN3,
    },
}

COLORS = {
    "teacher": "#064e96",  # e.g., 7B / 8B large models
    "student": "#064e96",  # e.g., 2B base student model
    "method": "#064e96",   # e.g., +KD, +M3KD, etc.
    "grid": "#D9DEE7",
    "text": "#202631",
    "missing": "#B8BEC8",
}

OUTPUT_SUBDIR = "training_cost_panels"

# Common field names from Hugging Face Trainer and optional custom logging.
RUNTIME_KEYS = ("train_runtime", "training_runtime_sec", "runtime_seconds")
THROUGHPUT_KEYS = (
    "train_samples_per_second",
    "training_samples_per_second",
    "samples_per_second",
)
SAMPLES_KEYS = ("train_samples", "num_train_samples", "training_samples")
STEPS_KEYS = ("train_steps_per_second", "steps_per_second")
LOSS_KEYS = ("train_loss", "training_loss")

# Prefer reserved memory for the peak footprint; allocated is also retained
# as a fallback. Values with *_gib are already GiB; *_mib / generic fields
# below are interpreted as MiB.
MEMORY_GIB_KEYS = (
    "peak_vram_reserved_gib",
    "peak_gpu_memory_reserved_gib",
    "peak_vram_allocated_gib",
    "peak_gpu_memory_allocated_gib",
    "peak_vram_gib",
)
MEMORY_MIB_KEYS = (
    "peak_vram_reserved_mib",
    "peak_gpu_memory_reserved_mib",
    "peak_vram_allocated_mib",
    "peak_gpu_memory_allocated_mib",
    "peak_vram_mib",
    "peak_vram_reserved",
    "peak_vram_allocated",
    "max_memory_reserved",
    "max_memory_allocated",
)


def finite_number(value):
    """Convert a value to a finite float; return None for missing/invalid data."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def first_number(payload, keys):
    for key in keys:
        value = finite_number(payload.get(key))
        if value is not None:
            return value
    return None


def find_results_file(model_dir, results_file):
    matches = []
    for current_root, _, files in os.walk(model_dir):
        if results_file in files:
            matches.append(os.path.join(current_root, results_file))
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple {results_file} files found under {model_dir}: {matches}. "
            "Keep only the intended final run or narrow the model directory."
        )
    return matches[0]


def load_data(root, family_config, results_file):
    rows = []

    for folder_name, display_name in family_config["models"].items():
        model_dir = os.path.join(root, folder_name)
        if not os.path.isdir(model_dir):
            print(f"WARNING: model folder not found for {display_name}: {model_dir}")
            continue

        try:
            path = find_results_file(model_dir, results_file)
        except RuntimeError as exc:
            print(f"WARNING: {exc}")
            continue

        if path is None:
            print(f"WARNING: {results_file} not found under {model_dir}")
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: cannot read {path}: {exc}")
            continue

        runtime_s = first_number(payload, RUNTIME_KEYS)
        throughput = first_number(payload, THROUGHPUT_KEYS)
        samples = first_number(payload, SAMPLES_KEYS)
        steps_per_s = first_number(payload, STEPS_KEYS)
        train_loss = first_number(payload, LOSS_KEYS)

        memory_gib = first_number(payload, MEMORY_GIB_KEYS)
        if memory_gib is None:
            memory_mib = first_number(payload, MEMORY_MIB_KEYS)
            if memory_mib is not None:
                memory_gib = memory_mib / 1024.0

        if runtime_s is None and throughput is None:
            print(f"WARNING: no runtime or throughput fields found in {path}")
            continue

        rows.append({
            "label": display_name,
            "folder": folder_name,
            "path": path,
            "runtime_s": runtime_s,
            "runtime_h": runtime_s / 3600.0 if runtime_s is not None else None,
            "throughput": throughput,
            "samples": samples,
            "steps_per_s": steps_per_s,
            "train_loss": train_loss,
            "peak_vram_gib": memory_gib,
        })

    return rows


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.tick_params(axis="both", length=0, pad=5, colors=COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])


def bar_colors(rows):
    """Assigns distinct colors for student base models, teacher models, and distillation methods."""
    colors = []
    for i, row in enumerate(rows):
        label_lower = row["label"].lower()
        folder_lower = row["folder"].lower()
        
        # Explicitly categorize based on naming convention
        if "7b" in label_lower or "8b" in label_lower or "7b" in folder_lower or "8b" in folder_lower:
            colors.append(COLORS["teacher"])
        elif i == 0 or "2b" in label_lower or "2b" in folder_lower:
            # First item or explicitly 2B base model acts as the student reference
            colors.append(COLORS["student"])
        else:
            colors.append(COLORS["method"])
    
    # Ensure the absolute first model is treated as student and second as teacher if standard order holds
    if len(colors) >= 2 and not any(k in rows[0]["label"].lower() for k in ["7b", "8b"]):
        colors[0] = COLORS["student"]
    if len(colors) >= 2 and any(k in rows[1]["label"].lower() for k in ["7b", "8b"]):
        colors[1] = COLORS["teacher"]
        
    return colors


def annotate_bars(ax, bars, values, formatter, offset=4):
    for bar, value in zip(bars, values):
        if value is None:
            continue
        ax.annotate(
            formatter(value),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=COLORS["text"],
        )


def plot_metric(ax, rows, key, title, ylabel, formatter, color_list):
    values = [row[key] for row in rows]
    valid = [value for value in values if value is not None]
    style_axis(ax)
    ax.set_ylabel(ylabel, labelpad=5)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([row["label"] for row in rows], rotation=0)

    if not valid:
        ax.text(
            0.5, 0.5, "Metric unavailable",
            transform=ax.transAxes, ha="center", va="center",
            color=COLORS["text"],
        )
        ax.set_ylim(0, 1)
        ax.grid(False)
        return

    bars = []
    for i, value in enumerate(values):
        height = value if value is not None else 0
        color = color_list[i] if value is not None else COLORS["missing"]
        bars.append(ax.bar(
            i, height, width=0.62, color=color, edgecolor="white",
            linewidth=0.8, zorder=3,
        )[0])

    max_value = max(valid)
    ax.set_ylim(0, max_value * 1.30 if max_value > 0 else 1.0)
    annotate_bars(ax, bars, values, formatter)


def write_report(rows, family_name, output_dir):
    report_path = os.path.join(output_dir, "training_metrics_report.txt")
    lines = [
        f"{family_name} Training Systems Measurements",
        "=" * (len(family_name) + 36),
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "Training metrics",
        "----------------",
    ]

    for row in rows:
        lines.extend([
            f"{row['label']} ({row['folder']})",
            f"  Source: {row['path']}",
            f"  Training samples: {row['samples'] if row['samples'] is not None else 'N/A'}",
            f"  Runtime: {row['runtime_s']:.3f} s ({row['runtime_h']:.4f} h)"
            if row["runtime_s"] is not None else "  Runtime: N/A",
            f"  Throughput: {row['throughput']:.6f} samples/s"
            if row["throughput"] is not None else "  Throughput: N/A",
            f"  Steps/s: {row['steps_per_s']:.6f}"
            if row["steps_per_s"] is not None else "  Steps/s: N/A",
            f"  Training loss: {row['train_loss']:.6f}"
            if row["train_loss"] is not None else "  Training loss: N/A",
            f"  Peak GPU memory: {row['peak_vram_gib']:.4f} GiB"
            if row["peak_vram_gib"] is not None else "  Peak GPU memory: N/A",
            "",
        ])

    lines.extend([
        "Interpretation notes",
        "--------------------",
        "Runtime is train_runtime converted from seconds to hours.",
        "Throughput is train_samples_per_second as reported by Trainer.",
        "Peak GPU memory is shown only when a recognized memory field is present.",
        "Memory values with *_gib are treated as GiB; *_mib and generic memory fields are treated as MiB.",
        "Do not compare runtime across runs with different sample counts, epochs, hardware, or optimization settings without qualification.",
        "",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


def write_csv(rows, output_dir):
    import csv

    path = os.path.join(output_dir, "training_metrics.csv")
    fields = [
        "label", "folder", "samples", "runtime_s", "runtime_h",
        "throughput_samples_per_s", "steps_per_s", "train_loss",
        "peak_vram_gib", "source_json",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "label": row["label"],
                "folder": row["folder"],
                "samples": row["samples"],
                "runtime_s": row["runtime_s"],
                "runtime_h": row["runtime_h"],
                "throughput_samples_per_s": row["throughput"],
                "steps_per_s": row["steps_per_s"],
                "train_loss": row["train_loss"],
                "peak_vram_gib": row["peak_vram_gib"],
                "source_json": row["path"],
            })
    return path


def make_figures(rows, family_name, output_dir):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })

    colors = bar_colors(rows)

    # Separate, publication-ready panels.
    panels = [
        ("runtime_h", "Training wall-clock time", "Hours",
         lambda x: f"{x:.2f} h", "training_runtime"),
        ("throughput", "Training throughput", "Samples per second",
         lambda x: f"{x:.3f}", "training_throughput"),
    ]

    has_memory = any(row["peak_vram_gib"] is not None for row in rows)
    if has_memory:
        panels.append((
            "peak_vram_gib", "Peak training GPU memory", "GiB",
            lambda x: f"{x:.2f} GiB", "training_peak_vram",
        ))

    for key, title, ylabel, formatter, filename in panels:
        fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=300)
        fig.subplots_adjust(left=0.17, right=0.97, top=0.88, bottom=0.19)
        plot_metric(ax, rows, key, title, ylabel, formatter, colors)
        path_pdf = os.path.join(output_dir, filename + ".pdf")
        path_png = os.path.join(output_dir, filename + ".png")
        fig.savefig(path_pdf, bbox_inches="tight")
        fig.savefig(path_png, dpi=600, bbox_inches="tight")
        plt.close(fig)

    # Combined 1x2 training-cost overview.
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.4), dpi=300)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.82, bottom=0.19, wspace=0.30)
    fig.suptitle(
        f"{family_name} training systems measurements",
        y=0.97, fontsize=11, fontweight="bold", color=COLORS["text"],
    )
    plot_metric(
        axes[0], rows, "runtime_h", "Training wall-clock time", "Hours",
        lambda x: f"{x:.2f} h", colors,
    )
    plot_metric(
        axes[1], rows, "throughput", "Training throughput", "Samples per second",
        lambda x: f"{x:.3f}", colors,
    )
    combined_pdf = os.path.join(output_dir, "training_cost_comparison.pdf")
    combined_png = os.path.join(output_dir, "training_cost_comparison.png")
    fig.savefig(combined_pdf, bbox_inches="tight")
    fig.savefig(combined_png, dpi=600, bbox_inches="tight")
    plt.close(fig)

    return combined_pdf, combined_png


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="/root/username/distillm-2/outputs", 
        help="Experiment root containing qwen2/ and qwen3/ subdirectories.",
    )
    parser.add_argument(
        "--family", choices=["qwen2", "qwen3", "both"], default="both",
        help="Which model family to plot (default: both).",
    )
    parser.add_argument(
        "--results-file", default="train_results.json",
        help="Training metrics JSON filename (default: train_results.json).",
    )
    parser.add_argument(
        "--output-dir", default="/root/username/MLsys_Cosmos/DriveBench/results_final_100scenes/plots_training_eff",
        help="Optional output directory for plots and reports (default: ROOT/training_cost_panels).",
    )
    args = parser.parse_args()

    root = os.path.abspath(os.path.expanduser(args.root))
    if not os.path.isdir(root):
        raise SystemExit(f"Experiment root does not exist: {root}")

    families = ["qwen2", "qwen3"] if args.family == "both" else [args.family]

    for family in families:
        config = FAMILIES[family]
        rows = load_data(root, config, args.results_file)
        if not rows:
            print(f"WARNING: no usable metrics found for {family}; skipping.")
            continue

        output_dir = os.path.join(args.output_dir, family)
        os.makedirs(output_dir, exist_ok=True)

        combined_pdf, combined_png = make_figures(
            rows, config["display"], output_dir
        )
        csv_path = write_csv(rows, output_dir)
        report_path = write_report(rows, config["display"], output_dir)

        print(f"\n[{family.upper()}] Collected {len(rows)} model runs")
        print(f"  CSV: {csv_path}")
        print(f"  Report: {report_path}")
        print(f"  Combined PDF: {combined_pdf}")
        print(f"  Combined PNG: {combined_png}")
        for row in rows:
            runtime = (
                f"{row['runtime_h']:.2f} h"
                if row["runtime_h"] is not None else "N/A"
            )
            throughput = (
                f"{row['throughput']:.3f} samples/s"
                if row["throughput"] is not None else "N/A"
            )
            memory = (
                f"{row['peak_vram_gib']:.2f} GiB"
                if row["peak_vram_gib"] is not None else "N/A"
            )
            print(
                f"  {row['label']:<20} "
                f"runtime={runtime:<10} throughput={throughput:<20} "
                f"peak_vram={memory}"
            )


if __name__ == "__main__":
    main()