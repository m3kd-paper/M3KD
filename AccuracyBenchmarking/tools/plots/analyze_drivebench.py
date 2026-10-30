#!/usr/init/env python3
"""Aggregate DriveBench corruption evaluations and generate paper-ready outputs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


# =====================================================================
# MODEL NAME MAPPING (Ordered dictionary: folder_name --> display_name)
# =====================================================================
MODEL_DISPLAY_MAP_qwen2 = OrderedDict([
    ("Qwen2-VL-2B-Instruct", "Qwen2-VL-2B-Instruct"),
    ("Qwen2-VL-7B-Instruct", "Qwen2-VL-7B-Instruct"),
    ("qwen2-vl-2b-sft", "+SFT"),
    ("qwen2-vl-2b-kd-8000", "+KD"),
    ("qwen2-vl-2b-speculative_kd-8000", "+SpeculativeKD"),
    ("qwen2-vl-2b-distillm_v2-8000", "+DistillM_v2"),
    #("qwen2-vl-2b-HPD_Original-8000", "+HPD"),
    #("qwen2-vl-2b-gkd-8000", "+GKD"),
    ("qwen2-vl-2b-m3kd-micro4-8000-REJECTEDCURRICULUM-true", "+M3KD"),
])
MODEL_DISPLAY_MAP_qwen3 = OrderedDict([
    ("Qwen3-VL-2B-Instruct", "Qwen3-VL-2B-Instruct"),
    ("Qwen3-VL-8B-Instruct", "Qwen3-VL-8B-Instruct"),
    ("qwen3-vl-2b-sft", "+SFT"),
    ("qwen3-vl-2b-kd-8000", "+KD"),
    ("qwen3-vl-2b-speculative_kd-8000", "+SpeculativeKD"),
    ("qwen3-vl-2b-distillm_v2-8000", "+DistillM_v2"),
    #("qwen3-vl-2b-HPD_Original-8000", "+HPD"),
    #("qwen3-vl-2b-gkd-8000", "+GKD"),
    ("qwen3-vl-2b-m3kd-micro8-8000-REJECTEDCURRICULUM-true-MACROMIX-true", "+M3KD"),
])


# =====================================================================
# LATEX TABLE METRICS CONFIGURATION
# Customize which metrics to include and display in model_aggregate_table.tex
# =====================================================================
LATEX_TABLE_METRICS = [
    "MCQ.accuracy",
    "VQA.language_metrics.Bleu_1",
    #"VQA.language_metrics.Bleu_2",
    #"VQA.language_metrics.Bleu_3",
    "VQA.language_metrics.Bleu_4",
    "VQA.language_metrics.ROUGE_L",
    #"VQA.language_metrics.CIDEr",
]


CORRUPTIONS = [
    #"biterror",
    #"bright",
    #"cameracrash",
    #"colorquant",
    #"fog",
    #"framelost",
    "h256",
    #"lens",
    #"lowlight",
    "motion",
    "rain",
    "saturate",
    #"snow",
    "water",
    #"zoom",
]

TASKS = ["perception"]

# Metrics that are expected to be comparable across corruption files.
METRIC_ORDER = [
    "MCQ.accuracy",
    "VQA.language_metrics.Bleu_1",
    "VQA.language_metrics.Bleu_2",
    "VQA.language_metrics.Bleu_3",
    "VQA.language_metrics.Bleu_4",
    "VQA.language_metrics.ROUGE_L",
    "VQA.language_metrics.CIDEr",
]


def get_model_display_name(model_map, folder_name: str) -> str:
    return model_map.get(folder_name, folder_name)


def parse_scores(path: Path) -> dict[str, Any]:
    """Parse the JSON object following 'Final Scores:'."""
    text = path.read_text(encoding="utf-8", errors="replace")

    marker = "Final Scores:"
    if marker not in text:
        raise ValueError(f"Missing '{marker}'")

    tail = text.split(marker, 1)[1]
    start = tail.find("{")
    if start < 0:
        raise ValueError("No JSON object found after Final Scores")

    try:
        scores, _ = json.JSONDecoder().raw_decode(tail[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid score JSON: {exc}") from exc

    if not isinstance(scores, dict):
        raise ValueError("Score JSON is not an object")

    return scores


def flatten_scores(
    scores: dict[str, Any],
    model: str,
    corruption: str,
) -> list[dict[str, Any]]:
    """Flatten nested task/metric dictionaries into one row per metric."""
    rows = []

    def walk(node: Any, path: list[str]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, path + [str(key)])
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            if len(path) >= 3:
                task = path[0]
                metric = ".".join(path[1:])
                rows.append({
                    "model": model,
                    "corruption": corruption,
                    "task": task,
                    "metric": metric,
                    "score": float(node),
                })

    for task, contents in scores.items():
        walk(contents, [str(task)])

    return rows


def discover_results(model_map, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read all model/corruption files and report missing or invalid files."""
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []

    model_dirs = sorted(p for p in root.iterdir() if p.is_dir())

    if not model_dirs:
        raise RuntimeError(f"No model directories found in {root}")

    for model_dir in model_dirs:
        display_name = get_model_display_name(model_map, model_dir.name)

        for corruption in CORRUPTIONS:
            path = model_dir / f"{corruption}.txt"

            if not path.exists():
                issues.append({
                    "model": display_name,
                    "corruption": corruption,
                    "issue": "missing file",
                    "path": str(path),
                })
                continue

            try:
                scores = parse_scores(path)
                file_rows = flatten_scores(
                    scores, display_name, corruption
                )

                if not file_rows:
                    raise ValueError("No numeric metrics found")

                rows.extend(file_rows)

            except Exception as exc:
                issues.append({
                    "model": display_name,
                    "corruption": corruption,
                    "issue": str(exc),
                    "path": str(path),
                })

    if not rows:
        raise RuntimeError("No scores parsed. Check the directory and files.")

    return pd.DataFrame(rows), pd.DataFrame(
        issues, columns=["model", "corruption", "issue", "path"]
    )


def aggregate_results(raw: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight mean and standard deviation across corruption types."""
    return (
        raw.groupby(["model", "task", "metric"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            std_score=("score", "std"),
            n_corruptions=("corruption", "nunique"),
        )
    )


def metric_label(metric: str) -> str:
    return {
        "MCQ.accuracy": "MCQ",
        "VQA.language_metrics.Bleu_1": "BLEU-1",
        "VQA.language_metrics.Bleu_2": "BLEU-2",
        "VQA.language_metrics.Bleu_3": "BLEU-3",
        "VQA.language_metrics.Bleu_4": "BLEU-4",
        "VQA.language_metrics.ROUGE_L": "ROUGE-L",
        "VQA.language_metrics.CIDEr": "CIDEr",
    }.get(metric, metric.replace(".", " / "))


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)

def write_latex_table(
    aggregate_qwen2: pd.DataFrame,
    aggregate_qwen3: pd.DataFrame,
    metrics: list[str],
    tasks: list[str] = TASKS,
    out_path: Path = Path("model_aggregate_table.tex")
) -> None:
    """Write a single-column LaTeX table comparing Qwen2 and Qwen3 distillation results 
    across a dynamically specified list of metrics, averaged over tasks and corruptions,
    with professional row shading and section dividers.
    """
    # Rows config: (Label, Qwen2 Model Name, Qwen3 Model Name, Row Type)
    # Types: 'baseline' (Teacher/Student Base), 'distilled' (Standard Methods), 'highlight' (M3KD)
    rows_config = [
        ("Teacher", "Qwen2-VL-7B-Instruct", "Qwen3-VL-8B-Instruct", "baseline"),
        ("Student Base", "Qwen2-VL-2B-Instruct", "Qwen3-VL-2B-Instruct", "baseline"),
        ("+SFT", "+SFT", "+SFT", "distilled"),
        ("+KD", "+KD", "+KD", "distilled"),
        ("+SpeculativeKD", "+SpeculativeKD", "+SpeculativeKD", "distilled"),
        ("+DistillM\\_v2", "+DistillM_v2", "+DistillM_v2", "distilled"),
        #("+HPD", "+HPD", "+HPD", "distilled"),
        #("+GKD", "+GKD", "+GKD", "distilled"),  
        ("+M3KD (Ours)", "+M3KD", "+M3KD", "highlight"),
    ]

    def get_task_averaged_metric(df, model_name, metric_key):
        if not model_name:
            return None, None
        sub = df[(df["model"] == model_name) & (df["metric"] == metric_key) & (df["task"].isin(tasks))]
        if sub.empty:
            return None, None
        mean_val = sub["mean_score"].mean()
        std_val = np.sqrt((sub["std_score"] ** 2).mean())
        return mean_val, std_val

    num_metrics = len(metrics)
    # Added a vertical bar '|' to cleanly separate Qwen2 columns from Qwen3 columns
    col_format = "l " + " ".join(["c" * num_metrics]) + " | " + " ".join(["c" * num_metrics])

    lines = [
        r"% Generated by analyze_drivebench.py",
        r"% Requires: \usepackage{booktabs,graphicx,colortbl,xcolor}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Cross-Family Distillation Comparison: Qwen2 vs. Qwen3 averaged across all DriveBench tasks and 15 corruptions.}",
        r"\label{tab:qwen2_vs_qwen3_distillation_summary}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{" + col_format + r"}",
        r"\toprule",
        r" & \multicolumn{" + str(num_metrics) + r"}{c}{\textbf{Qwen2 (7$\rightarrow$2B)}} & \multicolumn{" + str(num_metrics) + r"}{c}{\textbf{Qwen3 (8$\rightarrow$2B)}} \\",
        r"\cmidrule(lr){2-" + str(1 + num_metrics) + r"} \cmidrule(l){" + str(2 + num_metrics) + r"-" + str(1 + 2 * num_metrics) + r"}",
        r"\textbf{Model / Method} & " + " & ".join(f"\\textbf{{{metric_label(m)}}}" for m in metrics) + " & " + " & ".join(f"\\textbf{{{metric_label(m)}}}" for m in metrics) + r" \\",
        r"\midrule",
    ]

    for idx, (label, q2_model, q3_model, row_type) in enumerate(rows_config):
        row_cells = []
        
        # Format label with bolding for your method
        if row_type == "highlight":
            row_cells.append(f"\\textbf{{{latex_escape(label)}}}")
        else:
            row_cells.append(latex_escape(label))
        
        for metric in metrics:
            m_val, s_val = get_task_averaged_metric(aggregate_qwen2, q2_model, metric)
            cell_str = f"${m_val:.3f} \\pm {s_val:.3f}$" if m_val is not None else "--"
            if row_type == "highlight":
                cell_str = f"\\textbf{{{cell_str}}}"
            row_cells.append(cell_str)
            
        for metric in metrics:
            m_val, s_val = get_task_averaged_metric(aggregate_qwen3, q3_model, metric)
            cell_str = f"${m_val:.3f} \\pm {s_val:.3f}$" if m_val is not None else "--"
            if row_type == "highlight":
                cell_str = f"\\textbf{{{cell_str}}}"
            row_cells.append(cell_str)

        row_str = " & ".join(row_cells) + r" \\"

        # Apply light gray shading to baseline rows (Teacher & Student Base)
        if row_type == "baseline":
            row_str = r"\rowcolor{black!4} " + row_str

        lines.append(row_str)

        # Add divider lines based on your structural request:
        # 1. After the Student Base (separating baselines from distilled models)
        if label == "Student Base":
            lines.append(r"\midrule")
        # 2. Right before +M3KD (separating standard distilled models from your method)
        elif label == "+GKD":
            lines.append(r"\hdashline" if r"\usepackage{arydshln}" in locals() else r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")

def heatmap_to_text(matrix: pd.DataFrame, title: str) -> str:
    """Convert a heatmap data matrix into a clean formatted string table."""
    formatted_df = matrix.applymap(lambda val: f"{val:.3f}" if pd.notnull(val) else "--")
    table_str = formatted_df.to_string()
    return f"=== {title} ===\n\n{table_str}\n"


def generate_family_comparison_heatmap(
    raw: pd.DataFrame,
    model_map: OrderedDict[str, str],
    family_name: str,
    out_path: Path,
    out_txt_path: Path | None = None,
) -> None:
    """Create a single comprehensive heatmap for an entire model family (Qwen2 or Qwen3) 
    showing all models mapped from model_map on Perception MCQ.accuracy, and export a text equivalent.
    """
    sns.set_theme(style="white", context="paper", font_scale=1.7)
    
    models = list(model_map.values())
    
    subset = raw[(raw["metric"] == "MCQ.accuracy") & (raw["task"] == "perception")]
    if subset.empty:
        print(f"No perception MCQ data found for family {family_name}")
        return

    matrix = (
        subset.pivot_table(
            index="corruption",
            columns="model",
            values="score",
            aggfunc="mean",
        )
        .reindex(index=CORRUPTIONS, columns=models)
    )

    if matrix.isna().all().all():
        print(f"Matrix for family {family_name} is empty.")
        return

    avg_row = matrix.mean(axis=0).to_frame().T
    avg_row.index = ["Average"]
    matrix = pd.concat([matrix, avg_row])

    # Save text representation if path is specified
    if out_txt_path:
        title_str = f"DriveBench Perception MCQ Robustness: {family_name} Family"
        txt_content = heatmap_to_text(matrix, title_str)
        out_txt_path.write_text(txt_content, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(max(10.0, len(models) * 1.4), 7.8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        linewidths=0.4,
        linecolor="white",
        mask=matrix.isna(),
        cbar_kws={"label": "MCQ Accuracy"},
        ax=ax,
    )

    ax.axhline(y=len(matrix) - 1, color="black", linewidth=2.5)
    
    if len(models) > 2:
        ax.axvline(x=2, color="black", linestyle="-", linewidth=2.5, alpha=0.9)
    if len(models) > 6:
        ax.axvline(x=6, color="black", linestyle="-", linewidth=2.5, alpha=0.9)

    labels = [item.get_text() for item in ax.get_yticklabels()]
    if labels:
        labels[-1] = "Average"
    ax.set_yticklabels(labels, rotation=0, weight="bold")

    ax.set_title(
        f"DriveBench Perception MCQ Robustness: {family_name} Family",
        pad=14,
        weight="bold",
    )
    ax.set_xlabel("Model / Method")
    ax.set_ylabel("Corruption type")
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=30,
        ha="right",
    )
    fig.tight_layout()

    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate DriveBench corruption evaluations."
    )
    parser.add_argument(
        "--root_qwen2",
        type=Path,
        default=Path(
            "/root/username/MLsys_Cosmos/DriveBench/results_Qwen2_ablations_100Scenes"
        ),
        help="Directory containing one subdirectory per model.",
    )
    parser.add_argument(
        "--root_qwen3",
        type=Path,
        default=Path(
            "/root/username/MLsys_Cosmos/DriveBench/results_Qwen3_ablations_100Scenes"
        ),
        help="Directory containing one subdirectory per model.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(
            "/root/username/MLsys_Cosmos/DriveBench/results_final_100scenes/plots"
        ),
        help="Output directory (default: ROOT/analysis).",
    )
    args = parser.parse_args()

    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Qwen2 
    root_qwen2 = args.root_qwen2.expanduser().resolve()
    print(f"Reading results from: {root_qwen2}")
    raw_qwen2, issues_qwen2 = discover_results(MODEL_DISPLAY_MAP_qwen2, root_qwen2)
    aggregate_qwen2 = aggregate_results(raw_qwen2)

    # Qwen3
    root_qwen3 = args.root_qwen3.expanduser().resolve()
    print(f"Reading results from: {root_qwen3}")
    raw_qwen3, issues_qwen3 = discover_results(MODEL_DISPLAY_MAP_qwen3, root_qwen3)
    aggregate_qwen3 = aggregate_results(raw_qwen3)

    raw_qwen2.to_csv(outdir / "all_metrics_long_qwen2.csv", index=False)
    aggregate_qwen3.to_csv(outdir / "model_aggregate_qwen2.csv", index=False)

    write_latex_table(
        aggregate_qwen2,
        aggregate_qwen3,
        metrics=LATEX_TABLE_METRICS,
        out_path=outdir / "model_aggregate_table.tex",
    )

    # Generate comprehensive family heatmaps and text equivalents for Perception MCQ
    print("Generating comprehensive family heatmap and text table for Qwen2...")
    generate_family_comparison_heatmap(
        raw=raw_qwen2,
        model_map=MODEL_DISPLAY_MAP_qwen2,
        family_name="Qwen2",
        out_path=outdir / "heatmap_perception_mcq_qwen2_all_methods.pdf",
        out_txt_path=outdir / "heatmap_perception_mcq_qwen2_all_methods.txt",
    )
    
    print("Generating comprehensive family heatmap and text table for Qwen3...")
    generate_family_comparison_heatmap(
        raw=raw_qwen3,
        model_map=MODEL_DISPLAY_MAP_qwen3,
        family_name="Qwen3",
        out_path=outdir / "heatmap_perception_mcq_qwen3_all_methods.pdf",
        out_txt_path=outdir / "heatmap_perception_mcq_qwen3_all_methods.txt",
    )


if __name__ == "__main__":
    main()