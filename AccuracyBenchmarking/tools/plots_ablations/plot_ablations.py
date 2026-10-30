#!/usr/init/env python3
"""Aggregate DriveBench ablation corruption evaluations and generate paper-ready outputs."""

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
    ("qwen2-vl-2b-m3kd-micro1-8000-REJECTEDCURRICULUM-true", "No-MacroMixing-No-MonteCarlo"),
    ("qwen2-vl-2b-m3kd-micro4-8000-REJECTEDCURRICULUM-true", "No-MacroMixing-MonteCarlo"),
    ("qwen2-vl-2b-m3kd-micro1-8000-REJECTEDCURRICULUM-true-MACROMIX-true", "MacroMixing-No-MonteCarlo"),
    ("qwen2-vl-2b-m3kd-micro4-8000-REJECTEDCURRICULUM-true-MACROMIX-true", "MacroMixing-MonteCarlo"),
])

MODEL_DISPLAY_MAP_qwen3 = OrderedDict([
   ("qwen3-vl-2b-m3kd-micro2-8000_merged", "2 samples"),
   ("qwen3-vl-2b-m3kd-micro4-8000_merged", "4 samples"),
   ("qwen3-vl-2b-m3kd-micro1-8000-REJECTEDCURRICULUM-true-MACROMIX-true", "MacroMixing-No-MonteCarlo"),
   ("qwen3-vl-2b-m3kd-micro4-8000-REJECTEDCURRICULUM-true", "MacroMixing-MonteCarlo"),
   ("qwen3-vl-2b-m3kd-micro8-8000-REJECTEDCURRICULUM-true-MACROMIX-true", "MacroMixing-8MonteCarlo")
])


# =====================================================================
# LATEX TABLE METRICS CONFIGURATION
# =====================================================================
LATEX_TABLE_METRICS = [
    "MCQ.accuracy",
    "VQA.language_metrics.ROUGE_L",
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
        # Skip directories not in our ablation mapping
        if display_name == model_dir.name and model_dir.name not in model_map:
            continue

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
        "VQA.language_metrics.ROUGE_L": "ROUGE-L",
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
    out_path: Path = Path("ablation_sample_size_table.tex"),
) -> None:
    """Write a compact, single-column LaTeX table for the M3KD ablation."""

    rows_config = [
        (False, r"\times", "No-MacroMixing-No-MonteCarlo", None),
        (False, "4", "No-MacroMixing-MonteCarlo", None),
        (True, r"\times", "MacroMixing-No-MonteCarlo",
         "MacroMixing-No-MonteCarlo"),
        (True, "4", "MacroMixing-MonteCarlo",
         "MacroMixing-MonteCarlo"),
        (True, "8", None, "MacroMixing-8MonteCarlo"),
    ]

    # Bold all metrics for selected configurations.
    BOLD_CELLS = {
        ("qwen2", "No-MacroMixing-MonteCarlo"),
        ("qwen3", "MacroMixing-8MonteCarlo"),
    }

    def get_task_averaged_metric(df, model_name, metric_key):
        if model_name is None:
            return None, None

        sub = df[
            (df["model"] == model_name)
            & (df["metric"] == metric_key)
            & (df["task"].isin(tasks))
        ]

        if sub.empty:
            return None, None

        mean_val = sub["mean_score"].mean()
        std_val = np.sqrt((sub["std_score"] ** 2).mean())
        return mean_val, std_val

    def format_metric(df, model_name, metric, bold=False):
        mean_val, std_val = get_task_averaged_metric(
            df, model_name, metric
        )

        if mean_val is None:
            return "--"

        if bold:
            return rf"$\mathbf{{{mean_val:.3f}\pm{std_val:.3f}}}$"

        return rf"${mean_val:.3f}\pm{std_val:.3f}$"

    num_metrics = len(metrics)
    total_cols = 2 + 2 * num_metrics

    metric_headers = " & ".join(
        rf"\textbf{{{metric_label(m)}}}" for m in metrics
    )

    lines = [
        r"% Requires: \usepackage{booktabs,amssymb}",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation of macro mixing and Monte Carlo sample count ($N$) for Qwen2-VL-2B and Qwen3-VL-2B.}",
        r"\label{tab:m3kd_sample_size_ablation}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{0.5pt}",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\begin{tabular}{@{}cc" + "c" * (2 * num_metrics) + r"@{}}",
        r"\toprule",
        (
            r"\multicolumn{2}{c}{} & "
            + rf"\multicolumn{{{num_metrics}}}{{c}}{{\textbf{{Qwen2-VL-2B}}}}"
            + " & "
            + rf"\multicolumn{{{num_metrics}}}{{c}}{{\textbf{{Qwen3-VL-2B}}}}"
            + r" \\"
        ),
        (
            rf"\cmidrule(lr){{3-{2 + num_metrics}}} "
            + rf"\cmidrule(lr){{{3 + num_metrics}-{total_cols}}}"
        ),
        (
            r"\textbf{MacroMix} & \textbf{$N$} & "
            + metric_headers + " & " + metric_headers + r" \\"
        ),
        r"\midrule",
    ]

    for macro_enabled, mc_samples, qwen2_label, qwen3_label in rows_config:
        macro_symbol = r"\checkmark" if macro_enabled else r"$\times$"
        sample_symbol = (
            r"$\times$" if mc_samples == r"\times" else mc_samples
        )

        row_cells = [macro_symbol, sample_symbol]

        # Qwen2 metrics
        for metric in metrics:
            bold = ("qwen2", qwen2_label) in BOLD_CELLS
            row_cells.append(
                format_metric(
                    aggregate_qwen2, qwen2_label, metric, bold=bold
                )
            )

        # Qwen3 metrics
        if not macro_enabled:
            row_cells.append(
                rf"\multicolumn{{{num_metrics}}}{{c}}{{\textit{{Unstable}}}}"
            )
        else:
            for metric in metrics:
                bold = ("qwen3", qwen3_label) in BOLD_CELLS
                row_cells.append(
                    format_metric(
                        aggregate_qwen3, qwen3_label, metric, bold=bold
                    )
                )

        lines.append(" & ".join(row_cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        (
            r"\parbox{\columnwidth}{\scriptsize "
            r"\textit{Unstable} indicates that Qwen3 training without "
            r"macro mixing was unstable, preventing reliable evaluation.}"
        ),
        r"\end{table}",
        "",
    ])

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
    """Create a single comprehensive heatmap for the ablation study models."""
    sns.set_theme(style="white", context="paper", font_scale=1.7)
    
    models = list(model_map.values())
    
    subset = raw[(raw["metric"] == "MCQ.accuracy") & (raw["task"] == "perception")]
    if subset.empty:
        print(f"No perception MCQ data found for ablation family {family_name}")
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
        print(f"Matrix for ablation family {family_name} is empty.")
        return

    avg_row = matrix.mean(axis=0).to_frame().T
    avg_row.index = ["Average"]
    matrix = pd.concat([matrix, avg_row])

    if out_txt_path:
        title_str = f"Ablation Study (Sample Size) - {family_name} Perception MCQ Robustness"
        txt_content = heatmap_to_text(matrix, title_str)
        out_txt_path.write_text(txt_content, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(max(8.0, len(models) * 1.8), 7.8))
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

    labels = [item.get_text() for item in ax.get_yticklabels()]
    if labels:
        labels[-1] = "Average"
    ax.set_yticklabels(labels, rotation=0, weight="bold")

    ax.set_title(
        f"M3KD Sample Size Ablation: {family_name} Family",
        pad=14,
        weight="bold",
    )
    ax.set_xlabel("Sample Size ($N$)")
    ax.set_ylabel("Corruption type")
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=0,
        ha="center",
    )
    fig.tight_layout()

    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate DriveBench ablation evaluations."
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
            "/root/username/MLsys_Cosmos/DriveBench/results_final_100scenes/plots_ablation"
        ),
        help="Output directory for ablation plots and tables.",
    )
    args = parser.parse_args()

    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Qwen2 Ablations
    root_qwen2 = args.root_qwen2.expanduser().resolve()
    print(f"Reading Qwen2 ablation results from: {root_qwen2}")
    raw_qwen2, _ = discover_results(MODEL_DISPLAY_MAP_qwen2, root_qwen2)
    aggregate_qwen2 = aggregate_results(raw_qwen2)

    # Qwen3 Ablations
    root_qwen3 = args.root_qwen3.expanduser().resolve()
    print(f"Reading Qwen3 ablation results from: {root_qwen3}")
    raw_qwen3, _ = discover_results(MODEL_DISPLAY_MAP_qwen3, root_qwen3)
    aggregate_qwen3 = aggregate_results(raw_qwen3)

    raw_qwen2.to_csv(outdir / "all_metrics_long_qwen2_ablation.csv", index=False)
    aggregate_qwen3.to_csv(outdir / "model_aggregate_qwen3_ablation.csv", index=False)

    # Write Ablation LaTeX Table
    write_latex_table(
        aggregate_qwen2,
        aggregate_qwen3,
        metrics=["MCQ.accuracy", "VQA.language_metrics.ROUGE_L"],
        out_path=outdir / "ablation_sample_size_table.tex",
    )

    # Generate Ablation Heatmaps and Text Equivalents
    print("Generating ablation heatmap and text table for Qwen2...")
    generate_family_comparison_heatmap(
        raw=raw_qwen2,
        model_map=MODEL_DISPLAY_MAP_qwen2,
        family_name="Qwen2",
        out_path=outdir / "heatmap_ablation_qwen2_samples.pdf",
        out_txt_path=outdir / "heatmap_ablation_qwen2_samples.txt",
    )
    
    print("Generating ablation heatmap and text table for Qwen3...")
    generate_family_comparison_heatmap(
        raw=raw_qwen3,
        model_map=MODEL_DISPLAY_MAP_qwen3,
        family_name="Qwen3",
        out_path=outdir / "heatmap_ablation_qwen3_samples.pdf",
        out_txt_path=outdir / "heatmap_ablation_qwen3_samples.txt",
    )


if __name__ == "__main__":
    main()