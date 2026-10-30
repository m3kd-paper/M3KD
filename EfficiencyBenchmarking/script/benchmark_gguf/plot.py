import json
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

# Run from the directory containing res_new_FULL_Efficiency, or update these paths.
MODEL_FILES = {
    "Qwen2-7B": "res_new_FULL_Efficiency/Qwen2-VL-7B-Instruct-gguf/h256.efficiency.json",
    "Qwen2-2B M3KD": "res_new_FULL_Efficiency/Qwen2-VL-2b-m3kd-gguf/h256.efficiency.json",

    "Qwen3-8B": "res_new_FULL_Efficiency/Qwen3VL-8B-Instruct-gguf/h256.efficiency.json",
    "Qwen3-2B M3KD": "res_new_FULL_Efficiency/Qwen3-VL-m3kd-gguf/h256.efficiency.json",
}
COLORS = {"teacher": "#3B5B92", "student": "#D17A35", "grid": "#D9DEE7", "text": "#202631"}

OUTPUT_DIR = "figures/efficiency_panels"
COMBINED_PDF = os.path.join(OUTPUT_DIR, "mlsys_distillation_comparison.pdf")
COMBINED_PNG = os.path.join(OUTPUT_DIR, "mlsys_distillation_comparison.png")
TEXT_REPORT = os.path.join(OUTPUT_DIR, "mlsys_distillation_comparison.txt")

METRICS = [
    {
        "key": "Latency (s)",
        "filename": "latency",
        "title": "End-to-end latency",
        "ylabel": "Seconds per example",
        "lower_better": True,
        "format": lambda x: f"{x:.4f} s",
        "report_format": lambda x: f"{x:.6f} s",
    },
    {
        "key": "VRAM (GB)",
        "filename": "vram",
        "title": "Peak device-wide VRAM",
        "ylabel": "GB (GiB when converted from MiB)",
        "lower_better": True,
        "format": lambda x: f"{x:.2f} GB",
        "report_format": lambda x: f"{x:.6f} GiB",
    },
    {
        "key": "Energy (J/ex)",
        "filename": "energy_per_example",
        "title": "Estimated GPU energy per example",
        "ylabel": "Joules per example",
        "lower_better": True,
        "format": lambda x: f"{x:.2f} J",
        "report_format": lambda x: f"{x:.6f} J/example",
    },
    {
        "key": "Throughput (tok/s)",
        "filename": "throughput",
        "title": "Generation throughput",
        "ylabel": "Generated tokens per second",
        "lower_better": False,
        "format": lambda x: f"{x:.2f} tok/s",
        "report_format": lambda x: f"{x:.6f} tokens/s",
    },
]


def finite_number(value):
    """Convert a value to a finite float; return None for missing/invalid data."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def load_data(file_dict):
    rows = []
    for label, path in file_dict.items():
        if not os.path.isfile(path):
            print(f"WARNING: summary not found for {label}: {path}")
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: cannot read {path}: {exc}")
            continue

        telemetry = payload.get("telemetry") or {}
        vram_mb = finite_number(telemetry.get("gpu_memory_peak_mb_device_wide"))
        rows.append({
            "label": label,
            "path": path,
            "Latency (s)": finite_number(
                payload.get("per_example_end_to_end_latency_sec_mean")
            ),
            # The source field is MiB; divide by 1024 to report GiB.
            "VRAM (GB)": vram_mb / 1024.0 if vram_mb is not None else None,
            "Energy (J/ex)": finite_number(
                payload.get("gpu_energy_j_per_example_estimated")
            ),
            "Throughput (tok/s)": finite_number(
                payload.get("generated_tokens_per_sec")
            ),
            "raw_vram_mb": vram_mb,
            "raw_latency": payload.get("per_example_end_to_end_latency_sec_mean"),
            "raw_energy": payload.get("gpu_energy_j_per_example_estimated"),
            "raw_throughput": payload.get("generated_tokens_per_sec"),
        })
    return rows


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linewidth=0.6, alpha=0.35)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.35)
    ax.spines["bottom"].set_alpha(0.35)
    ax.tick_params(axis="both", length=0, pad=4)


def relative_change(rows, metric_key):
    teacher = next((r for r in rows if "Teacher" in r["label"]), None)
    student = next((r for r in rows if "M3KD" in r["label"]), None)
    if not teacher or not student:
        return None
    baseline, candidate = teacher[metric_key], student[metric_key]
    if baseline in (None, 0) or candidate is None:
        return None
    return 100.0 * (candidate - baseline) / baseline


def change_description(change, lower_better):
    if change is None:
        return "Not available (one or both values are missing, or baseline is zero)."
    if abs(change) < 0.05:
        return "approximately 0.0% change"
    if lower_better:
        return f"{abs(change):.2f}% lower" if change < 0 else f"{change:.2f}% higher"
    return f"{change:.2f}% higher" if change > 0 else f"{abs(change):.2f}% lower"


def annotate_bars(ax, bars, values, formatter, offset=4):
    for bar, value in zip(bars, values):
        if value is None:
            ax.text(
                bar.get_x() + bar.get_width() / 2, 0.025, "N/A",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8,
            )
        else:
            ax.annotate(
                formatter(value),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, offset), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )


def plot_metric(ax, rows, metric, colors, standalone=False):
    key = metric["key"]
    values = [row[key] for row in rows]
    valid = [value for value in values if value is not None]
    style_axis(ax)
    #ax.set_title(metric["title"], loc="left", pad=8, fontweight="bold")
    ax.set_ylabel(metric["ylabel"], labelpad=4)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([row["label"] for row in rows], rotation=0)

    if not valid:
        ax.text(0.5, 0.5, "Metric unavailable", transform=ax.transAxes,
                ha="center", va="center")
        ax.set_ylim(0, 1)
        ax.grid(False)
        return

    bars = []
    for i, value in enumerate(values):
        height = value if value is not None else 0
        bars.append(ax.bar(
            i, height, width=0.58, color=colors[i], edgecolor="white",
            linewidth=0.7, zorder=3,
        )[0])

    max_value = max(valid)
    y_top = max_value * (1.32 if standalone else 1.38) if max_value > 0 else 1.0
    if key == "VRAM (GB)":
        ax.axhline(8.0, linestyle="--", linewidth=1.0,
                   label="8 GiB nominal capacity", zorder=2)
        ax.legend(loc="upper left", frameon=False, fontsize=7.5)
        y_top = max(y_top, 8.8)
    ax.set_ylim(0, y_top)
    annotate_bars(ax, bars, values, metric["format"])

    change = relative_change(rows, key)
    if change is not None:
        label = "M3KD: " + change_description(change, metric["lower_better"])
        ax.text(0.98, 0.97, label,
                        transform=ax.transAxes, ha="right", va="top", fontsize=7.2,
                        color="#384252",
                        bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                                  edgecolor=COLORS["grid"], linewidth=0.6))


def write_text_report(rows):
    lines = [
        "MLSys Distillation Efficiency Comparison",
        "=" * 42,
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "Target device: NVIDIA RTX 3070 (nominal 8 GiB VRAM, as configured in the figure)",
        "",
        "Input summary files:",
    ]
    for row in rows:
        lines.append(f"  {row['label']}: {row['path']}")
    lines.extend([
        "",
        "Metric values",
        "-------------",
        "VRAM conversion: gpu_memory_peak_mb_device_wide is divided by 1024; the result is GiB.",
        "Energy is the estimated GPU energy per example reported by the input summary.",
        "Throughput is generated tokens per second as reported by the input summary.",
        "",
    ])

    for metric in METRICS:
        key = metric["key"]
        lines.append(f"{metric['title']} ({key})")
        lines.append("~" * (len(metric["title"]) + len(key) + 3))
        for row in rows:
            value = row[key]
            value_text = metric["report_format"](value) if value is not None else "N/A"
            lines.append(f"  {row['label']}: {value_text}")
        change = relative_change(rows, key)
        lines.append(
            "  M3KD relative to teacher: "
            + change_description(change, metric["lower_better"])
        )
        if change is not None:
            lines.append(f"  Signed relative change: {change:+.6f}% (student - teacher) / teacher")
        lines.append("")

    lines.extend([
        "Source fields used",
        "------------------",
    ])
    source_fields = [
        ("Latency (s)", "per_example_end_to_end_latency_sec_mean"),
        ("VRAM (MiB, converted to GiB)", "telemetry.gpu_memory_peak_mb_device_wide"),
        ("Energy (J/ex)", "gpu_energy_j_per_example_estimated"),
        ("Throughput (tok/s)", "generated_tokens_per_sec"),
    ]
    for label, field in source_fields:
        lines.append(f"  {label}: {field}")
    lines.extend([
        "",
        "Interpretation notes",
        "--------------------",
        "Lower is better for latency, VRAM, and energy per example; higher is better for throughput.",
        "Relative change is calculated as 100 * (M3KD - teacher) / teacher.",
        "A negative relative change means a reduction in the metric; for throughput, a positive change is favorable.",
        "Energy estimates and all performance values depend on workload, context length, batch settings, and runtime configuration.",
        "Missing or invalid fields are reported as N/A and are not imputed.",
        "",
        "Generated figure files",
        "---------------------",
        f"  Combined PDF: {COMBINED_PDF}",
        f"  Combined PNG: {COMBINED_PNG}",
    ])
    for metric in METRICS:
        lines.append(f"  {metric['title']} PDF: {os.path.join(OUTPUT_DIR, metric['filename'] + '.pdf')}")
        lines.append(f"  {metric['title']} PNG: {os.path.join(OUTPUT_DIR, metric['filename'] + '.png')}")

    with open(TEXT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    rows = load_data(MODEL_FILES)
    if not rows:
        raise SystemExit(
            "No summaries were loaded. Run this script from the directory containing "
            "'res_new_FULL_Efficiency', or update MODEL_FILES to use the correct paths."
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = [cycle[i % len(cycle)] for i in range(len(rows))]

    # Individual, publication-ready PDF and PNG for each metric.
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(4.8, 3.5), dpi=300)
        fig.subplots_adjust(left=0.18, right=0.97, top=0.86, bottom=0.20)
        plot_metric(ax, rows, metric, colors, standalone=True)
        #fig.suptitle("Edge deployment efficiency on NVIDIA RTX 3070",
        #             y=0.98, fontsize=10.5, fontweight="bold")
        pdf_path = os.path.join(OUTPUT_DIR, metric["filename"] + ".pdf")
        png_path = os.path.join(OUTPUT_DIR, metric["filename"] + ".png")
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=600, bbox_inches="tight")
        plt.close(fig)

    # Combined 2x2 overview, retained for convenient comparison.
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.2), dpi=300)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.83, bottom=0.13,
                        hspace=0.62, wspace=0.32)
    fig.suptitle("Edge deployment efficiency on NVIDIA RTX 3070",
                 y=0.97, fontsize=11.5, fontweight="bold")
    fig.text(0.5, 0.91, "Qwen2-VL-7B teacher vs. Qwen2-VL-2B M3KD student",
             ha="center", va="center", fontsize=8.5)
    for ax, metric in zip(axes.flat, METRICS):
        plot_metric(ax, rows, metric, colors)
    fig.text(0.11, 0.035,
             "Energy is estimated from GPU telemetry; results depend on workload and runtime settings.",
             ha="left", va="center", fontsize=6.8)
    fig.savefig(COMBINED_PDF, bbox_inches="tight")
    fig.savefig(COMBINED_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)

    write_text_report(rows)
    print("Saved output files:")
    print(f"  Text report: {TEXT_REPORT}")
    print(f"  Combined PDF: {COMBINED_PDF}")
    print(f"  Combined PNG: {COMBINED_PNG}")
    for metric in METRICS:
        print(f"  {os.path.join(OUTPUT_DIR, metric['filename'] + '.pdf')}")
        print(f"  {os.path.join(OUTPUT_DIR, metric['filename'] + '.png')}")


if __name__ == "__main__":
    main()
