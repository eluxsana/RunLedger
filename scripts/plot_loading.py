# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.10,<4"]
# ///
"""Render measured medians and min/max ranges; never synthesize measurements."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="Output base path without extension")
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    methods = ["jsonl-materialized", "jsonl-streaming", "parquet-materialized", "parquet-streaming"]
    labels = [
        "JSONL\nmaterialized",
        "JSONL\nstreaming",
        "Parquet\nmaterialized",
        "Parquet\nstreaming",
    ]
    colors = ["#7b9199", "#295e85", "#b3c3c5", "#087b72"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": "#d6dedc",
            "text.color": "#193039",
            "axes.labelcolor": "#526971",
            "xtick.color": "#526971",
            "ytick.color": "#193039",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 6), layout="constrained")
    metrics = [
        ("peak_rss_bytes", 1 / 2**20, "Peak process memory", "MiB", ".1f"),
        ("first_batch_seconds", 1000, "First 256 records", "milliseconds", ".1f"),
        ("total_seconds", 1, "Full validated scan", "seconds", ".2f"),
    ]
    for ax, (metric, scale, title, unit, fmt) in zip(axes, metrics, strict=True):
        stats = [data["aggregates"][method][metric] for method in methods]
        values = [s["median"] * scale for s in stats]
        errors = [
            [(s["median"] - s["min"]) * scale for s in stats],
            [(s["max"] - s["median"]) * scale for s in stats],
        ]
        ax.barh(
            labels,
            values,
            color=colors,
            height=0.58,
            xerr=errors,
            error_kw={"ecolor": "#193039", "capsize": 3, "linewidth": 1},
        )
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontweight="bold", pad=18, fontsize=12)
        ax.set_xlabel(unit)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.18)
        ax.tick_params(axis="y", length=0)
        upper = max(s["max"] for s in stats) * scale
        ax.set_xlim(0, upper * 1.25)
        for index, (value, stat) in enumerate(zip(values, stats, strict=True)):
            ax.text(
                stat["max"] * scale + upper * 0.025,
                index,
                format(value, fmt),
                ha="left",
                va="center",
                fontsize=10,
            )
    fig.suptitle(
        f"Dataset loading: memory, startup latency and scan time.\n"
        f"{data['rows']:,} synthetic records · {data['repeats']} fresh-process trials per method",
        x=0.01,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.supxlabel(
        "Bars: medians. Whiskers: observed min–max. One warm-up per method; warm/uncontrolled OS cache.\n"
        "Identical ordered records, schema validation and digest work. One macOS arm64 machine; no GPU or training claim.",
        fontsize=9,
        color="#526971",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(args.output.with_suffix(f".{extension}"), dpi=170, facecolor="white")
    svg = args.output.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
