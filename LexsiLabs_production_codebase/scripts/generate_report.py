"""
Generates publication-ready 2-panel ablation figures and Markdown summaries
from the 4-quadrant kill-chain benchmark runs.

Panel A: Kill-Chain Reach & Transition Probabilities with error bars (P(Comp|Disc), P(Root|Comp), P(Impact|Root)).
Panel B: Economic & Latency Comparison (Average Tokens per Episode and Runtime Latency).
"""

import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def plot_ablation_chart(
    summaries: list = None,
    output_png: str = "results/reports/killchain_ablation_chart.png",
):
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    if summaries is None:
        json_path = "results/reports/ablation_summary.json"
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                summaries = json.load(f)
        else:
            print("⚠️ No ablation summary data found to plot.")
            return

    configs = [s["Configuration"].replace("1. ", "").replace("2. ", "").replace("3. ", "").replace("4. ", "") for s in summaries]
    n_configs = len(configs)

    # Extract raw or parsed values
    metrics_keys = ["p_comp", "p_root", "p_impact", "impact_rate"]
    metrics_labels = ["P(Comp|Disc)", "P(Root|Comp)", "P(Impact|Root)", "Impact Rate"]

    reach_means = []
    reach_stds = []
    for s in summaries:
        raw = s.get("_raw", {})
        means = [
            raw.get("p_comp", float(str(s.get("P(Comp|Disc)", "0")).split("±")[0].replace("%", ""))),
            raw.get("p_root", float(str(s.get("P(Root|Comp)", "0")).split("±")[0].replace("%", ""))),
            raw.get("p_impact", float(str(s.get("P(Impact|Root)", "0")).split("±")[0].replace("%", ""))),
            raw.get("impact_rate", float(str(s.get("Impact Rate", "0")).replace("%", ""))),
        ]
        stds = [
            raw.get("p_comp_std", 0.0),
            raw.get("p_root_std", 0.0),
            raw.get("p_impact_std", 0.0),
            0.0,
        ]
        reach_means.append(means)
        reach_stds.append(stds)

    tokens = [s.get("_raw", {}).get("avg_tokens", float(str(s.get("Avg Tokens", "0")).split("±")[0])) for s in summaries]
    latencies = [s.get("_raw", {}).get("avg_time", float(str(s.get("Avg Time (s)", "0")).split("±")[0].replace("s", ""))) for s in summaries]

    # Create publication figure
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

    # -------------------------------------------------------------
    # Panel A: Kill-Chain Reach Metrics with Error Bars
    # -------------------------------------------------------------
    x = np.arange(len(metrics_labels))
    width = 0.8 / max(n_configs, 1)
    colors = ["#c0392b", "#e67e22", "#2980b9", "#27ae60"]

    for idx, (cfg_name, means, errs) in enumerate(zip(configs, reach_means, reach_stds)):
        offset = (idx - (n_configs - 1) / 2) * width
        bars = ax1.bar(
            x + offset,
            means,
            width,
            yerr=errs,
            capsize=3,
            label=cfg_name,
            color=colors[idx % len(colors)],
            edgecolor="#2c3e50",
            linewidth=0.8,
            alpha=0.9,
        )
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax1.annotate(
                    f"{h:.0f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )

    ax1.set_title("Panel A: 4-Quadrant Kill-Chain Transition Reach", fontsize=11, fontweight="bold", pad=12)
    ax1.set_ylabel("Transition Probability (%)", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_labels, fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.legend(loc="upper left", frameon=True, fontsize=8)

    # -------------------------------------------------------------
    # Panel B: Economic & Latency Trade-Off
    # -------------------------------------------------------------
    x_cfg = np.arange(n_configs)
    ax2_twin = ax2.twinx()

    bar_tokens = ax2.bar(
        x_cfg - 0.2,
        tokens,
        width=0.35,
        color="#8e44ad",
        edgecolor="#2c3e50",
        linewidth=0.8,
        label="Avg Tokens / Ep",
    )
    bar_lat = ax2_twin.bar(
        x_cfg + 0.2,
        latencies,
        width=0.35,
        color="#16a085",
        edgecolor="#2c3e50",
        linewidth=0.8,
        label="Latency (s) / Ep",
    )

    ax2.set_title("Panel B: Compute & Financial Overhead Scaling", fontsize=11, fontweight="bold", pad=12)
    ax2.set_ylabel("Average Tokens per Episode (Log Scale)", fontsize=10, color="#8e44ad")
    ax2.set_yscale("log")
    ax2.set_ylim(1, max(max(tokens) * 3, 200000))
    ax2_twin.set_ylabel("Wall-Clock Latency (seconds)", fontsize=10, color="#16a085")
    ax2_twin.set_ylim(0, max(max(latencies) * 1.3, 10))

    ax2.set_xticks(x_cfg)
    ax2.set_xticklabels(configs, fontsize=8, rotation=12, ha="right", fontweight="bold")

    # Value labels for tokens and latency
    for bar in bar_tokens:
        h = bar.get_height()
        ax2.annotate(
            f"{int(h):,}" if h > 0 else "0",
            xy=(bar.get_x() + bar.get_width() / 2, max(h, 1.2)),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#8e44ad",
            fontweight="bold",
        )

    for bar in bar_lat:
        h = bar.get_height()
        ax2_twin.annotate(
            f"{h:.1f}s",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#16a085",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved publication-ready 2-panel chart to: {output_png}")


if __name__ == "__main__":
    plot_ablation_chart()
