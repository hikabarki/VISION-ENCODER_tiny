"""
Plot benchmark results for Keural VLM vs baselines.
Reads from outputs/eval/summary.json and generates a bar chart.

Usage:
    python3.10 scripts/plot_benchmarks.py
    python3.10 scripts/plot_benchmarks.py --output-dir outputs/eval
"""

import argparse, json, os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default="outputs/eval")
args = parser.parse_args()

# ── load Keural results ───────────────────────────────────────────────────────
summary_path = os.path.join(args.output_dir, "summary.json")
if not os.path.exists(summary_path):
    print(f"ERROR: {summary_path} not found. Run eval_all.py first.")
    exit(1)

with open(summary_path) as f:
    keural = json.load(f)

# ── published baseline scores ─────────────────────────────────────────────────
# Source: LLaVA-1.5 paper, LLaVA-1.6 (LLaVA-NeXT) paper
BASELINES = {
    "LLaVA 1.5\n(307M enc)": {
        "vqav2":    78.5,
        "pope_acc": 85.9,
        "pope_f1":  85.9,
        "mme":      1510.7,
    },
    "LLaVA 1.6\n(307M enc)": {
        "vqav2":    81.8,
        "pope_acc": 86.5,
        "pope_f1":  86.5,
        "mme":      1519.3,
    },
}

keural_scores = {
    "vqav2":    keural.get("vqav2_accuracy"),
    "pope_acc": keural.get("pope_avg_accuracy"),
    "pope_f1":  keural.get("pope_avg_f1"),
    "mme":      keural.get("mme_total"),
}

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.patch.set_facecolor("#0d1117")

COLORS = {
    "Keural VLM\n(24.7M enc)":  "#00d4aa",
    "LLaVA 1.5\n(307M enc)":    "#4f8ef7",
    "LLaVA 1.6\n(307M enc)":    "#a78bfa",
}

models      = list(COLORS.keys())
model_names = ["Keural VLM\n(24.7M enc)"] + list(BASELINES.keys())

def bar_group(ax, metric_key, title, ylabel, max_val, models_data):
    vals   = [models_data[m] for m in model_names]
    colors = [COLORS[m] for m in model_names]
    x      = np.arange(len(model_names))
    bars   = ax.bar(x, vals, color=colors, width=0.5, edgecolor="none",
                    zorder=3, alpha=0.92)

    ax.set_facecolor("#161b22")
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel, color="#8b949e", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, color="white", fontsize=9)
    ax.set_ylim(0, max_val)
    ax.tick_params(axis="y", colors="#8b949e")
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color="#30363d", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_val * 0.01,
                f"{val:.1f}", ha="center", va="bottom",
                color="white", fontsize=10, fontweight="bold")

# VQAv2
vqav2_data = {
    "Keural VLM\n(24.7M enc)":  keural_scores["vqav2"] or 0,
    "LLaVA 1.5\n(307M enc)":    BASELINES["LLaVA 1.5\n(307M enc)"]["vqav2"],
    "LLaVA 1.6\n(307M enc)":    BASELINES["LLaVA 1.6\n(307M enc)"]["vqav2"],
}
bar_group(axes[0], "vqav2", "VQAv2 Accuracy (%)", "Accuracy (%)", 100, vqav2_data)

# POPE F1
pope_data = {
    "Keural VLM\n(24.7M enc)":  keural_scores["pope_f1"] or 0,
    "LLaVA 1.5\n(307M enc)":    BASELINES["LLaVA 1.5\n(307M enc)"]["pope_f1"],
    "LLaVA 1.6\n(307M enc)":    BASELINES["LLaVA 1.6\n(307M enc)"]["pope_f1"],
}
bar_group(axes[1], "pope_f1", "POPE F1 Score (%)", "F1 (%)", 100, pope_data)

# MME
mme_data = {
    "Keural VLM\n(24.7M enc)":  keural_scores["mme"] or 0,
    "LLaVA 1.5\n(307M enc)":    BASELINES["LLaVA 1.5\n(307M enc)"]["mme"],
    "LLaVA 1.6\n(307M enc)":    BASELINES["LLaVA 1.6\n(307M enc)"]["mme"],
}
bar_group(axes[2], "mme", "MME Total Score", "Score", 2000, mme_data)

# ── legend + title ────────────────────────────────────────────────────────────
patches = [mpatches.Patch(color=COLORS[m], label=m.replace("\n", " "))
           for m in model_names]
fig.legend(handles=patches, loc="upper center", ncol=3, frameon=False,
           labelcolor="white", fontsize=10, bbox_to_anchor=(0.5, 1.02))

fig.suptitle(
    "Keural VLM — Benchmark Results\n"
    "24.7M vision encoder trained from scratch on CC3M + CC12M (~15M pairs)",
    color="white", fontsize=14, fontweight="bold", y=1.08
)

plt.tight_layout()
out_path = os.path.join(args.output_dir, "benchmark_results.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
plt.close()
