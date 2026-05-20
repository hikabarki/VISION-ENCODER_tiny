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
# Source: LLaVA-1.5 paper, LLaVA-1.6 (LLaVA-NeXT) blog/paper
BASELINES = {
    "LLaVA 1.5\n(307M enc)": {
        "vqav2":      78.5,
        "pope_f1":    85.9,
        "mme":        1510.7,
        "textvqa":    58.2,
        "scienceqa":  66.8,
    },
    "LLaVA 1.6\n(307M enc)": {
        "vqav2":      81.8,
        "pope_f1":    86.5,
        "mme":        1519.3,
        "textvqa":    64.9,
        "scienceqa":  70.6,
    },
}

keural_scores = {
    "vqav2":     keural.get("vqav2_accuracy")    or 0,
    "pope_f1":   keural.get("pope_avg_f1")       or 0,
    "mme":       keural.get("mme_total")         or 0,
    "textvqa":   keural.get("textvqa_accuracy")  or 0,
    "scienceqa": keural.get("scienceqa_accuracy") or 0,
}

model_names = ["Keural VLM\n(24.7M enc)"] + list(BASELINES.keys())

COLORS = {
    "Keural VLM\n(24.7M enc)":  "#00d4aa",
    "LLaVA 1.5\n(307M enc)":    "#4f8ef7",
    "LLaVA 1.6\n(307M enc)":    "#a78bfa",
}

# ── 2-row layout: [VQAv2 | POPE F1 | MME] / [TextVQA | ScienceQA] ────────────
fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor("#0d1117")

# top row: 3 benchmarks
ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)
# bottom row: 2 benchmarks centred
ax4 = fig.add_subplot(2, 3, (4, 5))   # spans cols 1-2
ax5 = fig.add_subplot(2, 3, 6)        # col 3 (right-aligned)

# shift bottom-row axes to centre the two charts
ax4.set_position([0.08,  0.06, 0.36, 0.36])
ax5.set_position([0.56,  0.06, 0.36, 0.36])


def bar_group(ax, keural_val, baseline_key, title, ylabel, max_val):
    vals   = [keural_val] + [BASELINES[m][baseline_key] for m in list(BASELINES.keys())]
    colors = [COLORS[m] for m in model_names]
    x      = np.arange(len(model_names))
    bars   = ax.bar(x, vals, color=colors, width=0.5, edgecolor="none", zorder=3, alpha=0.92)

    ax.set_facecolor("#161b22")
    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, color="#8b949e", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, color="white", fontsize=8.5)
    ax.set_ylim(0, max_val)
    ax.tick_params(axis="y", colors="#8b949e")
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color="#30363d", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_val * 0.01,
                f"{val:.1f}", ha="center", va="bottom",
                color="white", fontsize=9.5, fontweight="bold")


bar_group(ax1, keural_scores["vqav2"],     "vqav2",     "VQAv2 Accuracy (%)",  "Accuracy (%)", 100)
bar_group(ax2, keural_scores["pope_f1"],   "pope_f1",   "POPE F1 Score (%)",   "F1 (%)",       100)
bar_group(ax3, keural_scores["mme"],       "mme",       "MME Total Score",     "Score",        2000)
bar_group(ax4, keural_scores["textvqa"],   "textvqa",   "TextVQA Accuracy (%)", "Accuracy (%)", 100)
bar_group(ax5, keural_scores["scienceqa"], "scienceqa", "ScienceQA Accuracy (%) — image subset", "Accuracy (%)", 100)

# ── legend + title ────────────────────────────────────────────────────────────
patches = [mpatches.Patch(color=COLORS[m], label=m.replace("\n", " "))
           for m in model_names]
fig.legend(handles=patches, loc="upper center", ncol=3, frameon=False,
           labelcolor="white", fontsize=10, bbox_to_anchor=(0.5, 1.01))

fig.suptitle(
    "Keural VLM — Benchmark Results\n"
    "24.7M vision encoder trained from scratch on CC3M + CC12M (~15M pairs)",
    color="white", fontsize=14, fontweight="bold", y=1.05
)

out_path = os.path.join(args.output_dir, "benchmark_results.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
plt.close()
