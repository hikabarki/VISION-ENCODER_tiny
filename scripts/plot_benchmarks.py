"""
Plot benchmark results for Keural VLM vs baselines.

Usage:
    python3.10 scripts/plot_benchmarks.py
    python3.10 scripts/plot_benchmarks.py \
        --sft-dir outputs/eval/sft_30k \
        --dpo-dir outputs/eval/dpo_final \
        --output-dir outputs/eval
"""

import argparse, json, os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--sft-dir",    default="outputs/eval/sft_30k")
parser.add_argument("--dpo-dir",    default="outputs/eval/dpo_final")
parser.add_argument("--output-dir", default="outputs/eval")
args = parser.parse_args()


def load_scores(path):
    summary = os.path.join(path, "summary.json")
    if not os.path.exists(summary):
        return None
    with open(summary) as f:
        d = json.load(f)
    return {
        "vqav2":     d.get("vqav2_accuracy")     or 0,
        "pope_f1":   d.get("pope_avg_f1")        or 0,
        "mme":       d.get("mme_total")          or 0,
        "textvqa":   d.get("textvqa_accuracy")   or 0,
        "scienceqa": d.get("scienceqa_accuracy") or 0,
    }


sft_scores = load_scores(args.sft_dir)
dpo_scores = load_scores(args.dpo_dir)

if sft_scores is None and dpo_scores is None:
    print("ERROR: no summary.json found in either --sft-dir or --dpo-dir")
    exit(1)

# ── published baseline scores ─────────────────────────────────────────────────
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

# build model list dynamically
model_names = []
scores_map  = {}

if sft_scores:
    model_names.append("Keural SFT\n(24.7M enc)")
    scores_map["Keural SFT\n(24.7M enc)"] = sft_scores
if dpo_scores:
    model_names.append("Keural DPO\n(24.7M enc)")
    scores_map["Keural DPO\n(24.7M enc)"] = dpo_scores

model_names += list(BASELINES.keys())
for k, v in BASELINES.items():
    scores_map[k] = v

COLORS = {
    "Keural SFT\n(24.7M enc)":  "#fbbf24",   # amber
    "Keural DPO\n(24.7M enc)":  "#00d4aa",   # teal
    "LLaVA 1.5\n(307M enc)":    "#4f8ef7",   # blue
    "LLaVA 1.6\n(307M enc)":    "#a78bfa",   # purple
}

# ── layout ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 11))
fig.patch.set_facecolor("#0d1117")

ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)
ax4 = fig.add_subplot(2, 3, (4, 5))
ax5 = fig.add_subplot(2, 3, 6)

ax4.set_position([0.08,  0.06, 0.36, 0.36])
ax5.set_position([0.56,  0.06, 0.36, 0.36])


def bar_group(ax, metric_key, title, ylabel, max_val):
    vals   = [scores_map[m][metric_key] for m in model_names]
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


bar_group(ax1, "vqav2",     "VQAv2 Accuracy (%)",                   "Accuracy (%)", 100)
bar_group(ax2, "pope_f1",   "POPE F1 Score (%)",                    "F1 (%)",       100)
bar_group(ax3, "mme",       "MME Total Score",                      "Score",        2000)
bar_group(ax4, "textvqa",   "TextVQA Accuracy (%)",                 "Accuracy (%)", 100)
bar_group(ax5, "scienceqa", "ScienceQA Accuracy (%) — image subset","Accuracy (%)", 100)

# ── legend + title ────────────────────────────────────────────────────────────
patches = [mpatches.Patch(color=COLORS[m], label=m.replace("\n", " "))
           for m in model_names]
fig.legend(handles=patches, loc="upper center", ncol=len(model_names),
           frameon=False, labelcolor="white", fontsize=10, bbox_to_anchor=(0.5, 1.01))

fig.suptitle(
    "Keural VLM — Benchmark Results\n"
    "24.7M vision encoder trained from scratch on CC3M + CC12M (~15M pairs)",
    color="white", fontsize=14, fontweight="bold", y=1.05
)

os.makedirs(args.output_dir, exist_ok=True)
out_path = os.path.join(args.output_dir, "benchmark_results.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
plt.close()
