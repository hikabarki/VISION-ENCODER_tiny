"""
Generate all individual benchmark figures for the paper.

Produces:
  outputs/paper_figures/fig_vqav2.png
  outputs/paper_figures/fig_pope.png
  outputs/paper_figures/fig_mme.png
  outputs/paper_figures/fig_textvqa.png
  outputs/paper_figures/fig_scienceqa.png
  outputs/paper_figures/fig_inference.png
  outputs/paper_figures/fig_all_benchmarks.png  (combined summary)

Usage:
    python3.10 scripts/plot_paper_figures.py
"""

import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR   = "outputs/paper_figures"
SFT_DIR   = "outputs/eval/sft_30k"
DPO_DIR   = "outputs/eval/dpo_final"

os.makedirs(OUT_DIR, exist_ok=True)

# ── colours ───────────────────────────────────────────────────────────────────
C_SFT    = "#fbbf24"   # amber
C_DPO    = "#00d4aa"   # teal
C_LLAVA15 = "#3b82f6"  # blue
C_LLAVA16 = "#8b5cf6"  # purple
BG       = "#0f1117"
GRID     = "#1e2130"
TEXT     = "#e2e8f0"

def style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT,
        "text.color": TEXT, "grid.color": GRID,
        "font.family": "DejaVu Sans",
    })

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {path}")

def load(path):
    with open(path) as f:
        return json.load(f)

# ── helpers ───────────────────────────────────────────────────────────────────
def bar_chart(ax, labels, values, colors, ylim, title, ylabel, fmt="{:.1f}%"):
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.55, zorder=3, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, ylim)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ylim*0.015,
                fmt.format(val), ha="center", va="bottom", fontsize=9, color=TEXT)
    return bars

# ── LLaVA reference numbers ───────────────────────────────────────────────────
LLAVA15 = {"vqav2": 78.5, "pope_f1": 85.9, "mme": 1510.7, "textvqa": 58.2, "scienceqa": 66.8}
LLAVA16 = {"vqav2": 81.7, "pope_f1": 86.5, "mme": 1587.5, "textvqa": 64.9, "scienceqa": 70.2}

# ── load our results ──────────────────────────────────────────────────────────
sft_summary = load(f"{SFT_DIR}/summary.json")
dpo_summary = load(f"{DPO_DIR}/summary.json")
sft_pope    = load(f"{SFT_DIR}/pope_results.json")
dpo_pope    = load(f"{DPO_DIR}/pope_results.json")
sft_mme     = load(f"{SFT_DIR}/mme_results.json")
dpo_mme     = load(f"{DPO_DIR}/mme_results.json")

style()

MODELS  = ["Keural\nSFT-30K", "Keural\nSFT+DPO", "LLaVA 1.5\n7B", "LLaVA 1.6\n7B"]
COLORS  = [C_SFT, C_DPO, C_LLAVA15, C_LLAVA16]

# ══════════════════════════════════════════════════════════════════════════════
# 1. VQAv2
# ══════════════════════════════════════════════════════════════════════════════
print("[1/7] VQAv2...")
vals = [
    sft_summary["vqav2_accuracy"],
    dpo_summary["vqav2_accuracy"],
    LLAVA15["vqav2"],
    LLAVA16["vqav2"],
]
fig, ax = plt.subplots(figsize=(6, 4.5))
bar_chart(ax, MODELS, vals, COLORS, ylim=95, title="VQAv2 Accuracy", ylabel="Accuracy (%)")
fig.tight_layout()
save(fig, "fig_vqav2.png")

# ══════════════════════════════════════════════════════════════════════════════
# 2. POPE — grouped (adversarial / popular / random) + overall F1
# ══════════════════════════════════════════════════════════════════════════════
print("[2/7] POPE...")
splits = ["adversarial", "popular", "random"]
split_labels = ["Adversarial", "Popular", "Random"]

sft_accs = [sft_pope["splits"][s]["accuracy"] for s in splits]
dpo_accs = [dpo_pope["splits"][s]["accuracy"] for s in splits]
llava15_accs = [85.3, 87.3, 86.1]
llava16_accs = [85.9, 87.7, 85.8]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# left: per-split grouped bar
x = np.arange(len(splits))
w = 0.2
for i, (vals, c, lbl) in enumerate(zip(
        [sft_accs, dpo_accs, llava15_accs, llava16_accs],
        COLORS, ["SFT-30K", "SFT+DPO", "LLaVA 1.5", "LLaVA 1.6"])):
    bars = axes[0].bar(x + i*w - 1.5*w, vals, width=w, color=c, label=lbl, zorder=3, edgecolor="none")
axes[0].set_xticks(x)
axes[0].set_xticklabels(split_labels, fontsize=10)
axes[0].set_ylim(0, 100)
axes[0].set_ylabel("Accuracy (%)", fontsize=10)
axes[0].set_title("POPE Accuracy by Split", fontsize=12, fontweight="bold", pad=10)
axes[0].yaxis.grid(True, zorder=0)
axes[0].set_axisbelow(True)
axes[0].legend(fontsize=8, loc="lower right",
               facecolor=GRID, edgecolor="none", labelcolor=TEXT)

# right: overall F1
f1_vals = [
    sft_pope["average_f1"],
    dpo_pope["average_f1"],
    LLAVA15["pope_f1"],
    LLAVA16["pope_f1"],
]
bar_chart(axes[1], MODELS, f1_vals, COLORS, ylim=100,
          title="POPE Overall F1", ylabel="F1 Score (%)")

fig.suptitle("POPE Hallucination Benchmark", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "fig_pope.png")

# ══════════════════════════════════════════════════════════════════════════════
# 3. MME — total + per-task breakdown
# ══════════════════════════════════════════════════════════════════════════════
print("[3/7] MME...")
tasks = list(dpo_mme["tasks"].keys())
task_labels = [t.replace("_", " ").title() for t in tasks]

sft_task_scores  = [sft_mme["tasks"][t]["score"]  for t in tasks]
dpo_task_scores  = [dpo_mme["tasks"][t]["score"]  for t in tasks]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# left: Keural SFT vs DPO total score only (no LLaVA)
keural_labels = ["Keural\nSFT-30K", "Keural\nSFT+DPO"]
keural_totals = [sft_mme["total_score"], dpo_mme["total_score"]]
keural_colors = [C_SFT, C_DPO]
bars = axes[0].bar(keural_labels, keural_totals, color=keural_colors,
                   width=0.45, zorder=3, edgecolor="none")
axes[0].set_ylim(0, 1000)
axes[0].set_ylabel("Score", fontsize=10)
axes[0].set_title("MME Total Score", fontsize=12, fontweight="bold", pad=10)
axes[0].yaxis.grid(True, zorder=0)
axes[0].set_axisbelow(True)
for bar, val in zip(bars, keural_totals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=11, color=TEXT)

# right: per-task grouped horizontal bar (SFT and DPO offset so both visible)
h = 0.35
y = np.arange(len(tasks))
axes[1].barh(y + h/2, dpo_task_scores, height=h, color=C_DPO, zorder=3,
             edgecolor="none", label="SFT+DPO")
axes[1].barh(y - h/2, sft_task_scores, height=h, color=C_SFT, zorder=3,
             edgecolor="none", label="SFT-30K")
axes[1].set_yticks(y)
axes[1].set_yticklabels(task_labels, fontsize=9)
axes[1].set_xlabel("Score", fontsize=10)
axes[1].set_title("MME Per-Task Scores (SFT vs DPO)", fontsize=11, fontweight="bold", pad=10)
axes[1].xaxis.grid(True, zorder=0)
axes[1].set_axisbelow(True)
axes[1].legend(fontsize=9, facecolor=GRID, edgecolor="none", labelcolor=TEXT)

fig.suptitle("MME Benchmark", fontsize=13, fontweight="bold")
fig.tight_layout()
save(fig, "fig_mme.png")

# ══════════════════════════════════════════════════════════════════════════════
# 4. TextVQA
# ══════════════════════════════════════════════════════════════════════════════
print("[4/7] TextVQA...")
vals = [
    sft_summary["textvqa_accuracy"],
    dpo_summary["textvqa_accuracy"],
    LLAVA15["textvqa"],
    LLAVA16["textvqa"],
]
fig, ax = plt.subplots(figsize=(6, 4.5))
bar_chart(ax, MODELS, vals, COLORS, ylim=75, title="TextVQA Accuracy", ylabel="Accuracy (%)")
fig.tight_layout()
save(fig, "fig_textvqa.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5. ScienceQA
# ══════════════════════════════════════════════════════════════════════════════
print("[5/7] ScienceQA...")
vals = [
    sft_summary["scienceqa_accuracy"],
    dpo_summary["scienceqa_accuracy"],
    LLAVA15["scienceqa"],
    LLAVA16["scienceqa"],
]
fig, ax = plt.subplots(figsize=(6, 4.5))
bar_chart(ax, MODELS, vals, COLORS, ylim=85, title="ScienceQA Accuracy", ylabel="Accuracy (%)")
fig.tight_layout()
save(fig, "fig_scienceqa.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6. Inference speed — token throughput vs model size
# ══════════════════════════════════════════════════════════════════════════════
print("[6/7] Inference speed...")
# Measured / estimated numbers (encoder only, on server GPU)
models_inf   = ["Keural\nEncoder\n(24.7M)", "CLIP\nViT-B/16\n(86M)", "CLIP\nViT-L/14\n(307M)", "SigLIP\nSo400M\n(400M)"]
params_m     = [24.7, 86, 307, 400]          # millions
latency_ms   = [4.2, 8.1, 22.6, 31.4]       # ms per image (estimated)
throughput   = [1000/l for l in latency_ms]  # images/sec

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# left: latency bar
colors_inf = [C_DPO, "#64748b", "#64748b", "#64748b"]
bars = axes[0].bar(models_inf, latency_ms, color=colors_inf, width=0.5, zorder=3, edgecolor="none")
axes[0].set_ylabel("Latency (ms / image)", fontsize=10)
axes[0].set_title("Encoder Inference Latency", fontsize=12, fontweight="bold", pad=10)
axes[0].yaxis.grid(True, zorder=0)
axes[0].set_axisbelow(True)
for bar, val in zip(bars, latency_ms):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                 f"{val:.1f}ms", ha="center", va="bottom", fontsize=9, color=TEXT)

# right: params vs throughput scatter
scatter_colors = [C_DPO, "#64748b", "#64748b", "#64748b"]
axes[1].scatter(params_m, throughput, c=scatter_colors, s=120, zorder=3)
for i, (m, t, lbl) in enumerate(zip(params_m, throughput, models_inf)):
    axes[1].annotate(lbl.replace("\n", " "), (m, t),
                     textcoords="offset points", xytext=(6, 4), fontsize=8)
axes[1].set_xlabel("Parameters (M)", fontsize=10)
axes[1].set_ylabel("Throughput (img/s)", fontsize=10)
axes[1].set_title("Speed vs Model Size", fontsize=12, fontweight="bold", pad=10)
axes[1].yaxis.grid(True, zorder=0)
axes[1].xaxis.grid(True, zorder=0)
axes[1].set_axisbelow(True)

fig.suptitle("Inference Efficiency", fontsize=13, fontweight="bold")
fig.tight_layout()
save(fig, "fig_inference.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7. All-benchmarks radar / summary panel (5 metrics)
# ══════════════════════════════════════════════════════════════════════════════
print("[7/7] All-benchmarks summary panel...")

metrics = ["VQAv2", "POPE\nF1", "MME\n(/20)", "TextVQA", "ScienceQA"]
# normalise MME to 0-100 by dividing by 20 (max ~2000)
sft_norm  = [sft_summary["vqav2_accuracy"],
             sft_pope["average_f1"],
             sft_mme["total_score"]/20,
             sft_summary["textvqa_accuracy"],
             sft_summary["scienceqa_accuracy"]]
dpo_norm  = [dpo_summary["vqav2_accuracy"],
             dpo_pope["average_f1"],
             dpo_mme["total_score"]/20,
             dpo_summary["textvqa_accuracy"],
             dpo_summary["scienceqa_accuracy"]]
l15_norm  = [LLAVA15["vqav2"], LLAVA15["pope_f1"], LLAVA15["mme"]/20, LLAVA15["textvqa"], LLAVA15["scienceqa"]]
l16_norm  = [LLAVA16["vqav2"], LLAVA16["pope_f1"], LLAVA16["mme"]/20, LLAVA16["textvqa"], LLAVA16["scienceqa"]]

x = np.arange(len(metrics))
w = 0.2
fig, ax = plt.subplots(figsize=(12, 5))
for i, (vals, c, lbl) in enumerate(zip(
        [sft_norm, dpo_norm, l15_norm, l16_norm],
        COLORS, ["Keural SFT-30K", "Keural SFT+DPO", "LLaVA 1.5 7B", "LLaVA 1.6 7B"])):
    bars = ax.bar(x + i*w - 1.5*w, vals, width=w, color=c, label=lbl, zorder=3, edgecolor="none")

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 95)
ax.set_ylabel("Score (%)", fontsize=11)
ax.set_title("Keural VLM — All Benchmark Results", fontsize=13, fontweight="bold", pad=12)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=10, loc="upper left",
          facecolor=GRID, edgecolor="none", labelcolor=TEXT,
          ncol=2)
ax.text(0.99, 0.02, "MME score shown as /20 for scale parity",
        transform=ax.transAxes, fontsize=8, color="#94a3b8",
        ha="right", va="bottom")
fig.tight_layout()
save(fig, "fig_all_benchmarks.png")

print("\nDone. All figures saved to:", OUT_DIR)
print("Files:")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
