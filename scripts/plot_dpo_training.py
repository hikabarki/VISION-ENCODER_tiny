"""
Plot DPO training curves from dpo_log.jsonl.

Usage:
    python3.10 scripts/plot_dpo_training.py
    python3.10 scripts/plot_dpo_training.py --log-path outputs/dpo/dpo_log.jsonl --output-dir outputs/dpo
"""

import argparse, json, os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--log-path",   default="outputs/dpo/dpo_log.jsonl")
parser.add_argument("--output-dir", default="outputs/dpo")
args = parser.parse_args()

if not os.path.exists(args.log_path):
    print(f"ERROR: {args.log_path} not found.")
    exit(1)

# ── load — keep only full-format entries with real loss ───────────────────────
records = []
with open(args.log_path) as f:
    for line in f:
        d = json.loads(line)
        if "timestamp" in d and d["loss"] > 0 and d["step"] > 0:
            records.append(d)

if not records:
    print("No valid log entries found.")
    exit(1)

steps      = [r["step"]                   for r in records]
loss       = [r["loss"]                   for r in records]
lr         = [r["learning_rate"]          for r in records]
acc        = [r["rewards/accuracies"]     for r in records]
margin     = [r["rewards/margins"]        for r in records]
r_chosen   = [r["rewards/chosen"]        for r in records]
r_rejected = [r["rewards/rejected"]      for r in records]
gpu_mem    = [r["gpu_mem_mb"]             for r in records]

# ── style ─────────────────────────────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
GRID    = "#30363d"
WHITE   = "#e6edf3"
MUTED   = "#8b949e"
TEAL    = "#00d4aa"
AMBER   = "#fbbf24"
BLUE    = "#4f8ef7"
PURPLE  = "#a78bfa"
RED     = "#f87171"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   MUTED,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        WHITE,
    "grid.color":        GRID,
    "grid.linewidth":    0.8,
})

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.patch.set_facecolor(BG)
fig.suptitle(
    "Keural VLM — DPO Training Curves (3,000 steps · RLHF-V · 5,733 pairs)",
    fontsize=15, fontweight="bold", color=WHITE, y=1.01
)


def style_ax(ax, title, ylabel):
    ax.set_title(title, fontsize=11, fontweight="bold", color=WHITE, pad=8)
    ax.set_xlabel("Step", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[:].set_visible(False)


def smooth(y, w=5):
    if len(y) < w:
        return y
    kernel = np.ones(w) / w
    padded = np.pad(y, (w//2, w//2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(y)]


# ── 1. Loss ───────────────────────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(steps, loss, color=TEAL, alpha=0.25, linewidth=1)
ax.plot(steps, smooth(loss, 10), color=TEAL, linewidth=2, label="loss (smoothed)")
ax.set_ylim(bottom=0)
style_ax(ax, "Training Loss", "Loss")
ax.legend(fontsize=9, frameon=False, labelcolor=WHITE)

# ── 2. Reward Accuracy ────────────────────────────────────────────────────────
ax = axes[0, 1]
ax.plot(steps, [a * 100 for a in acc], color=AMBER, alpha=0.25, linewidth=1)
ax.plot(steps, smooth([a * 100 for a in acc], 10), color=AMBER, linewidth=2)
ax.axhline(50, color=MUTED, linewidth=0.8, linestyle="--", label="random baseline")
ax.set_ylim(0, 100)
style_ax(ax, "Reward Accuracy (%)", "Accuracy (%)")
ax.legend(fontsize=9, frameon=False, labelcolor=WHITE)

# ── 3. Reward Margin ──────────────────────────────────────────────────────────
ax = axes[0, 2]
ax.plot(steps, margin, color=BLUE, alpha=0.25, linewidth=1)
ax.plot(steps, smooth(margin, 10), color=BLUE, linewidth=2)
ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
style_ax(ax, "Reward Margin (chosen − rejected)", "Margin")

# ── 4. Chosen vs Rejected Rewards ─────────────────────────────────────────────
ax = axes[1, 0]
ax.plot(steps, r_chosen,   color=TEAL,   alpha=0.3, linewidth=1)
ax.plot(steps, smooth(r_chosen, 10),   color=TEAL,   linewidth=2, label="chosen")
ax.plot(steps, r_rejected, color=RED,    alpha=0.3, linewidth=1)
ax.plot(steps, smooth(r_rejected, 10), color=RED,    linewidth=2, label="rejected")
style_ax(ax, "Chosen vs Rejected Rewards", "Reward")
ax.legend(fontsize=9, frameon=False, labelcolor=WHITE)

# ── 5. Learning Rate ──────────────────────────────────────────────────────────
ax = axes[1, 1]
ax.plot(steps, lr, color=PURPLE, linewidth=2)
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
style_ax(ax, "Learning Rate Schedule", "LR")

# ── 6. GPU Memory ─────────────────────────────────────────────────────────────
ax = axes[1, 2]
ax.fill_between(steps, [m / 1024 for m in gpu_mem], alpha=0.3, color=AMBER)
ax.plot(steps, [m / 1024 for m in gpu_mem], color=AMBER, linewidth=2)
style_ax(ax, "GPU Memory Usage", "VRAM (GiB)")

plt.tight_layout()
os.makedirs(args.output_dir, exist_ok=True)
out_path = os.path.join(args.output_dir, "dpo_training_curves.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out_path}")
plt.close()
