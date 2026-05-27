"""
Generate Phase 1 and Phase 2B data flow diagrams for the technical report.
Output: poc/outputs/diagrams/phase1_dataflow.png
        poc/outputs/diagrams/phase2b_dataflow.png
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rounded_box(ax, x, y, w, h, label, sublabel=None,
                fc="#2563EB", tc="white", fs=9, sub_fs=7.5, radius=0.03):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=fc, edgecolor="white", linewidth=1.2, zorder=3,
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(x, y + h * 0.12, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=4)
        ax.text(x, y - h * 0.22, sublabel, ha="center", va="center",
                fontsize=sub_fs, color=tc, alpha=0.85, zorder=4)
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=4)


def arrow(ax, x0, y0, x1, y1, label=None, color="#94A3B8"):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=1.5,
            mutation_scale=14,
        ),
        zorder=2,
    )
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2 + 0.03
        ax.text(mx, my, label, ha="center", va="bottom",
                fontsize=7, color="#64748B", zorder=5)


def branching_arrow(ax, x0, y0, targets, color="#94A3B8"):
    for (x1, y1) in targets:
        arrow(ax, x0, y0, x1, y1, color=color)


# ---------------------------------------------------------------------------
# Phase 1 — Contrastive Pretraining
# ---------------------------------------------------------------------------

def make_phase1():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    # Title
    ax.text(6, 4.65, "Phase 1 — Contrastive Pretraining Data Flow",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="white")
    ax.text(6, 4.35, "Image + Caption Pairs  →  SigLIP Contrastive Loss  (CC3M + CC12M, ~15.3M pairs)",
            ha="center", va="center", fontsize=8.5, color="#94A3B8")

    # --- Image branch ---
    Y_IMG = 2.8

    rounded_box(ax, 0.9, Y_IMG, 1.4, 0.55,
                "Image Input", "(B, 3, H, W)", fc="#1E3A5F")
    arrow(ax, 1.6, Y_IMG, 2.2, Y_IMG, "(B,3,H,W)")

    rounded_box(ax, 2.85, Y_IMG, 1.1, 0.55,
                "Conv Stem", "4-layer CNN", fc="#1D4ED8")
    arrow(ax, 3.4, Y_IMG, 3.9, Y_IMG, "(B,512,h,w)")

    rounded_box(ax, 4.55, Y_IMG, 1.1, 0.55,
                "Saliency\nPredictor", "1×1 conv", fc="#1D4ED8")
    arrow(ax, 5.1, Y_IMG, 5.6, Y_IMG, "sal_map")

    rounded_box(ax, 6.2, Y_IMG, 1.1, 0.55,
                "ATB\nTokenizer", "budget=512", fc="#7C3AED")
    arrow(ax, 6.75, Y_IMG, 7.3, Y_IMG, "(B,N,768)")

    rounded_box(ax, 7.95, Y_IMG, 1.1, 0.55,
                "Spatial\nEncoder", "24L ViT", fc="#1D4ED8")
    arrow(ax, 8.5, Y_IMG, 9.1, Y_IMG, "(B,N,768)")

    rounded_box(ax, 9.7, Y_IMG, 1.0, 0.55,
                "Pool\nToken", "[CLS]", fc="#0F766E")
    arrow(ax, 10.2, Y_IMG, 10.8, Y_IMG, "(B,768)")

    rounded_box(ax, 11.2, Y_IMG, 0.95, 0.55,
                "Image\nEmbed", "ẑᵢ", fc="#065F46")

    # --- Text branch ---
    Y_TXT = 1.5

    rounded_box(ax, 0.9, Y_TXT, 1.4, 0.55,
                "Caption Input", "list[str]", fc="#1E3A5F")
    arrow(ax, 1.6, Y_TXT, 2.3, Y_TXT, "tokenize")

    rounded_box(ax, 3.0, Y_TXT, 1.2, 0.55,
                "CLIP Text\nEncoder", "ViT-L/14", fc="#92400E")
    arrow(ax, 3.6, Y_TXT, 4.1, Y_TXT, "(B,768)")

    rounded_box(ax, 4.7, Y_TXT, 1.1, 0.55,
                "Text\nProjector", "Linear", fc="#78350F")
    arrow(ax, 5.25, Y_TXT, 10.8, Y_TXT)

    rounded_box(ax, 11.2, Y_TXT, 0.95, 0.55,
                "Text\nEmbed", "ẑₜ", fc="#78350F")

    # --- Loss ---
    Y_LOSS = 0.55

    # merge arrows from image embed and text embed down to loss
    arrow(ax, 11.2, Y_IMG - 0.28, 11.2, Y_LOSS + 0.45)
    arrow(ax, 11.2, Y_TXT - 0.28, 11.2, Y_LOSS + 0.45)

    # also saliency variance reg arrow
    arrow(ax, 4.55, Y_IMG - 0.28, 4.55, Y_LOSS + 0.45, "var_reg")

    rounded_box(ax, 11.2, Y_LOSS, 1.4, 0.55,
                "SigLIP Loss", "ℒ = ℒ_sig + ℒ_hct + λ·ℒ_var",
                fc="#991B1B", fs=8, sub_fs=6.5)

    # Frozen badge on text encoder
    ax.text(3.0, Y_TXT + 0.42, "FROZEN", ha="center", va="center",
            fontsize=6.5, color="#FCD34D", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="#92400E", ec="#FCD34D", lw=0.8))

    # Legend
    legend_items = [
        mpatches.Patch(fc="#1D4ED8", label="Vision Encoder"),
        mpatches.Patch(fc="#7C3AED", label="ATB Tokenizer"),
        mpatches.Patch(fc="#0F766E", label="Pool / Embed"),
        mpatches.Patch(fc="#92400E", label="Text Encoder (frozen)"),
        mpatches.Patch(fc="#991B1B", label="Loss"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=7,
              facecolor="#1E293B", edgecolor="#334155", labelcolor="white",
              ncol=5, bbox_to_anchor=(0.0, 0.0))

    plt.tight_layout(pad=0.3)
    out = OUT_DIR / "phase1_dataflow.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="#0F172A")
    plt.close()
    print(f"[diagrams] Saved: {out}")


# ---------------------------------------------------------------------------
# Phase 2B — Instruction Fine-Tuning
# ---------------------------------------------------------------------------

def make_phase2b():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    ax.text(6.5, 4.65, "Phase 2B — Instruction Fine-Tuning Data Flow",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="white")
    ax.text(6.5, 4.35,
            "Image + Conversation  →  Causal LM Loss  (LLaVA-Instruct-150K + COCO-train2017)",
            ha="center", va="center", fontsize=8.5, color="#94A3B8")

    Y = 2.9   # main horizontal lane

    # Vision pipeline (frozen)
    rounded_box(ax, 0.85, Y, 1.3, 0.55,
                "Image Input", "(B,3,H,W)", fc="#1E3A5F")
    arrow(ax, 1.5, Y, 2.05, Y, "(B,3,H,W)")

    rounded_box(ax, 2.65, Y, 1.1, 0.55,
                "Conv Stem\n+ ATB", "FROZEN", fc="#374151")
    arrow(ax, 3.2, Y, 3.75, Y, "(B,N,768)")

    rounded_box(ax, 4.35, Y, 1.1, 0.55,
                "Spatial\nEncoder", "FROZEN", fc="#374151")
    arrow(ax, 4.9, Y, 5.45, Y, "(B,N,768)")

    # Projector (trained)
    rounded_box(ax, 6.05, Y, 1.2, 0.55,
                "Level-Aware\nProjector", "384→2048→4096",
                fc="#7C3AED")
    arrow(ax, 6.65, Y, 7.25, Y, "(B,N,4096)")

    # Text tokenizer
    Y_TXT = 1.6
    rounded_box(ax, 0.85, Y_TXT, 1.3, 0.55,
                "Conversation", "Instruction+\nResponse", fc="#1E3A5F")
    arrow(ax, 1.5, Y_TXT, 2.2, Y_TXT, "tokenize")

    rounded_box(ax, 2.9, Y_TXT, 1.2, 0.55,
                "LLM\nTokenizer", "Mistral vocab\n32000 tokens", fc="#92400E")
    arrow(ax, 3.5, Y_TXT, 4.1, Y_TXT, "(B,L)")

    rounded_box(ax, 4.7, Y_TXT, 1.1, 0.55,
                "Token\nEmbedding", "(B,L,4096)", fc="#78350F")

    # Merge arrow up to LLM
    arrow(ax, 6.05, Y - 0.28, 6.05, 2.25, color="#A78BFA")
    arrow(ax, 4.7, Y_TXT + 0.28, 7.85, 2.55)
    ax.text(6.3, 2.38, "concat visual\n+ text tokens", ha="left", va="center",
            fontsize=7, color="#A78BFA")

    # LLM
    rounded_box(ax, 8.5, Y, 1.6, 0.7,
                "Mistral-7B\nInstruct-v0.3", "QLoRA r=16\nq,k,v,o trained",
                fc="#065F46", fs=8.5, sub_fs=7)
    arrow(ax, 9.3, Y, 10.0, Y, "(B,L,4096)")

    # LM Head
    rounded_box(ax, 10.65, Y, 1.1, 0.55,
                "LM Head", "→ logits", fc="#0F766E")
    arrow(ax, 11.2, Y, 11.9, Y, "(B,L,32000)")

    # Loss
    rounded_box(ax, 12.35, Y, 1.1, 0.55,
                "Causal LM\nLoss", "ℒ_clm", fc="#991B1B")

    # Frozen badges
    for (bx, by) in [(2.65, Y), (4.35, Y)]:
        ax.text(bx, by + 0.38, "FROZEN", ha="center", va="center",
                fontsize=6.5, color="#9CA3AF", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc="#374151", ec="#6B7280", lw=0.7))

    # LoRA badge on LLM
    ax.text(8.5, Y + 0.55, "LoRA ADAPTERS", ha="center", va="center",
            fontsize=6.5, color="#6EE7B7", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="#064E3B", ec="#6EE7B7", lw=0.8))

    # Legend
    legend_items = [
        mpatches.Patch(fc="#374151", label="Frozen (Phase 1 ckpt)"),
        mpatches.Patch(fc="#7C3AED", label="Projector (trained)"),
        mpatches.Patch(fc="#065F46", label="Mistral-7B + LoRA"),
        mpatches.Patch(fc="#92400E", label="Text Tokenizer"),
        mpatches.Patch(fc="#991B1B", label="Loss"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=7,
              facecolor="#1E293B", edgecolor="#334155", labelcolor="white",
              ncol=5, bbox_to_anchor=(0.0, 0.0))

    plt.tight_layout(pad=0.3)
    out = OUT_DIR / "phase2b_dataflow.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="#0F172A")
    plt.close()
    print(f"[diagrams] Saved: {out}")


if __name__ == "__main__":
    make_phase1()
    make_phase2b()
    print("[diagrams] Done.")
