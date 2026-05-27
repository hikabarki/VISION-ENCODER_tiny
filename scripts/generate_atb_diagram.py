"""
Generate ATB Tokenizer detail diagram.
Output: poc/outputs/diagrams/atb_tokenizer_detail.png

Shows the saliency-guided token allocation flow:
  Conv Stem output → Saliency Predictor → Budget Split → Top-K per level
  → Token Selection → Output (tokens, level_ids, attention_mask)
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rbox(ax, x, y, w, h, label, sublabel=None,
         fc="#1D4ED8", tc="white", fs=8.5, sfs=7, r=0.025):
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad={r}",
        facecolor=fc, edgecolor="white", linewidth=1.1, zorder=3,
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(x, y + h*0.14, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=4)
        ax.text(x, y - h*0.25, sublabel, ha="center", va="center",
                fontsize=sfs, color=tc, alpha=0.8, zorder=4)
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=4)


def arr(ax, x0, y0, x1, y1, lbl=None, c="#94A3B8", lw=1.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                                mutation_scale=12), zorder=2)
    if lbl:
        mx, my = (x0+x1)/2, (y0+y1)/2
        dx, dy = x1-x0, y1-y0
        # offset label perpendicular to arrow
        norm = (dx**2+dy**2)**0.5 or 1
        ox, oy = -dy/norm*0.12, dx/norm*0.12
        ax.text(mx+ox, my+oy, lbl, ha="center", va="center",
                fontsize=6.5, color="#94A3B8", zorder=5)


def make_atb_diagram():
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    # Title
    ax.text(7, 6.15, "Adaptive Token Budgeting (ATB) Tokenizer — Saliency-Guided Token Allocation",
            ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    ax.text(7, 5.82,
            "Allocates token budget across three hierarchy levels: Global (5%) · Region (25%) · Detail (70%)",
            ha="center", va="center", fontsize=8, color="#94A3B8")

    # ── Stage 1: Input ──────────────────────────────────────────────
    rbox(ax, 0.85, 3.8, 1.3, 0.6,
         "Conv Stem\nOutput", "(B, 512, h, w)", fc="#1E3A5F")
    arr(ax, 1.5, 3.8, 2.15, 3.8, "(B,512,h,w)")

    # ── Stage 2: Saliency predictor ─────────────────────────────────
    rbox(ax, 2.85, 3.8, 1.2, 0.6,
         "Saliency\nPredictor", "1×1 conv + σ", fc="#1D4ED8")
    arr(ax, 3.45, 3.8, 4.05, 3.8, "(B,1,h,w)")

    # Saliency map mini-vis (fake heatmap grid)
    sal_ax = ax.inset_axes([0.293, 0.47, 0.065, 0.16])  # [left,bottom,w,h] in axes fraction
    fake_sal = np.random.rand(8, 8)
    fake_sal[3:5, 3:6] = 0.9   # hot center
    fake_sal[2:6, 2:7] = fake_sal[2:6, 2:7] * 0.6 + 0.35
    sal_ax.imshow(fake_sal, cmap="hot", vmin=0, vmax=1, aspect="auto")
    sal_ax.axis("off")
    ax.text(2.85, 2.95, "sal_map", ha="center", va="top",
            fontsize=6, color="#94A3B8")

    # ── Stage 3: Flatten + rank ──────────────────────────────────────
    rbox(ax, 4.75, 3.8, 1.2, 0.6,
         "Flatten &\nRank Patches", "H×W → N_patches\nby saliency score", fc="#374151")
    arr(ax, 5.35, 3.8, 6.0, 3.8, "ranked\npatches")

    # ── Stage 4: Budget split ────────────────────────────────────────
    rbox(ax, 6.75, 3.8, 1.3, 0.6,
         "Budget\nAllocator", "K_g+K_r+K_d=N", fc="#7C3AED")

    # Budget formula box
    ax.text(6.75, 2.9,
            "K_g = max(floor_g,  5% × N)\n"
            "K_r = max(floor_r, 25% × N)\n"
            "K_d = N − K_g − K_r  (70%)",
            ha="center", va="top", fontsize=7, color="#C4B5FD",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="#1E1B4B", ec="#7C3AED", lw=0.9))

    # Split arrows down to three level boxes
    arr(ax, 6.75, 3.5, 6.75, 3.25)
    arr(ax, 6.75, 3.25, 5.4, 2.4, c="#6EE7B7")
    arr(ax, 6.75, 3.25, 6.75, 2.4, c="#FCD34D")
    arr(ax, 6.75, 3.25, 8.1, 2.4, c="#F87171")

    # ── Level token boxes ────────────────────────────────────────────
    Y_LVL = 2.0
    # Global
    rbox(ax, 5.4, Y_LVL, 1.4, 0.65,
         "Global Tokens", f"K_g ≈ 26\nlevel_id = 0", fc="#065F46", sfs=6.8)
    ax.text(5.4, Y_LVL + 0.5, "GLOBAL", ha="center", va="bottom",
            fontsize=7, color="#6EE7B7", fontweight="bold")

    # Region
    rbox(ax, 6.75, Y_LVL, 1.4, 0.65,
         "Region Tokens", f"K_r ≈ 128\nlevel_id = 1", fc="#78350F", sfs=6.8)
    ax.text(6.75, Y_LVL + 0.5, "REGION", ha="center", va="bottom",
            fontsize=7, color="#FCD34D", fontweight="bold")

    # Detail
    rbox(ax, 8.1, Y_LVL, 1.4, 0.65,
         "Detail Tokens", f"K_d ≈ 358\nlevel_id = 2", fc="#7F1D1D", sfs=6.8)
    ax.text(8.1, Y_LVL + 0.5, "DETAIL", ha="center", va="bottom",
            fontsize=7, color="#F87171", fontweight="bold")

    # ── Merge ────────────────────────────────────────────────────────
    arr(ax, 5.4, Y_LVL - 0.33, 5.4, 1.0,  c="#6EE7B7")
    arr(ax, 6.75, Y_LVL - 0.33, 6.75, 1.0, c="#FCD34D")
    arr(ax, 8.1, Y_LVL - 0.33, 8.1, 1.0,  c="#F87171")

    # concat node
    rbox(ax, 6.75, 0.72, 1.0, 0.42,
         "Concat", "along token dim", fc="#0F766E", fs=8, sfs=6.5)
    arr(ax, 7.25, 0.72, 8.0, 0.72, "→")

    # ── Output struct ────────────────────────────────────────────────
    rbox(ax, 9.2, 0.72, 2.2, 0.65,
         "KeuralEncoderOutput",
         "tokens (B,N,D) · level_ids · attention_mask\npooled (B,D) · spatial_metadata",
         fc="#1E3A5F", fs=7.5, sfs=6.3)

    # ── 2D RoPE annotation ───────────────────────────────────────────
    arr(ax, 8.15, 3.8, 9.6, 3.8)
    rbox(ax, 10.35, 3.8, 1.4, 0.6,
         "2D RoPE\nPosition Enc.", "row, col coords\nper token", fc="#1D4ED8")
    ax.annotate("", xy=(10.35, 1.2), xytext=(10.35, 3.5),
                arrowprops=dict(arrowstyle="-|>", color="#60A5FA", lw=1.2,
                                linestyle="dashed", mutation_scale=10), zorder=2)
    ax.text(10.7, 2.35, "position\nencoding", ha="left", va="center",
            fontsize=6.5, color="#60A5FA")

    # ── Spatial Transformer ──────────────────────────────────────────
    arr(ax, 9.2, 0.38, 11.0, 0.38)
    rbox(ax, 11.7, 0.38, 1.4, 0.55,
         "Spatial\nEncoder", "24-layer ViT\nembed_dim=384", fc="#1D4ED8")
    arr(ax, 12.4, 0.38, 13.15, 0.38, "(B,N,384)")
    rbox(ax, 13.6, 0.38, 0.7, 0.42,
         "Output", "(B,N,384)", fc="#065F46", fs=7.5)

    # ── token budget label ───────────────────────────────────────────
    ax.text(6.75, 5.35,
            "Token Budget N = 512 (default) · auto-scales with resolution · min per level enforced by budget_floors",
            ha="center", va="center", fontsize=7.5, color="#CBD5E1",
            bbox=dict(boxstyle="round,pad=0.3", fc="#1E293B", ec="#334155", lw=0.8))

    # ── Legend ───────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(fc="#1D4ED8", label="Vision Encoder components"),
        mpatches.Patch(fc="#7C3AED", label="Budget Allocator"),
        mpatches.Patch(fc="#065F46", label="Global tokens (5%)"),
        mpatches.Patch(fc="#78350F", label="Region tokens (25%)"),
        mpatches.Patch(fc="#7F1D1D", label="Detail tokens (70%)"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=7,
              facecolor="#1E293B", edgecolor="#334155", labelcolor="white",
              ncol=5, bbox_to_anchor=(0.0, 0.0))

    plt.tight_layout(pad=0.3)
    out = OUT_DIR / "atb_tokenizer_detail.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="#0F172A")
    plt.close()
    print(f"[atb] Saved: {out}")


if __name__ == "__main__":
    make_atb_diagram()
