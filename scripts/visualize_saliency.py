"""
Saliency map and token placement visualization.

For each input image, produces a 3-panel figure:
  [Original image] | [Saliency heatmap] | [Token placement overlay]

The token overlay shows where the model places Global (blue), Region (orange),
and Detail (red) tokens — directly validating the ATB paper claim.

Usage:
    python3.10 scripts/visualize_saliency.py --checkpoint outputs/run/checkpoint-28000
    python3.10 scripts/visualize_saliency.py --checkpoint outputs/run/checkpoint-28000 --image-dir test_images
    python3.10 scripts/visualize_saliency.py --checkpoint outputs/run/checkpoint-28000 --image path/to/image.jpg
"""
import sys, argparse, os, glob
sys.path.insert(0, "/home/mkd-s002/vision-encoder/poc")

import torch
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

import alignment.hf_wrapper
from transformers import AutoModel

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="outputs/run/checkpoint-28000")
parser.add_argument("--image-dir", default="test_images", help="Directory of images")
parser.add_argument("--image", default=None, help="Single image path (overrides --image-dir)")
parser.add_argument("--budget", type=int, default=256, help="Vision token budget")
parser.add_argument("--out-dir", default="outputs/saliency_viz", help="Output directory for figures")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── load model ────────────────────────────────────────────────────────────────
print(f"Loading vision encoder from: {args.checkpoint}")
# Use bfloat16 on CUDA (matches training dtype); float32 on CPU
load_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
vision = AutoModel.from_pretrained(args.checkpoint, dtype=load_dtype)
vision = vision.to(device).eval()
# pool_metadata was saved as non-persistent in older checkpoints — ensure correct values
vision.encoder.pool_metadata.data[:] = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
enc = vision.encoder
tok = enc.tokenizer
print(f"  embed_dim={enc.embed_dim}, depth={len(enc.blocks)}")

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── collect images ────────────────────────────────────────────────────────────
if args.image:
    image_paths = [args.image]
else:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.JPG")
    image_paths = []
    for ext in exts:
        image_paths += glob.glob(os.path.join(args.image_dir, ext))
    image_paths = sorted(image_paths)

if not image_paths:
    print(f"No images found in {args.image_dir}")
    sys.exit(1)

print(f"Processing {len(image_paths)} image(s)...")

# ── level colors ──────────────────────────────────────────────────────────────
LEVEL_COLORS = {
    0: ("#2196F3", "Global"),    # blue
    1: ("#FF9800", "Region"),    # orange
    2: ("#F44336", "Detail"),    # red
}

# ── process each image ────────────────────────────────────────────────────────
for img_path in image_paths:
    name = os.path.splitext(os.path.basename(img_path))[0]
    print(f"\n  Processing: {img_path}")

    orig_img = Image.open(img_path).convert("RGB")
    pv = transform(orig_img).unsqueeze(0).to(device, dtype=load_dtype)  # (1, 3, 256, 256)

    with torch.no_grad():
        # Get saliency map directly from tokenizer sub-modules
        stem_feat = tok.stem(pv)                    # (1, D, h, w)
        sal_map = tok.saliency(stem_feat)            # (1, 1, h, w)  in (0,1)
        h_sal, w_sal = sal_map.shape[-2:]

        # Get full encoder output for token positions
        out = vision(pv, token_budget=args.budget)  # KeuralEncoderOutput
        spatial_meta = out.spatial_metadata[0]      # (N, 4): cx, cy, w, h  [0..1]
        level_ids    = out.level_ids[0]             # (N,)

    sal_np = sal_map[0, 0].cpu().numpy()            # (h, w)  in (0,1)

    # ── figure: 3 panels ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{name}  |  checkpoint: {os.path.basename(args.checkpoint)}",
                 fontsize=11, y=1.01)

    # Panel 1: original image
    axes[0].imshow(orig_img)
    axes[0].set_title("Original Image", fontsize=10)
    axes[0].axis("off")

    # Panel 2: saliency heatmap overlaid on original
    axes[1].imshow(orig_img)
    sal_resized = np.array(
        Image.fromarray((sal_np * 255).astype(np.uint8)).resize(
            orig_img.size, resample=Image.BILINEAR
        )
    ) / 255.0
    axes[1].imshow(sal_resized, cmap="hot", alpha=0.6, vmin=0, vmax=1)
    axes[1].set_title(
        f"Saliency Map\n(min={sal_np.min():.3f}  max={sal_np.max():.3f}  "
        f"mean={sal_np.mean():.3f})",
        fontsize=9
    )
    axes[1].axis("off")

    # Colorbar for saliency
    sm = plt.cm.ScalarMappable(cmap="hot", norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: token placement overlay
    W_img, H_img = orig_img.size
    axes[2].imshow(orig_img)
    axes[2].set_title(
        f"Token Placement  (budget={args.budget})\n"
        f"Global={int((level_ids==0).sum())}  "
        f"Region={int((level_ids==1).sum())}  "
        f"Detail={int((level_ids==2).sum())}",
        fontsize=9
    )

    meta_np = spatial_meta.cpu().float().numpy()  # (N, 4): cx, cy, w, h in [0..1]
    for level in (0, 1, 2):
        mask = (level_ids == level).cpu().numpy()
        if mask.sum() == 0:
            continue
        cx = meta_np[mask, 0] * W_img
        cy = meta_np[mask, 1] * H_img
        tw = meta_np[mask, 2] * W_img
        th = meta_np[mask, 3] * H_img
        color, label = LEVEL_COLORS[level]

        # Draw bounding boxes for each token
        for i in range(len(cx)):
            x0 = cx[i] - tw[i] / 2
            y0 = cy[i] - th[i] / 2
            rect = patches.Rectangle(
                (x0, y0), tw[i], th[i],
                linewidth=1.2,
                edgecolor=color,
                facecolor=color,
                alpha=0.15,
            )
            axes[2].add_patch(rect)
            axes[2].plot(cx[i], cy[i], ".", color=color, markersize=3, alpha=0.7)

    # Legend
    legend_patches = [
        patches.Patch(color=c, label=f"{l} ({int((level_ids==lv).sum())})")
        for lv, (c, l) in LEVEL_COLORS.items()
    ]
    axes[2].legend(handles=legend_patches, loc="upper right", fontsize=8,
                   framealpha=0.8)
    axes[2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, f"{name}_saliency.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    # Print stats
    print(f"    Saliency  — min={sal_np.min():.4f}  max={sal_np.max():.4f}  "
          f"mean={sal_np.mean():.4f}  std={sal_np.std():.4f}")
    print(f"    Tokens    — G={int((level_ids==0).sum())}  "
          f"R={int((level_ids==1).sum())}  D={int((level_ids==2).sum())}")

print(f"\nAll figures saved to: {args.out_dir}/")
