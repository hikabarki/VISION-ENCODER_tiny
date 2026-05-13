"""
Real image-text retrieval evaluation on CC3M subset.

Extracts N image-caption pairs from training tars, encodes them with the
Keural vision encoder + MiniLM text encoder, then reports R@1, R@5, R@10
for both image→text and text→image directions.

Usage:
    python3.10 scripts/eval_retrieval_cc3m.py --checkpoint outputs/run/checkpoint-28000
    python3.10 scripts/eval_retrieval_cc3m.py --checkpoint outputs/run/checkpoint-28000 --n-pairs 500
"""
import sys, argparse, tarfile, io, os
sys.path.insert(0, "/home/mkd-s002/vision-encoder/poc")

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np

import alignment.hf_wrapper
from transformers import AutoModel
from sentence_transformers import SentenceTransformer

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="outputs/run/checkpoint-28000")
parser.add_argument("--n-pairs", type=int, default=200, help="Number of image-caption pairs to evaluate")
parser.add_argument("--tar-dir", default="data/cc3m", help="Directory containing CC3M tar files")
parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size")
parser.add_argument("--budget", type=int, default=256, help="Vision token budget")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── load models ───────────────────────────────────────────────────────────────
print(f"\nLoading vision encoder from: {args.checkpoint}")
# Use bfloat16 on CUDA (matches training dtype); float32 on CPU
load_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
vision = AutoModel.from_pretrained(args.checkpoint, dtype=load_dtype)
vision = vision.to(device).eval()
# pool_metadata was saved as non-persistent in older checkpoints — ensure correct values
vision.encoder.pool_metadata.data[:] = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
print(f"  Parameters: {sum(p.numel() for p in vision.parameters()):,}")

print("Loading text encoder (MiniLM)...")
text_enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── extract image-caption pairs from tar files ────────────────────────────────
def extract_pairs_from_tars(tar_dir, n_pairs):
    """Extract up to n_pairs (image, caption) tuples from CC3M tars."""
    pairs = []
    tar_files = sorted([
        os.path.join(tar_dir, f) for f in os.listdir(tar_dir)
        if f.endswith(".tar")
    ])
    if not tar_files:
        raise RuntimeError(f"No .tar files found in {tar_dir}")

    for tar_path in tar_files:
        if len(pairs) >= n_pairs:
            break
        print(f"  Reading {os.path.basename(tar_path)}...")
        try:
            with tarfile.open(tar_path) as tar:
                members = {m.name: m for m in tar.getmembers()}
                # Find all stems that have both .jpg and .txt
                stems = set()
                for name in members:
                    if name.endswith(".jpg"):
                        stems.add(name[:-4])

                for stem in sorted(stems):
                    if len(pairs) >= n_pairs:
                        break
                    jpg_key = stem + ".jpg"
                    txt_key = stem + ".txt"
                    if jpg_key not in members or txt_key not in members:
                        continue
                    try:
                        img_bytes = tar.extractfile(members[jpg_key]).read()
                        cap_bytes = tar.extractfile(members[txt_key]).read()
                        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        caption = cap_bytes.decode("utf-8", errors="replace").strip()
                        if caption:
                            pairs.append((img, caption))
                    except Exception:
                        continue
        except Exception as e:
            print(f"  Warning: could not read {tar_path}: {e}")
            continue

    return pairs

print(f"\nExtracting {args.n_pairs} image-caption pairs from {args.tar_dir}...")
pairs = extract_pairs_from_tars(args.tar_dir, args.n_pairs)
print(f"  Extracted {len(pairs)} pairs")
if len(pairs) < 10:
    print("ERROR: too few pairs, check tar_dir path")
    sys.exit(1)

images, captions = zip(*pairs)
N = len(images)

# ── encode images (batched) ───────────────────────────────────────────────────
# CPU batching causes NaN in attention on degenerate image combinations — use batch=1 on CPU
batch_size = args.batch_size if device == "cuda" else 1
if device == "cpu" and args.batch_size > 1:
    print(f"  Note: forcing batch_size=1 on CPU to avoid NaN in attention")

print(f"\nEncoding {N} images (batch={batch_size})...")
all_img_embeds = []
pixel_tensors = [transform(img) for img in images]

autocast_ctx = torch.autocast(device_type=device, dtype=load_dtype) if device == "cuda" else torch.autocast(device_type="cpu", enabled=False)
with torch.no_grad(), autocast_ctx:
    for i in range(0, N, batch_size):
        batch = torch.stack(pixel_tensors[i:i + batch_size]).to(device, dtype=load_dtype)
        out = vision(batch, token_budget=args.budget)
        emb = out.pooled.float()  # (B, D)
        all_img_embeds.append(emb.cpu())
        if (i // batch_size) % 5 == 0:
            print(f"  [{i+len(batch)}/{N}] pooled norm: {emb.norm(dim=-1).mean():.4f}")

img_embeds = torch.cat(all_img_embeds, dim=0)  # (N, D)
img_embeds = F.normalize(img_embeds, dim=-1)
print(f"  Image embeddings: {img_embeds.shape}, nan={img_embeds.isnan().any()}")

# ── encode captions ───────────────────────────────────────────────────────────
print(f"\nEncoding {N} captions...")
txt_embeds = torch.tensor(
    text_enc.encode(list(captions), batch_size=64, show_progress_bar=False)
)  # (N, 384)
txt_embeds = F.normalize(txt_embeds, dim=-1)
print(f"  Text embeddings: {txt_embeds.shape}")

# ── compute retrieval metrics ─────────────────────────────────────────────────
sim = (img_embeds @ txt_embeds.T).numpy()  # (N, N)

def recall_at_k(sim_matrix, k_values=(1, 5, 10), direction="i2t"):
    """Compute R@k. direction: 'i2t' (image→text) or 't2i' (text→image)."""
    N = sim_matrix.shape[0]
    results = {}
    if direction == "t2i":
        sim_matrix = sim_matrix.T
    for k in k_values:
        hits = 0
        for i in range(N):
            top_k = np.argsort(sim_matrix[i])[::-1][:k]
            if i in top_k:
                hits += 1
        results[k] = hits / N * 100
    return results

def median_and_mean_rank(sim_matrix, direction="i2t"):
    """Compute MedR and MeanR. Lower is better (rank 1 = perfect)."""
    N = sim_matrix.shape[0]
    if direction == "t2i":
        sim_matrix = sim_matrix.T
    ranks = []
    for i in range(N):
        sorted_idx = np.argsort(sim_matrix[i])[::-1]
        rank = int(np.where(sorted_idx == i)[0][0]) + 1  # 1-indexed
        ranks.append(rank)
    return float(np.median(ranks)), float(np.mean(ranks))

i2t = recall_at_k(sim, direction="i2t")
t2i = recall_at_k(sim, direction="t2i")
i2t_medr, i2t_meanr = median_and_mean_rank(sim, direction="i2t")
t2i_medr, t2i_meanr = median_and_mean_rank(sim, direction="t2i")

print(f"\n{'='*55}")
print(f"  CC3M Retrieval Evaluation  |  N={N} pairs")
print(f"{'='*55}")
print(f"  {'Metric':<25} {'Image→Text':>12} {'Text→Image':>12}")
print(f"  {'-'*50}")
for k in (1, 5, 10):
    print(f"  R@{k:<23} {i2t[k]:>11.1f}% {t2i[k]:>11.1f}%")
print(f"  {'-'*50}")
print(f"  {'MedR':<25} {i2t_medr:>11.1f}  {t2i_medr:>11.1f}")
print(f"  {'MeanR':<25} {i2t_meanr:>11.1f}  {t2i_meanr:>11.1f}")
print(f"{'='*55}")

random_r1 = 1 / N * 100
random_r5 = 5 / N * 100
random_medr = (N + 1) / 2
print(f"\n  Random-chance R@1:  {random_r1:.2f}%  |  R@5: {random_r5:.2f}%  |  MedR: {random_medr:.1f}")
# Use R@5 average as primary signal (more stable than R@1 on small N)
avg_r5 = (i2t[5] + t2i[5]) / 2
avg_medr = (i2t_medr + t2i_medr) / 2
if avg_r5 > random_r5 * 2 or avg_medr < random_medr * 0.7:
    print("  ✓ Model significantly above random — alignment is learning")
elif avg_r5 > random_r5 or avg_medr < random_medr:
    print("  ~ Model above random — alignment in progress (normal before step 60k)")
else:
    print("  ✗ Model at or below random — check loss convergence")

print(f"\nDone. Checkpoint: {args.checkpoint}")
