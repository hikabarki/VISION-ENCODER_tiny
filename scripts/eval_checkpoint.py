"""
Quality evaluation for a Keural checkpoint.
Tests image-text alignment using cosine similarity.

Usage:
    python3.10 scripts/eval_checkpoint.py --checkpoint outputs/run/checkpoint-6000
"""
import sys, argparse
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
parser.add_argument("--checkpoint", default="outputs/run/checkpoint-6000")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── load models ───────────────────────────────────────────────────────────────
print(f"Loading vision encoder from: {args.checkpoint}")
# Use bfloat16 on CUDA (matches training dtype); float32 on CPU
load_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
vision = AutoModel.from_pretrained(args.checkpoint, dtype=load_dtype)
vision = vision.to(device).eval()
# pool_metadata was saved as non-persistent in older checkpoints — ensure correct values
vision.encoder.pool_metadata.data[:] = torch.tensor([[0.5, 0.5, 1.0, 1.0]])

# Verify dtype and check for NaN in weights
param_dtypes = {p.dtype for p in vision.parameters()}
print(f"Parameter dtypes after load: {param_dtypes}")
nan_params = [(n, p) for n, p in vision.named_parameters() if p.isnan().any() or p.isinf().any()]
if nan_params:
    print(f"WARNING: NaN/Inf found in {len(nan_params)} parameters: {[n for n, _ in nan_params[:5]]}")
else:
    print("Weights OK — no NaN or Inf in parameters")

print("Loading text encoder...")
text_enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── NaN debugger ─────────────────────────────────────────────────────────────
def _dbg(model, pv):
    """Trace NaN through the encoder stage by stage."""
    enc = model.encoder
    tok = enc.tokenizer
    print("  [dbg] tracing NaN source...")
    with torch.no_grad():
        stem = tok.stem(pv)
        print(f"  stem:    nan={stem.isnan().any().item()}, norm={stem.norm():.2f}")
        sal = tok.saliency(stem)
        print(f"  saliency:nan={sal.isnan().any().item()}, range=[{sal.min():.4f},{sal.max():.4f}]")
        from tokenizer.atb_tokenizer import split_budget
        K_g, K_r, K_d = split_budget(256, tok.config.tokenizer.budget_split, tok.config.tokenizer.budget_floors)
        g = tok.global_constructor(stem, K_g=K_g)
        print(f"  global:  nan={g.isnan().any().item()}, norm={g.norm():.2f}")
        from tokenizer.region_sampler import RegionSampler
        picks = tok.region_sampler(sal, K=K_r, gumbel_temperature=0.0, training=False)
        r = tok.region_constructor(stem, picks)
        print(f"  region:  nan={r.isnan().any().item()}, norm={r.norm():.2f}")
        picks_d = tok.detail_sampler(sal, K=K_d, gumbel_temperature=0.0, training=False)
        d = tok.detail_constructor(stem, picks_d)
        print(f"  detail:  nan={d.isnan().any().item()}, norm={d.norm():.2f}")
        # Run transformer blocks with simplified metadata
        tokens = torch.cat([g, r, d], dim=1)
        level_ids = torch.zeros(1, K_g + K_r + K_d, dtype=torch.int64)
        level_ids[:, :K_g] = 0; level_ids[:, K_g:K_g+K_r] = 1; level_ids[:, K_g+K_r:] = 2
        level_emb = enc.level_embeddings[level_ids]
        x = tokens + level_emb
        pool_tok = enc.pool_token.expand(1, 1, x.shape[-1]).to(x.dtype)
        x = torch.cat([pool_tok, x], dim=1)
        print(f"  pre-blocks: nan={x.isnan().any().item()}, norm={x.norm():.2f}")
        # Use full-image metadata for all tokens (approximation for debugging)
        meta_full = enc.pool_metadata.unsqueeze(0).expand(1, x.shape[1], 4)
        for i, blk in enumerate(enc.blocks):
            mask = torch.ones(1, x.shape[1], dtype=torch.int64)
            x = blk(x, meta_full, mask)
            if x.isnan().any():
                print(f"  block {i}: NaN introduced here!")
                break
            print(f"  block {i}: ok, norm={x.norm():.2f}")

# ── test pairs (image_path, caption) ─────────────────────────────────────────
# Replace these with real images from your machine
TEST_PAIRS = [
    ("data/cc3m/sample_0.jpg", "a dog running in a park"),
    ("data/cc3m/sample_1.jpg", "a cat sitting on a chair"),
    ("data/cc3m/sample_2.jpg", "a red car on a city street"),
]

# If no real images, use colored dummy images to test basic direction
def make_dummy_image(color):
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[:, :] = color
    return Image.fromarray(arr)

DUMMY_PAIRS = [
    (make_dummy_image([200, 50, 50]),  "a red colored image"),
    (make_dummy_image([50, 200, 50]),  "a green colored image"),
    (make_dummy_image([50, 50, 200]),  "a blue colored image"),
    (make_dummy_image([200, 200, 50]), "a yellow bright image"),
]

print("\n" + "="*60)
print("TEST: Image-Text Similarity Matrix")
print("Each row = one image. Diagonal should be HIGHEST score.")
print("="*60)

# Use dummy pairs (always works without real images)
images, captions = zip(*DUMMY_PAIRS)

# Encode images
pixel_values = torch.stack([transform(img) for img in images]).to(device, dtype=load_dtype)
print(f"pixel_values dtype={pixel_values.dtype}, shape={pixel_values.shape}")
autocast_ctx = torch.autocast(device_type=device, dtype=load_dtype) if device == "cuda" else torch.autocast(device_type="cpu", enabled=False)
with torch.no_grad(), autocast_ctx:
    out = vision(pixel_values, token_budget=256)
    img_embeds = out.pooled.float().cpu()  # (N, 384) — cast to float32 for similarity
    print(f"pooled raw norm (pre-normalize): {img_embeds.norm(dim=-1).tolist()}")
    if img_embeds.isnan().any():
        print("NaN detected in pooled output — running per-module debug...")
        _dbg(vision, pixel_values[:1])
    img_embeds = F.normalize(img_embeds, dim=-1)

# Encode texts
txt_embeds = torch.tensor(text_enc.encode(list(captions)))      # (N, 384)
txt_embeds = F.normalize(txt_embeds, dim=-1)

# Similarity matrix
sim = (img_embeds @ txt_embeds.T).numpy()

print(f"\n{'':20s}" + "  ".join(f"T{i}" for i in range(len(captions))))
for i, caption in enumerate(captions):
    row = "  ".join(f"{sim[i,j]:+.3f}{'*' if i==j else ' '}" for j in range(len(captions)))
    print(f"  I{i} {row}   ← {caption[:30]}")

# Check if diagonal is highest in each row
correct = sum(np.argmax(sim[i]) == i for i in range(len(captions)))
print(f"\nRetrieval accuracy: {correct}/{len(captions)} images matched correct caption")

if correct == len(captions):
    print("✓ All matched — encoder has learned image-text alignment")
elif correct > 0:
    print("~ Partial match — encoder is learning but not fully aligned yet (expected at ~8% training)")
else:
    print("✗ No match — embeddings may be collapsed or random (check loss)")

print("\n" + "="*60)
print("Token budget stats (budget=256):")
with torch.no_grad(), autocast_ctx:
    out2 = vision(pixel_values[:1], token_budget=256)
    ids = out2.level_ids[0]
    print(f"  Global tokens: {(ids==0).sum().item()}")
    print(f"  Region tokens: {(ids==1).sum().item()}")
    print(f"  Detail tokens: {(ids==2).sum().item()}")
    print(f"  Pooled norm:   {out2.pooled.float().norm().item():.4f}  (should be ~1.0 after training)")
print("="*60)
