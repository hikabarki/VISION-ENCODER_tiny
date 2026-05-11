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

# ── load models ───────────────────────────────────────────────────────────────
print(f"Loading vision encoder from: {args.checkpoint}")
vision = AutoModel.from_pretrained(args.checkpoint)
vision.eval()

print("Loading text encoder...")
text_enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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
pixel_values = torch.stack([transform(img) for img in images])
with torch.no_grad():
    img_embeds = vision(pixel_values, token_budget=256).pooled  # (N, 384)
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
with torch.no_grad():
    out = vision(pixel_values[:1], token_budget=256)
    ids = out.level_ids[0]
    print(f"  Global tokens: {(ids==0).sum().item()}")
    print(f"  Region tokens: {(ids==1).sum().item()}")
    print(f"  Detail tokens: {(ids==2).sum().item()}")
    print(f"  Pooled norm:   {out.pooled.norm().item():.4f}  (should be ~1.0 after training)")
print("="*60)
