# Keural Vision Encoder — Mid Level Plan (250M Model)

**Model:** keural-mid (~183M params, embed_dim=768)  
**Goal:** Production-grade vision encoder targeting CLIP ViT-L/14 performance class  
**Starts:** After June 10 POC demo  
**Hardware needed:** 8× H100 80GB (or 8× A100 80GB)

---

## Why Mid Level

| | POC (current) | Mid Level |
|---|---|---|
| Parameters | 24.7M | ~183M |
| embed_dim | 384 | 768 |
| Depth | 12 layers | 24 layers |
| Token budget | 256 | 512 |
| Training data | 6.9M (CC3M+CC12M) | ~400M |
| Expected R@1 | ~5–10% | ~35–50% |
| OCR capability | Weak | Strong (with Phase 3) |
| Resolution | 256×256 | Up to 512×512 |

---

## Key Improvements Over POC

### 1. Knowledge Distillation (Two Teachers)

The biggest single quality improvement available.

```
Natural image batch:
    Teacher → SigLIP-400M (google/siglip-so400m-patch14-384)
    Loss    → 1 - cosine_similarity(keural_pooled, siglip_pooled)
    Weight  → λ_distill = 0.5

Document image batch:
    Teacher → InternViT-300M (OpenGVLab/InternViT-300M-448px)
    Loss    → 1 - cosine_similarity(keural_pooled, internvit_pooled)
    Weight  → λ_distill = 0.5
```

**Why SigLIP for natural images:**
- Trained on 4B image-text pairs
- Best-in-class natural scene alignment
- Proven in LLaVA, PaLIGemma, Idefics2

**Why InternViT for documents:**
- Backbone of InternVL (top open-source document VLM)
- Trained on natural images + documents + charts + OCR
- Same embed_dim (768) — easy to align

**Expected R@1 improvement:**
```
Without distillation: ~5–10% (POC level)
With SigLIP only:     ~25–40%
With both teachers:   ~35–50%
```

### 2. Architecture Upgrades

| Component | POC | Mid |
|---|---|---|
| embed_dim | 384 | 768 |
| Depth | 12 | 24 |
| Token budget | 256 | 512 |
| ROI align size | 2 | 4 |
| Max resolution | 256×256 | 512×512 |
| Image buckets | 1 size | 7 sizes (224–512) |

### 3. Training Data (~400M samples)

| Dataset | Samples | Purpose |
|---|---|---|
| CC3M | 3M | Base (already have) |
| CC12M | 12M | Base (already have) |
| LAION-400M subset | ~200M | Scale |
| COYO-700M subset | ~100M | Quality |
| Document pairs (Docling) | ~50M | OCR/document |
| Web scraped + filtered | ~35M | Diversity |

**Data pipeline (must build before training):**
- Deduplication (near-duplicate image removal)
- NSFW filtering
- Quality scoring (CLIP score threshold > 0.25)
- Caption quality filter (min 5 words, max aspect ratio 3.0)

### 4. Training Stability Improvements

- Gradient checkpointing (saves ~40% VRAM)
- Larger effective batch: 4096 (vs 256 in POC)
- betas=(0.9, 0.95) instead of (0.9, 0.999)
- Warmup 10k steps (vs 4k in POC)

---

## Training Phases (Post-Demo)

```
PHASE 1M   Natural image pretraining        ████░░░░░░   ~7–10 days (8× H100)
           CC3M + CC12M + LAION + COYO
           Loss: SigLIP + HCT + Sal + SigLIP distillation

PHASE 2M   Projector alignment              ██░░░░░░░░   ~3–5 days
           Upgrade LevelAwareProjector: 768→4096
           keural-14.8b-base 4-bit QLoRA

PHASE 3M   Document fine-tuning            ██░░░░░░░░   ~3–5 days
           Docling dataset (from RAG team)
           Loss: SigLIP + InternViT distillation
           Teacher: InternViT-300M

PHASE 4M   Instruction fine-tuning         ██░░░░░░░░   ~3–5 days
           LLaVA-Instruct-150K + document QA pairs
```

---

## Distillation Implementation

```python
import open_clip
from transformers import AutoModel

# ── Teacher 1: SigLIP (natural images) ────────────────────────────────────────
siglip, _, siglip_preprocess = open_clip.create_model_and_transforms(
    'ViT-SO400M-14-SigLIP', pretrained='webli'
)
siglip = siglip.to(device).eval()
for p in siglip.parameters():
    p.requires_grad_(False)

# ── Teacher 2: InternViT (documents) ──────────────────────────────────────────
internvit = AutoModel.from_pretrained(
    'OpenGVLab/InternViT-300M-448px',
    torch_dtype=torch.bfloat16
).to(device).eval()
for p in internvit.parameters():
    p.requires_grad_(False)

# ── Distillation projector (teacher_dim → keural_dim) ─────────────────────────
# SigLIP embed_dim=1152, InternViT embed_dim=768, Keural mid embed_dim=768
siglip_proj = nn.Linear(1152, 768).to(device)  # only for SigLIP (different dim)

# ── Loss ──────────────────────────────────────────────────────────────────────
def distillation_loss(keural_pooled, pixel_values, batch_type="natural"):
    keural_emb = F.normalize(keural_pooled.float(), dim=-1)

    if batch_type == "natural":
        with torch.no_grad():
            teacher_emb = siglip.encode_image(pixel_values)
            teacher_emb = F.normalize(siglip_proj(teacher_emb.float()), dim=-1)
    else:  # document
        with torch.no_grad():
            teacher_emb = internvit(pixel_values=pixel_values).pooler_output
            teacher_emb = F.normalize(teacher_emb.float(), dim=-1)

    return 1 - F.cosine_similarity(keural_emb, teacher_emb).mean()

# ── Combined loss ─────────────────────────────────────────────────────────────
total_loss = (
    siglip_loss          # primary contrastive loss
    + 1.0 * hct_loss     # hierarchical concept tokenization
    + 0.5 * sal_loss     # saliency regularization
    + 0.5 * distill_loss # knowledge distillation ← NEW
)
```

---

## Success Criteria (Mid Level)

| Metric | POC target | Mid target |
|---|---|---|
| R@1 (1K pairs, COCO) | > 8% | > 35% |
| R@5 (1K pairs, COCO) | > 20% | > 60% |
| MedR (1K pairs) | < 70 | < 10 |
| Document OCR accuracy | — | > 70% (Docling test set) |
| Pooled norm | < 2.0 | < 1.5 |
| Primary loss | < 1.5 | < 1.0 |

---

## Eval Benchmarks to Add (Mid Level)

| Benchmark | What it tests |
|---|---|
| COCO Retrieval (5K pairs) | Natural image alignment |
| Flickr30K Retrieval (1K pairs) | Generalization |
| TextVQA | OCR in natural images |
| DocVQA | Document understanding |
| ChartQA | Chart/table understanding |

---

## Hardware Plan

```
8× H100 80GB SXM5
  └── DDP (DistributedDataParallel)
  └── FSDP for 183M model — fits comfortably per GPU
  └── ~60GB VRAM per GPU (model + activations + two frozen teachers)
  └── Estimated: 7–14 days for Phase 1M (500k steps)
```

**Cost estimate (cloud H100):**
- 8× H100 × ~$3.5/hr × 10 days = ~$6,720 USD
- Alternative: 4× H100 × 20 days = same cost, slower

---

## File Structure Target

```
poc/
├── configs/
│   ├── keural_mid.yaml              ← Phase 1M training config (to build)
│   ├── projector_mid.yaml           ← Phase 2M projector config (to build)
│   └── architecture/
│       └── phase2_mid.json          ← Architecture spec ✅
├── scripts/
│   ├── train.py                     ← Extend for DDP + distillation
│   ├── train_projector.py           ← Extend for 768→4096 ✅
│   └── eval_retrieval_coco.py       ← New: COCO 5K eval (to build)
└── ROADMAP_MID.md                   ← This file ✅
```

---

## Dependencies to Install (Before Mid Training)

```bash
pip install open_clip_torch          # SigLIP teacher
pip install intern-image             # InternViT teacher (or via HF)
pip install datasets                 # for COCO/Flickr eval
pip install torch.distributed        # already in PyTorch for DDP
```

---

## Timeline (Post June 10)

| Date | Milestone |
|---|---|
| Jun 10 | POC demo ← current target |
| Jun 11–20 | Setup 8× H100, data pipeline, distillation code |
| Jun 20 | Launch Phase 1M (natural image pretraining) |
| Jul 1–4 | Phase 1M complete, run COCO eval |
| Jul 4–9 | Phase 2M projector training |
| Jul 9–14 | Phase 3M document fine-tuning (Docling) |
| Jul 14–21 | Phase 4M instruction tuning |
| Aug 1 | keural-mid v1.0 release |

---

*Created: 2026-05-14 | Depends on: POC demo Jun 10 | Hardware: 8× H100*
