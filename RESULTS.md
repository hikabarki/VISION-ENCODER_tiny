# Keural Vision Encoder — Training Results

**Model:** KeuralVisionEncoder (ATB + HCT)
**Params:** 24.7M | **Embed:** 384 | **Depth:** 12 | **Heads:** 6
**Data:** CC3M + CC12M (~6.9M samples)
**Hardware:** NVIDIA RTX 5090 32GB

---

## Training Progress

| Step | % Done | Total Loss | Primary (SigLIP) | HCT Loss | Saliency | Speed (s/50 steps) |
|------|--------|------------|------------------|----------|----------|--------------------|
| 5,000 | 6.7% | ~8.8 | ~4.4 | ~4.4 | ~4.0 | — (checkpoint) |
| 6,000 | 8.0% | 6.75 | 3.36 | 3.35 | 4.01 | 538 |
| 7,000 | 9.3% | 6.18 | 3.08 | 3.06 | 4.01 | 549 |
| 8,000 | 10.7% | 5.95 | 2.96 | 2.95 | 4.00 | 525 |
| 9,000 | 12.0% | 5.38 | 2.67 | 2.67 | 4.00 | 525 |
| 10,000 | 13.3% | 5.84 | 2.90 | 2.89 | 4.00 | 524 |
| 11,000 | 14.7% | 5.73 | 2.84 | 2.85 | 4.00 | 523 |
| 12,000 | 16.0% | 5.29 | 2.62 | 2.62 | 4.00 | 526 |
| 13,000 | 17.3% | 5.24 | 2.60 | 2.60 | 4.00 | 526 |
| 14,000 | 18.7% | 5.82 | 2.89 | 2.89 | 4.00 | 521 |
| 15,000 | 20.0% | 5.04 | 2.50 | 2.50 | 4.00 | 533 |
| 16,000 | 21.3% | 4.79 | 2.38 | 2.37 | 4.00 | 527 |
| 17,000 | 22.7% | 5.47 | 2.72 | 2.71 | 4.00 | 526 |
| 18,000 | 24.0% | 5.56 | 2.76 | 2.76 | 4.00 | 519 |
| 19,000 | 25.3% | 4.61 | 2.29 | 2.28 | 3.99 | 514 |
| 20,000 | 26.7% | 5.04 | 2.50 | 2.50 | 4.00 | 517 |
| 21,000 | 28.0% | 4.36 | 2.16 | 2.16 | 4.00 | 511 |
| 22,000 | 29.3% | 4.64 | 2.30 | 2.30 | 4.00 | 515 |
| 23,000 | 30.7% | 4.67 | 2.32 | 2.31 | 4.00 | 515 |
| 24,000 | 32.0% | 4.34 | 2.15 | 2.15 | 4.00 | 511 |
| 25,000 | 33.3% | 4.80 | 2.39 | 2.38 | 4.00 | 510 |
| 26,000 | 34.7% | 4.88 | 2.42 | 2.42 | 4.00 | 509 |
| 27,000 | 36.0% | 4.60 | 2.28 | 2.28 | 3.99 | 509 |
| 28,000 | 37.3% | 4.92 | 2.44 | 2.44 | 4.00 | 509 |

**Training speed:** ~508–540s per 50 steps (~10.2s per step)
**Expected completion:** ~May 16–17, 2026

---

## Checkpoint Evaluations

### checkpoint-6000 (2026-05-08)

**Smoke test** (`scripts/test_checkpoint.py`):
| Budget | Tokens shape | Pooled shape | Global | Region | Detail |
|--------|-------------|-------------|--------|--------|--------|
| 64 | (1, 64, 384) | (1, 384) | 4 | 16 | 44 |
| 256 | (1, 256, 384) | (1, 384) | 13 | 64 | 179 |
| 1024 | (1, 1024, 384) | (1, 384) | 52 | 256 | 716 |

**Image-text alignment test** (`scripts/eval_checkpoint.py`):
| Metric | Value | Notes |
|--------|-------|-------|
| Retrieval accuracy | 1/4 (25%) | Random chance level — expected at 8% training |
| Pooled norm (raw) | 15.42 | Normal for contrastive training, eval uses normalized embeds |
| Architecture | Healthy | Correct shapes, no NaN, ATB allocating correctly |

> At 8% training (6,000 / 75,000 steps), random-chance retrieval is expected.
> Meaningful benchmark evaluation (Flickr30K, COCO) will be run on the final checkpoint.

---

## Checkpoints

| Checkpoint | Step | Total Loss | Saved | HuggingFace |
|------------|------|------------|-------|-------------|
| checkpoint-5000 | 5,000 | ~8.8 | Local + HF | mkd-hika/keural-vision-encoder-poc |
| checkpoint-6000 | 6,000 | 6.75 | Local only | — |
| checkpoint-27000 | 27,000 | 4.60 | Local only | — |

---

## Training Configuration

```yaml
model:     KeuralVisionEncoder — embed=384, depth=12, heads=6 (24.7M params)
data:      CC3M + CC12M — ~6.9M image-text pairs
objective: SigLIP contrastive + HCT auxiliary (λ=1.0) + saliency reg (β=0.01)
optimizer: AdamW — lr=1e-3, betas=(0.9, 0.95), wd=0.05
schedule:  cosine_with_warmup — warmup=2000, max_steps=75000
batch:     256 per device × 4 grad accum = 1024 effective
precision: bfloat16 + gradient checkpointing
```

---

## Planned Final Benchmarks (step 75,000)

| Benchmark | Expected |
|-----------|----------|
| Flickr30K i2t R@1 | TBD |
| Flickr30K t2i R@1 | TBD |
| COCO i2t R@1 | TBD |
| COCO t2i R@1 | TBD |
| Image-text retrieval accuracy | TBD |

---

*Last updated: 2026-05-11 | Step 28,000 / 75,000 (37.3%)*
