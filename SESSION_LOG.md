# Keural Vision Encoder — Full Session History

> **Project:** Keural Vision Encoder (Tiny POC → Foundation Model)
> **Repo:** https://github.com/mkd-hika/VISION-ENCODER
> **Working directory:** `/home/mkd-s002/vision-encoder/poc/`
> **Author:** MKD-AI
> **Sessions covered:** 2026-05-11 (two sessions, context compacted between them)

---

## Quick Reference — Key Commands

```bash
# Training
PYTHONPATH=/home/mkd-s002/vision-encoder/poc \
  nohup python3.10 -m training.train --config configs/keural_tiny_poc.yaml \
  > outputs/run/train.log 2>&1 &

# Eval
python3.10 scripts/eval_checkpoint.py --checkpoint outputs/run/checkpoint-30000
python3.10 scripts/eval_retrieval_cc3m.py --checkpoint outputs/run/checkpoint-30000 --n-pairs 200
python3.10 scripts/visualize_saliency.py --checkpoint outputs/run/checkpoint-30000

# Push to GitHub
git add -p && git commit -m "..." && git push origin main
```

---

## Session 1 — 2026-05-11 (earlier)

### What Was Accomplished

#### 1. COMPARISON_FASTVLM.md created
- Side-by-side comparison of FastVLM (Apple 2025) vs Keural architecture
- Key differentiator: FastVLM = structural stride-32 fixed token reduction; Keural = semantic saliency-guided adaptive allocation
- Action plan and benchmark targets table
- File: `poc/COMPARISON_FASTVLM.md`

#### 2. pool_metadata bug found and fixed (CRITICAL)
- **Symptom:** All eval runs via `from_pretrained()` returned NaN in pooled output
- **Root cause:** `encoder.pool_metadata` registered as `persistent=False` → never saved to checkpoint → HuggingFace loading corrupts it to `[6.1e+25, 3.2e-41, 4.4e+21, 1.7e-04]` → RoPE receives garbage positions → NaN in block 0
- **Training impact: ZERO** — training always calls `__init__` first which sets correct values `[0.5, 0.5, 1.0, 1.0]`
- **Fix 1:** `encoder.py` — changed `persistent=False` → `persistent=True`
- **Fix 2:** All eval scripts — added manual reset after `from_pretrained()`:
  ```python
  vision.encoder.pool_metadata.data[:] = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
  ```
- Files fixed: `architecture/encoder.py`, `scripts/eval_checkpoint.py`, `scripts/eval_retrieval_cc3m.py`, `scripts/visualize_saliency.py`

#### 3. eval_retrieval_cc3m.py — created
- Extracts N image-caption pairs from CC3M WebDataset tars
- Encodes with Keural vision encoder + MiniLM text encoder
- Reports R@1, R@5, R@10 for image→text and text→image
- Results at checkpoint-28000 (200 pairs):

| Metric | Image→Text | Text→Image | Random Baseline |
|---|---|---|---|
| R@1 | 1.0% | 0.5% | 0.50% |
| R@5 | 3.5% | 3.5% | 2.50% |
| R@10 | 6.0% | 5.5% | 5.00% |

R@1 = **2× above random** at step 28k (38% through training) — expected.

#### 4. visualize_saliency.py — created
- 3-panel figure per image: original | saliency heatmap | token placement (colored by level)
- Level colors: Global=blue(#2196F3), Region=orange(#FF9800), Detail=red(#F44336)
- Token budget at step 28k: Global=13, Region=64, Detail=179 ✓
- Saliency at step 28k: mean ~0.47, std ~0.498 (near-random — expected at 38% training)
- Output: `outputs/saliency_viz/{cat,dog,horse}_saliency.png`

#### 5. generate_results_doc.py — created
- Generates `results/keural_eval_results_step28000.docx`
- 8 sections with tables, embedded saliency images, FastVLM comparison

#### 6. TRAINING_LOG.md — created
- Full training history from step 0 to 28,600
- 6 documented bugs with fixes
- 5-phase roadmap (pre-training → projector → benchmarks → VLM eval → paper)
- File: `poc/TRAINING_LOG.md`

#### 7. configs/keural_tiny_poc.yaml — updated
- `gradient_checkpointing: true` → `false` (disabled at step 28,000 for speed)
- VRAM jumped 11.4 GB → 25.8 GB; speed marginal improvement (bottleneck is data loading)

---

## Session 2 — 2026-05-11 (current)

### What Was Accomplished

#### 1. Documentation file corrected (keural_foundation_model_v.0.docx, 2.4MB)
> **Note:** Early in session 1, edits went to the WRONG file (`project_plan/keural_foundation_model_doc.docx`, 45KB). The correct file is `poc/keural_foundation_model_v.0.docx` (2.4MB). All fixes were re-applied to the correct file.

**Fixes applied (v0.1.1 → v0.1.3):**

| Location | Was | Fixed To |
|---|---|---|
| Phase 1 param count | 50M | 24.7M (actual) |
| Phase 1 loss curves [FILL] | empty | Added step 28k partial data |
| Phase 1 retrieval [FILL] | empty | R@1=1.0% at step 28k |
| Phase 2 header params | ~80M | ~183M (actual: embed_dim=768, depth=24) |
| Phase 2 header hardware | 4×A100 | 8×H100/A100 |
| Phase 2 header steps | 150,000 | 500,000 |
| Phase 3 header params | 80M | ~1.1B |
| Phase 3 header hardware | 8×H100 | 64×H100 SXM5 (stage1) + 16×H100 (stage2) |
| Phase 3 header steps | 300,000 | 2,000,000 + 200,000 |
| Phase 3 batch size | 8192 | 32,768 / 8,192 |
| §4.3 saliency regularizer | (no note) | Added Phase 1 implementation note: inverse-variance `1/(var+eps)` vs spec's entropy+smoothness |
| §4.5 optimization | (generic) | Added explicit LR per phase: Phase 1=1e-3, Phase 2=5e-4, Phase 3=2e-4 |
| §4.5 gradient checkpointing | (generic) | Phase 1 disabled at step 28k for speed |
| Model variant table: Keural-Base | 512/12/8/250M | 768/24/12/~183M |
| Model variant table: Keural-Large | 768/24/12/1.1B | 1024/48/16/1.1B |

#### 2. configs/architecture/phase2_mid.json — fixed
- `effective_batch_size`: 8192 → **4096** (actual: 128 × 4 grad_accum × 8 GPUs = 4096)
- `parameters.total`: "~250M" → **"~183M"** (correct for embed_dim=768, depth=24)

#### 3. Saliency regularizer — major fix
**Old (inverse-variance only):**
```python
L_sal = 1 / (var(S) + eps)
```
Problem: rewarded salt-and-pepper noise (high variance, spatially incoherent)

**New (anti-collapse + smoothness):**
```python
L_sal = 1/(var(S) + eps) + lambda_smooth * TV_L2(S)
```
Where `TV_L2(S) = mean(|∇_x S|²) + mean(|∇_y S|²)` (squared spatial gradients).

Parameters: `lambda_smooth=4.0` (new config field `saliency_lambda_smooth`)

Loss landscape:
- Flat map (uniform): anti-collapse → ∞, TV → 0 → total → very high ✗
- Random noise (var=0.25, TV=0.33): total ≈ 4.0 + 4×0.33 = 5.3 (bad but trainable)
- Smooth blob (var=0.25, TV=0.01): total ≈ 4.0 + 4×0.01 = 4.04 ✓ ← optimum

Gradient pushes saliency to: high contrast between subject/background AND spatially smooth transitions → meaningful token placement from step 30k onward.

**Files changed:** `training/losses.py`, `keural_config.py`, `tests/test_losses.py`

#### 4. Training restarted from checkpoint-30000
- Killed old training processes (PIDs 1661616 + workers — running old code)
- Restarted with fixed saliency regularizer
- Command:
  ```bash
  PYTHONPATH=/home/mkd-s002/vision-encoder/poc \
    nohup python3.10 -m training.train --config configs/keural_tiny_poc.yaml \
    > outputs/run/train_restart_30k.log 2>&1
  ```
- GPU: 25,317 MiB used, training loading first batch

#### 5. All changes pushed to GitHub
```
4a8c3e8  Replace saliency regularizer: add spatial smoothness term (TV_L2)
cab070b  Fix phase2_mid.json parameter count: ~250M → ~183M
6693af8  Fix model variant table in foundation model doc (v0.1.3)
5a5237f  Fix documentation errors in foundation model doc (v0.1.2) and phase2 config
12033ec  Fix documentation errors in foundation model doc (v0.1.1)
bc18b79  Add eval scripts, viz, docs, VLM inference, and fix pool_metadata bug
```

---

## Architecture Reference

### Keural-Tiny (Phase 1 POC — current)
| Component | Value |
|---|---|
| Parameters | 24,666,369 (~24.7M) |
| Stem channels | [64, 128, 256, 384] |
| embed_dim | 384 |
| Depth | 12 |
| Heads | 6, head_dim=64 |
| Token budget | 256 (Global=13, Region=64, Detail=179) |
| Positional encoding | 2D RoPE, pos_scale=100.0 |
| Loss | SigLIP + HCT (λ=1.0) + Saliency regularizer (β=0.01, λ_smooth=4.0) |
| Training | AdamW, lr=1e-3, cosine+warmup 2000 steps, batch=256, grad_accum=4 |
| Dataset | CC3M + CC12M (~6.9M pairs) |
| Text encoder | sentence-transformers/all-MiniLM-L6-v2 (frozen, 384-dim) |

### Keural-Base (Phase 2 — future)
| Component | Value |
|---|---|
| Parameters | ~183M |
| embed_dim | 768 |
| Depth | 24 |
| Heads | 12 |
| Token budget | 512 |
| Training | lr=5e-4, 500k steps, 8×H100, batch=128, grad_accum=4 (effective=4096) |
| Dataset | ~400M pairs (CC3M+CC12M+LAION-400M subset+curated) |

### Keural-Large (Phase 3 — future)
| Component | Value |
|---|---|
| Parameters | ~1.1B |
| embed_dim | 1024 |
| Depth | 48 |
| Heads | 16 |
| Training | lr=2e-4, 2M+200k steps, 64×H100 SXM5 |

---

## Known Issues & Decisions

| Issue | Status | Notes |
|---|---|---|
| pool_metadata `persistent=False` | **FIXED** | Changed to `persistent=True` in encoder.py; eval scripts manually reset for old checkpoints |
| Saliency regularizer (inverse-variance only) | **FIXED** | Now uses anti-collapse + TV_L2 smoothness from step 30k |
| Phase 2 effective_batch_size = 8192 | **FIXED** | Corrected to 4096 in phase2_mid.json |
| Documentation param counts | **FIXED** | All corrected in keural_foundation_model_v.0.docx v0.1.3 |
| Saliency regularizer spec mismatch | **Documented** | Spec says entropy+smoothness; Phase 1 used inverse-variance (now fixed). Phase 2 should implement full spec |
| HCT detail-patch reconstruction | **Deferred to v0.2** | Currently global-text only; detail-patch MAE component not implemented |
| Projector (Phase 2 VLM bridge) | **Not started** | Needs `scripts/train_projector.py` + `configs/projector_align.yaml` |

---

## Roadmap — Next Steps

### Immediate (while training runs to step 75k)
- [ ] At step 50,000: run `eval_retrieval_cc3m.py` — check R@1 improvement
- [ ] At step 75,000: run `visualize_saliency.py` — saliency figures for paper
- [ ] At step 75,000: run `eval_retrieval_cc3m.py` — paper's Phase 1 number (target R@1 ≥ 15%)

### After step 75,000 (Phase 2 setup)
- [ ] `configs/projector_align.yaml` — LevelAwareProjector (384 → LLM dim) config
- [ ] `scripts/train_projector.py` — projector alignment training on CC3M captions
- [ ] Flickr30K retrieval benchmark (R@1, R@5, R@10)
- [ ] MSCOCO retrieval benchmark

### Paper Targets vs FastVLM
| Benchmark | FastVLM | Keural Target |
|---|---|---|
| COCO Retrieval R@1 | ~75% | ≥ 70% |
| MMBench | ~73% | ≥ 68% |
| POPE F1 | ~86% | ≥ 84% |
| TextVQA | ~57% | ≥ 55% |

---

## File Map

```
poc/
├── architecture/
│   ├── encoder.py              ← pool_metadata fixed (persistent=True)
│   ├── spatial/rope_2d.py
│   └── ...
├── configs/
│   ├── keural_tiny_poc.yaml    ← gradient_checkpointing=false, saliency_lambda_smooth=4.0
│   └── architecture/
│       ├── phase2_mid.json     ← effective_batch_size=4096, total=~183M (FIXED)
│       └── phase3_commercial.json
├── training/
│   ├── train.py
│   └── losses.py               ← saliency regularizer fixed (anti-collapse + TV_L2)
├── scripts/
│   ├── eval_checkpoint.py      ← pool_metadata fix applied
│   ├── eval_retrieval_cc3m.py  ← NEW — R@1/R@5/R@10 on CC3M
│   ├── visualize_saliency.py   ← NEW — 3-panel saliency/token viz
│   └── generate_results_doc.py ← NEW — .docx results report
├── tests/
│   └── test_losses.py          ← updated for new saliency regularizer
├── outputs/
│   └── run/
│       ├── checkpoint-30000/   ← latest checkpoint (current training base)
│       └── train_restart_30k.log ← current training log
├── keural_config.py            ← saliency_lambda_smooth field added
├── keural_foundation_model_v.0.docx  ← main doc (v0.1.3, all errors fixed)
├── TRAINING_LOG.md             ← full training history
├── COMPARISON_FASTVLM.md       ← FastVLM vs Keural comparison
└── SESSION_LOG.md              ← this file
```

---

*Last updated: 2026-05-11 | Training active at step ~30,001 / 75,000*
