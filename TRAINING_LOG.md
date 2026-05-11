# Keural Vision Encoder — Full Training Log & Roadmap

> **Project:** Keural Vision Encoder (Tiny POC)
> **Author:** MKD-AI
> **Date started:** 2026 (pre-step 5000)
> **Current status:** Step 28,600 / 75,000 — Training in progress
> **Last updated:** 2026-05-11

---

## Table of Contents

1. [Model Architecture Summary](#1-model-architecture-summary)
2. [Training Configuration](#2-training-configuration)
3. [Phase 1 Timeline — What Happened](#3-phase-1-timeline--what-happened)
4. [Errors & Fixes](#4-errors--fixes)
5. [Evaluation Results So Far](#5-evaluation-results-so-far)
6. [Current Training State](#6-current-training-state)
7. [Future Roadmap](#7-future-roadmap)
8. [Paper Positioning vs FastVLM](#8-paper-positioning-vs-fastvlm)

---

## 1. Model Architecture Summary

| Component | Details |
|---|---|
| **Name** | Keural Vision Encoder (Tiny POC) |
| **Parameters** | 24,666,369 (~24.7M) |
| **Input** | Images (224×224, 256×256, 320×320) |
| **Stage 1 — CNN Stem** | 4 conv stages → channels [64, 128, 256, 384], stride-16 downsampling |
| **Stage 2 — Saliency Predictor** | 2-conv head → (B, 1, h, w) sigmoid map — *learns where to focus* |
| **Stage 3 — ATB Tokenizer** | Adaptive Token Budget: splits budget into Global / Region / Detail tokens |
| **Stage 4 — Transformer** | 12 layers, 6 heads, head_dim=64, embed_dim=384, 2D-RoPE, SwiGLU MLP |
| **Output** | KeuralEncoderOutput: tokens (B, N, 384) + pooled (B, 384) + metadata |

**Token budget split (default 256 tokens):**
- Global: 5% → 13 tokens (full-scene summary via cross-attention)
- Region: 25% → 64 tokens (object-scale salient patches)
- Detail: 70% → 179 tokens (fine-grained crops at peak saliency)

**Loss function:**
- Primary: SigLIP contrastive (image-text alignment)
- Auxiliary: HCT loss — enforces global token aligns with text (λ=1.0)
- Regularizer: Saliency variance loss — prevents saliency collapse (β=0.01)

---

## 2. Training Configuration

```yaml
optimizer:        AdamW  (betas=[0.9, 0.95], weight_decay=0.05)
lr:               1.0e-3
lr_schedule:      cosine_with_warmup
warmup_steps:     2000
max_steps:        75000
batch_size:       256 per device
grad_accumulation: 4  (effective batch = 1024)
precision:        bfloat16
hardware:         1× RTX 5090 (32 GB VRAM)
dataset:          CC3M + CC12M (~15M image-text pairs)
text_encoder:     sentence-transformers/all-MiniLM-L6-v2 (frozen)
```

---

## 3. Phase 1 Timeline — What Happened

### Step 0–2000: Warmup
- LR ramped from 0 → 1e-3 over 2000 steps
- ATB tokenizer uses **fixed grid sampling** for the first 5000 steps (warmup curriculum) — saliency predictor is random at start, so grid prevents garbage token placement
- Initial loss: ~8.7 (random chance for SigLIP binary classification)

### Step 2000–5000: Early Training
- LR reached peak (1e-3)
- Loss dropped rapidly: 8.7 → ~7.0
- Saliency warmup still active (grid sampling)
- VRAM: ~9.4 GB, Speed: ~530s/50 steps
- **First checkpoint saved: checkpoint-5000**

### Step 5000–10000: Warmup ends, ATB activates
- Gumbel-softmax saliency sampling turned on (temperature 1.0 → annealing)
- Loss continued dropping: ~7.0 → ~6.0
- Several training restarts from checkpoint-5000 and checkpoint-6000 (watchdog/process issues — **no data lost, all resumed cleanly**)

### Step 10000–28000: Stable Training
- Loss steadily declining: ~6.0 → ~4.3
- HCT loss tracking primary loss closely (good sign — hierarchical alignment working)
- Saliency regularization stable at ~4.0 throughout
- Speed consistent: ~508–514s per 50 steps
- VRAM: ~11.4 GB (gradient checkpointing ON during this period)
- **Checkpoint-28000: last clean checkpoint before experiments**

### Step 28000: Batch Size Experiment (caused issues)
- Changed batch_size 256 → 640, grad_accum 4 → 2
- **Result: speed got WORSE** — 508s → 696s per 50 steps
- Root cause: GPU was already compute-bound; larger batch hit memory bandwidth limits
- Loss also spiked: optimizer state disrupted by restart
- **Decision: reverted to batch=256, grad_accum=4**

### Step 28000 (restart): Gradient Checkpointing Disabled
- Changed `gradient_checkpointing: true` → `false` in config
- VRAM jumped: 11.4 GB → 25.8 GB (confirms checkpointing disabled)
- Speed: ~516s/50 steps (marginal improvement — bottleneck is data loading, not compute)
- **Loss spike after multiple restarts: 4.3 → 8.2** (optimizer momentum disrupted)
- **Recovery: loss came back down within ~600 steps: 8.2 → 6.1 → 4.77**

### Step 28600–present: Recovering & Stable
- Loss back to ~4.7 and continuing to fall
- Training running cleanly, no further restarts
- Speed: ~516s/50 steps, VRAM: 25.8 GB

---

## 4. Errors & Fixes

### Error 1 — NaN in eval_checkpoint.py on CPU
**When:** First time running evaluation script
**Symptom:** All pooled embeddings returned `nan`, similarity matrix all NaN
**Root cause:** Vision encoder saved in bfloat16 during training. Eval script loaded with deprecated `torch_dtype=` parameter which didn't reliably convert all state to float32.
**Fix:**
```python
# Wrong (old):
vision = AutoModel.from_pretrained(checkpoint, torch_dtype=torch.float32)
# Fixed:
vision = AutoModel.from_pretrained(checkpoint, dtype=torch.float32)
pixel_values = pixel_values.float()  # explicit cast
```

---

### Error 2 — pool_metadata garbage values (major bug)
**When:** All evaluation runs using `from_pretrained()`
**Symptom:** NaN in first transformer block during inference (not during training)
**Root cause:**
```python
# In encoder.py — pool_metadata registered as NON-persistent
self.register_buffer("pool_metadata", pool_meta, persistent=False)
```
`persistent=False` means the buffer is **never saved to checkpoints**. When `from_pretrained()` loads a checkpoint:
1. `__init__` correctly sets `pool_metadata = [0.5, 0.5, 1.0, 1.0]`
2. HuggingFace loading flow corrupts it to garbage values (e.g. `6.1e+25, 4.4e+21`)
3. RoPE receives these extreme positions → NaN in attention

**Impact on training: ZERO.** Training always creates a fresh model in `__init__` first (correct values), then loads weights. `pool_metadata` is a constant — it never changes during training. All 28,000 steps of learned weights are valid.

**Fix applied (2026-05-11):**
```python
# encoder.py — changed to persistent=True
self.register_buffer("pool_metadata", pool_meta, persistent=True)

# All eval scripts — manual reset for backwards compatibility with old checkpoints:
vision.encoder.pool_metadata.data[:] = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
```
Files fixed: `encoder.py`, `eval_checkpoint.py`, `eval_retrieval_cc3m.py`, `visualize_saliency.py`

---

### Error 3 — predict.py device mismatch
**When:** Running VLM inference with `predict.py`
**Symptom:** `RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same`
**Root cause:** Vision encoder loaded in float32 but projector cast to float16 on CUDA
**Fix:** Load vision encoder with `load_dtype` (float16 on CUDA, float32 on CPU) so all components use the same dtype

---

### Error 4 — predict.py empty output
**When:** Running inference without a trained projector
**Symptom:** LLM outputs EOS immediately → `skip_special_tokens=True` strips it → empty string
**Root cause:** Untrained projector sends garbage embeddings to LLM; LLM collapses to EOS
**Fix:**
```python
out_ids = llm.generate(
    ...,
    min_new_tokens=5,              # force at least 5 tokens
    attention_mask=attention_mask, # was missing, caused attention issues
)
```
Added informative message: output is expected to be nonsense until Phase 2 (projector alignment) is complete.

---

### Error 5 — Batch size increase made training slower
**When:** Attempting to speed up training by increasing batch 256 → 640
**Symptom:** Speed went from 512s → 696s per 50 steps (37% slower)
**Root cause:** GPU was already fully compute-bound. Larger batch increased memory transfers without improving GPU utilization. The bottleneck is data loading (CPU→GPU), not compute.
**Fix:** Reverted to batch=256, grad_accum=4. Disabled gradient checkpointing instead (saves compute without changing batch dynamics).

---

### Error 6 — Multiple restarts corrupted optimizer state
**When:** Restarting training 3–4 times from checkpoint-28000 to test batch sizes
**Symptom:** Loss spiked from ~4.3 → 8.2 after final restart
**Root cause:** Each restart from the same checkpoint with different batch sizes disrupted optimizer momentum (Adam's first and second moment estimates)
**Fix:** Let training run naturally — loss recovered to ~4.77 within 600 steps. No intervention needed. **Lesson: avoid unnecessary restarts.**

---

## 5. Evaluation Results So Far

### 5.1 Dummy Image Retrieval (checkpoint-28000)
Script: `scripts/eval_checkpoint.py`
- 4 synthetic solid-color images vs 4 color captions
- **Retrieval accuracy: 1/4 (25%)** — same as random for 4 items
- All cosine similarities negative (−0.01 to −0.10) → embedding spaces not yet aligned
- Token budget: Global=13, Region=64, Detail=179 ✓

### 5.2 Real CC3M Retrieval (checkpoint-28000)
Script: `scripts/eval_retrieval_cc3m.py`
- 200 real image-caption pairs from CC3M training tars

| Metric | Image→Text | Text→Image | Random Baseline |
|---|---|---|---|
| **R@1** | **1.0%** | 0.5% | 0.50% |
| **R@5** | **3.5%** | 3.5% | 2.50% |
| **R@10** | **6.0%** | 5.5% | 5.00% |

- R@1 is **2× above random** → model is learning alignment, not junk
- Expected at full training: R@1 ~15–35%

### 5.3 Saliency Visualization (checkpoint-28000)
Script: `scripts/visualize_saliency.py`
- Images: cat, dog, horse from `test_images/`
- Output figures in: `outputs/saliency_viz/`

| Image | Saliency Mean | Saliency Std | Assessment |
|---|---|---|---|
| cat | 0.453 | 0.498 | Near-random, some texture focus |
| dog | 0.469 | 0.499 | Slight outline of dog — earliest learning signal |
| horse | 0.484 | 0.500 | Nearly uniform — no clear subject focus |

- Saliency mean ~0.5 = still mostly random/uniform
- Token placement not yet content-adaptive — nearly uniform grid
- **Expected at 75k steps:** Saliency clearly focuses on foreground subjects; token clusters visibly concentrate on salient regions

---

## 6. Current Training State

```
Step:        28,600 / 75,000  (38.1% complete)
Loss:        ~4.77 (recovering from restart spike, trending down)
Primary:     ~2.37
HCT:         ~2.36
Saliency:    ~4.00
LR:          7.21e-04 (cosine decay from 1e-3)
Speed:       ~516s per 50 steps  (~10.3s/step)
VRAM:        ~25.8 GB / 32 GB
ETA:         ~133 hours (~5.5 days) from step 28,600
Grad. ckpt:  OFF (disabled for speed)
Batch:       256 per device, grad_accum=4 (effective batch=1024)
```

**Loss trend (overall):**
```
Step  5,000 → loss ~8.7   (random start)
Step 10,000 → loss ~6.0
Step 15,000 → loss ~5.5
Step 20,000 → loss ~5.1
Step 25,000 → loss ~4.8
Step 28,000 → loss ~4.3   (pre-restart)
Step 28,600 → loss ~4.77  (post-restart, recovering)
```

---

## 7. Future Roadmap

### Phase 1 — Complete Vision Encoder Pre-training
- [ ] Run to step 75,000 (~5.5 days remaining)
- [ ] Run `eval_retrieval_cc3m.py` at step 50,000 — log R@1 improvement
- [ ] Run `visualize_saliency.py` at step 75,000 — should show clear subject focus
- [ ] Run `eval_retrieval_cc3m.py` at step 75,000 — this is your paper's Phase 1 number
- [ ] **Target:** R@1 ≥ 15% on CC3M subset at step 75,000

### Phase 2 — Projector Alignment (LLM bridge)
- [ ] Create `configs/projector_align.yaml`
- [ ] Create `scripts/train_projector.py`
- [ ] Train `LevelAwareProjector` (384 → 2048) on CC3M caption pairs
- [ ] Verify with `predict.py` — LLM output should produce coherent sentences
- [ ] **Goal:** Vision encoder embeddings projected into LLM token space

### Phase 3 — Scaling & Benchmarks
- [ ] Run Flickr30K retrieval benchmark (R@1, R@5, R@10) — standard paper metric
- [ ] Run MSCOCO retrieval benchmark
- [ ] Benchmark inference latency (TTFT) — target < FastVLM's reported speed
- [ ] Add higher resolution bucket: 448×448
- [ ] **Target benchmarks:** Flickr30K R@1 ≥ 50%, COCO R@1 ≥ 35%

### Phase 4 — VLM Downstream Evaluation
- [ ] Run MMBench — general VLM capability
- [ ] Run POPE — hallucination benchmark
- [ ] Run TextVQA — fine detail reading (ATB's strongest use case)
- [ ] Ablation: disable ATB (fixed grid) → measure accuracy drop
- [ ] Ablation: disable HCT loss → measure accuracy drop
- [ ] **These ablations prove your paper's claims**

### Phase 5 — Paper
- [ ] Ablation table: ATB on/off, HCT on/off, budget 64/128/256/512
- [ ] Figure 1: Saliency map showing adaptive token placement (from visualize_saliency.py at 75k)
- [ ] Figure 2: Architecture diagram (ATB pipeline)
- [ ] Table: Parameter efficiency vs FastVLM, LLaVA-1.5, CLIP
- [ ] Paper title recommendation: **"Keural: Content-Adaptive Vision Encoding via Saliency-Guided Token Budgets"**

---

## 8. Paper Positioning vs FastVLM

| Dimension | FastVLM (Apple 2025) | Keural |
|---|---|---|
| Token reduction | Structural — stride-32, fixed count | Semantic — saliency-guided, adaptive |
| Token types | Uniform patches | Global / Region / Detail (3 levels) |
| Content awareness | None | Full — budget follows image content |
| Model size | ~125M params | ~24.7M params |
| Open source | No | Yes |

**Key claim:** FastVLM achieves efficiency through structural token reduction — fewer tokens by design. Keural achieves efficiency through semantic token allocation — the right tokens, where it matters.

**Strongest differentiator:** A plain background and a detailed scene get the same token budget in FastVLM. In Keural, the model detects where complexity is and concentrates tokens there. This is directly measurable in the TextVQA benchmark (fine-detail reading).

**Benchmark targets to beat or match FastVLM:**

| Benchmark | FastVLM | Keural Target |
|---|---|---|
| COCO Retrieval R@1 | ~75% | ≥ 70% |
| MMBench | ~73% | ≥ 68% |
| POPE F1 | ~86% | ≥ 84% |
| TextVQA | ~57% | ≥ 55% |

---

*Last updated: 2026-05-11 | Step 28,600 / 75,000 | Training active*
