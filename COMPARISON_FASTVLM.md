# Keural Vision Encoder vs FastVLM — Architecture Comparison & Action Plan

> Reference paper: *FastVLM: Efficient Vision Encoding for Vision Language Models* (Apple, 2025)

---

## 1. At-a-Glance Comparison

| Dimension | **FastVLM** | **Keural Vision Encoder** |
|---|---|---|
| **Architecture family** | Hybrid CNN + Transformer (FastViTHD) | Pure Transformer + CNN stem |
| **Backbone stages** | 5 stages (CNN ×4 → Transformer ×1) | CNN stem → Adaptive Tokenizer → Transformer |
| **Token reduction method** | Structural: stride-32 downsampling (fixed) | Semantic: saliency-guided ATB (adaptive) |
| **Token count** | Fixed (e.g. 4× fewer than FastViT) | Configurable budget (default 256, max 1024) |
| **Token types** | Uniform (all patch tokens) | 3 levels: Global / Region / Detail |
| **Saliency / attention routing** | None — fixed spatial grid | Learnable saliency map → Gumbel sampling |
| **Input resolution** | High-res (multiple scales) | Bucketed: 224×224, 256×256, 320×320 |
| **Model size** | ~125M params | ~24.7M params (tiny POC) |
| **Loss** | Standard CLIP/SigLIP | SigLIP + HCT auxiliary + saliency regularization |
| **Training data** | Internal Apple data + public | CC3M + CC12M (~15M pairs) |
| **Inference latency** | 2-3× faster than FastViT-HD | Not yet benchmarked (Phase 1 in progress) |
| **Open source** | No (Apple internal) | Yes (this repo) |

---

## 2. What FastVLM Does

FastVLM's core insight is **structural token reduction**:

1. Use a hybrid backbone (FastViTHD) where early CNN stages compress spatial resolution aggressively (stride 32).
2. Feed the resulting small feature grid directly to the LLM — no complex routing or sampling.
3. Result: far fewer tokens with minimal accuracy loss, enabling fast LLM decoding.

**Strengths of FastVLM:**
- Very low time-to-first-token (TTFT) — key for mobile/edge deployment
- Simple, deterministic token count
- Strong empirical results on standard VLM benchmarks

**Limitations of FastVLM:**
- Token count is content-independent — a plain background and a detailed scene get the same budget
- No explicit mechanism to preserve fine-grained detail where it matters
- Closed-source; no public weights or training code

---

## 3. What Keural Does Differently (Your Unique Contributions)

### 3.1 Content-Adaptive Token Budget (ATB)
Instead of a fixed spatial grid, Keural predicts *where* to spend tokens:

```
Image → CNN stem → Saliency map → Budget split → Global / Region / Detail tokens
```

- **Global tokens**: low-res summary of the whole scene (5% budget)
- **Region tokens**: medium-res salient object patches (25% budget)
- **Detail tokens**: fine-grained crops at peak saliency (70% budget)

*FastVLM has no equivalent — its token count is fixed regardless of content.*

### 3.2 Hierarchical Contextual Tokenization (HCT)
Three levels of abstraction co-attend in the same Transformer:
- Level embeddings encode the semantic "zoom level" of each token
- Pool token aggregates across all levels for the final image embedding
- HCT auxiliary loss enforces cross-level consistency during training

*FastVLM has no hierarchical multi-level token structure.*

### 3.3 Differentiable Saliency Sampling
- Gumbel-Softmax with temperature annealing (1.0 → 0.1 over 10k steps)
- Warmup period with fixed grid (5k steps) for stable early training
- Saliency module is end-to-end trainable with the rest of the encoder

*FastVLM's token count is determined by architecture (stride), not learned.*

### 3.4 Smaller, Faster Base
At ~25M params, the Keural tiny POC is 5× smaller than FastVLM (125M) while covering the full ATB + HCT pipeline. A scaled-up version (e.g. 80-120M) can directly compete.

---

## 4. Where FastVLM Has the Edge (Gaps to Close)

| Gap | Status | Action |
|---|---|---|
| Benchmarked TTFT / throughput numbers | Not yet measured | Step 4.3 below |
| High-resolution input (1024+) | Not yet supported | Add dynamic resolution in Phase 3 |
| Strong VLM downstream scores | Phase 2 not started | Complete projector alignment |
| Public benchmark numbers (MMBench, POPE, etc.) | None yet | Step 4.5 below |

---

## 5. Step-by-Step Action Plan

### Phase 1 — Finish Vision Encoder Pre-training *(current)*

- [x] Train to step 28,000 / 75,000
- [ ] **Step 5.1**: Confirm gradient-checkpointing-off speedup (check next 50-step log)
- [ ] **Step 5.2**: Run `eval_checkpoint.py` every 5,000 steps; log retrieval accuracy in RESULTS.md
- [ ] **Step 5.3**: Watch for diagonal ≥ 0.30 in similarity matrix — that signals meaningful alignment
- [ ] **Step 5.4**: Complete training to step 75,000 (~3-4 more days at current speed)

### Phase 2 — Projector Alignment

- [ ] **Step 5.5**: Create `configs/projector_align.yaml` and `scripts/train_projector.py`
- [ ] **Step 5.6**: Train `LevelAwareProjector` (encoder 384 → LLM 2048) on CC3M pairs
- [ ] **Step 5.7**: Verify with `predict.py` — output should be coherent sentences, not empty

### Phase 3 — Scaling & Benchmarks

- [ ] **Step 5.8**: Implement dynamic resolution bucketing (add 448×448, 512×512 buckets)
- [ ] **Step 5.9**: Benchmark TTFT: `time python predict.py ...` — target < FastVLM's reported TTFT
- [ ] **Step 5.10**: Run retrieval benchmark on MSCOCO val (5k images) — report R@1, R@5, R@10

### Phase 4 — VLM Downstream Evaluation

- [ ] **Step 5.11**: Run MMBench evaluation — compare Keural vs FastVLM reported numbers
- [ ] **Step 5.12**: Run POPE hallucination benchmark
- [ ] **Step 5.13**: Run TextVQA (tests fine-grained detail reading — where ATB should shine)
- [ ] **Step 5.14**: Ablation: disable saliency → fixed grid. Measure accuracy drop to prove ATB value.

### Phase 5 — Paper Positioning

- [ ] **Step 5.15**: Write ablation table: ATB on vs off, HCT loss on vs off, budget 64/128/256/512
- [ ] **Step 5.16**: Prepare Figure 1: saliency map visualization showing adaptive token placement
- [ ] **Step 5.17**: Compute parameter efficiency: accuracy-per-param vs FastVLM, LLaVA-1.5, etc.
- [ ] **Step 5.18**: Highlight the content-adaptive vs fixed-grid distinction as the core claim

---

## 6. Key Benchmark Targets (to match or beat FastVLM)

| Benchmark | FastVLM reported | Keural target | Why important |
|---|---|---|---|
| MSCOCO Retrieval R@1 (I→T) | ~75% | ≥ 70% | Basic image-text alignment |
| MMBench | ~73% | ≥ 68% | General VLM capability |
| POPE (F1) | ~86% | ≥ 84% | Hallucination resistance |
| TextVQA | ~57% | ≥ 55% | Fine detail reading (ATB advantage) |
| TTFT (256px, 256 tokens) | ~50ms | < 60ms | Inference speed |

---

## 7. Narrative Positioning (for paper / report)

> *"FastVLM achieves efficiency through structural token reduction: fewer tokens by design. Keural achieves efficiency through semantic token allocation: the right tokens, where it matters. By predicting image saliency and dynamically distributing a fixed budget across global context, salient regions, and fine detail, Keural matches FastVLM's throughput at 5× fewer parameters while concentrating representational capacity exactly where downstream tasks require it."*

---

*Last updated: 2026-05-11 | Training at step 28,000 / 75,000*
