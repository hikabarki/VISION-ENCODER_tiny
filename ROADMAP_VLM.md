# Keural VLM — Full Alignment Roadmap

**Goal:** Image → KeuralVisionEncoder → LevelAwareProjector → keural-alpha → Text
**Current state:** Pipeline wired ✅ | Vision encoder 8% trained | Projector untrained | LLM not aligned

---

## Overview

```
PHASE 1  Vision encoder training          ██░░░░░░░░  8%    → May 16-17
PHASE 2  Projector alignment              ░░░░░░░░░░  0%    → May 18-19
PHASE 3  LLM instruction fine-tuning      ░░░░░░░░░░  0%    → May 20-24
PHASE 4  Full VLM evaluation & release    ░░░░░░░░░░  0%    → May 25
```

---

## Phase 1 — Finish Vision Encoder Training
**Status:** In progress | **ETA:** ~May 16-17, 2026

### What to do
- [ ] Resume training: `bash scripts/restart_training.sh`
- [ ] Let it run to step 75,000
- [ ] Apply ETA formula fix at next clean restart
- [ ] Monitor loss — should reach ~3.0–4.0 total by step 75,000

### Success criteria
- Total loss < 4.5 at step 75,000
- Primary (SigLIP) loss < 2.5
- HCT loss < 2.5
- Saliency loss stable ~4.0

### Expected loss trajectory
| Step | Total Loss |
|------|------------|
| 6,000 (now) | ~6.75 |
| 20,000 | ~4.5–5.0 |
| 50,000 | ~3.5–4.0 |
| 75,000 | ~3.0–3.5 |

### On completion
```bash
# Run full benchmark evaluation
python3.10 scripts/eval_checkpoint.py --checkpoint outputs/run/checkpoint-75000

# Upload final weights to HuggingFace
python3.10 -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='outputs/run/checkpoint-75000',
    repo_id='mkd-hika/keural-vision-encoder-poc',
    token='YOUR_HF_TOKEN'
)
"
```

---

## Phase 2 — Projector Alignment Training
**Status:** Not started | **ETA:** ~1–2 days | **Starts after:** Phase 1

### What it does
Trains `LevelAwareProjector` (384 → 2048) to map vision tokens into keural-alpha's
embedding space. The LLM is **fully frozen** during this phase — only the projector trains.

### Architecture
```
Image → [Frozen Vision Encoder] → tokens (B, 256, 384)
                                      ↓
                              [LevelAwareProjector]  ← trains here
                                      ↓
                              projected (B, 256, 2048)
                                      ↓
                           [Frozen keural-alpha] → caption loss
```

### Training data needed
~500K–1M image-caption pairs. Options:
- CC3M (already have) — 3M pairs, use a subset
- CC12M (already have) — 12M pairs, use a subset
- LAION-400M subset (download separately)

**Recommended:** use existing CC3M + CC12M data (no new downloads needed)

### Config (create: `configs/projector_align.yaml`)
```yaml
training:
  lr: 1.0e-4
  max_steps: 10000
  batch_size_per_device: 128
  gradient_accumulation: 8        # effective batch = 1024
  warmup_steps: 500
  optimizer: adamw
  weight_decay: 0.05
  precision: bfloat16

model:
  vision_ckpt: outputs/run/checkpoint-75000
  llm_name: mkd-ai/keural-alpha
  projector: level_aware           # LevelAwareProjector
  freeze_vision: true              # frozen
  freeze_llm: true                 # frozen

loss:
  type: next_token_prediction      # standard causal LM loss on captions
```

### Expected outcome
- Projector learns to translate visual features into LLM-compatible space
- LLM starts producing image-relevant captions (basic, not instruction-tuned yet)

### Estimated time
~1 day on RTX 5090

---

## Phase 3 — LLM Instruction Fine-tuning (VLM Alignment)
**Status:** Not started | **ETA:** ~3–5 days | **Starts after:** Phase 2

### What it does
Fine-tunes keural-alpha with LoRA on visual instruction-following data.
The projector continues training; vision encoder remains frozen.

### Architecture
```
Image → [Frozen Vision Encoder] → [Trainable Projector] → vision tokens
                                                               ↓
                                              [keural-alpha + LoRA]  ← trains here
                                                               ↓
                                                      instruction response
```

### Training data needed
Visual instruction-following datasets:
- **LLaVA-Instruct-150K** — 150K GPT-4 generated VQA pairs (download from HF)
- **VQAv2** subset — visual question answering
- **COCO captions** — 600K captions

```bash
# Download LLaVA instruction data
from datasets import load_dataset
ds = load_dataset("liuhaotian/LLaVA-Instruct-150K")
```

### LoRA config
```python
from peft import LoraConfig
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value", "dense"],  # GPT-NeoX attention
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Training settings
```yaml
training:
  lr: 2.0e-5
  max_steps: 5000
  batch_size_per_device: 16
  gradient_accumulation: 8
  warmup_steps: 100
  precision: bfloat16
  freeze_vision: true
  freeze_llm: false         # LoRA active
  lora_r: 16
  lora_alpha: 32
```

### Estimated time
~3–5 days on RTX 5090

### Expected outcome
- Model answers questions about images
- Follows instructions: "What color is the car?", "Describe the scene", "Count the objects"

---

## Phase 4 — Evaluation & Release
**Status:** Not started | **ETA:** ~1 day | **Starts after:** Phase 3

### Benchmarks to run
| Benchmark | Target | Tool |
|-----------|--------|------|
| Flickr30K i2t R@1 | >50% | `scripts/eval_checkpoint.py` |
| COCO i2t R@1 | >30% | `scripts/eval_checkpoint.py` |
| VQAv2 accuracy | >50% | standard VQA eval |
| GQA | >40% | standard GQA eval |

### Release checklist
- [ ] Upload final VLM to HuggingFace (`mkd-hika/keural-vlm`)
- [ ] Update RESULTS.md with all benchmark numbers
- [ ] Push projector weights to HF
- [ ] Write VLM usage guide
- [ ] Update GitHub README

---

## Full Timeline

| Date | Milestone |
|------|-----------|
| May 8 (today) | Pipeline test passed ✅ Vision encoder 8% |
| May 16-17 | Vision encoder training complete (step 75,000) |
| May 17 | Upload final encoder to HuggingFace |
| May 18-19 | Projector alignment training (~10,000 steps) |
| May 20-24 | LLM instruction fine-tuning (LoRA, ~5,000 steps) |
| May 25 | Full VLM evaluation + release |

---

## File Structure When Complete

```
poc/
├── outputs/
│   ├── run/checkpoint-75000/     ← final vision encoder
│   ├── projector/checkpoint/     ← trained projector
│   └── vlm/checkpoint/           ← full VLM (encoder + projector + LoRA)
├── configs/
│   ├── keural_tiny_poc.yaml      ← Phase 1 config ✅
│   ├── projector_align.yaml      ← Phase 2 config (to create)
│   └── vlm_instruct.yaml         ← Phase 3 config (to create)
└── scripts/
    ├── train.py                  ← Phase 1 ✅
    ├── train_projector.py        ← Phase 2 (to create)
    ├── train_vlm.py              ← Phase 3 (to create)
    └── eval_vlm.py               ← Phase 4 (to create)
```

---

## What to Build Next (per phase)

### After Phase 1 (build before May 17)
- `scripts/train_projector.py` — projector-only training loop
- `configs/projector_align.yaml` — projector training config

### After Phase 2 (build before May 20)
- `scripts/train_vlm.py` — full VLM fine-tuning with LoRA
- `configs/vlm_instruct.yaml` — instruction tuning config
- Download LLaVA-Instruct-150K dataset

### After Phase 3 (build before May 25)
- `scripts/eval_vlm.py` — full benchmark evaluation
- `scripts/demo_vlm.py` — interactive demo (ask questions about any image)

---

*Created: 2026-05-08 | Current phase: 1 of 4*
