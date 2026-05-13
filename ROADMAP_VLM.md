# Keural VLM — Phase 2 Plan & Roadmap

**Goal:** Image → Keural Vision Encoder → Projector → keural-14.8b-base → Text response  
**Demo target:** June 10, 2026 — POC presentation  
**Last updated:** 2026-05-13

---

## Current Status

```
PHASE 1  Vision encoder training (CC3M+CC12M)    ████████░░  58%   → ~May 16–17
PHASE 2  Projector training (frozen LLM)         ░░░░░░░░░░   0%   → ~May 17–27
PHASE 3  Document understanding (Docling data)   ░░░░░░░░░░   0%   → After June 10
```

---

## Phase 1 — Vision Encoder Training (In Progress)

**Status:** 58.7% complete | **ETA:** ~May 16–17, 2026

### Current metrics (checkpoint-44000)

| Metric | Value | Random baseline |
|---|---|---|
| Step | 44,000 / 75,000 | — |
| Primary loss (SigLIP) | ~1.97 | — |
| CC3M R@1 i2t (200 pairs) | 1.0% | 0.50% |
| CC3M R@1 t2i (200 pairs) | 2.0% | 0.50% |
| CC3M R@5 i2t (200 pairs) | 4.0% | 2.50% |
| CC3M MedR i2t (200 pairs) | 97.0 | 100.5 |
| Pooled norm | ~4.48 | target ~1.0 |

### Success criteria at step 75,000
- Primary loss < 1.5
- CC3M R@1 (200 pairs) > 8%
- CC3M MedR < 70
- Pooled norm < 2.0
- Saliency maps show clear foreground focus

### On completion
```bash
# Full retrieval eval
cd /home/mkd-s002/vision-encoder/poc
python3.10 scripts/eval_retrieval_cc3m.py --checkpoint outputs/run/checkpoint-75000 --n-pairs 200

# Saliency visualization
python3.10 scripts/visualize_saliency.py --checkpoint outputs/run/checkpoint-75000 --image-dir test_images --out-dir outputs/saliency_viz_75k

# Push to HuggingFace
# (use HF token)
```

---

## Phase 2 — Projector Training (Starts ~May 17)

**Status:** Not started | **ETA:** ~10 days | **Target:** May 27

### Architecture

```
Image (256×256)
      ↓
Keural Vision Encoder — FROZEN (24.7M params)
      ↓
256 tokens × 384-dim
      ↓
Projector MLP — TRAINS HERE
  Linear(384 → 2048) → GELU → Linear(2048 → 4096)
      ↓
256 visual tokens × 4096-dim  (Mixtral hidden dim)
      ↓
keural-14.8b-base — FROZEN + 4-bit QLoRA
      ↓
Text response
```

### VRAM plan (RTX 5090, 32GB)

| Component | Full bfloat16 | 4-bit QLoRA |
|---|---|---|
| keural-14.8b-base | ~29.6 GB | ~7.4 GB |
| Vision encoder | ~50 MB | ~50 MB |
| Projector | ~50 MB | ~50 MB |
| Activations + gradients | ~2 GB | ~4 GB |
| **Total** | **~32 GB (too tight)** | **~12 GB (safe)** |

**Must use 4-bit QLoRA** — full bfloat16 does not fit.

### Training data
Use existing CC3M data (no new downloads needed):
- 500K image-caption pairs from CC3M tars in `data/cc3m/`
- Format: `(image, "Caption: <text>")`
- Simple next-token prediction loss on the caption

### Config (`configs/projector_align.yaml`)
```yaml
training:
  lr: 1.0e-4
  max_steps: 10000
  batch_size_per_device: 32
  gradient_accumulation: 8        # effective batch = 256
  warmup_steps: 500
  optimizer: adamw
  weight_decay: 0.05
  precision: bfloat16

model:
  vision_ckpt: outputs/run/checkpoint-75000
  llm_name: mkd-hossain/keural-14.8b-base
  llm_quantization: 4bit           # QLoRA — required for 32GB GPU
  freeze_vision: true
  freeze_llm: true
  lora_r: 16
  lora_alpha: 32
  lora_target_modules: ["q_proj", "v_proj"]

projector:
  type: mlp2x
  input_dim: 384
  hidden_dim: 2048
  output_dim: 4096                 # Mixtral hidden dim

loss:
  type: next_token_prediction
```

### Scripts to build (before May 17)
- [ ] `scripts/train_projector.py` — projector training loop
- [ ] `configs/projector_align.yaml` — config file

### Expected outcome
- Model generates basic image captions
- "What is this?" → "This is a cat sitting on a sofa"
- Not instruction-tuned yet — raw caption generation

### Estimated time
~7–10 days on RTX 5090 (10,000 steps × ~60s/step)

---

## Phase 2B — Instruction Fine-tuning (Starts ~May 27)

**Status:** Not started | **ETA:** ~5–7 days | **Target:** June 3–5

### What changes
- Projector continues training (not frozen)
- LLM gets LoRA adapters for instruction following
- Training data changes to VQA / instruction format

### Training data
```bash
# Download LLaVA instruction data from HuggingFace
from datasets import load_dataset
ds = load_dataset("liuhaotian/LLaVA-Instruct-150K")
```

### Format
```
User: <image> What is in this image?
Assistant: This image shows a dog running in a park...

User: <image> How many people are there?
Assistant: There are 3 people in the image...
```

### LoRA config
```python
from peft import LoraConfig
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Expected outcome after Phase 2B
- Answers questions about images
- "What color is the car?" → "The car is red"
- "Describe the scene" → coherent paragraph
- "How many objects?" → approximate count

---

## Phase 3 — Document Understanding (After June 10)

**Status:** Planned | **Depends on:** Docling dataset from RAG team

### What this phase adds
Train on Docling-processed document images so Keural can understand:
- PDF pages, research papers, reports
- Tables, charts, figures
- Dense text regions

### Data pipeline
```
Docling processed dataset (from RAG team)
      ↓
Document page image + extracted text pairs
      ↓
Fine-tune vision encoder + projector on document data
      ↓
Keural handles both natural scenes AND documents
```

### Why ATB is well-suited for documents
- Detail tokens (179/256) naturally focus on dense text regions
- Region tokens capture paragraph/section blocks
- Saliency learns text vs whitespace distinction

---

## Full Timeline

| Date | Milestone | Status |
|---|---|---|
| May 8 | Phase 1 started | Done |
| May 13 | checkpoint-44000, eval metrics updated | Done |
| May 16–17 | Phase 1 complete (step 75,000) | Pending |
| May 17 | Push checkpoint-75000 to HuggingFace | Pending |
| May 17–18 | Build projector training script | Pending |
| May 18–27 | Phase 2 projector training (10k steps) | Pending |
| May 27–Jun 3 | Phase 2B instruction fine-tuning | Pending |
| Jun 3–5 | Integration testing, demo preparation | Pending |
| **Jun 10** | **POC demo presentation** | **Target** |
| Post Jun 10 | Phase 3 document understanding (Docling) | Planned |

---

## POC Demo Plan (June 10)

### What to show

1. **Saliency visualization** — show ATB tokens focusing on foreground objects (not random patches)
2. **Retrieval metrics** — R@1, R@5, MedR on 200 CC3M pairs vs random baseline
3. **Live VLM response** — upload image, ask question, get answer from keural-14.8b-base
4. **Architecture diagram** — CNN Stem → ATB Tokenizer → Transformer → Projector → LLM

### Demo script (simple)
```python
# Upload image → ask question → get response
image = load_image("demo.jpg")
question = "What is in this image?"
response = keural_vlm(image, question)
print(response)
```

### Key POC claim
> "24.7M parameter vision encoder with content-adaptive tokenization,
> trained from scratch, connected to a 14.8B LLM via a lightweight projector.
> Demonstrates image-text alignment learning and visual question answering."

---

## File Structure Target

```
poc/
├── outputs/
│   ├── run/checkpoint-75000/        ← Phase 1 final (May 16–17)
│   ├── projector/checkpoint/        ← Phase 2 projector weights
│   └── vlm/checkpoint/              ← Phase 2B full VLM
├── configs/
│   ├── keural_tiny_poc.yaml         ← Phase 1 ✅
│   ├── projector_align.yaml         ← Phase 2 (to create)
│   └── vlm_instruct.yaml            ← Phase 2B (to create)
└── scripts/
    ├── train.py                     ← Phase 1 ✅
    ├── eval_checkpoint.py           ← Phase 1 eval ✅
    ├── eval_retrieval_cc3m.py       ← Phase 1 eval ✅
    ├── visualize_saliency.py        ← Phase 1 viz ✅
    ├── gui_eval.py                  ← GUI tool ✅
    ├── train_projector.py           ← Phase 2 (to build)
    ├── train_vlm.py                 ← Phase 2B (to build)
    └── demo_vlm.py                  ← June 10 demo (to build)
```

---

*Created: 2026-05-08 | Updated: 2026-05-13 | Phase: 1 of 3*
