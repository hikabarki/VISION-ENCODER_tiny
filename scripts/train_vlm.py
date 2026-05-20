"""
Keural Phase 2B — Instruction Fine-tuning (VLM).

Trains the full VLM on LLaVA-Instruct-150K conversations.
Architecture: frozen vision encoder → trainable projector → mistralai/Mistral-7B-Instruct-v0.3 (4-bit QLoRA).
Loss: next-token prediction on assistant responses only (human turns masked with -100).

Phase 2B differences from Phase 2A:
  - Data: LLaVA-Instruct-150K conversations (not raw CC3M captions)
  - Projector: loaded from Phase 2A, continues training (not frozen)
  - LoRA: expanded to q_proj, k_proj, v_proj, o_proj (4 modules)
  - LR: lower (2e-5 vs 1e-4) — projector already aligned
  - Multi-turn: human/assistant alternation, loss on assistant only

Usage:
    cd /home/mkd-s002/vision-encoder/poc
    python3.10 scripts/train_vlm.py \\
        --vision-ckpt outputs/run/checkpoint-75000 \\
        --projector-ckpt outputs/projector/checkpoint-10000/projector.pt \\
        --output-dir outputs/vlm \\
        --coco-dir data/coco/train2017 \\
        --llava-json data/llava_instruct/llava_instruct_150k.json

    # Resume from latest checkpoint:
    python3.10 scripts/train_vlm.py ... --resume

    # Quick sanity check (5 steps):
    python3.10 scripts/train_vlm.py ... --max-steps 5 --batch-size 1 --grad-accum 1
"""

from __future__ import annotations

import sys, os, json, math, time, random, shutil, argparse
sys.path.insert(0, "/home/mkd-s002/vision-encoder/poc")

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

import alignment.hf_wrapper
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from alignment.projectors import LevelAwareProjector


# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Phase 2B instruction fine-tuning")
parser.add_argument("--vision-ckpt", default="outputs/run/checkpoint-75000",
                    help="Path to Phase 1 vision encoder checkpoint (frozen)")
parser.add_argument("--projector-ckpt", default="outputs/projector/checkpoint-10000/projector.pt",
                    help="Path to Phase 2A projector weights (.pt file)")
parser.add_argument("--llm-name", default="mistralai/Mistral-7B-Instruct-v0.3")
parser.add_argument("--output-dir", default="outputs/vlm")
parser.add_argument("--coco-dir", default="data/coco/train2017",
                    help="Directory containing COCO train2017 images")
parser.add_argument("--llava-json", default="data/llava_instruct/llava_instruct_150k.json",
                    help="Path to LLaVA-Instruct-150K JSON file")
parser.add_argument("--max-steps", type=int, default=5000)
parser.add_argument("--batch-size", type=int, default=2,
                    help="Images per micro-batch (before grad accumulation)")
parser.add_argument("--grad-accum", type=int, default=16,
                    help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
parser.add_argument("--lr", type=float, default=2e-5,
                    help="Peak learning rate (lower than Phase 2A — projector already aligned)")
parser.add_argument("--warmup-steps", type=int, default=300)
parser.add_argument("--weight-decay", type=float, default=0.05)
parser.add_argument("--save-every", type=int, default=500)
parser.add_argument("--keep-checkpoints", type=int, default=2,
                    help="Number of most recent checkpoints to keep")
parser.add_argument("--log-every", type=int, default=10)
parser.add_argument("--token-budget", type=int, default=256)
parser.add_argument("--max-text-length", type=int, default=512,
                    help="Max tokenized conversation length (in tokens)")
parser.add_argument("--hidden-dim", type=int, default=2048,
                    help="LevelAwareProjector hidden dim (must match Phase 2A)")
parser.add_argument("--lora-r", type=int, default=16)
parser.add_argument("--lora-alpha", type=int, default=32)
parser.add_argument("--lora-targets", nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "o_proj"],
                    help="LoRA target modules (expanded vs Phase 2A for instruction following)")
parser.add_argument("--resume", action="store_true",
                    help="Resume from latest checkpoint in --output-dir")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type != "cuda":
    print("WARNING: training on CPU will be extremely slow. Use CUDA for Phase 2B.")

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── LR schedule (cosine with linear warmup) ───────────────────────────────────
def cosine_warmup_lr(step: int, peak_lr: float, warmup_steps: int,
                     max_steps: int, min_lr_ratio: float = 0.05) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * min_lr_ratio + peak_lr * (1.0 - min_lr_ratio) * cosine


# ── load frozen vision encoder ────────────────────────────────────────────────
print(f"\nLoading vision encoder: {args.vision_ckpt}")
vision = AutoModel.from_pretrained(args.vision_ckpt, dtype=torch.bfloat16)
vision = vision.to(device).eval()
# Reinitialize non-persistent buffers corrupted by from_pretrained(dtype=bfloat16)
vision.encoder.pool_metadata.data[:] = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
for block in vision.encoder.blocks:
    rope = block.attn.rope
    i = torch.arange(0, rope.half_dim, 2, dtype=torch.float32)
    rope.inv_freq.data.copy_(1.0 / (rope.base ** (i / rope.half_dim)))
for p in vision.parameters():
    p.requires_grad_(False)
n_vis = sum(p.numel() for p in vision.parameters())
print(f"  {n_vis:,} params — all frozen")


# ── load tokenizer ─────────────────────────────────────────────────────────────
print(f"\nLoading tokenizer: {args.llm_name}")
tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# ── load LLM in 4-bit NF4 + LoRA ─────────────────────────────────────────────
print(f"\nLoading LLM: {args.llm_name}")
print("  Quantization: 4-bit NF4 double-quant (bfloat16 compute)")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
llm = AutoModelForCausalLM.from_pretrained(
    args.llm_name,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
llm = prepare_model_for_kbit_training(llm)

lora_cfg = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=args.lora_targets,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
llm = get_peft_model(llm, lora_cfg)
llm.print_trainable_parameters()


# ── build projector and load Phase 2A weights ─────────────────────────────────
print(f"\nBuilding LevelAwareProjector: 384 → {args.hidden_dim} → 4096")
projector = LevelAwareProjector(
    encoder_dim=384,
    llm_dim=4096,
    hidden_dim=args.hidden_dim,
).to(device).to(torch.bfloat16)

if os.path.exists(args.projector_ckpt):
    projector.load_state_dict(torch.load(args.projector_ckpt, map_location=device))
    print(f"  Loaded Phase 2A weights from: {args.projector_ckpt}")
else:
    print(f"  WARNING: projector checkpoint not found at {args.projector_ckpt}")
    print("  Starting projector from random init (Phase 2A must finish first)")

n_proj = sum(p.numel() for p in projector.parameters())
print(f"  {n_proj:,} params ({n_proj/1e6:.1f}M) — trainable (continues from Phase 2A)")


# ── optimizer over projector + LoRA params ────────────────────────────────────
lora_params = [p for p in llm.parameters() if p.requires_grad]
all_trainable = list(projector.parameters()) + lora_params
optimizer = torch.optim.AdamW(
    all_trainable,
    lr=args.lr,
    weight_decay=args.weight_decay,
    betas=(0.9, 0.999),
)
print(f"\nOptimizer: AdamW, lr={args.lr}, wd={args.weight_decay}")
print(f"  Trainable params: {sum(p.numel() for p in all_trainable):,}")


# ── resume from checkpoint ────────────────────────────────────────────────────
start_step = 0
if args.resume:
    ckpts = sorted(output_dir.glob("checkpoint-[0-9]*"),
                   key=lambda p: int(p.name.split("-")[-1]))
    if ckpts:
        latest = ckpts[-1]
        resume_step = int(latest.name.split("-")[-1])
        print(f"\nResuming from {latest.name} (step {resume_step})")
        proj_path  = latest / "projector.pt"
        state_path = latest / "training_state.pt"
        lora_path  = latest / "lora_adapter"
        if proj_path.exists():
            projector.load_state_dict(torch.load(proj_path, map_location=device))
            print("  Projector weights loaded")
        if lora_path.exists():
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file as load_safetensors
            adapter_file = lora_path / "adapter_model.safetensors"
            if adapter_file.exists():
                lora_state = load_safetensors(str(adapter_file))
                set_peft_model_state_dict(llm, lora_state)
                print("  LoRA adapter weights loaded")
            else:
                print("  WARNING: lora_adapter folder found but no adapter_model.safetensors")
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu")
            optimizer.load_state_dict(state["optimizer"])
            start_step = state["step"] + 1
            print(f"  Optimizer state loaded. Resuming at step {start_step}")
    else:
        print("  No checkpoint found — starting from scratch")


# ── LLaVA dataset ─────────────────────────────────────────────────────────────
HUMAN_PREFIX = "USER: "
ASST_PREFIX = " ASSISTANT: "

def format_and_mask(conversations: list[dict], max_length: int) -> tuple[list[int], list[int]]:
    """
    Build (input_ids, labels) for a multi-turn conversation.
    Human turns → labels = -100 (masked).
    Assistant turns → labels = real token IDs (loss computed here).
    Returns Python lists (no padding — done in collate).
    """
    all_ids: list[int] = []
    all_labels: list[int] = []

    bos_id = tokenizer.bos_token_id
    if bos_id is not None:
        all_ids.append(bos_id)
        all_labels.append(-100)

    i = 0
    while i < len(conversations):
        turn = conversations[i]
        role = turn["from"]
        text = turn["value"].replace("<image>", "").strip()

        if role in ("human", "user"):
            # Human turn + ASSISTANT: prefix — masked entirely
            human_str = HUMAN_PREFIX + text + ASST_PREFIX
            human_ids = tokenizer(human_str, add_special_tokens=False).input_ids
            all_ids.extend(human_ids)
            all_labels.extend([-100] * len(human_ids))

        else:  # gpt / assistant
            # Assistant response — compute loss on these tokens
            asst_ids = tokenizer(text, add_special_tokens=False).input_ids
            eos_id = tokenizer.eos_token_id
            if eos_id is not None:
                asst_ids = asst_ids + [eos_id]
            all_ids.extend(asst_ids)
            all_labels.extend(asst_ids)

        i += 1

    all_ids = all_ids[:max_length]
    all_labels = all_labels[:max_length]
    return all_ids, all_labels


print(f"\nLoading LLaVA-Instruct dataset: {args.llava_json}")
with open(args.llava_json, "r") as f:
    llava_data = json.load(f)

coco_dir = Path(args.coco_dir)
valid_samples = []
missing = 0
for sample in llava_data:
    img_name = sample.get("image", "")
    img_path = coco_dir / img_name
    if img_path.exists() and len(sample.get("conversations", [])) >= 2:
        valid_samples.append((str(img_path), sample["conversations"]))
    else:
        missing += 1

print(f"  Total samples: {len(llava_data):,}")
print(f"  Valid (image found + has conversations): {len(valid_samples):,}")
if missing > 0:
    print(f"  Skipped (image missing or no conversations): {missing:,}")
if len(valid_samples) == 0:
    print("ERROR: No valid samples. Check --coco-dir and --llava-json paths.")
    sys.exit(1)


def sample_stream():
    """Infinite shuffled generator of (img_path, conversations)."""
    indices = list(range(len(valid_samples)))
    while True:
        random.shuffle(indices)
        for idx in indices:
            yield valid_samples[idx]


def load_image(img_path: str) -> torch.Tensor | None:
    try:
        img = Image.open(img_path).convert("RGB")
        return transform(img)
    except Exception:
        return None


def pull_batch(stream, n: int):
    """Pull n valid (pixel_tensor, input_ids, labels) triples from the stream."""
    items = []
    for img_path, conversations in stream:
        pv = load_image(img_path)
        if pv is None:
            continue
        input_ids, labels = format_and_mask(conversations, args.max_text_length)
        # Skip if labels are all -100 (no assistant tokens to learn from)
        if all(l == -100 for l in labels):
            continue
        items.append((pv, input_ids, labels))
        if len(items) >= n:
            break
    return items


def collate(items: list, embed_device: torch.device) -> dict:
    """Pad text sequences to max length in batch, prepend visual tokens."""
    pixel_values = torch.stack([x[0] for x in items]).to(device, dtype=torch.bfloat16)
    max_t = max(len(x[1]) for x in items)
    pad_id = tokenizer.pad_token_id or 0

    padded_ids, padded_labels, text_masks = [], [], []
    for _, ids, labs in items:
        pad_len = max_t - len(ids)
        padded_ids.append(ids + [pad_id] * pad_len)
        padded_labels.append(labs + [-100] * pad_len)
        text_masks.append([1] * len(ids) + [0] * pad_len)

    input_ids = torch.tensor(padded_ids, dtype=torch.long, device=embed_device)
    labels_text = torch.tensor(padded_labels, dtype=torch.long, device=embed_device)
    attn_mask = torch.tensor(text_masks, dtype=torch.long, device=embed_device)

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "labels_text": labels_text,
        "attn_mask": attn_mask,
    }


# ── checkpoint save / rotate ──────────────────────────────────────────────────
def save_checkpoint(step: int) -> None:
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(projector.state_dict(), ckpt_dir / "projector.pt")
    llm.save_pretrained(ckpt_dir / "lora_adapter")
    torch.save({"step": step, "optimizer": optimizer.state_dict()},
               ckpt_dir / "training_state.pt")
    print(f"[train] Saved checkpoint-{step} to {ckpt_dir}")
    all_ckpts = sorted(output_dir.glob("checkpoint-[0-9]*"),
                       key=lambda p: int(p.name.split("-")[-1]))
    for old in all_ckpts[:-args.keep_checkpoints]:
        shutil.rmtree(old, ignore_errors=True)
        print(f"[train] Removed old checkpoint {old.name}")


# ── training loop ─────────────────────────────────────────────────────────────
eff_batch = args.batch_size * args.grad_accum
print(f"\nStarting Phase 2B training:")
print(f"  Steps: {args.max_steps} | Micro-batch: {args.batch_size} | Grad-accum: {args.grad_accum} | Eff-batch: {eff_batch}")
print(f"  LR: {args.lr} | Warmup: {args.warmup_steps} | Token budget: {args.token_budget}")
print(f"  LoRA targets: {args.lora_targets}")

embed_fn = llm.get_input_embeddings()
embed_device = next(embed_fn.parameters()).device

data_stream = sample_stream()
log_path = output_dir / "training_log.jsonl"

projector.train()
llm.train()
optimizer.zero_grad(set_to_none=True)

accum_count = 0
accum_loss = 0.0
step = start_step
start_time = time.time()

try:
    while step < args.max_steps:
        items = pull_batch(data_stream, args.batch_size)
        if not items:
            continue

        batch = collate(items, embed_device)
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        labels_text = batch["labels_text"]
        attn_mask = batch["attn_mask"]

        B = pixel_values.shape[0]

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            # Vision encoder: frozen
            with torch.no_grad():
                vis_out = vision(pixel_values, token_budget=args.token_budget)

            # Projector: trainable
            visual_embeds = projector(vis_out)  # (B, N_vis, 4096)
            N_vis = visual_embeds.shape[1]

            # Text embeddings
            with torch.no_grad():
                text_embeds = embed_fn(input_ids).to(torch.bfloat16)  # (B, T, 4096)

            T_len = input_ids.shape[1]

            # Concatenate: [visual tokens | conversation tokens]
            inputs_embeds = torch.cat(
                [visual_embeds.to(embed_device), text_embeds], dim=1
            )  # (B, N_vis+T, 4096)

            # Attention mask: all visual tokens + non-padding text tokens
            vis_attn = torch.ones(B, N_vis, dtype=torch.long, device=embed_device)
            full_attn = torch.cat([vis_attn, attn_mask], dim=1)

            # Labels: -100 for visual positions + human turns; real IDs for assistant turns
            vis_labels = torch.full((B, N_vis), -100, dtype=torch.long, device=embed_device)
            labels = torch.cat([vis_labels, labels_text], dim=1)

            outputs = llm(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attn,
                labels=labels,
            )
            loss = outputs.loss / args.grad_accum

        loss.backward()
        accum_loss += loss.item()
        accum_count += 1

        if accum_count >= args.grad_accum:
            grad_norm = nn.utils.clip_grad_norm_(all_trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0

            current_lr = cosine_warmup_lr(
                step, args.lr, args.warmup_steps, args.max_steps
            )
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            if step % args.log_every == 0:
                elapsed = time.time() - start_time
                steps_done = max(step - start_step + 1, 1)
                secs_per_step = elapsed / steps_done
                eta_secs = int(secs_per_step * (args.max_steps - step))
                eta_h, eta_m = divmod(eta_secs // 60, 60)
                pct = step / args.max_steps * 100
                line = (
                    f"[train] step={step:5d}/{args.max_steps} ({pct:.1f}%) "
                    f"ETA={eta_h}h{eta_m:02d}m  "
                    f"lr={current_lr:.2e}  "
                    f"loss={accum_loss:.4f}  "
                    f"grad_norm={float(grad_norm):.2e}"
                )
                print(line)
                with log_path.open("a") as f:
                    f.write(json.dumps({
                        "step": step, "loss": accum_loss,
                        "lr": current_lr, "grad_norm": float(grad_norm),
                    }) + "\n")

            accum_loss = 0.0

            if step > 0 and step % args.save_every == 0:
                save_checkpoint(step)

            step += 1

except KeyboardInterrupt:
    print(f"\n[train] Interrupted at step {step} — saving checkpoint...")
    save_checkpoint(step)
    print("[train] Checkpoint saved. Exiting.")
    sys.exit(0)

print(f"\n[train] Phase 2B complete — {step} steps")
save_checkpoint(step)
print(f"\nCheckpoints saved to: {output_dir}")
print(f"  Projector:    checkpoint-{step}/projector.pt")
print(f"  LoRA adapter: checkpoint-{step}/lora_adapter/")
print(f"\nNext: python3.10 scripts/demo_vlm.py to test the full pipeline")
