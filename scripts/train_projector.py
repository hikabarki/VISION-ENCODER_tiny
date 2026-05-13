"""
Keural Phase 2 — Projector Alignment Training.

Trains the LevelAwareProjector (384 → 2048 → 4096) that bridges the frozen
Keural vision encoder and keural-14.8b-base. The LLM is loaded in 4-bit NF4
(QLoRA) and remains frozen except for lightweight LoRA adapters on the
attention layers. Only the projector and LoRA weights train.

Architecture during Phase 2:
    Image (256×256)
        ↓  Vision Encoder (frozen, 24.7M params)
    256 tokens × 384-dim  (KeuralEncoderOutput)
        ↓  LevelAwareProjector (trains here, ~18M params)
    256 visual tokens × 4096-dim
        ↓  keural-14.8b-base (4-bit NF4 + LoRA, ~7.4GB)
    "Caption: <text>"

Usage:
    cd /home/mkd-s002/vision-encoder/poc
    python3.10 scripts/train_projector.py \\
        --vision-ckpt outputs/run/checkpoint-75000 \\
        --output-dir outputs/projector \\
        --tar-dir data/cc3m

    # Resume from a checkpoint:
    python3.10 scripts/train_projector.py ... --resume

    # Quick sanity check (5 steps, small batch):
    python3.10 scripts/train_projector.py ... --max-steps 5 --batch-size 2 --grad-accum 1
"""

from __future__ import annotations

import sys, os, io, tarfile, glob, argparse, json, math, time, random, shutil
sys.path.insert(0, "/home/mkd-s002/vision-encoder/poc")

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

import alignment.hf_wrapper  # registers KeuralVisionConfig/Model with AutoModel
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from alignment.projectors import LevelAwareProjector


# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Phase 2 projector alignment training")
parser.add_argument("--vision-ckpt", default="outputs/run/checkpoint-75000",
                    help="Path to trained Phase 1 vision encoder checkpoint")
parser.add_argument("--llm-name", default="mkd-hossain/keural-14.8b-base",
                    help="HF repo ID or local path to the LLM")
parser.add_argument("--output-dir", default="outputs/projector")
parser.add_argument("--tar-dir", default="data/cc3m",
                    help="Directory containing CC3M tar shards (*.tar)")
parser.add_argument("--max-steps", type=int, default=10000)
parser.add_argument("--batch-size", type=int, default=4,
                    help="Images per micro-batch (before grad accumulation)")
parser.add_argument("--grad-accum", type=int, default=8,
                    help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--warmup-steps", type=int, default=500)
parser.add_argument("--weight-decay", type=float, default=0.05)
parser.add_argument("--save-every", type=int, default=1000)
parser.add_argument("--log-every", type=int, default=10)
parser.add_argument("--token-budget", type=int, default=256,
                    help="Vision token budget passed to the encoder")
parser.add_argument("--max-caption-length", type=int, default=128,
                    help="Max tokenized caption length (in tokens)")
parser.add_argument("--hidden-dim", type=int, default=2048,
                    help="Projector MLP hidden dimension")
parser.add_argument("--lora-r", type=int, default=16)
parser.add_argument("--lora-alpha", type=int, default=32)
parser.add_argument("--lora-targets", nargs="+", default=["q_proj", "v_proj"],
                    help="LoRA target module names in the LLM")
parser.add_argument("--resume", action="store_true",
                    help="Resume from latest checkpoint in --output-dir")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type != "cuda":
    print("WARNING: training on CPU will be extremely slow. Use CUDA for Phase 2.")

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
# pool_metadata is non-persistent in older checkpoints — reset to global-pool default
vision.encoder.pool_metadata.data[:] = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
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
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
llm = get_peft_model(llm, lora_cfg)
llm.print_trainable_parameters()


# ── build projector ───────────────────────────────────────────────────────────
print(f"\nBuilding LevelAwareProjector: 384 → {args.hidden_dim} → 4096")
projector = LevelAwareProjector(
    encoder_dim=384,
    llm_dim=4096,
    hidden_dim=args.hidden_dim,
).to(device).to(torch.bfloat16)
n_proj = sum(p.numel() for p in projector.parameters())
print(f"  {n_proj:,} params ({n_proj/1e6:.1f}M) — all trainable")


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
        proj_path = latest / "projector.pt"
        state_path = latest / "training_state.pt"
        if proj_path.exists():
            projector.load_state_dict(torch.load(proj_path, map_location=device))
            print("  Projector weights loaded")
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu")
            optimizer.load_state_dict(state["optimizer"])
            start_step = state["step"] + 1
            print(f"  Optimizer state loaded. Resuming at step {start_step}")
    else:
        print("  No checkpoint found — starting from scratch")


# ── CC3M data stream ──────────────────────────────────────────────────────────
def cc3m_stream(tar_dir: str):
    """Infinite shuffled generator of (PIL.Image, caption) from CC3M tar shards."""
    tar_files = sorted(glob.glob(os.path.join(tar_dir, "*.tar")))
    if not tar_files:
        raise RuntimeError(f"No .tar files found in {tar_dir}")
    print(f"\nData: {len(tar_files)} tar shards in {tar_dir}")
    while True:
        random.shuffle(tar_files)
        for tar_path in tar_files:
            try:
                with tarfile.open(tar_path) as tar:
                    members = {m.name: m for m in tar.getmembers()}
                    stems = [n[:-4] for n in members if n.endswith(".jpg")]
                    random.shuffle(stems)
                    for stem in stems:
                        jpg_key = stem + ".jpg"
                        txt_key = stem + ".txt"
                        if jpg_key not in members or txt_key not in members:
                            continue
                        try:
                            img_bytes = tar.extractfile(members[jpg_key]).read()
                            cap_bytes = tar.extractfile(members[txt_key]).read()
                            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                            caption = cap_bytes.decode("utf-8", errors="replace").strip()
                            if caption:
                                yield img, caption
                        except Exception:
                            continue
            except Exception as e:
                print(f"  Warning: skipping {os.path.basename(tar_path)}: {e}")
                continue


def pull_batch(stream, n: int):
    """Pull n (image, caption) pairs from the stream."""
    images, captions = [], []
    for img, cap in stream:
        images.append(img)
        captions.append(cap)
        if len(images) >= n:
            break
    return images, captions


# ── checkpoint save / rotate ──────────────────────────────────────────────────
def save_checkpoint(step: int) -> None:
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(exist_ok=True)
    # Projector weights
    torch.save(projector.state_dict(), ckpt_dir / "projector.pt")
    # LoRA adapter (safe to load back with PeftModel.from_pretrained)
    llm.save_pretrained(ckpt_dir / "lora_adapter")
    # Optimizer state for resume
    torch.save({"step": step, "optimizer": optimizer.state_dict()},
               ckpt_dir / "training_state.pt")
    print(f"[train] Saved checkpoint-{step} to {ckpt_dir}")
    # Keep only the 3 most recent checkpoints
    all_ckpts = sorted(output_dir.glob("checkpoint-[0-9]*"),
                       key=lambda p: int(p.name.split("-")[-1]))
    for old in all_ckpts[:-3]:
        shutil.rmtree(old, ignore_errors=True)
        print(f"[train] Removed old checkpoint {old.name}")


# ── training loop ─────────────────────────────────────────────────────────────
eff_batch = args.batch_size * args.grad_accum
print(f"\nStarting training:")
print(f"  Steps: {args.max_steps} | Micro-batch: {args.batch_size} | Grad-accum: {args.grad_accum} | Eff-batch: {eff_batch}")
print(f"  LR: {args.lr} | Warmup: {args.warmup_steps} steps | Token budget: {args.token_budget}")

# The embedding table lives inside the LLM; get_input_embeddings() is device-aware
embed_fn = llm.get_input_embeddings()

data_stream = cc3m_stream(args.tar_dir)
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
        images, captions = pull_batch(data_stream, args.batch_size)
        pixel_values = torch.stack([transform(img) for img in images]).to(
            device, dtype=torch.bfloat16
        )

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            # Vision encoder: frozen — no gradient tracking needed
            with torch.no_grad():
                vis_out = vision(pixel_values, token_budget=args.token_budget)

            # Projector: trainable — grad flows back through these weights
            visual_embeds = projector(vis_out)  # (B, N_vis, 4096)

            # Tokenize captions — format: "Caption: <text>"
            enc = tokenizer(
                ["Caption: " + c for c in captions],
                padding=True,
                truncation=True,
                max_length=args.max_caption_length,
                return_tensors="pt",
                add_special_tokens=True,
            )
            # Place on the embedding layer's device (handles device_map splits)
            embed_device = next(embed_fn.parameters()).device
            input_ids = enc.input_ids.to(embed_device)
            attn_mask = enc.attention_mask.to(embed_device)

            B, N_vis = visual_embeds.shape[:2]
            T = input_ids.shape[1]

            # Text embeddings: no grad — embedding table stays frozen
            with torch.no_grad():
                text_embeds = embed_fn(input_ids).to(torch.bfloat16)  # (B, T, 4096)

            # Concatenate: [visual tokens | caption tokens]
            inputs_embeds = torch.cat(
                [visual_embeds.to(embed_device), text_embeds], dim=1
            )  # (B, N_vis+T, 4096)

            # Attention mask: attend to all visual + non-padding caption tokens
            vis_attn = torch.ones(B, N_vis, dtype=torch.long, device=embed_device)
            full_attn = torch.cat([vis_attn, attn_mask], dim=1)  # (B, N_vis+T)

            # Labels: -100 for visual positions; real IDs for caption (padding masked)
            text_labels = input_ids.clone()
            text_labels[attn_mask == 0] = -100
            vis_labels = torch.full(
                (B, N_vis), -100, dtype=torch.long, device=embed_device
            )
            labels = torch.cat([vis_labels, text_labels], dim=1)  # (B, N_vis+T)

            # LLM forward — loss computed over caption tokens only
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

# Final save
print(f"\n[train] Training complete — {step} steps")
save_checkpoint(step)
print(f"\nCheckpoints saved to: {output_dir}")
print(f"  projector weights:  checkpoint-{step}/projector.pt")
print(f"  LoRA adapter:       checkpoint-{step}/lora_adapter/")
print(f"\nNext: build scripts/demo_vlm.py to test the full pipeline")
