"""
Phase 2B DPO Alignment — Keural VLM
Dataset: openbmb/RLHF-V-Dataset (5,733 pairs)
Base:    checkpoint-30000 (LoRA) + Mistral-7B-Instruct-v0.3
Tested:  trl==1.4.0
"""

import sys, os, json, argparse, subprocess, time, warnings
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore", message=".*PEFT for a second time.*")
warnings.filterwarnings("ignore", message=".*multiple adapters.*")
warnings.filterwarnings("ignore", message=".*use_reentrant.*")
warnings.filterwarnings("ignore", message=".*pad_token.*")
warnings.filterwarnings("ignore", message=".*PAD/BOS/EOS.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, "/home/mkd-s002/vision-encoder/poc")

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig
from transformers import TrainerCallback

import alignment.hf_wrapper
from transformers import AutoModel
from alignment.projectors import LevelAwareProjector

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--vision-ckpt",    default="outputs/run/final")
parser.add_argument("--projector-ckpt", default="outputs/vlm/checkpoint-30000/projector.pt")
parser.add_argument("--sft-lora-ckpt",  default="outputs/vlm/checkpoint-30000/lora_adapter")
parser.add_argument("--llm-name",       default="mistralai/Mistral-7B-Instruct-v0.3")
parser.add_argument("--output-dir",     default="outputs/dpo")
parser.add_argument("--token-budget",   type=int, default=256)
parser.add_argument("--hidden-dim",     type=int, default=2048)
parser.add_argument("--max-steps",      type=int, default=3000)
parser.add_argument("--lr",             type=float, default=5e-6)
parser.add_argument("--batch-size",     type=int, default=1)
parser.add_argument("--grad-accum",     type=int, default=16)
parser.add_argument("--beta",           type=float, default=0.1)
parser.add_argument("--max-length",     type=int, default=512)
parser.add_argument("--save-steps",     type=int, default=500)
parser.add_argument("--log-steps",      type=int, default=10)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(args.output_dir)
OUT.mkdir(parents=True, exist_ok=True)


def gpu_stats():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip().split(",")
        return {"gpu_util": int(out[0].strip()), "gpu_mem_mb": int(out[1].strip()),
                "gpu_temp_c": int(out[3].strip())}
    except Exception:
        return {"gpu_util": -1, "gpu_mem_mb": -1, "gpu_temp_c": -1}


# ── load vision encoder (frozen) ──────────────────────────────────────────────
print("\n[1/4] Loading vision encoder...")
encoder = AutoModel.from_pretrained(
    str(Path(args.vision_ckpt).resolve()),
    trust_remote_code=True, torch_dtype=torch.bfloat16
).to(device).eval()
for mod in encoder.modules():
    if hasattr(mod, "inv_freq"):
        hd = mod.inv_freq.shape[0] * 2
        mod.inv_freq = (1.0 / (10000.0 ** (
            torch.arange(0, hd, 2, dtype=torch.float32) / hd))).to(device)
for p in encoder.parameters():
    p.requires_grad = False
print(f"  {sum(p.numel() for p in encoder.parameters()):,} params — frozen")

# ── load projector (frozen) ───────────────────────────────────────────────────
print("\n[2/4] Loading projector...")
projector = LevelAwareProjector(encoder_dim=384, hidden_dim=args.hidden_dim, llm_dim=4096)
projector.load_state_dict(torch.load(args.projector_ckpt, map_location=device))
projector = projector.to(device, dtype=torch.bfloat16).eval()
for p in projector.parameters():
    p.requires_grad = False

# ── load tokenizer & LLM ─────────────────────────────────────────────────────
print(f"\n[3/4] Loading tokenizer & LLM: {args.llm_name}")
tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"
)
base_llm = AutoModelForCausalLM.from_pretrained(
    args.llm_name, quantization_config=bnb_cfg,
    device_map="auto", torch_dtype=torch.bfloat16
)

# ── load SFT LoRA then add DPO LoRA ──────────────────────────────────────────
print(f"\n[4/4] Loading SFT LoRA from {args.sft_lora_ckpt}...")
model = PeftModel.from_pretrained(base_llm, args.sft_lora_ckpt, is_trainable=True)

dpo_lora_cfg = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, dpo_lora_cfg)
model.print_trainable_parameters()

# ── build dataset (prompt / chosen / rejected strings) ───────────────────────
print("\n[Dataset] Loading RLHF-V...")
raw = load_dataset("openbmb/RLHF-V-Dataset", split="train")
print(f"  Loaded {len(raw)} pairs")

def preprocess(example):
    text = json.loads(example["text"])
    question = text.get("question", "Describe this image.")
    chosen   = text.get("chosen",   "")
    rejected = text.get("rejected", "")
    prompt   = f"[INST] {question} [/INST]"
    return {
        "prompt":   prompt,
        "chosen":   chosen,
        "rejected": rejected,
    }

dataset = raw.map(preprocess, remove_columns=raw.column_names, desc="Building DPO pairs")
print(f"  Ready: {len(dataset)} pairs")

# ── DPO config ────────────────────────────────────────────────────────────────
dpo_config = DPOConfig(
    output_dir=str(OUT),
    max_steps=args.max_steps,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    beta=args.beta,
    logging_steps=args.log_steps,
    save_steps=args.save_steps,
    save_total_limit=3,
    bf16=True,
    report_to="none",
    disable_tqdm=True,
    dataloader_num_workers=2,
    remove_unused_columns=False,
    max_length=args.max_length,
)

# ── JSONL logger callback ─────────────────────────────────────────────────────
log_path = OUT / "dpo_log.jsonl"

class DPOLogCallback(TrainerCallback):
    def __init__(self, max_steps, log_path):
        self.max_steps  = max_steps
        self.log_path   = log_path
        self.start_time = time.time()

    def on_log(self, arguments, state, control, logs=None, **kwargs):
        if not logs:
            return
        step      = state.global_step
        elapsed_s = time.time() - self.start_time
        pct       = step / self.max_steps * 100 if self.max_steps > 0 else 0
        sps       = step / elapsed_s if elapsed_s > 0 else 0
        remaining = (self.max_steps - step) / sps if sps > 0 else 0
        eta_str   = str(timedelta(seconds=int(remaining)))
        now_str   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        g         = gpu_stats()
        lr        = logs.get("learning_rate", 0)

        entry = {
            "timestamp":          now_str,
            "step":               step,
            "pct":                round(pct, 1),
            "loss":               logs.get("loss", 0),
            "learning_rate":      lr,
            "rewards/chosen":     logs.get("rewards/chosen", 0),
            "rewards/rejected":   logs.get("rewards/rejected", 0),
            "rewards/accuracies": logs.get("rewards/accuracies", 0),
            "rewards/margins":    logs.get("rewards/margins", 0),
            "logps/chosen":       logs.get("logps/chosen", 0),
            "logps/rejected":     logs.get("logps/rejected", 0),
            "gpu_util_pct":       g["gpu_util"],
            "gpu_mem_mb":         g["gpu_mem_mb"],
            "gpu_temp_c":         g["gpu_temp_c"],
            "elapsed_s":          round(elapsed_s, 1),
            "eta":                eta_str,
            "steps_per_sec":      round(sps, 3),
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        loss  = entry["loss"]
        r_acc = entry["rewards/accuracies"]
        r_mar = entry["rewards/margins"]
        r_cho = entry["rewards/chosen"]
        r_rej = entry["rewards/rejected"]
        print(
            f"[{now_str}] step={step}/{self.max_steps} ({pct:.1f}%) ETA={eta_str} | "
            f"loss={loss:.4f} lr={lr:.2e} | "
            f"acc={r_acc:.3f} margin={r_mar:.3f} chosen={r_cho:.3f} rejected={r_rej:.3f} | "
            f"GPU={g['gpu_util']}% VRAM={g['gpu_mem_mb']}MiB T={g['gpu_temp_c']}°C",
            flush=True
        )

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    callbacks=[DPOLogCallback(max_steps=args.max_steps, log_path=log_path)],
)

# ── train ─────────────────────────────────────────────────────────────────────
print(f"\n[DPO] Starting — {args.max_steps} steps | beta={args.beta} | lr={args.lr}")
print(f"[DPO] Output: {OUT}\n")
trainer.train()
trainer.save_model(str(OUT / "final"))
tokenizer.save_pretrained(str(OUT / "final"))
print(f"\n[DPO] Complete. Final model saved to {OUT / 'final'}")
