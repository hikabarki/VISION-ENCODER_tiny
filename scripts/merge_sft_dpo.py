"""
Merge SFT LoRA into Mistral-7B base, then stack DPO LoRA on top,
and save the result as a single standalone LoRA adapter.

After running this, the GUI only needs to load ONE LoRA (outputs/merged/dpo_ready)
on top of base Mistral-7B — no stacking required.

Usage:
    python3.10 scripts/merge_sft_dpo.py

Output:
    outputs/merged/dpo_ready/   ← single LoRA adapter, load directly on base Mistral
"""

import sys, os, torch
sys.path.insert(0, "/home/mkd-s002/vision-encoder/poc")

SFT_LORA  = "outputs/vlm/checkpoint-30000/lora_adapter"
DPO_LORA  = "outputs/dpo/final"
LLM_NAME  = "mistralai/Mistral-7B-Instruct-v0.3"
OUT_DIR   = "outputs/merged/dpo_ready"

os.makedirs(OUT_DIR, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── Step 1: Load base model in bfloat16 (NO quantization — needed for merge) ──
print("\n[1/5] Loading Mistral-7B in bfloat16 (no quantization)...")
print("      This needs ~14GB RAM/VRAM. Using CPU offload if needed.")
base = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
print(f"  Done. Device map: {base.hf_device_map if hasattr(base, 'hf_device_map') else 'auto'}")

# ── Step 2: Load SFT LoRA ─────────────────────────────────────────────────────
print(f"\n[2/5] Loading SFT LoRA from {SFT_LORA}...")
model = PeftModel.from_pretrained(base, SFT_LORA, local_files_only=True)
print("  SFT LoRA loaded.")

# ── Step 3: Merge SFT LoRA into base weights ──────────────────────────────────
print("\n[3/5] Merging SFT LoRA into base weights...")
model = model.merge_and_unload()
print("  Merged. Model is now plain Mistral-7B + SFT baked in.")

# ── Step 4: Load DPO LoRA on top of merged model ─────────────────────────────
print(f"\n[4/5] Loading DPO LoRA on top from {DPO_LORA}...")
model = PeftModel.from_pretrained(model, DPO_LORA, local_files_only=True)
print("  DPO LoRA loaded.")

# ── Step 5: Save DPO LoRA adapter only ───────────────────────────────────────
print(f"\n[5/5] Saving merged DPO adapter to {OUT_DIR}...")
model.save_pretrained(OUT_DIR)

tok = AutoTokenizer.from_pretrained(LLM_NAME)
tok.save_pretrained(OUT_DIR)

print(f"\n✅ Done! Saved to {OUT_DIR}")
print("   In the GUI, use:")
print(f"   --lora-ckpt {OUT_DIR}")
print("   (no DPO stacking needed — SFT is already baked in)")
