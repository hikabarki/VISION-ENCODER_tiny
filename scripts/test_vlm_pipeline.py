"""
Quick VLM pipeline test — checks if vision encoder + projector + LLM wire together.
Results will be random/nonsense (projector is untrained) but the pipeline should run.

Usage:
    python3.10 scripts/test_vlm_pipeline.py --hf-token YOUR_HF_TOKEN
"""
import sys, argparse
sys.path.insert(0, "/home/mkd-s002/vision-encoder/poc")

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--hf-token", required=True, help="HuggingFace token")
parser.add_argument("--vision-ckpt", default="mkd-hika/keural-vision-encoder-poc")
parser.add_argument("--llm",         default="mkd-ai/keural-alpha")
parser.add_argument("--budget",      type=int, default=64, help="token budget (64=faster)")
parser.add_argument("--prompt",      default="Describe what you see:")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── 1. Load vision encoder ────────────────────────────────────────────────────
print(f"\n[1/4] Loading vision encoder from {args.vision_ckpt}...")
import alignment.hf_wrapper
from transformers import AutoModel
vision = AutoModel.from_pretrained(
    args.vision_ckpt,
    token=args.hf_token,
).to(device).eval()
print(f"      Vision encoder loaded — 24.7M params")

# ── 2. Load projector (untrained) ─────────────────────────────────────────────
print(f"\n[2/4] Creating LevelAwareProjector (encoder=384 → llm=2048)...")
from alignment.projectors import LevelAwareProjector
projector = LevelAwareProjector(encoder_dim=384, llm_dim=2048).to(device).eval()
print(f"      Projector ready (randomly initialized — results will be nonsense)")

# ── 3. Load LLM ───────────────────────────────────────────────────────────────
print(f"\n[3/4] Loading LLM ({args.llm})...")
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(args.llm, token=args.hf_token)
llm = AutoModelForCausalLM.from_pretrained(
    args.llm,
    token=args.hf_token,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device).eval()
llm_embed = llm.gpt_neox.embed_in  # GPT-NeoX embedding layer
print(f"      LLM loaded")

# ── 4. Run VLM pipeline ───────────────────────────────────────────────────────
print(f"\n[4/4] Running VLM pipeline...")

# Dummy image (red square — no real image needed for pipeline test)
dummy_img = Image.fromarray(np.full((256, 256, 3), [180, 80, 80], dtype=np.uint8))
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
pixel_values = transform(dummy_img).unsqueeze(0).to(device)

with torch.no_grad():
    # Step A: encode image → (1, budget, 384)
    vision_out = vision(pixel_values, token_budget=args.budget)
    print(f"  Vision tokens: {vision_out.tokens.shape}  "
          f"(global={( vision_out.level_ids==0).sum().item()} "
          f"region={(vision_out.level_ids==1).sum().item()} "
          f"detail={(vision_out.level_ids==2).sum().item()})")

    # Step B: project → (1, budget, 2048)
    vision_tokens = projector(vision_out).to(torch.float16 if device=="cuda" else torch.float32)
    print(f"  Projected:     {vision_tokens.shape}")

    # Step C: tokenize prompt → get LLM embeddings
    text_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    text_embeds = llm_embed(text_ids)  # (1, text_len, 2048)
    print(f"  Text embeds:   {text_embeds.shape}  (prompt: '{args.prompt}')")

    # Step D: concatenate [vision | text] → (1, budget+text_len, 2048)
    inputs_embeds = torch.cat([vision_tokens, text_embeds], dim=1)
    print(f"  Combined:      {inputs_embeds.shape}")

    # Step E: generate
    print(f"\n  Generating (max 50 new tokens)...")
    out_ids = llm.generate(
        inputs_embeds=inputs_embeds,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(out_ids[0], skip_special_tokens=True)

print(f"\n{'='*60}")
print(f"PIPELINE OUTPUT (untrained projector — expect nonsense):")
print(f"{'='*60}")
print(response)
print(f"{'='*60}")
print(f"\nPipeline test PASSED — all components wired successfully.")
print(f"  Vision encoder → projector → LLM: OK")
print(f"  Input shape:  image (1,3,256,256) + prompt")
print(f"  Output shape: {out_ids.shape}")
