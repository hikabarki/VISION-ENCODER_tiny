"""
Keural VLM inference — image + prompt → text.

Usage
-----
# Basic (untrained projector — output will be nonsense until Phase 2 is done):
    python predict.py \\
        --model-path outputs/run/checkpoint-28000 \\
        --image-file /path/to/image.png \\
        --prompt "Describe the image."

# With a trained projector (after Phase 2):
    python predict.py \\
        --model-path  outputs/run/checkpoint-75000 \\
        --projector   outputs/projector/checkpoint \\
        --llm         mkd-ai/keural-alpha \\
        --image-file  /path/to/image.png \\
        --prompt      "What objects are in this image?"

# Private model with HF token:
    python predict.py ... --hf-token hf_xxx
"""
import sys, argparse, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torchvision.transforms as T
from PIL import Image

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Run Keural VLM inference on a single image."
)
parser.add_argument("--model-path",  required=True,  help="Path to vision encoder checkpoint dir")
parser.add_argument("--image-file",  required=True,  help="Path to input image")
parser.add_argument("--prompt",      required=True,  help="Text prompt / question")
parser.add_argument("--projector",   default=None,   help="Path to trained projector checkpoint (optional)")
parser.add_argument("--llm",         default="mkd-ai/keural-alpha", help="LLM name or local path")
parser.add_argument("--hf-token",    default=None,   help="HuggingFace token for private models")
parser.add_argument("--budget",      type=int, default=256, help="Vision token budget (default 256)")
parser.add_argument("--max-tokens",  type=int, default=200, help="Max new tokens to generate")
parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
load_dtype = torch.float16 if device == "cuda" else torch.float32
print(f"Device: {device}  |  dtype: {load_dtype}")

# ── 1. Vision encoder ─────────────────────────────────────────────────────────
print(f"\n[1/4] Loading vision encoder from {args.model_path} ...")
import alignment.hf_wrapper
from transformers import AutoModel
vision = AutoModel.from_pretrained(
    args.model_path,
    dtype=load_dtype,         # float32 on CPU (NaN safety), float16 on CUDA
    token=args.hf_token,
).to(device).eval()
print(f"      Vision encoder ready — embed_dim={vision.encoder.embed_dim}")

# ── 2. Projector ──────────────────────────────────────────────────────────────
from alignment.projectors import LevelAwareProjector

def _load_projector(path, encoder_dim, llm_dim):
    proj = LevelAwareProjector(encoder_dim=encoder_dim, llm_dim=llm_dim)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    # support both raw state_dict and {"model": state_dict} saves
    if "model" in sd:
        sd = sd["model"]
    proj.load_state_dict(sd)
    return proj

# Peek at the LLM config to get its hidden size (needed to size the projector)
print(f"\n[2/4] Resolving LLM hidden dim from {args.llm} ...")
from transformers import AutoConfig
try:
    llm_cfg = AutoConfig.from_pretrained(args.llm, token=args.hf_token)
    # GPT-NeoX uses `hidden_size`; LLaMA / Mistral use `hidden_size` too
    llm_dim = getattr(llm_cfg, "hidden_size", None) or getattr(llm_cfg, "d_model", 2048)
except Exception:
    llm_dim = 2048
    print(f"      Could not read LLM config — defaulting llm_dim={llm_dim}")
print(f"      llm_dim={llm_dim}")

encoder_dim = vision.encoder.embed_dim  # 384
if args.projector and os.path.exists(args.projector):
    print(f"      Loading projector from {args.projector} ...")
    projector = _load_projector(args.projector, encoder_dim, llm_dim)
    print(f"      Projector loaded.")
else:
    if args.projector:
        print(f"      WARNING: projector path not found ({args.projector})")
    print("      Using randomly initialized projector — output will be nonsense.")
    print("      Run Phase 2 (train_projector.py) to get a trained projector.")
    projector = LevelAwareProjector(encoder_dim=encoder_dim, llm_dim=llm_dim)

projector = projector.to(device).to(load_dtype).eval()

# ── 3. LLM ────────────────────────────────────────────────────────────────────
print(f"\n[3/4] Loading LLM ({args.llm}) ...")
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(args.llm, token=args.hf_token)
llm = AutoModelForCausalLM.from_pretrained(
    args.llm,
    token=args.hf_token,
    dtype=load_dtype,
).to(device).eval()

# GPT-NeoX style: llm.gpt_neox.embed_in
# LLaMA / Mistral:  llm.model.embed_tokens
if hasattr(llm, "gpt_neox"):
    embed_layer = llm.gpt_neox.embed_in
elif hasattr(llm, "model") and hasattr(llm.model, "embed_tokens"):
    embed_layer = llm.model.embed_tokens
else:
    # fallback: try get_input_embeddings()
    embed_layer = llm.get_input_embeddings()
print(f"      LLM ready — embed layer: {type(embed_layer).__name__}")

# ── 4. Run pipeline ───────────────────────────────────────────────────────────
print(f"\n[4/4] Running inference ...")

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open(args.image_file).convert("RGB")
pixel_values = transform(img).unsqueeze(0).to(dtype=load_dtype, device=device)  # (1, 3, 256, 256)

with torch.no_grad():
    # Step A: vision encoder → (1, budget, 384)
    vision_out = vision(pixel_values, token_budget=args.budget)
    print(f"      Vision tokens: {vision_out.tokens.shape} "
          f"(global={( vision_out.level_ids==0).sum().item()} "
          f"region={(vision_out.level_ids==1).sum().item()} "
          f"detail={(vision_out.level_ids==2).sum().item()})")

    # Step B: projector → (1, budget, llm_dim)
    vision_tokens = projector(vision_out).to(load_dtype)

    # Step C: tokenize prompt → LLM embeddings
    text_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    text_embeds = embed_layer(text_ids)  # (1, text_len, llm_dim)

    # Step D: [vision | text] → generate
    inputs_embeds = torch.cat([vision_tokens, text_embeds], dim=1)
    attention_mask = torch.ones(1, inputs_embeds.shape[1], device=device, dtype=torch.long)

    do_sample = args.temperature > 0
    out_ids = llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=args.max_tokens,
        min_new_tokens=5,
        do_sample=do_sample,
        temperature=args.temperature if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(out_ids[0], skip_special_tokens=True)

print(f"\n{'='*60}")
print(f"PROMPT : {args.prompt}")
if response.strip():
    print(f"OUTPUT : {response}")
else:
    print("OUTPUT : (empty — projector is untrained, LLM output collapsed to EOS)")
    print("         This is expected until Phase 2 (projector alignment) is complete.")
print(f"{'='*60}")
