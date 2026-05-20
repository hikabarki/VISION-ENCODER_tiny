"""
Keural VLM — June 10 Demo Script.

Loads the full pipeline (vision encoder → projector → keural-14.8b-base)
and answers questions about images.

Modes:
  Single question:
    python3.10 scripts/demo_vlm.py --image demo.jpg --question "What is in this image?"

  Interactive (keep asking about the same image):
    python3.10 scripts/demo_vlm.py --image demo.jpg --interactive

  Batch (multiple images, one question each):
    python3.10 scripts/demo_vlm.py --image-dir test_images/ --question "Describe this image."
"""

from __future__ import annotations

import sys, os, argparse, glob
sys.path.insert(0, "/home/mkd-s002/vision-encoder/poc")

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

import alignment.hf_wrapper
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from alignment.projectors import LevelAwareProjector


# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Keural VLM demo")
parser.add_argument("--vision-ckpt", default="outputs/run/checkpoint-75000",
                    help="Phase 1 vision encoder checkpoint (frozen)")
parser.add_argument("--projector-ckpt", default="outputs/vlm/checkpoint-5000/projector.pt",
                    help="Phase 2B projector weights (.pt)")
parser.add_argument("--lora-ckpt", default="outputs/vlm/checkpoint-5000/lora_adapter",
                    help="Phase 2B LoRA adapter directory")
parser.add_argument("--llm-name", default="mistralai/Mistral-7B-Instruct-v0.3")
parser.add_argument("--image", default=None, help="Path to a single image")
parser.add_argument("--image-dir", default=None, help="Directory of images (batch mode)")
parser.add_argument("--question", default="What is in this image?")
parser.add_argument("--interactive", action="store_true",
                    help="Interactive mode — keep asking questions about the same image")
parser.add_argument("--max-new-tokens", type=int, default=256)
parser.add_argument("--token-budget", type=int, default=256)
parser.add_argument("--hidden-dim", type=int, default=2048,
                    help="Projector hidden dim (must match training)")
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top-p", type=float, default=0.9)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dtype = torch.bfloat16

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

HUMAN_PREFIX = "USER: "
ASST_PREFIX  = " ASSISTANT:"


# ── model loading ─────────────────────────────────────────────────────────────
def load_models():
    print("=" * 55)
    print("  Keural VLM — Loading models")
    print("=" * 55)

    print(f"\n[1/4] Vision encoder: {args.vision_ckpt}")
    vision = AutoModel.from_pretrained(args.vision_ckpt, dtype=load_dtype)
    vision = vision.to(device).eval()
    vision.encoder.pool_metadata.data[:] = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
    for block in vision.encoder.blocks:
        rope = block.attn.rope
        i = torch.arange(0, rope.half_dim, 2, dtype=torch.float32)
        rope.inv_freq.data.copy_(1.0 / (rope.base ** (i / rope.half_dim)))
    for p in vision.parameters():
        p.requires_grad_(False)
    print(f"  {sum(p.numel() for p in vision.parameters()):,} params — frozen")

    print(f"\n[2/4] Tokenizer: {args.llm_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"\n[3/4] LLM: {args.llm_name} (4-bit NF4)")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=load_dtype,
        bnb_4bit_use_double_quant=True,
    )
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_name,
        quantization_config=bnb_config,
        torch_dtype=load_dtype,
        device_map="auto",
    )
    if os.path.isdir(args.lora_ckpt):
        print(f"  Loading LoRA adapter: {args.lora_ckpt}")
        llm = PeftModel.from_pretrained(llm, args.lora_ckpt)
    else:
        print(f"  WARNING: LoRA adapter not found at {args.lora_ckpt}")
        print("  Running without LoRA (Phase 2B not complete yet)")
    llm.eval()

    print(f"\n[4/4] Projector: {args.projector_ckpt}")
    projector = LevelAwareProjector(
        encoder_dim=384,
        llm_dim=4096,
        hidden_dim=args.hidden_dim,
    ).to(device).to(load_dtype)
    if os.path.exists(args.projector_ckpt):
        projector.load_state_dict(torch.load(args.projector_ckpt, map_location=device))
        print(f"  {sum(p.numel() for p in projector.parameters()):,} params — loaded")
    else:
        print(f"  WARNING: projector not found at {args.projector_ckpt}")
        print("  Using random projector (Phase 2A not complete yet)")
    projector.eval()

    print("\n  All models loaded.\n")
    return vision, tokenizer, llm, projector


# ── encode one image to visual embeddings ─────────────────────────────────────
@torch.no_grad()
def encode_image(vision, projector, pil_img: Image.Image) -> torch.Tensor:
    pv = transform(pil_img).unsqueeze(0).to(device, dtype=load_dtype)
    with torch.autocast(device_type=device.type, dtype=load_dtype,
                        enabled=(device.type == "cuda")):
        vis_out = vision(pv, token_budget=args.token_budget)
        visual_embeds = projector(vis_out)  # (1, N_vis, 4096)
    return visual_embeds


# ── generate one answer ────────────────────────────────────────────────────────
@torch.no_grad()
def generate(vision, tokenizer, llm, projector,
             pil_img: Image.Image, question: str) -> str:

    visual_embeds = encode_image(vision, projector, pil_img)
    embed_fn = llm.get_input_embeddings()
    embed_device = next(embed_fn.parameters()).device

    prompt = HUMAN_PREFIX + question + ASST_PREFIX
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc.input_ids.to(embed_device)
    attn_mask = enc.attention_mask.to(embed_device)

    with torch.autocast(device_type=device.type, dtype=load_dtype,
                        enabled=(device.type == "cuda")):
        text_embeds = embed_fn(input_ids).to(load_dtype)

    B, N_vis = 1, visual_embeds.shape[1]
    T_len = input_ids.shape[1]

    inputs_embeds = torch.cat(
        [visual_embeds.to(embed_device), text_embeds], dim=1
    )
    vis_attn = torch.ones(B, N_vis, dtype=torch.long, device=embed_device)
    full_attn = torch.cat([vis_attn, attn_mask], dim=1)

    output_ids = llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=full_attn,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=(args.temperature > 0),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Strip the prompt echo if the model repeats it
    if ASST_PREFIX.strip() in response:
        response = response.split(ASST_PREFIX.strip())[-1].strip()
    return response


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    vision, tokenizer, llm, projector = load_models()

    # ── single image + interactive ─────────────────────────────────────────────
    if args.image:
        pil_img = Image.open(args.image).convert("RGB")
        print(f"Image: {args.image}  ({pil_img.size[0]}×{pil_img.size[1]})")
        print("-" * 55)

        if args.interactive:
            print("Interactive mode — type a question or 'quit' to exit.\n")
            while True:
                try:
                    question = input("Question: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not question or question.lower() in ("quit", "exit", "q"):
                    break
                print("Answer:", generate(vision, tokenizer, llm, projector, pil_img, question))
                print()
        else:
            print(f"Question: {args.question}")
            answer = generate(vision, tokenizer, llm, projector, pil_img, args.question)
            print(f"Answer:   {answer}")

    # ── batch mode ─────────────────────────────────────────────────────────────
    elif args.image_dir:
        img_paths = sorted(
            glob.glob(os.path.join(args.image_dir, "*.jpg")) +
            glob.glob(os.path.join(args.image_dir, "*.png")) +
            glob.glob(os.path.join(args.image_dir, "*.jpeg"))
        )
        if not img_paths:
            print(f"No images found in {args.image_dir}")
            sys.exit(1)

        print(f"Batch mode — {len(img_paths)} images | Question: {args.question}")
        print("=" * 55)
        for img_path in img_paths:
            try:
                pil_img = Image.open(img_path).convert("RGB")
                answer = generate(vision, tokenizer, llm, projector, pil_img, args.question)
                print(f"\n[{os.path.basename(img_path)}]")
                print(f"  {answer}")
            except Exception as e:
                print(f"\n[{os.path.basename(img_path)}] ERROR: {e}")

    else:
        print("Provide --image or --image-dir.")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
