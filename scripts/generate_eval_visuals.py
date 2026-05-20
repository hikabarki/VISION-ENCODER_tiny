"""
Generate two evaluation visuals for HuggingFace:
  1. qualitative_examples.png  — 4 images with real model Q&A outputs
  2. token_budget_curve.png    — VQA accuracy vs token budget (64/128/256/512)

Usage:
    python3.10 scripts/generate_eval_visuals.py \
        --vision-ckpt outputs/run/final \
        --projector-ckpt outputs/vlm/checkpoint-10000/projector.pt \
        --lora-ckpt outputs/vlm/checkpoint-10000/lora_adapter \
        --output-dir outputs/eval
"""

import sys, os, argparse, textwrap, time
sys.path.insert(0, "/home/mkd-s002/vision-encoder/poc")

import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import alignment.hf_wrapper
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from alignment.projectors import LevelAwareProjector

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--vision-ckpt",    default="outputs/run/final")
parser.add_argument("--projector-ckpt", default="outputs/vlm/checkpoint-10000/projector.pt")
parser.add_argument("--lora-ckpt",      default="outputs/vlm/checkpoint-10000/lora_adapter")
parser.add_argument("--llm-name",       default="mistralai/Mistral-7B-Instruct-v0.3")
parser.add_argument("--token-budget",   type=int, default=256)
parser.add_argument("--hidden-dim",     type=int, default=2048)
parser.add_argument("--output-dir",     default="outputs/eval")
parser.add_argument("--test-images",    default="test_images")
parser.add_argument("--skip-qual",      action="store_true")
parser.add_argument("--skip-budget",    action="store_true")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── load model ────────────────────────────────────────────────────────────────
print("Loading models...")
encoder = AutoModel.from_pretrained(args.vision_ckpt, trust_remote_code=True,
                                    torch_dtype=torch.bfloat16).to(device).eval()
for mod in encoder.modules():
    if hasattr(mod, "inv_freq"):
        head_dim = mod.inv_freq.shape[0] * 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2,
                          dtype=torch.float32) / head_dim))
        mod.inv_freq = inv_freq.to(device)

projector = LevelAwareProjector(encoder_dim=384, hidden_dim=args.hidden_dim, llm_dim=4096)
projector.load_state_dict(torch.load(args.projector_ckpt, map_location=device))
projector = projector.to(device, dtype=torch.bfloat16).eval()

tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
base_llm = AutoModelForCausalLM.from_pretrained(args.llm_name, quantization_config=bnb_cfg,
                                                  device_map="auto", torch_dtype=torch.bfloat16)
llm = PeftModel.from_pretrained(base_llm, args.lora_ckpt) if os.path.exists(args.lora_ckpt) else base_llm
llm.eval()
print("Models loaded.\n")


@torch.no_grad()
def run_inference(pil_image, question, token_budget=256, max_new_tokens=30):
    x = transform(pil_image.convert("RGB")).unsqueeze(0).to(device, dtype=torch.bfloat16)
    vis_embs  = projector(encoder(x, token_budget=token_budget))
    prompt    = f"[INST] <image>\n{question} [/INST]"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    txt_embs  = llm.get_input_embeddings()(input_ids)
    img_id    = tokenizer.convert_tokens_to_ids("<image>")
    pos_lst   = (input_ids[0] == img_id).nonzero(as_tuple=True)[0]
    if len(pos_lst) > 0:
        p = pos_lst[0].item()
        combined = torch.cat([txt_embs[:, :p], vis_embs, txt_embs[:, p+1:]], dim=1)
    else:
        combined = torch.cat([vis_embs, txt_embs], dim=1)
    out_ids = llm.generate(inputs_embeds=combined, max_new_tokens=max_new_tokens,
                            do_sample=False, pad_token_id=tokenizer.eos_token_id,
                            repetition_penalty=1.3)
    decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
    if "[/INST]" in decoded:
        decoded = decoded.split("[/INST]")[-1].strip()
    return decoded.split("\n")[0].strip()


# ══════════════════════════════════════════════════════════════════════════════
# 1. QUALITATIVE EXAMPLES
# ══════════════════════════════════════════════════════════════════════════════
if not args.skip_qual:
    print("="*55)
    print("  Generating qualitative examples...")
    print("="*55)

    EXAMPLES = [
        {
            "file": "cat.jpg",
            "questions": [
                "What animal is in this image?",
                "What color is the sofa?",
            ],
        },
        {
            "file": "2.jpg",
            "questions": [
                "What sport is being played?",
                "What are the jersey numbers of the two main players?",
            ],
        },
        {
            "file": "horse.jpg",
            "questions": [
                "How many horses are in the image?",
                "What time of day does this appear to be?",
            ],
        },
        {
            "file": "ocr.jpg",
            "questions": [
                "What is the title text at the top of the slide?",
                "What hardware device is mentioned in the pipeline?",
            ],
        },
    ]

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.08)

    for idx, ex in enumerate(EXAMPLES):
        img_path = os.path.join(args.test_images, ex["file"])
        pil_img  = Image.open(img_path).convert("RGB")

        print(f"\n  [{idx+1}/4] {ex['file']}")
        answers = []
        for q in ex["questions"]:
            ans = run_inference(pil_img, q, token_budget=args.token_budget, max_new_tokens=25)
            print(f"    Q: {q}")
            print(f"    A: {ans}")
            answers.append((q, ans))

        row, col = divmod(idx, 2)
        # image subplot (top row)
        ax_img = fig.add_subplot(gs[0, idx])
        ax_img.imshow(pil_img)
        ax_img.axis("off")
        ax_img.set_title(ex["file"], color="#8b949e", fontsize=9, pad=4)

        # text subplot (bottom row)
        ax_txt = fig.add_subplot(gs[1, idx])
        ax_txt.set_facecolor("#161b22")
        ax_txt.axis("off")

        text_lines = []
        for q, a in answers:
            q_wrapped = "\n  ".join(textwrap.wrap(q, width=28))
            a_wrapped = "\n  ".join(textwrap.wrap(a if a else "(no answer)", width=28))
            text_lines.append(f"Q: {q_wrapped}")
            text_lines.append(f"A: {a_wrapped}")
            text_lines.append("")

        ax_txt.text(0.05, 0.95, "\n".join(text_lines).strip(),
                    transform=ax_txt.transAxes,
                    va="top", ha="left", fontsize=8.5,
                    color="white", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22", edgecolor="#30363d"))

    fig.suptitle("Keural VLM — Qualitative Examples\n"
                 "Real model outputs on diverse image types",
                 color="white", fontsize=14, fontweight="bold", y=1.01)

    out_path = os.path.join(args.output_dir, "qualitative_examples.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. TOKEN BUDGET CURVE
# ══════════════════════════════════════════════════════════════════════════════
if not args.skip_budget:
    print("\n" + "="*55)
    print("  Generating token budget curve (latency)...")
    print("="*55)

    TEST_IMAGE = os.path.join(args.test_images, "cat.jpg")
    pil_test   = Image.open(TEST_IMAGE).convert("RGB")
    budgets    = [32, 64, 128, 256, 512]
    latencies  = []
    token_counts = []
    WARMUP = 3
    RUNS   = 10

    # warmup
    for _ in range(WARMUP):
        run_inference(pil_test, "What is in this image?", token_budget=256, max_new_tokens=10)

    for budget in budgets:
        # measure actual visual token count
        x = transform(pil_test).unsqueeze(0).to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            enc_out = encoder(x, token_budget=budget)
        n_tokens = enc_out.tokens.shape[1]
        token_counts.append(n_tokens)

        # measure latency
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            run_inference(pil_test, "What is in this image?", token_budget=budget, max_new_tokens=10)
            times.append((time.perf_counter() - t0) * 1000)
        avg_ms = np.mean(times)
        latencies.append(avg_ms)
        print(f"  Budget={budget:4d}  tokens={n_tokens:4d}  latency={avg_ms:.0f}ms")

    # normalise to budget=256 as baseline (1.0×)
    base_idx  = budgets.index(256)
    speedups  = [latencies[base_idx] / l for l in latencies]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        ax.spines[:].set_visible(False)
        ax.yaxis.grid(True, color="#30363d", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets], color="white", fontsize=10)
        ax.tick_params(axis="y", colors="#8b949e")
        ax.set_xlabel("Token Budget", color="#8b949e", fontsize=12)

    # left: latency
    ax1.plot(budgets, latencies, color="#00d4aa", linewidth=2.5, marker="o",
             markersize=8, zorder=3)
    for b, l in zip(budgets, latencies):
        ax1.annotate(f"{l:.0f}ms", (b, l), textcoords="offset points",
                     xytext=(0, 10), ha="center", color="white", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Inference Latency (ms)", color="#8b949e", fontsize=12)
    ax1.set_title("Latency vs Token Budget", color="white", fontsize=13, fontweight="bold", pad=12)
    ax1.axvspan(128, 256, alpha=0.07, color="#00d4aa")

    # right: speedup
    ax2.bar(budgets, speedups, color=["#4f8ef7" if b != 256 else "#00d4aa" for b in budgets],
            width=60, zorder=3, alpha=0.9)
    for b, s in zip(budgets, speedups):
        ax2.annotate(f"{s:.2f}×", (b, s), textcoords="offset points",
                     xytext=(0, 6), ha="center", color="white", fontsize=10, fontweight="bold")
    ax2.axhline(1.0, color="#8b949e", linewidth=1, linestyle="--", zorder=2)
    ax2.set_ylabel("Speedup vs budget=256", color="#8b949e", fontsize=12)
    ax2.set_title("Relative Speedup vs Token Budget", color="white", fontsize=13, fontweight="bold", pad=12)

    fig.suptitle("Keural VLM — ATB Token Budget: Latency & Speedup\n"
                 "Lower budget = fewer visual tokens = faster inference",
                 color="white", fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, "token_budget_curve.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Saved: {out_path}")

print("\nDone.")
