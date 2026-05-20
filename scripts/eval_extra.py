"""
Keural VLM — Extra Benchmark Evaluation: TextVQA + ScienceQA
Runs after eval_all.py, appends results to outputs/eval/summary.json.

TextVQA : 5K validation samples — text reading in images
ScienceQA: image-only subset (~2.2K) — science multi-choice reasoning

Usage:
    python3.10 scripts/eval_extra.py \
        --vision-ckpt outputs/run/final \
        --projector-ckpt outputs/vlm/checkpoint-10000/projector.pt \
        --lora-ckpt outputs/vlm/checkpoint-10000/lora_adapter \
        --textvqa-samples 1000 \
        --scienceqa-samples 500
"""

import sys, os, argparse, json, string, time
sys.path.insert(0, "/home/mkd-s002/vision-encoder/poc")

import torch
import torchvision.transforms as T
from tqdm import tqdm
from datasets import load_dataset

import alignment.hf_wrapper
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from alignment.projectors import LevelAwareProjector

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--vision-ckpt",       default="outputs/run/final")
parser.add_argument("--projector-ckpt",    default="outputs/vlm/checkpoint-10000/projector.pt")
parser.add_argument("--lora-ckpt",         default="outputs/vlm/checkpoint-10000/lora_adapter")
parser.add_argument("--llm-name",          default="mistralai/Mistral-7B-Instruct-v0.3")
parser.add_argument("--token-budget",      type=int, default=256)
parser.add_argument("--hidden-dim",        type=int, default=2048)
parser.add_argument("--textvqa-samples",   type=int, default=1000)
parser.add_argument("--scienceqa-samples", type=int, default=500)
parser.add_argument("--output-dir",        default="outputs/eval")
parser.add_argument("--skip-textvqa",      action="store_true")
parser.add_argument("--skip-scienceqa",    action="store_true")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── load models ───────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  Keural VLM — Extra Benchmark Evaluation")
print("="*55)

print("\n[1/3] Loading vision encoder...")
encoder = AutoModel.from_pretrained(args.vision_ckpt, trust_remote_code=True,
                                    torch_dtype=torch.bfloat16).to(device).eval()
for mod in encoder.modules():
    if hasattr(mod, "inv_freq"):
        head_dim = mod.inv_freq.shape[0] * 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2,
                          dtype=torch.float32) / head_dim))
        mod.inv_freq = inv_freq.to(device)

print("[2/3] Loading projector...")
projector = LevelAwareProjector(encoder_dim=384, hidden_dim=args.hidden_dim, llm_dim=4096)
projector.load_state_dict(torch.load(args.projector_ckpt, map_location=device))
projector = projector.to(device, dtype=torch.bfloat16).eval()

print("[3/3] Loading tokenizer + LLM...")
tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
base_llm = AutoModelForCausalLM.from_pretrained(args.llm_name, quantization_config=bnb_cfg,
                                                  device_map="auto", torch_dtype=torch.bfloat16)
llm = PeftModel.from_pretrained(base_llm, args.lora_ckpt) if os.path.exists(args.lora_ckpt) else base_llm
llm.eval()
print("\nAll models loaded.\n")


# ── shared inference ──────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(pil_image, question, max_new_tokens=10):
    x = transform(pil_image.convert("RGB")).unsqueeze(0).to(device, dtype=torch.bfloat16)
    vis_embs  = projector(encoder(x, token_budget=args.token_budget))
    input_ids = tokenizer(question, return_tensors="pt").input_ids.to(device)
    txt_embs  = llm.get_input_embeddings()(input_ids)
    img_id    = tokenizer.convert_tokens_to_ids("<image>")
    pos_lst   = (input_ids[0] == img_id).nonzero(as_tuple=True)[0]
    if len(pos_lst) > 0:
        p = pos_lst[0].item()
        combined = torch.cat([txt_embs[:, :p], vis_embs, txt_embs[:, p+1:]], dim=1)
    else:
        combined = torch.cat([vis_embs, txt_embs], dim=1)
    out_ids = llm.generate(inputs_embeds=combined, max_new_tokens=max_new_tokens,
                            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
    if "[/INST]" in decoded:
        decoded = decoded.split("[/INST]")[-1].strip()
    return decoded.split("\n")[0].strip()

def make_prompt(question, instruction=""):
    suffix = f"\n{instruction}" if instruction else ""
    return f"[INST] <image>\n{question}{suffix} [/INST]"


# ── VQA normalization (reused for TextVQA) ────────────────────────────────────
ARTICLES = {"a", "an", "the"}
CONTRACTIONS = {"aint": "ain't", "arent": "aren't", "cant": "can't",
                "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
                "isnt": "isn't", "shouldnt": "shouldn't", "wasnt": "wasn't",
                "werent": "weren't", "wouldnt": "wouldn't"}

def normalize_vqa(answer):
    answer = answer.lower().strip()
    answer = answer.translate(str.maketrans("", "", string.punctuation))
    words  = [w for w in answer.split() if w not in ARTICLES]
    answer = " ".join(words)
    return CONTRACTIONS.get(answer, answer).strip()


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 1 — TextVQA
# ══════════════════════════════════════════════════════════════════════════════
textvqa_score = None
if not args.skip_textvqa:
    print("="*55)
    print("  BENCHMARK 1 / 2 — TextVQA")
    print("="*55)
    t0 = time.time()

    dataset = load_dataset("lmms-lab/textvqa", split="validation")
    if args.textvqa_samples > 0:
        dataset = dataset.select(range(min(args.textvqa_samples, len(dataset))))
    print(f"  {len(dataset):,} samples\n")

    correct = 0
    results = []
    for item in tqdm(dataset, desc="TextVQA"):
        prompt  = make_prompt(item["question"], "Answer in one or two words.")
        pred    = normalize_vqa(run_inference(item["image"], prompt, max_new_tokens=10))
        answers = [normalize_vqa(a) for a in item["answers"]]
        acc     = min(answers.count(pred) / 3.0, 1.0)
        correct += acc
        results.append({"question": item["question"], "pred": pred,
                        "answers": list(set(answers)), "acc": acc})

    textvqa_score = correct / len(dataset) * 100
    elapsed = (time.time() - t0) / 60
    print(f"\n  TextVQA Accuracy: {textvqa_score:.2f}%  ({len(dataset):,} samples, {elapsed:.1f} min)\n")
    with open(os.path.join(args.output_dir, "textvqa_results.json"), "w") as f:
        json.dump({"accuracy": textvqa_score, "num_samples": len(dataset),
                   "results": results}, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 2 — ScienceQA (image subset)
# ══════════════════════════════════════════════════════════════════════════════
scienceqa_score = None
if not args.skip_scienceqa:
    print("="*55)
    print("  BENCHMARK 2 / 2 — ScienceQA (image subset)")
    print("="*55)
    t0 = time.time()

    dataset = load_dataset("derek-thomas/ScienceQA", split="validation")
    # keep only samples that have an image
    dataset = dataset.filter(lambda x: x["image"] is not None)
    if args.scienceqa_samples > 0:
        dataset = dataset.select(range(min(args.scienceqa_samples, len(dataset))))
    print(f"  {len(dataset):,} samples (image-only subset)\n")

    CHOICE_LETTERS = "ABCDE"

    def format_choices(choices):
        return "  ".join(f"{CHOICE_LETTERS[i]}) {c}" for i, c in enumerate(choices))

    def extract_choice(text, n_choices):
        t = text.strip().upper()
        # check first character
        if t and t[0] in CHOICE_LETTERS[:n_choices]:
            return t[0]
        # check for "Answer: X" or "option X"
        for letter in CHOICE_LETTERS[:n_choices]:
            if f"ANSWER: {letter}" in t or f"OPTION {letter}" in t or f"({letter})" in t:
                return letter
        # check any letter occurrence
        for letter in CHOICE_LETTERS[:n_choices]:
            if letter in t.split():
                return letter
        return "A"  # default

    correct = 0
    results = []
    for item in tqdm(dataset, desc="ScienceQA"):
        choices_str = format_choices(item["choices"])
        question    = f"{item['question']}\n{choices_str}"
        prompt      = make_prompt(question, "Answer with only the letter of the correct choice.")
        pred_text   = run_inference(item["image"], prompt, max_new_tokens=5)
        pred_letter = extract_choice(pred_text, len(item["choices"]))
        gold_letter = CHOICE_LETTERS[item["answer"]]
        is_correct  = pred_letter == gold_letter
        correct    += int(is_correct)
        results.append({"question": item["question"], "pred": pred_letter,
                        "gold": gold_letter, "correct": is_correct})

    scienceqa_score = correct / len(dataset) * 100
    elapsed = (time.time() - t0) / 60
    print(f"\n  ScienceQA Accuracy: {scienceqa_score:.2f}%  ({len(dataset):,} samples, {elapsed:.1f} min)\n")
    with open(os.path.join(args.output_dir, "scienceqa_results.json"), "w") as f:
        json.dump({"accuracy": scienceqa_score, "num_samples": len(dataset),
                   "results": results}, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# UPDATE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
summary_path = os.path.join(args.output_dir, "summary.json")
summary = {}
if os.path.exists(summary_path):
    with open(summary_path) as f:
        summary = json.load(f)

if textvqa_score is not None:
    summary["textvqa_accuracy"] = textvqa_score
if scienceqa_score is not None:
    summary["scienceqa_accuracy"] = scienceqa_score

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*55)
print("  EXTRA BENCHMARK SUMMARY")
print("="*55)
if textvqa_score  is not None: print(f"  TextVQA Accuracy  : {textvqa_score:.2f}%")
if scienceqa_score is not None: print(f"  ScienceQA Accuracy: {scienceqa_score:.2f}%")
print("="*55)
print(f"\nResults saved to {args.output_dir}/\n")
