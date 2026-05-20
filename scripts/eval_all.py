"""
Keural VLM — Full Benchmark Evaluation.
Runs VQAv2, POPE, and MME sequentially with the model loaded only once.

Standard settings:
  VQAv2 : full validation set (~214K) or --vqa-samples N for quick test
  POPE  : all 3 splits (adversarial, popular, random) — ~9K questions
  MME   : all 14 tasks — ~2.8K questions

Usage (quick test — ~1 hour):
    python3.10 scripts/eval_all.py \
        --vision-ckpt outputs/run/final \
        --projector-ckpt outputs/vlm/checkpoint-10000/projector.pt \
        --lora-ckpt outputs/vlm/checkpoint-10000/lora_adapter \
        --vqa-samples 1000

Usage (full standard benchmark — ~12-16 hours):
    python3.10 scripts/eval_all.py \
        --vision-ckpt outputs/run/final \
        --projector-ckpt outputs/vlm/checkpoint-10000/projector.pt \
        --lora-ckpt outputs/vlm/checkpoint-10000/lora_adapter \
        --vqa-samples -1

Results saved to: outputs/eval/
"""

import sys, os, argparse, json, string, time
sys.path.insert(0, "/home/mkd-s002/vision-encoder/poc")

import torch
import torchvision.transforms as T
from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict

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
parser.add_argument("--vqa-samples",    type=int, default=1000,
                    help="VQAv2 samples (-1 = full 214K validation set)")
parser.add_argument("--output-dir",     default="outputs/eval")
parser.add_argument("--skip-vqa",  action="store_true")
parser.add_argument("--skip-pope", action="store_true")
parser.add_argument("--skip-mme",  action="store_true")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── load model once ───────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  Keural VLM — Benchmark Evaluation")
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
print(f"  {sum(p.numel() for p in encoder.parameters()):,} params")

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


def extract_yes_no(text):
    t = text.lower().strip()
    if t.startswith("yes"): return "yes"
    if t.startswith("no"):  return "no"
    words = t.split()[:3]
    if "yes" in words: return "yes"
    if "no"  in words: return "no"
    return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 1 — VQAv2
# ══════════════════════════════════════════════════════════════════════════════
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

vqa_score = None
if not args.skip_vqa:
    print("="*55)
    print("  BENCHMARK 1 / 3 — VQAv2")
    print("="*55)
    t0 = time.time()

    dataset = load_dataset("lmms-lab/VQAv2", split="validation")
    if args.vqa_samples > 0:
        dataset = dataset.select(range(min(args.vqa_samples, len(dataset))))
    print(f"  {len(dataset):,} samples  "
          f"({'full validation' if args.vqa_samples < 0 else 'subset'})\n")

    correct = 0
    vqa_results = []
    for item in tqdm(dataset, desc="VQAv2"):
        prompt = make_prompt(item["question"], "Answer in one or two words.")
        pred   = normalize_vqa(run_inference(item["image"], prompt, max_new_tokens=10))
        answers = [normalize_vqa(a["answer"] if isinstance(a, dict) else a) for a in item["answers"]]
        acc    = min(answers.count(pred) / 3.0, 1.0)
        correct += acc
        vqa_results.append({"question": item["question"], "pred": pred,
                             "answers": list(set(answers)), "acc": acc})

    vqa_score = correct / len(dataset) * 100
    elapsed   = (time.time() - t0) / 60
    print(f"\n  VQAv2 Accuracy: {vqa_score:.2f}%  ({len(dataset):,} samples, {elapsed:.1f} min)\n")
    with open(os.path.join(args.output_dir, "vqav2_results.json"), "w") as f:
        json.dump({"accuracy": vqa_score, "num_samples": len(dataset),
                   "results": vqa_results}, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 2 — POPE
# ══════════════════════════════════════════════════════════════════════════════
def pope_metrics(preds, labels):
    tp = sum(p == "yes" and l == "yes" for p, l in zip(preds, labels))
    tn = sum(p == "no"  and l == "no"  for p, l in zip(preds, labels))
    fp = sum(p == "yes" and l == "no"  for p, l in zip(preds, labels))
    fn = sum(p == "no"  and l == "yes" for p, l in zip(preds, labels))
    acc  = (tp + tn) / len(preds) * 100
    prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

pope_results = {}
if not args.skip_pope:
    print("="*55)
    print("  BENCHMARK 2 / 3 — POPE (Hallucination)")
    print("="*55)
    t0 = time.time()

    for split in ["adversarial", "popular", "random"]:
        print(f"\n  Split: {split}")
        dataset = load_dataset("lmms-lab/POPE", "Full", split=split)
        preds, labels = [], []
        for item in tqdm(dataset, desc=f"  POPE-{split}"):
            prompt = make_prompt(item["question"], "Answer yes or no only.")
            pred   = extract_yes_no(run_inference(item["image"], prompt, max_new_tokens=5))
            preds.append(pred)
            labels.append(item["answer"].lower().strip())
        m = pope_metrics(preds, labels)
        pope_results[split] = m
        print(f"  Acc={m['accuracy']:.2f}%  F1={m['f1']:.2f}%  "
              f"Prec={m['precision']:.2f}%  Rec={m['recall']:.2f}%")

    avg_acc = sum(m["accuracy"] for m in pope_results.values()) / 3
    avg_f1  = sum(m["f1"]       for m in pope_results.values()) / 3
    elapsed = (time.time() - t0) / 60
    print(f"\n  POPE Average — Acc={avg_acc:.2f}%  F1={avg_f1:.2f}%  ({elapsed:.1f} min)\n")
    with open(os.path.join(args.output_dir, "pope_results.json"), "w") as f:
        json.dump({"average_accuracy": avg_acc, "average_f1": avg_f1,
                   "splits": pope_results}, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 3 — MME
# ══════════════════════════════════════════════════════════════════════════════
def mme_score(items):
    correct = sum(1 for it in items if it["correct"])
    by_img  = defaultdict(list)
    for it in items:
        by_img[it["img_id"]].append(it["correct"])
    acc_plus = sum(1 for qs in by_img.values() if all(qs)) / max(len(by_img), 1)
    acc      = correct / max(len(items), 1)
    return {"score": (acc + acc_plus) * 100,
            "accuracy": acc * 100, "accuracy_plus": acc_plus * 100,
            "correct": correct, "total": len(items)}

mme_results = {}
total_mme   = 0
if not args.skip_mme:
    print("="*55)
    print("  BENCHMARK 3 / 3 — MME")
    print("="*55)
    t0 = time.time()

    dataset = load_dataset("lmms-lab/MME", split="test", trust_remote_code=True)
    tasks   = defaultdict(list)
    for item in dataset:
        tasks[item["category"]].append(item)

    for task_name, items in sorted(tasks.items()):
        task_items = []
        for i, item in enumerate(tqdm(items, desc=f"  {task_name}", leave=False)):
            prompt  = make_prompt(item["question"], "Answer yes or no only.")
            pred    = extract_yes_no(run_inference(item["image"], prompt, max_new_tokens=5))
            label   = item["answer"].lower().strip()
            task_items.append({"img_id": item.get("image_id", i//2),
                                "correct": pred == label, "pred": pred, "label": label})
        m = mme_score(task_items)
        mme_results[task_name] = m
        total_mme += m["score"]
        print(f"  {task_name:28s}  {m['score']:6.1f}  "
              f"(Acc={m['accuracy']:.1f}%  Acc+={m['accuracy_plus']:.1f}%)")

    elapsed = (time.time() - t0) / 60
    print(f"\n  MME Total Score: {total_mme:.1f}  ({elapsed:.1f} min)\n")
    with open(os.path.join(args.output_dir, "mme_results.json"), "w") as f:
        json.dump({"total_score": total_mme, "tasks": mme_results}, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  KEURAL VLM — BENCHMARK SUMMARY")
print("="*55)
if vqa_score is not None:
    print(f"  VQAv2 Accuracy     : {vqa_score:.2f}%")
if pope_results:
    avg_acc = sum(m["accuracy"] for m in pope_results.values()) / 3
    avg_f1  = sum(m["f1"]       for m in pope_results.values()) / 3
    print(f"  POPE Avg Accuracy  : {avg_acc:.2f}%")
    print(f"  POPE Avg F1        : {avg_f1:.2f}%")
if mme_results:
    print(f"  MME Total Score    : {total_mme:.1f}")
print("="*55)

summary = {
    "vqav2_accuracy":   vqa_score,
    "pope_avg_accuracy": sum(m["accuracy"] for m in pope_results.values()) / 3 if pope_results else None,
    "pope_avg_f1":       sum(m["f1"]       for m in pope_results.values()) / 3 if pope_results else None,
    "mme_total":         total_mme if mme_results else None,
}
with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nAll results saved to {args.output_dir}/\n")
