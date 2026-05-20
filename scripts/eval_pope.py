"""
POPE Hallucination Evaluation for Keural VLM.
Tests object hallucination across 3 splits: adversarial, popular, random.

Usage:
    python3.10 scripts/eval_pope.py \
        --vision-ckpt outputs/run/final \
        --projector-ckpt outputs/vlm/checkpoint-10000/projector.pt \
        --lora-ckpt outputs/vlm/checkpoint-10000/lora_adapter

Results saved to: outputs/eval/pope_results.json
"""

import sys, os, argparse, json
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
parser.add_argument("--vision-ckpt",    default="outputs/run/final")
parser.add_argument("--projector-ckpt", default="outputs/vlm/checkpoint-10000/projector.pt")
parser.add_argument("--lora-ckpt",      default="outputs/vlm/checkpoint-10000/lora_adapter")
parser.add_argument("--llm-name",       default="mistralai/Mistral-7B-Instruct-v0.3")
parser.add_argument("--token-budget",   type=int, default=256)
parser.add_argument("--hidden-dim",     type=int, default=2048)
parser.add_argument("--output-dir",     default="outputs/eval")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── load models ───────────────────────────────────────────────────────────────
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
if os.path.exists(args.lora_ckpt):
    llm = PeftModel.from_pretrained(base_llm, args.lora_ckpt)
else:
    llm = base_llm
llm.eval()
print()


# ── extract yes/no from model output ─────────────────────────────────────────
def extract_yes_no(text: str) -> str:
    text = text.lower().strip()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    if "yes" in text.split()[:3]:
        return "yes"
    if "no" in text.split()[:3]:
        return "no"
    return "unknown"


# ── inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(pil_image, question):
    x = transform(pil_image.convert("RGB")).unsqueeze(0).to(device, dtype=torch.bfloat16)
    enc_out  = encoder(x, token_budget=args.token_budget)
    vis_embs = projector(enc_out)

    prompt    = f"[INST] <image>\n{question}\nAnswer yes or no only. [/INST]"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    txt_embs  = llm.get_input_embeddings()(input_ids)

    img_id  = tokenizer.convert_tokens_to_ids("<image>")
    pos_lst = (input_ids[0] == img_id).nonzero(as_tuple=True)[0]
    if len(pos_lst) > 0:
        p = pos_lst[0].item()
        combined = torch.cat([txt_embs[:, :p], vis_embs, txt_embs[:, p+1:]], dim=1)
    else:
        combined = torch.cat([vis_embs, txt_embs], dim=1)

    out_ids = llm.generate(inputs_embeds=combined, max_new_tokens=5,
                            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
    if "[/INST]" in decoded:
        decoded = decoded.split("[/INST]")[-1].strip()
    return extract_yes_no(decoded)


# ── evaluate each POPE split ──────────────────────────────────────────────────
def compute_metrics(preds, labels):
    tp = sum(p == "yes" and l == "yes" for p, l in zip(preds, labels))
    tn = sum(p == "no"  and l == "no"  for p, l in zip(preds, labels))
    fp = sum(p == "yes" and l == "no"  for p, l in zip(preds, labels))
    fn = sum(p == "no"  and l == "yes" for p, l in zip(preds, labels))
    acc       = (tp + tn) / len(preds) * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


SPLITS = ["adversarial", "popular", "random"]
all_results = {}

for split in SPLITS:
    print(f"Running POPE [{split}]...")
    dataset = load_dataset("lmms-lab/POPE", split=split, trust_remote_code=True)

    preds, labels = [], []
    for item in tqdm(dataset, desc=f"  POPE-{split}"):
        image    = item["image"]
        question = item["question"]
        label    = item["label"].lower().strip()   # "yes" or "no"

        pred = predict(image, question)
        preds.append(pred)
        labels.append(label)

    metrics = compute_metrics(preds, labels)
    all_results[split] = metrics

    print(f"  Accuracy : {metrics['accuracy']:.2f}%")
    print(f"  Precision: {metrics['precision']:.2f}%")
    print(f"  Recall   : {metrics['recall']:.2f}%")
    print(f"  F1       : {metrics['f1']:.2f}%\n")

# ── summary ───────────────────────────────────────────────────────────────────
print(f"{'='*50}")
print(f"  POPE Summary")
print(f"{'='*50}")
for split, m in all_results.items():
    print(f"  {split:12s}  Acc={m['accuracy']:.2f}%  F1={m['f1']:.2f}%")

avg_acc = sum(m["accuracy"] for m in all_results.values()) / len(SPLITS)
avg_f1  = sum(m["f1"]       for m in all_results.values()) / len(SPLITS)
print(f"  {'average':12s}  Acc={avg_acc:.2f}%  F1={avg_f1:.2f}%")
print(f"{'='*50}\n")

out_path = os.path.join(args.output_dir, "pope_results.json")
with open(out_path, "w") as f:
    json.dump({"average_accuracy": avg_acc, "average_f1": avg_f1,
               "splits": all_results}, f, indent=2)
print(f"Results saved to {out_path}")
