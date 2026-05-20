"""
MME (Multimodal Evaluation) Benchmark for Keural VLM.
Covers 14 subtasks: existence, count, position, color, posters, celebrity,
scene, landmark, artwork, commonsense reasoning, numerical calculation,
text translation, code reasoning.

Usage:
    python3.10 scripts/eval_mme.py \
        --vision-ckpt outputs/run/final \
        --projector-ckpt outputs/vlm/checkpoint-10000/projector.pt \
        --lora-ckpt outputs/vlm/checkpoint-10000/lora_adapter

Results saved to: outputs/eval/mme_results.json
"""

import sys, os, argparse, json
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


# ── extract yes/no ────────────────────────────────────────────────────────────
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


# ── MME scoring ───────────────────────────────────────────────────────────────
# MME score per task = accuracy * 100 + accuracy_plus * 100
# accuracy      = correct / total
# accuracy_plus = both questions about same image correct / image_count
def compute_mme_score(items):
    correct  = sum(1 for it in items if it["correct"])
    total    = len(items)
    accuracy = correct / total if total > 0 else 0

    # accuracy_plus: group by image, both Q correct
    by_image = defaultdict(list)
    for it in items:
        by_image[it["image_id"]].append(it["correct"])
    both_correct = sum(1 for qs in by_image.values() if all(qs))
    accuracy_plus = both_correct / len(by_image) if by_image else 0

    score = (accuracy + accuracy_plus) * 100
    return {"score": score, "accuracy": accuracy * 100,
            "accuracy_plus": accuracy_plus * 100,
            "correct": correct, "total": total}


# ── run MME ───────────────────────────────────────────────────────────────────
print("Loading MME dataset...")
dataset = load_dataset("lmms-lab/MME", split="test", trust_remote_code=True)
print(f"  {len(dataset)} samples across all tasks\n")

# group by task
tasks = defaultdict(list)
for item in dataset:
    tasks[item["category"]].append(item)

task_results = {}
total_score  = 0

for task_name, items in sorted(tasks.items()):
    print(f"Running MME [{task_name}] ({len(items)} samples)...")
    task_items = []

    for i, item in enumerate(tqdm(items, desc=f"  {task_name}", leave=False)):
        image    = item["image"]
        question = item["question"]
        label    = item["answer"].lower().strip()

        pred    = predict(image, question)
        correct = (pred == label)

        task_items.append({
            "image_id": item.get("image_id", i // 2),
            "question": question,
            "label":    label,
            "pred":     pred,
            "correct":  correct,
        })

    metrics = compute_mme_score(task_items)
    task_results[task_name] = metrics
    total_score += metrics["score"]

    print(f"  Score: {metrics['score']:.1f}  "
          f"(Acc={metrics['accuracy']:.1f}%  Acc+={metrics['accuracy_plus']:.1f}%)\n")

# ── summary ───────────────────────────────────────────────────────────────────
PERCEPTION_TASKS  = {"existence", "count", "position", "color", "posters",
                     "celebrity", "scene", "landmark", "artwork"}
COGNITION_TASKS   = {"commonsense_reasoning", "numerical_calculation",
                     "text_translation", "code_reasoning"}

perception_score = sum(v["score"] for k, v in task_results.items() if k in PERCEPTION_TASKS)
cognition_score  = sum(v["score"] for k, v in task_results.items() if k in COGNITION_TASKS)

print(f"{'='*55}")
print(f"  MME Results")
print(f"{'='*55}")
for task, m in sorted(task_results.items()):
    print(f"  {task:25s}  {m['score']:6.1f}")
print(f"{'─'*55}")
print(f"  {'Perception Total':25s}  {perception_score:6.1f}")
print(f"  {'Cognition Total':25s}  {cognition_score:6.1f}")
print(f"  {'TOTAL':25s}  {total_score:6.1f}")
print(f"{'='*55}\n")

out_path = os.path.join(args.output_dir, "mme_results.json")
with open(out_path, "w") as f:
    json.dump({"total_score": total_score, "perception_score": perception_score,
               "cognition_score": cognition_score, "tasks": task_results}, f, indent=2)
print(f"Results saved to {out_path}")
