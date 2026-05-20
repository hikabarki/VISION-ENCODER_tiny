"""
VQAv2 Evaluation for Keural VLM.

Usage:
    python3.10 scripts/eval_vqa.py \
        --vision-ckpt outputs/run/final \
        --projector-ckpt outputs/vlm/checkpoint-10000/projector.pt \
        --lora-ckpt outputs/vlm/checkpoint-10000/lora_adapter \
        --num-samples 500

Results saved to: outputs/eval/vqav2_results.json
"""

import sys, os, argparse, json, re, string
sys.path.insert(0, "/home/mkd-s002/vision-encoder/poc")

import torch
import torchvision.transforms as T
from PIL import Image
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
parser.add_argument("--num-samples",    type=int, default=500,
                    help="Number of validation samples (use -1 for full set)")
parser.add_argument("--token-budget",   type=int, default=256)
parser.add_argument("--hidden-dim",     type=int, default=2048)
parser.add_argument("--max-new-tokens", type=int, default=10)
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
print("\n[1/4] Loading vision encoder...")
encoder = AutoModel.from_pretrained(args.vision_ckpt, trust_remote_code=True,
                                    torch_dtype=torch.bfloat16).to(device).eval()
for mod in encoder.modules():
    if hasattr(mod, "inv_freq"):
        head_dim = mod.inv_freq.shape[0] * 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2,
                          dtype=torch.float32) / head_dim))
        mod.inv_freq = inv_freq.to(device)

print("[2/4] Loading projector...")
projector = LevelAwareProjector(encoder_dim=384, hidden_dim=args.hidden_dim, llm_dim=4096)
projector.load_state_dict(torch.load(args.projector_ckpt, map_location=device))
projector = projector.to(device, dtype=torch.bfloat16).eval()

print("[3/4] Loading tokenizer + LLM...")
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

print("[4/4] Loading VQAv2 dataset...")
dataset = load_dataset("lmms-lab/VQAv2", split="validation")
if args.num_samples > 0:
    dataset = dataset.select(range(min(args.num_samples, len(dataset))))
print(f"  {len(dataset)} samples\n")


# ── answer normalization (VQA official rules) ─────────────────────────────────
ARTICLES    = {"a", "an", "the"}
CONTRACTIONS = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
                "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
                "hasnt": "hasn't", "havent": "haven't", "isnt": "isn't", "mightve": "might've",
                "mustve": "must've", "neednt": "needn't", "shouldve": "should've",
                "shouldnt": "shouldn't", "wasnt": "wasn't", "werent": "weren't",
                "wouldve": "would've", "wouldnt": "wouldn't"}

def normalize(answer: str) -> str:
    answer = answer.lower().strip()
    answer = answer.translate(str.maketrans("", "", string.punctuation))
    words = answer.split()
    words = [w for w in words if w not in ARTICLES]
    answer = " ".join(words)
    answer = CONTRACTIONS.get(answer, answer)
    return answer.strip()


# ── inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(pil_image, question):
    x = transform(pil_image.convert("RGB")).unsqueeze(0).to(device, dtype=torch.bfloat16)
    enc_out  = encoder(x, token_budget=args.token_budget)
    vis_embs = projector(enc_out)

    prompt   = f"[INST] <image>\n{question}\nAnswer in one or two words. [/INST]"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    txt_embs  = llm.get_input_embeddings()(input_ids)

    img_id  = tokenizer.convert_tokens_to_ids("<image>")
    pos_lst = (input_ids[0] == img_id).nonzero(as_tuple=True)[0]
    if len(pos_lst) > 0:
        p = pos_lst[0].item()
        combined = torch.cat([txt_embs[:, :p], vis_embs, txt_embs[:, p+1:]], dim=1)
    else:
        combined = torch.cat([vis_embs, txt_embs], dim=1)

    out_ids = llm.generate(inputs_embeds=combined, max_new_tokens=args.max_new_tokens,
                            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
    if "[/INST]" in decoded:
        decoded = decoded.split("[/INST]")[-1].strip()
    return decoded.split("\n")[0].strip()


# ── evaluate ──────────────────────────────────────────────────────────────────
results  = []
correct  = 0

for item in tqdm(dataset, desc="VQAv2"):
    image    = item["image"]
    question = item["question"]
    answers  = [normalize(a) for a in item["answers"]]   # list of 10 annotator answers

    pred = normalize(predict(image, question))
    # VQA soft accuracy: min(count of matching annotations / 3, 1)
    acc  = min(answers.count(pred) / 3.0, 1.0)
    correct += acc

    results.append({
        "question_id": item.get("question_id", ""),
        "question":    question,
        "prediction":  pred,
        "answers":     list(set(answers)),
        "accuracy":    acc,
    })

overall_acc = correct / len(dataset) * 100

print(f"\n{'='*45}")
print(f"  VQAv2 Accuracy : {overall_acc:.2f}%  ({len(dataset)} samples)")
print(f"{'='*45}\n")

out_path = os.path.join(args.output_dir, "vqav2_results.json")
with open(out_path, "w") as f:
    json.dump({"accuracy": overall_acc, "num_samples": len(dataset),
               "results": results}, f, indent=2)
print(f"Results saved to {out_path}")
