"""
Keural VLM — Vision Encoder Comparison GUI.
Compare Keural Vision Encoder vs LLaVA 1.6 Mistral 7B side by side.

Both systems use Mistral 7B as the language model — a fair apples-to-apples
comparison of vision alignment quality.

Usage:
    python3.10 scripts/vlm_compare_gui.py \
        --vision-ckpt outputs/run/final \
        --projector-ckpt outputs/vlm/checkpoint-2000/projector.pt \
        --lora-ckpt outputs/vlm/checkpoint-2000/lora_adapter

Then open the shared Gradio URL.
"""

import sys, os, argparse
sys.path.insert(0, "/home/mkd-s002/vision-encoder/poc")

import torch
import torchvision.transforms as T
from PIL import Image
import gradio as gr

import alignment.hf_wrapper
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig,
                          LlavaNextProcessor, LlavaNextForConditionalGeneration)
from peft import PeftModel

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--vision-ckpt",    default="outputs/run/final")
parser.add_argument("--projector-ckpt", default="outputs/vlm/checkpoint-2000/projector.pt")
parser.add_argument("--lora-ckpt",      default="outputs/vlm/checkpoint-2000/lora_adapter")
parser.add_argument("--llm-name",       default="mistralai/Mistral-7B-Instruct-v0.3")
parser.add_argument("--llava-model",    default="llava-hf/llava-v1.6-mistral-7b-hf")
parser.add_argument("--token-budget",   type=int, default=256)
parser.add_argument("--hidden-dim",     type=int, default=2048)
parser.add_argument("--max-new-tokens", type=int, default=150)
parser.add_argument("--port",           type=int, default=7861)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

keural_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# ── [1/5] Keural encoder ─────────────────────────────────────────────────────
print("\n[1/5] Loading Keural vision encoder...")
keural_encoder = AutoModel.from_pretrained(
    args.vision_ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16
).to(device).eval()
for mod in keural_encoder.modules():
    if hasattr(mod, "inv_freq"):
        head_dim = mod.inv_freq.shape[0] * 2
        inv_freq = 1.0 / (10000.0 ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        ))
        mod.inv_freq = inv_freq.to(device)
print(f"  {sum(p.numel() for p in keural_encoder.parameters()):,} params")

# ── [2/5] Keural projector ────────────────────────────────────────────────────
print("\n[2/5] Loading Keural projector...")
from alignment.projectors import LevelAwareProjector
keural_projector = LevelAwareProjector(
    encoder_dim=384, hidden_dim=args.hidden_dim, llm_dim=4096
)
if os.path.exists(args.projector_ckpt):
    keural_projector.load_state_dict(torch.load(args.projector_ckpt, map_location=device))
    print(f"  Loaded from {args.projector_ckpt}")
else:
    print(f"  WARNING: projector not found at {args.projector_ckpt} — using random weights")
keural_projector = keural_projector.to(device, dtype=torch.bfloat16).eval()

# ── [3/5] Keural LLM (Mistral + LoRA) ────────────────────────────────────────
print(f"\n[3/5] Loading Keural LLM: {args.llm_name} (4-bit NF4)...")
keural_tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
if keural_tokenizer.pad_token is None:
    keural_tokenizer.pad_token = keural_tokenizer.eos_token

keural_base_llm = AutoModelForCausalLM.from_pretrained(
    args.llm_name, quantization_config=bnb_cfg,
    device_map="auto", torch_dtype=torch.bfloat16,
)
if os.path.exists(args.lora_ckpt):
    keural_llm = PeftModel.from_pretrained(keural_base_llm, args.lora_ckpt)
    print(f"  LoRA loaded from {args.lora_ckpt}")
else:
    keural_llm = keural_base_llm
    print("  No LoRA adapter found — using base Mistral")
keural_llm.eval()

# ── [4/5] LLaVA 1.6 Mistral processor ────────────────────────────────────────
print(f"\n[4/5] Loading LLaVA 1.6 Mistral processor: {args.llava_model}...")
llava_processor = LlavaNextProcessor.from_pretrained(args.llava_model)
print("  Processor loaded.")

# ── [5/5] LLaVA 1.6 Mistral model ────────────────────────────────────────────
print(f"\n[5/5] Loading LLaVA 1.6 Mistral model (4-bit NF4)...")
llava_model = LlavaNextForConditionalGeneration.from_pretrained(
    args.llava_model,
    quantization_config=bnb_cfg,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
llava_model.eval()
print("  LLaVA 1.6 Mistral loaded.")
print("\nAll models loaded.\n")


# ── Keural encode ─────────────────────────────────────────────────────────────
def encode_keural(pil_image):
    x = keural_transform(pil_image.convert("RGB")).unsqueeze(0).to(device, dtype=torch.bfloat16)
    with torch.no_grad():
        enc_out = keural_encoder(x, token_budget=args.token_budget)
        visual_embeds = keural_projector(enc_out)   # (1, N, 4096)
    return visual_embeds


# ── Keural generate ───────────────────────────────────────────────────────────
def generate_keural_answer(visual_embeds, question):
    prompt = f"[INST] <image>\n{question} [/INST]"
    input_ids = keural_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        text_embeds = keural_llm.get_input_embeddings()(input_ids)
        img_token_id = keural_tokenizer.convert_tokens_to_ids("<image>")
        img_positions = (input_ids[0] == img_token_id).nonzero(as_tuple=True)[0]
        if len(img_positions) > 0:
            pos = img_positions[0].item()
            combined = torch.cat(
                [text_embeds[:, :pos], visual_embeds, text_embeds[:, pos+1:]], dim=1
            )
        else:
            combined = torch.cat([visual_embeds, text_embeds], dim=1)
        out_ids = keural_llm.generate(
            inputs_embeds=combined,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.3,
            pad_token_id=keural_tokenizer.eos_token_id,
        )
    answer = keural_tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
    if "[/INST]" in answer:
        answer = answer.split("[/INST]")[-1].strip()
    return answer


# ── LLaVA generate ────────────────────────────────────────────────────────────
def generate_llava_answer(pil_image, question):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]
    prompt = llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = llava_processor(
        images=pil_image.convert("RGB"), text=prompt, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        output_ids = llava_model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.3,
        )
    answer = llava_processor.decode(output_ids[0], skip_special_tokens=True).strip()
    if "[/INST]" in answer:
        answer = answer.split("[/INST]")[-1].strip()
    return answer


# ── state ─────────────────────────────────────────────────────────────────────
keural_embeds = [None]
current_pil   = [None]

def load_image(pil_image):
    if pil_image is None:
        return "No image loaded."
    keural_embeds[0] = encode_keural(pil_image)
    current_pil[0]   = pil_image
    return "Image encoded. Ask your question below."

def ask(question, keural_history, llava_history):
    if not question.strip():
        return keural_history, llava_history, ""
    if keural_embeds[0] is None:
        msg = {"role": "assistant", "content": "Please upload an image first."}
        keural_history.append({"role": "user", "content": question})
        keural_history.append(msg)
        llava_history.append({"role": "user", "content": question})
        llava_history.append(msg)
        return keural_history, llava_history, ""

    keural_ans = generate_keural_answer(keural_embeds[0], question)
    llava_ans  = generate_llava_answer(current_pil[0], question)

    keural_history.append({"role": "user",      "content": question})
    keural_history.append({"role": "assistant",  "content": keural_ans})
    llava_history.append( {"role": "user",       "content": question})
    llava_history.append( {"role": "assistant",  "content": llava_ans})

    return keural_history, llava_history, ""


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Keural vs LLaVA 1.6 Mistral — Vision Encoder Comparison") as demo:
    gr.Markdown("""
# Keural vs LLaVA 1.6 Mistral 7B — Vision Encoder Comparison
**Fair comparison: both use Mistral 7B as the language backbone with trained vision-language alignment.**
Upload an image and ask questions to compare both systems side by side.
""")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image", height=300)
        status = gr.Textbox(
            label="Status", value="Upload an image to begin.",
            interactive=False, scale=1
        )

    with gr.Row():
        with gr.Column():
            gr.Markdown(
                "### Keural Vision Encoder\n"
                "`24.7M params | ATB + HCT | trained from scratch | Phase 2B LoRA`"
            )
            keural_chat = gr.Chatbot(label="Keural Response", height=400)
        with gr.Column():
            gr.Markdown(
                "### LLaVA 1.6 Mistral 7B\n"
                "`llava-hf/llava-v1.6-mistral-7b-hf | CLIP ViT-L/14 + trained MLP + Mistral 7B`"
            )
            llava_chat = gr.Chatbot(label="LLaVA 1.6 Response", height=400)

    with gr.Row():
        question_input = gr.Textbox(
            placeholder="Ask a question about the image...",
            label="Question (sent to both models)",
            scale=5,
        )
        send_btn = gr.Button("Ask Both", variant="primary", scale=1)

    gr.Examples(
        examples=[
            ["What is in this image?"],
            ["Describe this image in detail."],
            ["What color is the main object?"],
            ["Is there a person in this image?"],
            ["What is the mood or atmosphere?"],
            ["Is this indoors or outdoors?"],
            ["How many objects can you count?"],
        ],
        inputs=question_input,
        label="Quick questions",
    )

    image_input.change(load_image, inputs=image_input, outputs=status)
    send_btn.click(
        ask,
        inputs=[question_input, keural_chat, llava_chat],
        outputs=[keural_chat, llava_chat, question_input],
    )
    question_input.submit(
        ask,
        inputs=[question_input, keural_chat, llava_chat],
        outputs=[keural_chat, llava_chat, question_input],
    )

demo.launch(server_name="0.0.0.0", server_port=args.port, share=True)
