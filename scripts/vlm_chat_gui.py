"""
Keural VLM — Interactive Chat GUI.
Upload any image and ask questions about it in a chat interface.

Usage:
    # SFT only
    python3.10 scripts/vlm_chat_gui.py \
        --vision-ckpt outputs/run/final \
        --projector-ckpt outputs/vlm/checkpoint-30000/projector.pt \
        --lora-ckpt outputs/vlm/checkpoint-30000/lora_adapter

    # SFT + DPO (stacked)
    python3.10 scripts/vlm_chat_gui.py \
        --vision-ckpt outputs/run/final \
        --projector-ckpt outputs/vlm/checkpoint-30000/projector.pt \
        --lora-ckpt outputs/vlm/checkpoint-30000/lora_adapter \
        --dpo-lora-ckpt outputs/dpo/final

Then open: http://localhost:7860
"""

import sys, os, argparse
sys.path.insert(0, "/home/mkd-s002/vision-encoder/poc")

import torch
import torchvision.transforms as T
from PIL import Image
import gradio as gr
import easyocr
import numpy as np

ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)

import alignment.hf_wrapper
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--vision-ckpt",    default="outputs/run/final")
parser.add_argument("--projector-ckpt", default="outputs/vlm/checkpoint-1000/projector.pt")
parser.add_argument("--lora-ckpt",      default="outputs/vlm/checkpoint-30000/lora_adapter")
parser.add_argument("--dpo-lora-ckpt",  default=None)
parser.add_argument("--llm-name",       default="mistralai/Mistral-7B-Instruct-v0.3")
parser.add_argument("--token-budget",   type=int, default=256)
parser.add_argument("--hidden-dim",     type=int, default=2048)
parser.add_argument("--max-new-tokens", type=int, default=200)
parser.add_argument("--port",           type=int, default=7860)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── image transform ───────────────────────────────────────────────────────────
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── load models once at startup ───────────────────────────────────────────────
print("\n[1/4] Loading vision encoder...")
encoder = AutoModel.from_pretrained(args.vision_ckpt, trust_remote_code=True,
                                    torch_dtype=torch.bfloat16).to(device).eval()
# reinitialize RoPE buffers to avoid NaN
for mod in encoder.modules():
    if hasattr(mod, "inv_freq"):
        head_dim = mod.inv_freq.shape[0] * 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2,
                          dtype=torch.float32) / head_dim))
        mod.inv_freq = inv_freq.to(device)
print(f"  {sum(p.numel() for p in encoder.parameters()):,} params — frozen")

print("\n[2/4] Loading projector...")
from alignment.projectors import LevelAwareProjector
projector = LevelAwareProjector(encoder_dim=384, hidden_dim=args.hidden_dim, llm_dim=4096)
if os.path.exists(args.projector_ckpt):
    projector.load_state_dict(torch.load(args.projector_ckpt, map_location=device))
    print(f"  Loaded from {args.projector_ckpt}")
projector = projector.to(device, dtype=torch.bfloat16).eval()

print(f"\n[3/4] Loading tokenizer: {args.llm_name}")
tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"\n[4/4] Loading LLM: {args.llm_name}")
bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
base_llm = AutoModelForCausalLM.from_pretrained(args.llm_name, quantization_config=bnb_cfg,
                                                  device_map="auto", torch_dtype=torch.bfloat16)
if os.path.exists(args.lora_ckpt):
    print(f"  Loading SFT LoRA from {args.lora_ckpt}...", flush=True)
    llm = PeftModel.from_pretrained(base_llm, args.lora_ckpt, local_files_only=True)
    print(f"  SFT LoRA loaded.", flush=True)
else:
    llm = base_llm
    print("  No SFT LoRA — using base model")

llm.eval()
print("\nAll models loaded. Starting GUI...\n")

# ── encode image → visual embeddings ─────────────────────────────────────────
def encode_image(pil_image):
    x = transform(pil_image.convert("RGB")).unsqueeze(0).to(device, dtype=torch.bfloat16)
    with torch.no_grad():
        enc_out = encoder(x, token_budget=args.token_budget)
        visual_embeds = projector(enc_out)   # (1, N_vis, 4096)
    return visual_embeds

# ── generate answer ───────────────────────────────────────────────────────────
def generate_answer(visual_embeds, question):
    prompt = f"[INST] <image>\n{question} [/INST]"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        text_embeds = llm.get_input_embeddings()(input_ids)  # (1, T, 4096)
        # find <image> token position and replace with visual tokens
        img_token_id = tokenizer.convert_tokens_to_ids("<image>")
        img_positions = (input_ids[0] == img_token_id).nonzero(as_tuple=True)[0]

        if len(img_positions) > 0:
            pos = img_positions[0].item()
            combined = torch.cat([
                text_embeds[:, :pos],
                visual_embeds,
                text_embeds[:, pos+1:]
            ], dim=1)
        else:
            combined = torch.cat([visual_embeds, text_embeds], dim=1)

        out_ids = llm.generate(
            inputs_embeds=combined,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
    # strip echoed prompt if present
    if "[/INST]" in answer:
        answer = answer.split("[/INST]")[-1].strip()
    return answer

# ── Gradio state ──────────────────────────────────────────────────────────────
current_visual_embeds = [None]
current_ocr_text = [None]

def run_ocr(pil_image):
    try:
        img_np = np.array(pil_image.convert("RGB"))
        results = ocr_reader.readtext(img_np, detail=0, paragraph=True)
        text = " ".join(results).strip()
        return text if text else None
    except Exception:
        return None

def load_image(pil_image, use_ocr):
    if pil_image is None:
        return [], None, "No image loaded.", ""
    current_visual_embeds[0] = encode_image(pil_image)
    ocr_text = None
    ocr_display = ""
    if use_ocr:
        ocr_text = run_ocr(pil_image)
        ocr_display = ocr_text if ocr_text else "(no text detected)"
    current_ocr_text[0] = ocr_text
    status_msg = "✅ Image loaded."
    if use_ocr:
        status_msg += f" OCR: {'found text' if ocr_text else 'no text found'}"
    return [], current_visual_embeds[0], status_msg, ocr_display

def chat(message, history):
    if not message.strip():
        return history, ""
    if current_visual_embeds[0] is None:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "⚠️ Please upload an image first."})
        return history, ""
    if current_ocr_text[0]:
        enriched_question = f"The image contains this text: \"{current_ocr_text[0]}\"\n\n{message}"
    else:
        enriched_question = message
    answer = generate_answer(current_visual_embeds[0], enriched_question)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return history, ""

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Keural VLM Chat") as demo:
    gr.Markdown("# Keural VLM — Vision Chat\n**24.7M encoder · SFT 30K steps · DPO aligned · OCR enabled**\nUpload an image, then ask questions about it.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image", height=350)
            use_ocr = gr.Checkbox(label="Enable OCR (extract text from image)", value=True)
            ocr_output = gr.Textbox(label="OCR Extracted Text", interactive=False, lines=3)
            status = gr.Textbox(label="Status", value="Upload an image to begin.", interactive=False)
            visual_state = gr.State(None)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", height=450)
            with gr.Row():
                question_input = gr.Textbox(
                    placeholder="Ask something about the image...",
                    label="Your question",
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            gr.Examples(
                examples=[
                    ["What is in this image?"],
                    ["Describe this image in detail."],
                    ["What text is in this image?"],
                    ["Read the text in this image."],
                    ["What color is the main object?"],
                    ["Is there a person in this image?"],
                    ["How many objects can you count?"],
                    ["What is happening in this image?"],
                ],
                inputs=question_input,
                label="Quick questions"
            )

    image_input.change(load_image, inputs=[image_input, use_ocr],
                       outputs=[chatbot, visual_state, status, ocr_output])
    send_btn.click(chat, inputs=[question_input, chatbot],
                   outputs=[chatbot, question_input])
    question_input.submit(chat, inputs=[question_input, chatbot],
                          outputs=[chatbot, question_input])

demo.launch(server_name="0.0.0.0", server_port=args.port, share=True,
            js="() => { document.documentElement.lang = 'en'; }")
