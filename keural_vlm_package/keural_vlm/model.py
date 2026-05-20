"""
KeuralVLM — plug-and-play Vision-Language Model.

from keural_vlm import KeuralVLM
vlm = KeuralVLM()
answer = vlm.ask("photo.jpg", "What is in this image?")
"""

from __future__ import annotations
import os
import torch
import torchvision.transforms as T
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download

from .projector import LevelAwareProjector, EncoderOutput


class KeuralVLM:
    """
    Keural Vision-Language Model.

    Loads everything automatically from HuggingFace on first use:
      - Vision encoder : mkd-hika/keural-vision-encoder-poc
      - Projector      : mkd-hika/keural-vlm-poc/projector.pt
      - LoRA adapter   : mkd-hika/keural-vlm-poc/lora_adapter/
      - LLM            : mistralai/Mistral-7B-Instruct-v0.3

    Args:
        device         : "cuda" or "cpu". Auto-detects if None.
        load_in_4bit   : 4-bit NF4 quantization (needs ~12GB VRAM).
        max_new_tokens : Maximum tokens to generate per answer.
        token_budget   : Number of visual tokens from the encoder.

    Example:
        vlm = KeuralVLM()
        print(vlm.ask("cat.jpg", "What animal is this?"))
    """

    ENCODER_REPO = "mkd-hika/keural-vision-encoder-poc"
    VLM_REPO     = "mkd-hika/keural-vlm-poc"
    LLM_NAME     = "mistralai/Mistral-7B-Instruct-v0.3"

    _TRANSFORM = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    def __init__(
        self,
        device: str | None  = None,
        load_in_4bit: bool  = True,
        max_new_tokens: int = 256,
        token_budget: int   = 256,
    ) -> None:
        self.device         = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_in_4bit   = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.token_budget   = token_budget
        self._dtype         = torch.bfloat16
        self._loaded        = False

    def _load(self) -> None:
        if self._loaded:
            return

        print("[KeuralVLM] Loading vision encoder...")
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(
            self.ENCODER_REPO,
            trust_remote_code=True,
            torch_dtype=self._dtype,
        ).to(self.device).eval()
        for mod in self.encoder.modules():
            if hasattr(mod, "inv_freq"):
                head_dim = mod.inv_freq.shape[0] * 2
                inv_freq = 1.0 / (10000.0 ** (
                    torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
                ))
                mod.inv_freq = inv_freq.to(self.device)

        print("[KeuralVLM] Loading projector...")
        projector_path = hf_hub_download(self.VLM_REPO, "projector.pt")
        self.projector = LevelAwareProjector()
        self.projector.load_state_dict(
            torch.load(projector_path, map_location=self.device)
        )
        self.projector = self.projector.to(self.device, dtype=self._dtype).eval()

        print("[KeuralVLM] Loading LLM...")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        self.tokenizer = AutoTokenizer.from_pretrained(self.LLM_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict = dict(torch_dtype=self._dtype, device_map="auto")
        if self.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self._dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        base_llm   = AutoModelForCausalLM.from_pretrained(self.LLM_NAME, **load_kwargs)
        lora_dir   = snapshot_download(self.VLM_REPO)
        lora_path  = os.path.join(lora_dir, "lora_adapter")
        self.llm   = PeftModel.from_pretrained(base_llm, lora_path)
        self.llm.eval()

        self._loaded = True
        print("[KeuralVLM] Ready.\n")

    # ── public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_image(self, image) -> torch.Tensor:
        """
        Encode an image into visual embeddings.

        Args:
            image : file path (str), PIL.Image, or numpy array

        Returns:
            torch.Tensor of shape (1, N_vis, 4096)
        """
        self._load()
        if isinstance(image, str):
            image = Image.open(image)
        if not isinstance(image, torch.Tensor):
            image = image.convert("RGB")
            x = self._TRANSFORM(image).unsqueeze(0).to(self.device, dtype=self._dtype)
        else:
            x = image.unsqueeze(0).to(self.device, dtype=self._dtype)

        enc_out = self.encoder(x, token_budget=self.token_budget)
        return self.projector(enc_out)

    @torch.no_grad()
    def ask(self, image, question: str) -> str:
        """
        Ask a question about an image.

        Args:
            image    : file path, PIL.Image, or pre-encoded tensor (1, N, 4096)
            question : str

        Returns:
            str — model answer
        """
        self._load()

        if isinstance(image, torch.Tensor) and image.shape[-1] == 4096:
            visual_embeds = image
        else:
            visual_embeds = self.encode_image(image)

        prompt    = f"[INST] <image>\n{question} [/INST]"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        txt_embs  = self.llm.get_input_embeddings()(input_ids)

        img_id  = self.tokenizer.convert_tokens_to_ids("<image>")
        pos_lst = (input_ids[0] == img_id).nonzero(as_tuple=True)[0]
        if len(pos_lst) > 0:
            p = pos_lst[0].item()
            combined = torch.cat(
                [txt_embs[:, :p], visual_embeds, txt_embs[:, p+1:]], dim=1
            )
        else:
            combined = torch.cat([visual_embeds, txt_embs], dim=1)

        out_ids = self.llm.generate(
            inputs_embeds=combined,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.3,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        answer = self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
        if "[/INST]" in answer:
            answer = answer.split("[/INST]")[-1].strip()
        return answer

    def chat(self, image, questions: list[str]) -> list[str]:
        """
        Ask multiple questions about the same image.
        Encodes the image once and reuses the embeddings.

        Args:
            image     : file path, PIL.Image
            questions : list of str

        Returns:
            list of str answers
        """
        embeds = self.encode_image(image)
        return [self.ask(embeds, q) for q in questions]
