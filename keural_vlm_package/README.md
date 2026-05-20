# keural-vlm

Python package for the Keural Vision-Language Model.
Install once, use anywhere — no manual git clones or path hacks.

## Install

```bash
pip install git+https://github.com/mkd-hika/VISION-ENCODER.git#subdirectory=keural_vlm_package
```

## Usage

```python
from keural_vlm import KeuralVLM

vlm = KeuralVLM()

# single question
answer = vlm.ask("photo.jpg", "What is in this image?")
print(answer)

# multiple questions (image encoded once)
answers = vlm.chat("photo.jpg", [
    "What is in this image?",
    "What color is the main object?",
    "Is this indoors or outdoors?",
])
for a in answers:
    print(a)
```

## Requirements

- Python 3.10+
- CUDA GPU with 12 GB+ VRAM
- HuggingFace account (to download Mistral 7B)

## Models

Automatically downloaded from HuggingFace on first use:
- Vision encoder: `mkd-hika/keural-vision-encoder-poc`
- Projector + LoRA: `mkd-hika/keural-vlm-poc`
- LLM: `mistralai/Mistral-7B-Instruct-v0.3`
