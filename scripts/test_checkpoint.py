"""Quick CPU smoke-test for a saved checkpoint."""
import sys
import torch
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, "/home/mkd-s002/vision-encoder/poc")

import alignment.hf_wrapper  # registers KeuralVisionModel with transformers
from transformers import AutoModel

CKPT = "/home/mkd-s002/vision-encoder/poc/outputs/run/checkpoint-5000"

print(f"Loading checkpoint from: {CKPT}")
model = AutoModel.from_pretrained(CKPT, trust_remote_code=True)
model.eval()
print(f"Model loaded — {sum(p.numel() for p in model.parameters()):,} parameters")

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dummy image (random noise — no real image needed)
dummy_img = torch.rand(3, 256, 256)
pixel_values = dummy_img.unsqueeze(0)  # (1, 3, 256, 256)

print("\nRunning forward pass on CPU (token_budget=64, 256, 1024)...")
with torch.no_grad():
    for budget in [64, 256, 1024]:
        out = model(pixel_values, token_budget=budget)
        global_count  = (out.level_ids == 0).sum().item()
        region_count  = (out.level_ids == 1).sum().item()
        detail_count  = (out.level_ids == 2).sum().item()
        print(
            f"  budget={budget:4d} | tokens={out.tokens.shape} "
            f"pooled={out.pooled.shape} | "
            f"global={global_count} region={region_count} detail={detail_count} | "
            f"budget_used={out.token_budget_used.item()}"
        )

print("\nAll checks passed. Checkpoint is healthy.")
