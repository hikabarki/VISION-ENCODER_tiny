"""
LevelAwareProjector — self-contained, no external dependencies.
Bridges Keural vision encoder (384-dim) to LLM embedding space (4096-dim).
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class EncoderOutput:
    tokens:   torch.Tensor             # (B, N, D)
    level_ids: torch.Tensor            # (B, N)  values in {-1, 0, 1, 2}
    attention_mask: Optional[torch.Tensor] = None


class LevelAwareProjector(nn.Module):
    """
    Projects Keural encoder tokens into LLM embedding space.
    Adds a learned per-level bias so the LLM knows which type of
    token it is consuming (global / region / detail).

    Args:
        encoder_dim : Keural output dim (default 384)
        hidden_dim  : MLP hidden dim (default 2048)
        llm_dim     : LLM embedding dim (default 4096 for Mistral 7B)
        num_levels  : Number of HCT levels (default 3)
    """

    def __init__(
        self,
        encoder_dim: int = 384,
        hidden_dim: int  = 2048,
        llm_dim: int     = 4096,
        num_levels: int  = 3,
    ) -> None:
        super().__init__()
        self.fc1         = nn.Linear(encoder_dim, hidden_dim)
        self.act         = nn.GELU()
        self.fc2         = nn.Linear(hidden_dim, llm_dim)
        self.level_embed = nn.Embedding(num_levels + 1, llm_dim)
        nn.init.zeros_(self.level_embed.weight)

    def forward(self, output: EncoderOutput) -> torch.Tensor:
        projected  = self.fc2(self.act(self.fc1(output.tokens)))
        level_bias = self.level_embed(output.level_ids + 1)
        return projected + level_bias
