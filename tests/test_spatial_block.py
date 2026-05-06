"""Unit tests for architecture/spatial/block.py."""

from __future__ import annotations

import pytest
import torch

from architecture.spatial.block import (
    RMSNorm,
    SpatialAttention,
    SpatialBlock,
    SwiGLUMLP,
)
from keural_config import SpatialConfig


# ----- RMSNorm -----


class TestRMSNorm:
    def test_basic_forward(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        y = norm(x)
        assert y.shape == x.shape

    def test_normalizes_to_unit_rms(self):
        norm = RMSNorm(64)
        x = torch.randn(1, 1, 64) * 100  # large magnitude
        y = norm(x)
        # After normalization with weight=1, RMS should be ~1
        rms = y.pow(2).mean(dim=-1).sqrt().item()
        assert abs(rms - 1.0) < 0.05

    def test_weight_scales_output(self):
        norm = RMSNorm(8)
        with torch.no_grad():
            norm.weight.fill_(2.0)
        x = torch.randn(1, 1, 8)
        y = norm(x)
        # With weight=2, RMS doubles
        rms = y.pow(2).mean(dim=-1).sqrt().item()
        assert abs(rms - 2.0) < 0.1

    def test_dtype_preserved(self):
        norm = RMSNorm(8)
        x = torch.randn(1, 4, 8, dtype=torch.float16)
        y = norm(x)
        assert y.dtype == torch.float16


# ----- SwiGLU MLP -----


class TestSwiGLUMLP:
    def test_basic_forward(self):
        mlp = SwiGLUMLP(embed_dim=64, mlp_ratio=4)
        x = torch.randn(2, 8, 64)
        y = mlp(x)
        assert y.shape == x.shape

    def test_gradients_flow(self):
        mlp = SwiGLUMLP(embed_dim=32, mlp_ratio=4)
        x = torch.randn(1, 4, 32, requires_grad=True)
        y = mlp(x)
        y.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_hidden_dim_is_multiple_of_64(self):
        mlp = SwiGLUMLP(embed_dim=100, mlp_ratio=4)
        # Inspect hidden dim by checking the gate matrix shape
        assert mlp.w_gate.out_features % 64 == 0


# ----- SpatialAttention -----


class TestSpatialAttention:
    def _make_attn(self, embed_dim: int = 64, num_heads: int = 4) -> SpatialAttention:
        return SpatialAttention(
            embed_dim=embed_dim, num_heads=num_heads, head_dim=embed_dim // num_heads
        )

    def test_basic_forward(self):
        attn = self._make_attn()
        x = torch.randn(2, 8, 64)
        meta = torch.rand(2, 8, 4)
        y = attn(x, meta)
        assert y.shape == x.shape

    def test_attention_mask_blocks_padding(self):
        """Tokens with attention_mask=0 should not influence the output of real tokens."""
        torch.manual_seed(0)
        attn = self._make_attn()

        # First call: 4 real tokens, 4 padding (with garbage values).
        x = torch.zeros(1, 8, 64)
        torch.manual_seed(1)
        x[0, :4] = torch.randn(4, 64)
        # Garbage in padding positions
        x[0, 4:] = torch.randn(4, 64) * 1000
        meta = torch.rand(1, 8, 4)
        mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])
        y_masked = attn(x, meta, attention_mask=mask)

        # Second call: same first 4 tokens, but no padding (only 4 tokens total).
        x2 = x[:, :4]
        meta2 = meta[:, :4]
        y_unmasked = attn(x2, meta2)

        # The output for the first 4 (real) tokens should match closely.
        assert torch.allclose(y_masked[0, :4], y_unmasked[0], atol=1e-5)

    def test_dim_mismatch_raises(self):
        with pytest.raises(ValueError, match="num_heads"):
            SpatialAttention(embed_dim=64, num_heads=5, head_dim=12)

    def test_gradients_flow(self):
        attn = self._make_attn()
        x = torch.randn(1, 4, 64, requires_grad=True)
        meta = torch.rand(1, 4, 4)
        y = attn(x, meta)
        y.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        for name, p in attn.named_parameters():
            assert p.grad is not None, f"No grad for {name}"


# ----- Full SpatialBlock -----


class TestSpatialBlock:
    def test_basic_forward(self):
        cfg = SpatialConfig(embed_dim=64, num_heads=4, head_dim=16)
        block = SpatialBlock(cfg)
        x = torch.randn(2, 8, 64)
        meta = torch.rand(2, 8, 4)
        y = block(x, meta)
        assert y.shape == x.shape

    def test_residual_connection_preserves_signal(self):
        """At init, the block should approximately preserve its input
        (residuals dominate before learning)."""
        cfg = SpatialConfig(embed_dim=64, num_heads=4, head_dim=16)
        block = SpatialBlock(cfg)
        # Zero out attn out_proj weights so the block becomes pure identity at init.
        with torch.no_grad():
            block.attn.out_proj.weight.zero_()
            block.mlp.w_down.weight.zero_()
        x = torch.randn(1, 8, 64)
        meta = torch.rand(1, 8, 4)
        y = block(x, meta)
        # With out projections zeroed, the block is exact identity.
        assert torch.allclose(y, x, atol=1e-5)

    def test_with_attention_mask(self):
        cfg = SpatialConfig(embed_dim=64, num_heads=4, head_dim=16)
        block = SpatialBlock(cfg)
        x = torch.randn(2, 8, 64)
        meta = torch.rand(2, 8, 4)
        mask = torch.ones(2, 8)
        mask[:, 6:] = 0  # last 2 positions padded
        y = block(x, meta, attention_mask=mask)
        assert y.shape == x.shape

    def test_full_gradient_flow(self):
        cfg = SpatialConfig(embed_dim=64, num_heads=4, head_dim=16)
        block = SpatialBlock(cfg)
        x = torch.randn(1, 8, 64, requires_grad=True)
        meta = torch.rand(1, 8, 4)
        y = block(x, meta)
        y.mean().backward()
        assert x.grad is not None
        for name, p in block.named_parameters():
            assert p.grad is not None, f"No grad for {name}"
            assert torch.isfinite(p.grad).all()

    def test_stack_of_blocks(self):
        """A stack of blocks should compose without shape errors."""
        cfg = SpatialConfig(embed_dim=64, num_heads=4, head_dim=16)
        blocks = torch.nn.ModuleList([SpatialBlock(cfg) for _ in range(4)])
        x = torch.randn(1, 8, 64)
        meta = torch.rand(1, 8, 4)
        for b in blocks:
            x = b(x, meta)
        assert x.shape == (1, 8, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
