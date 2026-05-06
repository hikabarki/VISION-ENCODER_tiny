"""Unit tests for alignment/vlm_helpers.py."""

from __future__ import annotations

import pytest
import torch

from alignment.output_contract import (
    LEVEL_DETAIL,
    LEVEL_GLOBAL,
    LEVEL_REGION,
    empty_output,
)
from alignment.projectors import LevelAwareProjector, SimpleProjector
from alignment.vlm_helpers import VLMInputs, build_vlm_inputs


def _make_output(B: int = 2, N: int = 8, D: int = 32):
    out = empty_output(B, N, D)
    out.attention_mask[:] = 1
    out.level_ids[:, :2] = LEVEL_GLOBAL
    out.level_ids[:, 2:4] = LEVEL_REGION
    out.level_ids[:, 4:] = LEVEL_DETAIL
    out.tokens = torch.randn(B, N, D)
    out.spatial_metadata = torch.rand(B, N, 4)
    out.saliency_scores = torch.rand(B, N)
    return out


class TestBuildVLMInputs:
    def test_basic_shapes(self):
        out = _make_output(B=2, N=8, D=32)
        proj = SimpleProjector(encoder_dim=32, llm_dim=64)
        text = torch.randn(2, 5, 64)
        text_mask = torch.ones(2, 5, dtype=torch.int64)

        result = build_vlm_inputs(out, proj, text, text_mask)
        assert isinstance(result, VLMInputs)
        # 8 vision + 5 text = 13 total
        assert result.inputs_embeds.shape == (2, 13, 64)
        assert result.attention_mask.shape == (2, 13)
        assert result.position_ids.shape == (2, 13)
        assert result.vision_token_count.shape == (2,)

    def test_vision_first_text_second(self):
        """Vision tokens come BEFORE text tokens in the concatenated sequence."""
        out = _make_output(B=1, N=4, D=8)
        # Mark vision tokens with a known value
        out.tokens[:] = 1.0
        proj = SimpleProjector(encoder_dim=8, llm_dim=8)
        # Make projector approximately identity (but with bias from MLP).
        with torch.no_grad():
            proj.fc1.weight.copy_(torch.eye(8))
            proj.fc1.bias.zero_()
            proj.fc2.weight.copy_(torch.eye(8))
            proj.fc2.bias.zero_()
        # Text tokens with distinct value (-2.0)
        text = torch.full((1, 3, 8), -2.0)
        text_mask = torch.ones(1, 3, dtype=torch.int64)
        result = build_vlm_inputs(out, proj, text, text_mask)
        # Position 0..3 (vision) should be derived from +1 input through GELU MLP
        # Position 4..6 (text) should be exactly -2.0
        assert torch.allclose(result.inputs_embeds[0, 4], torch.full((8,), -2.0))
        assert torch.allclose(result.inputs_embeds[0, 5], torch.full((8,), -2.0))
        assert torch.allclose(result.inputs_embeds[0, 6], torch.full((8,), -2.0))

    def test_attention_mask_combined(self):
        out = _make_output(B=2, N=4, D=16)
        # Mark some vision tokens as padding
        out.attention_mask[0, 3] = 0
        out.attention_mask[1, 2:] = 0
        proj = SimpleProjector(encoder_dim=16, llm_dim=32)
        text = torch.randn(2, 5, 32)
        text_mask = torch.tensor(
            [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.int64
        )
        result = build_vlm_inputs(out, proj, text, text_mask)
        # Item 0: 3 valid vision + 5 valid text = 8 real tokens
        # Item 1: 2 valid vision + 3 valid text = 5 real tokens
        assert result.attention_mask[0].sum().item() == 8
        assert result.attention_mask[1].sum().item() == 5

    def test_vision_token_count(self):
        out = _make_output(B=2, N=8, D=16)
        # Item 0: 5 valid; item 1: 3 valid
        out.attention_mask[0, 5:] = 0
        out.attention_mask[1, 3:] = 0
        proj = SimpleProjector(encoder_dim=16, llm_dim=32)
        text = torch.randn(2, 4, 32)
        text_mask = torch.ones(2, 4, dtype=torch.int64)
        result = build_vlm_inputs(out, proj, text, text_mask)
        assert result.vision_token_count.tolist() == [5, 3]

    def test_position_ids_sequential(self):
        out = _make_output(B=1, N=3, D=8)
        proj = SimpleProjector(encoder_dim=8, llm_dim=8)
        text = torch.randn(1, 4, 8)
        text_mask = torch.ones(1, 4, dtype=torch.int64)
        result = build_vlm_inputs(out, proj, text, text_mask)
        # Positions: 0, 1, 2, 3, 4, 5, 6
        assert result.position_ids[0].tolist() == [0, 1, 2, 3, 4, 5, 6]

    def test_works_with_level_aware_projector(self):
        out = _make_output(B=2, N=8, D=32)
        proj = LevelAwareProjector(encoder_dim=32, llm_dim=64)
        text = torch.randn(2, 5, 64)
        text_mask = torch.ones(2, 5, dtype=torch.int64)
        result = build_vlm_inputs(out, proj, text, text_mask)
        assert result.inputs_embeds.shape == (2, 13, 64)

    def test_dtype_consistency(self):
        """The combined output should match the dtype of text_embeds."""
        out = _make_output(B=1, N=4, D=8)
        proj = SimpleProjector(encoder_dim=8, llm_dim=16)
        text = torch.randn(1, 3, 16, dtype=torch.float16)
        text_mask = torch.ones(1, 3, dtype=torch.int64)
        result = build_vlm_inputs(out, proj, text, text_mask)
        assert result.inputs_embeds.dtype == torch.float16

    def test_batch_size_mismatch_raises(self):
        out = _make_output(B=2, N=4, D=8)
        proj = SimpleProjector(encoder_dim=8, llm_dim=16)
        text = torch.randn(3, 4, 16)  # wrong batch
        text_mask = torch.ones(3, 4, dtype=torch.int64)
        with pytest.raises(ValueError, match="Batch size mismatch"):
            build_vlm_inputs(out, proj, text, text_mask)

    def test_gradients_flow_through_chain(self):
        out = _make_output(B=1, N=4, D=8)
        out.tokens.requires_grad_(True)
        proj = SimpleProjector(encoder_dim=8, llm_dim=16)
        text = torch.randn(1, 3, 16, requires_grad=True)
        text_mask = torch.ones(1, 3, dtype=torch.int64)
        result = build_vlm_inputs(out, proj, text, text_mask)
        result.inputs_embeds.sum().backward()
        assert out.tokens.grad is not None
        assert text.grad is not None
        assert torch.isfinite(out.tokens.grad).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
