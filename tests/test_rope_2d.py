"""Unit tests for architecture/spatial/rope_2d.py."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from architecture.spatial.rope_2d import RoPE2DFromMetadata, compute_scale_bias


def _make_meta(
    cx_values: list[float], cy_values: list[float], w: float = 0.1, h: float = 0.1
) -> torch.Tensor:
    """Helper: make spatial_metadata (1, N, 4) from cx/cy lists."""
    assert len(cx_values) == len(cy_values)
    N = len(cx_values)
    meta = torch.zeros(1, N, 4)
    for i in range(N):
        meta[0, i, 0] = cx_values[i]
        meta[0, i, 1] = cy_values[i]
        meta[0, i, 2] = w
        meta[0, i, 3] = h
    return meta


class TestRoPE2DConstruction:
    def test_default_construction(self):
        rope = RoPE2DFromMetadata(head_dim=64)
        assert rope.head_dim == 64
        assert rope.half_dim == 32

    def test_rejects_invalid_head_dim(self):
        with pytest.raises(ValueError, match="divisible by 4"):
            RoPE2DFromMetadata(head_dim=66)


class TestRoPEShapes:
    def test_basic_apply(self):
        rope = RoPE2DFromMetadata(head_dim=32)
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 8, 32)
        meta = torch.rand(2, 8, 4)
        q_r, k_r = rope.rotate_qk(q, k, meta)
        assert q_r.shape == q.shape
        assert k_r.shape == k.shape

    def test_dim_mismatch_raises(self):
        rope = RoPE2DFromMetadata(head_dim=32)
        q = torch.randn(1, 4, 8, 64)  # head_dim=64 not 32
        k = torch.randn(1, 4, 8, 64)
        meta = torch.rand(1, 8, 4)
        with pytest.raises(ValueError, match="head_dim"):
            rope.rotate_qk(q, k, meta)

    def test_metadata_shape_mismatch_raises(self):
        rope = RoPE2DFromMetadata(head_dim=32)
        q = torch.randn(1, 4, 8, 32)
        k = torch.randn(1, 4, 8, 32)
        meta = torch.rand(1, 4, 4)  # wrong N
        with pytest.raises(ValueError, match="spatial_metadata"):
            rope.rotate_qk(q, k, meta)


class TestRoPEPositionEffect:
    """The whole point of RoPE is that two tokens at different positions
    produce different rotations. Test this directly."""

    def test_different_positions_produce_different_rotations(self):
        rope = RoPE2DFromMetadata(head_dim=32)
        q = torch.ones(1, 4, 2, 32)
        k = torch.ones(1, 4, 2, 32)
        # Two tokens, very different positions.
        meta = _make_meta(cx_values=[0.0, 0.5], cy_values=[0.0, 0.7])
        q_r, _ = rope.rotate_qk(q, k, meta)
        # The two tokens should now have different Q vectors (different rotations).
        assert not torch.allclose(q_r[:, :, 0], q_r[:, :, 1], atol=1e-3)

    def test_same_position_produces_same_rotation(self):
        rope = RoPE2DFromMetadata(head_dim=32)
        q = torch.ones(1, 4, 2, 32)
        k = torch.ones(1, 4, 2, 32)
        meta = _make_meta(cx_values=[0.5, 0.5], cy_values=[0.5, 0.5])
        q_r, _ = rope.rotate_qk(q, k, meta)
        assert torch.allclose(q_r[:, :, 0], q_r[:, :, 1], atol=1e-5)

    def test_zero_position_is_identity(self):
        """A token at position (0, 0) gets rotation=0, so the value is unchanged."""
        rope = RoPE2DFromMetadata(head_dim=32)
        q = torch.randn(1, 4, 1, 32)
        k = torch.randn(1, 4, 1, 32)
        meta = _make_meta(cx_values=[0.0], cy_values=[0.0])
        q_r, k_r = rope.rotate_qk(q, k, meta)
        assert torch.allclose(q_r, q, atol=1e-5)
        assert torch.allclose(k_r, k, atol=1e-5)


class TestRoPEResolutionInvariance:
    """The headline property: positions in [0, 1] are comparable across resolutions.
    A token at (0.5, 0.5) on a 224^2 image should get the same RoPE treatment
    as a token at (0.5, 0.5) on a 1024^2 image."""

    def test_normalized_positions_yield_same_output(self):
        rope = RoPE2DFromMetadata(head_dim=64)
        torch.manual_seed(0)
        q = torch.randn(1, 4, 1, 64)
        k = torch.randn(1, 4, 1, 64)
        meta_low_res = _make_meta(cx_values=[0.5], cy_values=[0.5])
        meta_high_res = _make_meta(cx_values=[0.5], cy_values=[0.5])
        q_low, _ = rope.rotate_qk(q, k, meta_low_res)
        q_high, _ = rope.rotate_qk(q, k, meta_high_res)
        # Identical positions => identical RoPE => identical output.
        assert torch.allclose(q_low, q_high, atol=1e-6)


class TestRoPEGradient:
    def test_gradients_flow_through_q(self):
        rope = RoPE2DFromMetadata(head_dim=32)
        q = torch.randn(1, 4, 4, 32, requires_grad=True)
        k = torch.randn(1, 4, 4, 32)
        meta = torch.rand(1, 4, 4)
        q_r, _ = rope.rotate_qk(q, k, meta)
        q_r.sum().backward()
        assert q.grad is not None
        assert torch.isfinite(q.grad).all()


# ----- Scale bias -----


class TestScaleBias:
    def test_basic_shape(self):
        meta = _make_meta(cx_values=[0.5, 0.5], cy_values=[0.5, 0.5], w=0.1, h=0.1)
        emb = nn.Parameter(torch.randn(8, 32))  # 8 buckets, dim 32
        bias = compute_scale_bias(meta, bias_dim=32, learned_scale_embedding=emb)
        assert bias.shape == (1, 2, 32)

    def test_different_scales_get_different_bias(self):
        # Token A: tiny region (w=h=0.05). Token B: full image (w=h=1.0).
        meta = _make_meta(cx_values=[0.0, 0.0], cy_values=[0.0, 0.0], w=0.05, h=0.05)
        meta2 = _make_meta(cx_values=[0.0], cy_values=[0.0], w=1.0, h=1.0)
        meta_combined = torch.cat([meta, meta2], dim=1)  # (1, 3, 4)

        torch.manual_seed(0)
        emb = nn.Parameter(torch.randn(8, 16))
        bias = compute_scale_bias(meta_combined, bias_dim=16, learned_scale_embedding=emb)
        # First two tokens (tiny scale) should get same bias; third (full scale) different.
        assert torch.allclose(bias[0, 0], bias[0, 1])
        assert not torch.allclose(bias[0, 0], bias[0, 2])

    def test_gradient_flows_to_embedding(self):
        meta = _make_meta(cx_values=[0.5], cy_values=[0.5], w=0.1, h=0.1)
        emb = nn.Parameter(torch.randn(8, 16))
        bias = compute_scale_bias(meta, bias_dim=16, learned_scale_embedding=emb)
        bias.sum().backward()
        assert emb.grad is not None
        assert torch.isfinite(emb.grad).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
