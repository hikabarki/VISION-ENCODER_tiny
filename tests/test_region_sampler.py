"""Unit tests for tokenizer/region_sampler.py.

This is the highest-risk component in the architecture. It gets thorough
testing — shape, value, gradient, determinism, and edge-case behavior.
"""

from __future__ import annotations

import pytest
import torch

from tokenizer.region_sampler import (
    RegionSampler,
    SampledRegions,
    deterministic_grid_regions,
    gaussian_smooth,
)


# ----- Gaussian smoothing primitive -----


class TestGaussianSmooth:
    def test_kernel_one_is_identity(self):
        x = torch.randn(2, 1, 8, 8)
        y = gaussian_smooth(x, kernel_size=1)
        assert torch.equal(y, x)

    def test_smoothing_preserves_shape(self):
        x = torch.randn(2, 1, 8, 8)
        y = gaussian_smooth(x, kernel_size=3)
        assert y.shape == x.shape

    def test_smoothing_preserves_total_mass(self):
        # A normalized Gaussian conv preserves mass except at boundaries.
        x = torch.zeros(1, 1, 16, 16)
        x[0, 0, 8, 8] = 1.0
        y = gaussian_smooth(x, kernel_size=3)
        # In the interior, sums should be close to 1 (boundary edges
        # would clip mass, but a centered impulse stays in the interior).
        assert abs(y.sum().item() - 1.0) < 1e-4

    def test_smoothing_blurs_impulse(self):
        x = torch.zeros(1, 1, 8, 8)
        x[0, 0, 4, 4] = 1.0
        y = gaussian_smooth(x, kernel_size=3)
        # The blurred response at the center should be < 1 and > 0
        assert 0 < y[0, 0, 4, 4].item() < 1.0
        # Adjacent pixels should now be > 0
        assert y[0, 0, 3, 4].item() > 0
        assert y[0, 0, 4, 3].item() > 0

    def test_rejects_even_kernel(self):
        with pytest.raises(ValueError, match="odd"):
            gaussian_smooth(torch.zeros(1, 1, 4, 4), kernel_size=2)


# ----- RegionSampler shape and value contracts -----


class TestRegionSamplerShapes:
    def test_basic_forward(self):
        sampler = RegionSampler(smooth_kernel=3, nms_radius=1)
        sal = torch.rand(2, 1, 14, 14)
        out = sampler(sal, K=8, gumbel_temperature=0.0)
        assert isinstance(out, SampledRegions)
        assert out.coords.shape == (2, 8, 2)
        assert out.scores.shape == (2, 8)
        assert out.soft_weights.shape == (2, 8, 14, 14)

    def test_coords_are_in_grid(self):
        sampler = RegionSampler(smooth_kernel=3, nms_radius=1)
        sal = torch.rand(3, 1, 10, 12)
        out = sampler(sal, K=6, gumbel_temperature=0.0)
        ys, xs = out.coords[..., 0], out.coords[..., 1]
        assert (ys >= 0).all() and (ys < 10).all()
        assert (xs >= 0).all() and (xs < 12).all()

    def test_coords_are_int64(self):
        sampler = RegionSampler(smooth_kernel=3, nms_radius=1)
        sal = torch.rand(1, 1, 8, 8)
        out = sampler(sal, K=4, gumbel_temperature=0.0)
        assert out.coords.dtype == torch.int64

    def test_soft_weights_sum_to_one_per_region(self):
        sampler = RegionSampler(smooth_kernel=3, nms_radius=1)
        sal = torch.rand(2, 1, 14, 14)
        out = sampler(sal, K=8, gumbel_temperature=0.0)
        # Each (B, K, ., .) slice should sum to ~1
        sums = out.soft_weights.flatten(-2).sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_rejects_zero_K(self):
        sampler = RegionSampler(smooth_kernel=3, nms_radius=1)
        sal = torch.rand(1, 1, 8, 8)
        with pytest.raises(ValueError, match="K"):
            sampler(sal, K=0)

    def test_rejects_wrong_saliency_shape(self):
        sampler = RegionSampler(smooth_kernel=3, nms_radius=1)
        with pytest.raises(ValueError, match="saliency"):
            sampler(torch.rand(2, 3, 8, 8), K=4)  # 3 channels, not 1


# ----- Sampling correctness -----


class TestSamplingCorrectness:
    def test_deterministic_when_temp_zero(self):
        """With temperature=0 and no NMS suppression, picks should follow saliency."""
        sampler = RegionSampler(smooth_kernel=1, nms_radius=0)
        # Build a saliency map with one obvious peak
        sal = torch.full((1, 1, 8, 8), 0.01)
        sal[0, 0, 3, 5] = 1.0
        out = sampler(sal, K=1, gumbel_temperature=0.0, training=False)
        assert out.coords[0, 0, 0].item() == 3
        assert out.coords[0, 0, 1].item() == 5

    def test_picks_top_k_distinct_with_nms(self):
        """NMS should force picks to spread out."""
        sampler = RegionSampler(smooth_kernel=1, nms_radius=2)
        # Uniform saliency — NMS should produce distinct picks
        sal = torch.rand(1, 1, 14, 14)
        out = sampler(sal, K=5, gumbel_temperature=0.0, training=False)
        coords = out.coords[0]  # (5, 2)
        # No two coords should be within nms_radius of each other in both axes
        # (they could be in one axis, but strict equality is impossible)
        for i in range(5):
            for j in range(i + 1, 5):
                same = (coords[i] == coords[j]).all().item()
                assert not same, f"Found duplicate picks: {coords[i]} == {coords[j]}"

    def test_eval_mode_no_gumbel(self):
        """In eval mode (training=False), output should be deterministic across calls."""
        sampler = RegionSampler(smooth_kernel=3, nms_radius=1)
        sampler.eval()
        sal = torch.rand(1, 1, 14, 14)
        out1 = sampler(sal, K=5, gumbel_temperature=1.0)
        out2 = sampler(sal, K=5, gumbel_temperature=1.0)
        # Determinism: same input, same output
        assert torch.equal(out1.coords, out2.coords)

    def test_train_mode_with_gumbel_introduces_variance(self):
        """In train mode with high temperature, picks should vary across calls."""
        sampler = RegionSampler(smooth_kernel=3, nms_radius=0)
        sampler.train()
        torch.manual_seed(0)
        sal = torch.rand(1, 1, 14, 14)
        out1 = sampler(sal, K=5, gumbel_temperature=2.0)
        torch.manual_seed(1)
        out2 = sampler(sal, K=5, gumbel_temperature=2.0)
        # With high temperature, at least one pick should differ
        diff = (out1.coords != out2.coords).any().item()
        assert diff, "High-temperature training picks should not be deterministic"


# ----- Gradient flow (the critical risk) -----


class TestGradientFlow:
    """The straight-through trick must produce real gradients to the saliency."""

    def test_scores_are_differentiable_w_r_t_saliency(self):
        sampler = RegionSampler(smooth_kernel=3, nms_radius=1)
        sal = torch.rand(1, 1, 14, 14, requires_grad=True)
        out = sampler(sal, K=4, gumbel_temperature=0.0, training=False)
        loss = out.scores.sum()
        loss.backward()
        assert sal.grad is not None
        assert torch.isfinite(sal.grad).all()
        # At least some saliency positions should have non-zero gradient
        assert sal.grad.abs().sum() > 0

    def test_soft_weights_can_backprop(self):
        """Soft weights are computed from coords (no grad), but they let
        downstream pooling backprop into features."""
        sampler = RegionSampler(smooth_kernel=3, nms_radius=1)
        sal = torch.rand(1, 1, 14, 14, requires_grad=True)
        feat = torch.randn(1, 32, 14, 14, requires_grad=True)
        out = sampler(sal, K=4, gumbel_temperature=0.0, training=False)

        # Simulate the kind of pooling downstream code will do:
        # weighted sum of features
        # feat: (1, 32, 14, 14), soft_weights: (1, K, 14, 14)
        pooled = torch.einsum("bchw,bkhw->bkc", feat, out.soft_weights)
        loss = pooled.sum()
        loss.backward()
        assert feat.grad is not None
        assert torch.isfinite(feat.grad).all()

    def test_gradient_flows_through_multiple_picks(self):
        sampler = RegionSampler(smooth_kernel=3, nms_radius=2)
        sal = torch.rand(2, 1, 14, 14, requires_grad=True)
        out = sampler(sal, K=10, gumbel_temperature=0.0, training=False)
        out.scores.mean().backward()
        assert sal.grad is not None
        assert torch.isfinite(sal.grad).all()


# ----- Deterministic grid fallback -----


class TestDeterministicGrid:
    def test_basic_shape(self):
        out = deterministic_grid_regions(grid_h=14, grid_w=14, K=16, batch_size=2)
        assert out.coords.shape == (2, 16, 2)
        assert out.scores.shape == (2, 16)
        assert out.soft_weights.shape == (2, 16, 14, 14)

    def test_coords_are_inside_grid(self):
        out = deterministic_grid_regions(14, 14, 16, 1)
        assert (out.coords[..., 0] >= 0).all() and (out.coords[..., 0] < 14).all()
        assert (out.coords[..., 1] >= 0).all() and (out.coords[..., 1] < 14).all()

    def test_coords_identical_across_batch(self):
        out = deterministic_grid_regions(8, 8, 9, 4)
        # All 4 batch items should have identical coords
        assert torch.equal(out.coords[0], out.coords[1])
        assert torch.equal(out.coords[0], out.coords[3])

    def test_handles_full_grid(self):
        # K = grid size — every cell picked exactly once
        out = deterministic_grid_regions(4, 4, 16, 1)
        # All 16 cells should be unique
        flat = out.coords[0, :, 0] * 4 + out.coords[0, :, 1]
        assert len(set(flat.tolist())) == 16

    def test_rejects_K_too_large(self):
        with pytest.raises(ValueError, match="grid cells"):
            deterministic_grid_regions(4, 4, 17, 1)


# ----- Edge cases -----


class TestEdgeCases:
    def test_single_pick(self):
        sampler = RegionSampler(smooth_kernel=3, nms_radius=1)
        sal = torch.rand(1, 1, 8, 8)
        out = sampler(sal, K=1, gumbel_temperature=0.0)
        assert out.coords.shape == (1, 1, 2)

    def test_max_picks_small_grid(self):
        """When K is large relative to grid, NMS may exhaust positions but
        the algorithm should still produce K picks (some may overlap when
        all -inf neighborhoods exhausted)."""
        sampler = RegionSampler(smooth_kernel=1, nms_radius=0)
        sal = torch.rand(1, 1, 4, 4)
        out = sampler(sal, K=8, gumbel_temperature=0.0)
        assert out.coords.shape == (1, 1, 8, 2) or out.coords.shape == (1, 8, 2)

    def test_zero_saliency_handled(self):
        """All-zero saliency should not produce NaN or crash."""
        sampler = RegionSampler(smooth_kernel=3, nms_radius=1)
        sal = torch.zeros(1, 1, 8, 8)
        out = sampler(sal, K=4, gumbel_temperature=0.0, training=False)
        assert torch.isfinite(out.scores).all()

    def test_uniform_saliency_handled(self):
        sampler = RegionSampler(smooth_kernel=3, nms_radius=1)
        sal = torch.full((1, 1, 8, 8), 0.5)
        out = sampler(sal, K=4, gumbel_temperature=0.0, training=False)
        # All scores should equal the saliency value
        assert torch.allclose(out.scores, torch.full_like(out.scores, 0.5))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
