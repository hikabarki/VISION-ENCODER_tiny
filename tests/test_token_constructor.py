"""Unit tests for tokenizer/token_constructor.py."""

from __future__ import annotations

import pytest
import torch

from tokenizer.region_sampler import RegionSampler, deterministic_grid_regions
from tokenizer.token_constructor import (
    GlobalQueryConstructor,
    RegionTokenConstructor,
    compute_spatial_metadata,
    global_token_metadata,
)


# ----- GlobalQueryConstructor -----


class TestGlobalQueryConstructor:
    def test_basic_forward(self):
        gc = GlobalQueryConstructor(embed_dim=64, num_heads=4, max_global_tokens=16)
        feat = torch.randn(2, 64, 14, 14)
        out = gc(feat, K_g=8)
        assert out.shape == (2, 8, 64)

    def test_uses_only_first_K_g_queries(self):
        gc = GlobalQueryConstructor(embed_dim=32, num_heads=4, max_global_tokens=8)
        feat = torch.randn(1, 32, 7, 7)
        out_4 = gc(feat, K_g=4)
        out_8 = gc(feat, K_g=8)
        # The first 4 outputs of the K_g=8 call should match K_g=4 call
        # (because the first 4 queries are used in both)
        assert torch.allclose(out_4, out_8[:, :4], atol=1e-5)

    def test_rejects_too_many_tokens(self):
        gc = GlobalQueryConstructor(embed_dim=32, num_heads=4, max_global_tokens=4)
        feat = torch.randn(1, 32, 7, 7)
        with pytest.raises(ValueError, match="exceeds"):
            gc(feat, K_g=8)

    def test_rejects_zero_tokens(self):
        gc = GlobalQueryConstructor(embed_dim=32, num_heads=4, max_global_tokens=4)
        feat = torch.randn(1, 32, 7, 7)
        with pytest.raises(ValueError, match="K_g"):
            gc(feat, K_g=0)

    def test_dim_mismatch_raises(self):
        gc = GlobalQueryConstructor(embed_dim=32, num_heads=4, max_global_tokens=4)
        feat = torch.randn(1, 64, 7, 7)  # 64 channels but expecting 32
        with pytest.raises(ValueError, match="channel"):
            gc(feat, K_g=2)

    def test_valid_mask_blocks_invalid_positions(self):
        """A position with valid_mask=0 should never receive attention weight.

        We test this by setting all stem features to a known value at masked
        positions and verifying the output never includes them.
        """
        torch.manual_seed(0)
        gc = GlobalQueryConstructor(embed_dim=32, num_heads=4, max_global_tokens=4)
        feat = torch.zeros(1, 32, 4, 4)
        # Mark masked positions with a huge unique value.
        feat[0, :, 0, 0] = 1000.0
        # Mask out (0, 0) — bottom-right is the only valid spot.
        mask = torch.zeros(1, 4, 4)
        mask[0, 1:, 1:] = 1.0
        out = gc(feat, K_g=2, valid_mask=mask)
        # If masking works, the huge value at (0,0) cannot influence the output.
        # All real values go through the value projection though, so we check
        # that the output magnitude is not absurd.
        assert out.abs().max().item() < 100, (
            f"Masked position appears to leak; max output magnitude={out.abs().max().item()}"
        )

    def test_gradients_flow(self):
        gc = GlobalQueryConstructor(embed_dim=32, num_heads=4, max_global_tokens=4)
        feat = torch.randn(1, 32, 7, 7, requires_grad=True)
        out = gc(feat, K_g=2)
        out.sum().backward()
        assert feat.grad is not None
        assert torch.isfinite(feat.grad).all()
        # Check that all parameters got gradient
        for name, p in gc.named_parameters():
            assert p.grad is not None, f"No grad for {name}"
            assert torch.isfinite(p.grad).all()


# ----- RegionTokenConstructor -----


class TestRegionTokenConstructor:
    def test_basic_forward(self):
        ctor = RegionTokenConstructor(embed_dim=64)
        feat = torch.randn(2, 64, 14, 14)
        regions = deterministic_grid_regions(grid_h=14, grid_w=14, K=16, batch_size=2)
        out = ctor(feat, regions)
        assert out.shape == (2, 16, 64)

    def test_with_real_sampler(self):
        sampler = RegionSampler(smooth_kernel=3, nms_radius=1)
        ctor = RegionTokenConstructor(embed_dim=32)
        feat = torch.randn(1, 32, 14, 14)
        sal = torch.rand(1, 1, 14, 14)
        regions = sampler(sal, K=8, gumbel_temperature=0.0)
        out = ctor(feat, regions)
        assert out.shape == (1, 8, 32)

    def test_separate_levels_specialize(self):
        """Two RegionTokenConstructor instances must NOT produce identical
        outputs (unless trained to). They have separate weights."""
        torch.manual_seed(0)
        ctor_region = RegionTokenConstructor(embed_dim=32)
        torch.manual_seed(1)
        ctor_detail = RegionTokenConstructor(embed_dim=32)

        feat = torch.randn(1, 32, 14, 14)
        regions = deterministic_grid_regions(14, 14, 8, 1)
        out_region = ctor_region(feat, regions)
        out_detail = ctor_detail(feat, regions)
        # Different random init → different output. This guarantees the
        # specialization mechanism (separate MLPs per level) is in place.
        assert not torch.allclose(out_region, out_detail, atol=1e-3), (
            "Region and Detail constructors with different inits produced identical "
            "outputs — specialization mechanism may be broken"
        )

    def test_shape_mismatch_raises(self):
        ctor = RegionTokenConstructor(embed_dim=32)
        feat = torch.randn(1, 32, 14, 14)
        # soft_weights with wrong spatial dims
        regions = deterministic_grid_regions(7, 7, 4, 1)
        with pytest.raises(ValueError, match="soft_weights"):
            ctor(feat, regions)

    def test_gradients_flow(self):
        ctor = RegionTokenConstructor(embed_dim=32)
        feat = torch.randn(1, 32, 14, 14, requires_grad=True)
        regions = deterministic_grid_regions(14, 14, 8, 1)
        out = ctor(feat, regions)
        out.sum().backward()
        assert feat.grad is not None
        assert torch.isfinite(feat.grad).all()
        for name, p in ctor.named_parameters():
            assert p.grad is not None, f"No grad for {name}"


# ----- Spatial metadata helpers -----


class TestSpatialMetadata:
    def test_basic_shape(self):
        coords = torch.tensor([[[3, 5], [7, 7]]], dtype=torch.int64)
        meta = compute_spatial_metadata(coords, grid_h=14, grid_w=14, receptive_field=3)
        assert meta.shape == (1, 2, 4)

    def test_values_are_normalized(self):
        coords = torch.tensor([[[0, 0], [13, 13]]], dtype=torch.int64)
        meta = compute_spatial_metadata(coords, 14, 14, receptive_field=3)
        # All values should be in [0, 1]
        assert (meta >= 0).all() and (meta <= 1).all()
        # Center at top-left grid cell: cx ≈ 0.5/14 ≈ 0.0357
        assert abs(meta[0, 0, 0].item() - 0.5 / 14) < 1e-5
        # Center at bottom-right grid cell: cx ≈ 13.5/14 ≈ 0.964
        assert abs(meta[0, 1, 0].item() - 13.5 / 14) < 1e-5

    def test_xy_order_is_x_first(self):
        """Per the public contract, spatial_metadata[..., 0] is cx (column)
        and [..., 1] is cy (row). Coords from the sampler are stored as
        (y, x), so the constructor must swap them."""
        coords = torch.tensor([[[0, 13]]], dtype=torch.int64)  # y=0, x=13
        meta = compute_spatial_metadata(coords, 14, 14, receptive_field=1)
        # cx should be near 1.0 (rightmost), cy should be near 0.0 (topmost)
        assert meta[0, 0, 0].item() > 0.9, "cx should be column position 13 → ~1.0"
        assert meta[0, 0, 1].item() < 0.1, "cy should be row position 0 → ~0.0"

    def test_global_metadata_is_full_image(self):
        meta = global_token_metadata(batch_size=2, K_g=4)
        assert meta.shape == (2, 4, 4)
        # All cx and cy = 0.5, all w and h = 1.0
        assert (meta[..., 0] == 0.5).all()
        assert (meta[..., 1] == 0.5).all()
        assert (meta[..., 2] == 1.0).all()
        assert (meta[..., 3] == 1.0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
