"""Unit tests for tokenizer/saliency.py."""

from __future__ import annotations

import pytest
import torch

from keural_config import SaliencyConfig
from tokenizer.saliency import SaliencyPredictor


class TestSaliencyShape:
    def test_basic_forward(self):
        head = SaliencyPredictor(SaliencyConfig(), in_channels=128)
        feat = torch.randn(2, 128, 14, 14)
        s = head(feat)
        assert s.shape == (2, 1, 14, 14)

    def test_outputs_in_unit_interval(self):
        head = SaliencyPredictor(SaliencyConfig(), in_channels=128)
        feat = torch.randn(4, 128, 14, 14) * 100  # large magnitude
        s = head(feat)
        assert (s >= 0).all() and (s <= 1).all()

    def test_arbitrary_spatial_shape(self):
        head = SaliencyPredictor(SaliencyConfig(), in_channels=64)
        feat = torch.randn(1, 64, 7, 11)
        s = head(feat)
        assert s.shape == (1, 1, 7, 11)

    def test_rejects_wrong_channels(self):
        head = SaliencyPredictor(SaliencyConfig(), in_channels=128)
        feat = torch.randn(1, 64, 14, 14)  # wrong channels
        with pytest.raises(ValueError, match="channels"):
            head(feat)

    def test_rejects_non_4d(self):
        head = SaliencyPredictor(SaliencyConfig(), in_channels=128)
        feat = torch.randn(128, 14, 14)
        with pytest.raises(ValueError, match="B, C, h, w"):
            head(feat)


class TestValidMask:
    def test_3d_mask_broadcasts(self):
        head = SaliencyPredictor(SaliencyConfig(), in_channels=64)
        feat = torch.randn(2, 64, 8, 8)
        mask = torch.zeros(2, 8, 8)
        mask[:, :4, :4] = 1  # only top-left valid
        s = head(feat, valid_mask=mask)
        # Padded region (positions where mask=0) must be exactly 0
        assert (s[:, :, :4, :4] > 0).all()
        assert (s[:, :, 4:, :] == 0).all()
        assert (s[:, :, :, 4:] == 0).all()

    def test_4d_mask(self):
        head = SaliencyPredictor(SaliencyConfig(), in_channels=64)
        feat = torch.randn(1, 64, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, :4, :4] = 1
        s = head(feat, valid_mask=mask)
        assert (s[:, :, 4:, :] == 0).all()

    def test_mask_spatial_mismatch_raises(self):
        head = SaliencyPredictor(SaliencyConfig(), in_channels=64)
        feat = torch.randn(1, 64, 8, 8)
        mask = torch.zeros(1, 4, 4)  # wrong spatial
        with pytest.raises(ValueError, match="spatial"):
            head(feat, valid_mask=mask)


class TestSaliencyTraining:
    def test_gradients_flow_through_features(self):
        head = SaliencyPredictor(SaliencyConfig(), in_channels=128)
        feat = torch.randn(2, 128, 14, 14, requires_grad=True)
        s = head(feat)
        loss = s.mean()
        loss.backward()
        assert feat.grad is not None
        assert torch.isfinite(feat.grad).all()

    def test_gradients_flow_through_params(self):
        head = SaliencyPredictor(SaliencyConfig(), in_channels=128)
        feat = torch.randn(1, 128, 8, 8)
        s = head(feat)
        # A non-trivial loss that touches the score
        s.sum().backward()
        for name, p in head.named_parameters():
            assert p.grad is not None, f"No grad for {name}"
            assert torch.isfinite(p.grad).all()


class TestSaliencyInit:
    """At init, saliency should be diffuse — close to 0.5 — so the allocator
    does not collapse to one corner of the image at step 0."""

    def test_init_saliency_near_half(self):
        head = SaliencyPredictor(SaliencyConfig(), in_channels=128)
        # Use random features to average over many positions
        torch.manual_seed(0)
        feat = torch.randn(8, 128, 14, 14)
        head.eval()
        with torch.no_grad():
            s = head(feat)
        # Mean should be in a reasonable band around 0.5
        mean = s.mean().item()
        assert 0.3 < mean < 0.7, (
            f"Initial saliency mean={mean:.3f} is too far from 0.5; "
            f"score head init may be too aggressive"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
