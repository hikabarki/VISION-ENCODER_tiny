"""Unit tests for tokenizer/stem.py."""

from __future__ import annotations

import pytest
import torch

from keural_config import StemConfig
from tokenizer.stem import ConvStem, pad_to_stride


class TestConvStemShape:
    def test_default_construction(self):
        stem = ConvStem(StemConfig())
        assert stem.out_channels == 512
        assert stem.stride == 16

    def test_basic_forward_shape_224(self):
        stem = ConvStem(StemConfig(channels=[32, 64, 96, 128]))
        x = torch.randn(2, 3, 224, 224)
        y = stem(x)
        # 224 / 16 = 14
        assert y.shape == (2, 128, 14, 14)

    def test_forward_shape_512(self):
        stem = ConvStem(StemConfig(channels=[32, 64, 96, 128]))
        x = torch.randn(1, 3, 512, 512)
        y = stem(x)
        assert y.shape == (1, 128, 32, 32)

    def test_non_square_input(self):
        stem = ConvStem(StemConfig(channels=[32, 64, 96, 128]))
        x = torch.randn(1, 3, 224, 320)
        y = stem(x)
        assert y.shape == (1, 128, 14, 20)

    def test_rejects_wrong_channels(self):
        stem = ConvStem(StemConfig(channels=[32, 64, 96, 128]))
        x = torch.randn(1, 4, 224, 224)
        with pytest.raises(ValueError, match="channels"):
            stem(x)

    def test_rejects_non_multiple_of_stride(self):
        stem = ConvStem(StemConfig(channels=[32, 64, 96, 128]))
        x = torch.randn(1, 3, 100, 100)  # 100 not divisible by 16
        with pytest.raises(ValueError, match="multiples"):
            stem(x)

    def test_rejects_non_4d(self):
        stem = ConvStem(StemConfig(channels=[32, 64, 96, 128]))
        x = torch.randn(3, 224, 224)
        with pytest.raises(ValueError, match="B, C, H, W"):
            stem(x)


class TestActivations:
    @pytest.mark.parametrize("act", ["gelu", "relu", "silu"])
    def test_all_activations_run(self, act):
        stem = ConvStem(StemConfig(channels=[16, 32, 48, 64], activation=act))
        x = torch.randn(1, 3, 32, 32)
        y = stem(x)
        assert y.shape == (1, 64, 2, 2)


class TestStemBackward:
    def test_gradients_flow(self):
        stem = ConvStem(StemConfig(channels=[32, 64, 96, 128]))
        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        y = stem(x)
        loss = y.sum()
        loss.backward()
        # Every parameter should have a gradient
        for name, p in stem.named_parameters():
            assert p.grad is not None, f"No grad for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"


class TestPadToStride:
    def test_no_padding_when_aligned(self):
        x = torch.randn(1, 3, 224, 224)
        padded, pads = pad_to_stride(x, stride=16)
        assert padded.shape == x.shape
        assert pads == (0, 0, 0, 0)
        assert torch.equal(padded, x)

    def test_pads_when_not_aligned(self):
        x = torch.randn(1, 3, 100, 100)
        padded, pads = pad_to_stride(x, stride=16)
        # Round up: ceil(100/16)*16 = 112
        assert padded.shape == (1, 3, 112, 112)
        assert pads == (0, 12, 0, 12)
        # Original content preserved
        assert torch.allclose(padded[:, :, :100, :100], x)
        # Padded region is zeros
        assert (padded[:, :, 100:, :] == 0).all()
        assert (padded[:, :, :, 100:] == 0).all()

    def test_pads_non_square(self):
        x = torch.randn(2, 3, 50, 200)
        padded, pads = pad_to_stride(x, stride=16)
        # 50 -> 64 (pad 14), 200 -> 208 (pad 8)
        assert padded.shape == (2, 3, 64, 208)
        assert pads == (0, 8, 0, 14)


class TestStemEndToEnd:
    """The stem is the only consumer of pixel_values in the model. These
    tests pin down its behavior at the resolutions we'll actually use."""

    @pytest.mark.parametrize("H,W", [(224, 224), (256, 256), (320, 320), (224, 320)])
    def test_real_buckets(self, H, W):
        stem = ConvStem(StemConfig(channels=[32, 64, 96, 128]))
        x = torch.randn(2, 3, H, W)
        y = stem(x)
        assert y.shape == (2, 128, H // 16, W // 16)

    def test_with_pad_to_stride(self):
        """A user gives us a non-aligned image. After pad_to_stride it works."""
        stem = ConvStem(StemConfig(channels=[32, 64, 96, 128]))
        x = torch.randn(1, 3, 100, 100)
        padded, _ = pad_to_stride(x, stride=stem.stride)
        y = stem(padded)
        assert y.shape == (1, 128, 7, 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
