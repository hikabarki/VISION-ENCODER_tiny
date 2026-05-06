"""Unit tests for training/losses.py."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from alignment.output_contract import (
    LEVEL_DETAIL,
    LEVEL_GLOBAL,
    LEVEL_REGION,
    empty_output,
)
from keural_config import LossConfig
from training.losses import (
    KeuralLoss,
    LossOutput,
    hct_global_text_loss,
    saliency_variance_regularizer,
    sigmoid_siglip_loss,
)


def _make_output(B: int = 4, N: int = 16, D: int = 32):
    out = empty_output(B, N, D)
    out.attention_mask[:] = 1
    out.level_ids[:, :2] = LEVEL_GLOBAL
    out.level_ids[:, 2:8] = LEVEL_REGION
    out.level_ids[:, 8:] = LEVEL_DETAIL
    out.tokens = torch.randn(B, N, D)
    out.pooled = torch.randn(B, D)
    out.spatial_metadata = torch.rand(B, N, 4)
    out.saliency_scores = torch.rand(B, N)
    return out


# ----- sigmoid_siglip_loss -----


class TestSigmoidSigLIPLoss:
    def test_basic_forward(self):
        img = F.normalize(torch.randn(4, 32), dim=-1)
        txt = F.normalize(torch.randn(4, 32), dim=-1)
        scale = torch.tensor(1.0)
        bias = torch.tensor(-10.0)
        loss = sigmoid_siglip_loss(img, txt, scale, bias)
        assert loss.dim() == 0  # scalar
        assert torch.isfinite(loss)

    def test_perfect_alignment_gives_low_loss(self):
        """If image[i] == text[i], the loss should be much lower than for random pairing."""
        torch.manual_seed(0)
        img = F.normalize(torch.randn(8, 32), dim=-1)
        txt_aligned = img.clone()  # perfect alignment
        txt_random = F.normalize(torch.randn(8, 32), dim=-1)
        scale = torch.tensor(2.3)
        bias = torch.tensor(-10.0)
        loss_aligned = sigmoid_siglip_loss(img, txt_aligned, scale, bias)
        loss_random = sigmoid_siglip_loss(img, txt_random, scale, bias)
        assert loss_aligned < loss_random

    def test_gradient_flows(self):
        img = F.normalize(torch.randn(4, 32), dim=-1).requires_grad_(True)
        txt = F.normalize(torch.randn(4, 32), dim=-1).requires_grad_(True)
        scale = torch.tensor(1.0, requires_grad=True)
        bias = torch.tensor(-10.0, requires_grad=True)
        loss = sigmoid_siglip_loss(img, txt, scale, bias)
        loss.backward()
        assert img.grad is not None
        assert txt.grad is not None
        assert scale.grad is not None
        assert bias.grad is not None

    def test_handles_batch_size_one(self):
        img = F.normalize(torch.randn(1, 32), dim=-1)
        txt = F.normalize(torch.randn(1, 32), dim=-1)
        scale = torch.tensor(1.0)
        bias = torch.tensor(-10.0)
        loss = sigmoid_siglip_loss(img, txt, scale, bias)
        assert torch.isfinite(loss)


# ----- HCT global-text -----


class TestHCTGlobalText:
    def test_basic_forward(self):
        out = _make_output(B=4, N=16, D=32)
        text = torch.randn(4, 32)
        scale = torch.tensor(1.0)
        bias = torch.tensor(-10.0)
        loss = hct_global_text_loss(out, text, scale, bias)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_only_uses_global_tokens(self):
        """Modifying region/detail tokens should not change the loss."""
        torch.manual_seed(0)
        out_a = _make_output(B=2, N=16, D=32)
        text = torch.randn(2, 32)
        scale = torch.tensor(1.0)
        bias = torch.tensor(-10.0)
        loss_a = hct_global_text_loss(out_a, text, scale, bias)

        out_b = _make_output(B=2, N=16, D=32)
        # Force same tokens for the global slice but different for region/detail
        out_b.tokens = out_a.tokens.clone()
        out_b.tokens[:, 2:] = torch.randn(2, 14, 32) * 100  # huge changes
        out_b.level_ids = out_a.level_ids.clone()
        out_b.attention_mask = out_a.attention_mask.clone()
        loss_b = hct_global_text_loss(out_b, text, scale, bias)
        # Same global tokens → same loss
        assert torch.allclose(loss_a, loss_b, atol=1e-5)

    def test_gradient_flow(self):
        out = _make_output(B=2, N=16, D=32)
        out.tokens.requires_grad_(True)
        text = torch.randn(2, 32, requires_grad=True)
        scale = torch.tensor(1.0, requires_grad=True)
        bias = torch.tensor(-10.0, requires_grad=True)
        loss = hct_global_text_loss(out, text, scale, bias)
        loss.backward()
        assert out.tokens.grad is not None
        # Region and detail token positions should have ZERO gradient (only global flows).
        non_global = out.level_ids != LEVEL_GLOBAL
        assert (out.tokens.grad[non_global] == 0).all()


# ----- saliency_variance_regularizer -----


class TestSaliencyRegularizer:
    def test_flat_map_high_loss(self):
        """A constant saliency map should give a HIGH loss."""
        flat = torch.full((1, 1, 14, 14), 0.5)
        loss_flat = saliency_variance_regularizer(flat)

        diverse = torch.rand(1, 1, 14, 14)
        loss_diverse = saliency_variance_regularizer(diverse)

        assert loss_flat > loss_diverse

    def test_gradient_flows(self):
        sal = torch.rand(2, 1, 14, 14, requires_grad=True)
        loss = saliency_variance_regularizer(sal)
        loss.backward()
        assert sal.grad is not None
        assert torch.isfinite(sal.grad).all()

    def test_eps_prevents_divergence(self):
        """Even a perfectly flat map should give a finite (large but bounded) loss."""
        flat = torch.full((1, 1, 4, 4), 0.5)
        loss = saliency_variance_regularizer(flat, eps=1e-3)
        assert torch.isfinite(loss)
        assert loss > 100  # large but bounded


# ----- KeuralLoss -----


class TestKeuralLoss:
    def test_basic_forward(self):
        loss_fn = KeuralLoss(LossConfig())
        out = _make_output(B=4, N=16, D=32)
        text = torch.randn(4, 32)
        sal = torch.rand(4, 1, 14, 14)
        result = loss_fn(out, text, saliency_map=sal)
        assert isinstance(result, LossOutput)
        assert result.total.dim() == 0
        assert torch.isfinite(result.total)

    def test_components_logged(self):
        loss_fn = KeuralLoss(LossConfig())
        out = _make_output(B=2, N=16, D=32)
        text = torch.randn(2, 32)
        sal = torch.rand(2, 1, 14, 14)
        result = loss_fn(out, text, saliency_map=sal)
        # All components present in the dict
        assert "loss/total" in result.components
        assert "loss/primary" in result.components
        assert "loss/hct" in result.components
        assert "loss/saliency" in result.components
        assert "loss/logit_scale" in result.components

    def test_no_saliency_map_works(self):
        """Should work with saliency_map=None (loss component is zero)."""
        loss_fn = KeuralLoss(LossConfig())
        out = _make_output(B=2, N=16, D=32)
        text = torch.randn(2, 32)
        result = loss_fn(out, text, saliency_map=None)
        assert result.saliency.item() == 0.0

    def test_gradient_through_total(self):
        loss_fn = KeuralLoss(LossConfig())
        out = _make_output(B=2, N=16, D=32)
        out.tokens.requires_grad_(True)
        out.pooled.requires_grad_(True)
        text = torch.randn(2, 32, requires_grad=True)
        sal = torch.rand(2, 1, 14, 14, requires_grad=True)
        result = loss_fn(out, text, saliency_map=sal)
        result.total.backward()
        assert out.tokens.grad is not None
        assert out.pooled.grad is not None
        assert sal.grad is not None
        assert loss_fn.logit_scale.grad is not None

    def test_dim_mismatch_raises(self):
        loss_fn = KeuralLoss(LossConfig())
        out = _make_output(B=2, N=16, D=32)
        text = torch.randn(2, 64)  # dim 64 != 32
        with pytest.raises(ValueError, match="dim mismatch"):
            loss_fn(out, text)

    def test_batch_mismatch_raises(self):
        loss_fn = KeuralLoss(LossConfig())
        out = _make_output(B=2, N=16, D=32)
        text = torch.randn(3, 32)
        with pytest.raises(ValueError, match="Batch size"):
            loss_fn(out, text)


class TestEndToEndWithEncoder:
    """The headline test: real encoder + real loss + backward = optimization step."""

    def test_one_step_optimization_decreases_loss(self):
        """A single SGD step on a small model should decrease the loss."""
        from architecture.encoder import KeuralVisionEncoder
        from keural_config import KeuralConfig, SpatialConfig, StemConfig

        cfg = KeuralConfig(
            stem=StemConfig(channels=[16, 32, 48, 64]),
            spatial=SpatialConfig(embed_dim=64, num_heads=4, head_dim=16, depth=2),
        )
        cfg.training.gradient_checkpointing = False
        torch.manual_seed(0)
        encoder = KeuralVisionEncoder(cfg)
        loss_fn = KeuralLoss(LossConfig())

        x = torch.randn(4, 3, 224, 224)
        text = torch.randn(4, 64)

        # First forward+loss — get the saliency map for the regularizer too.
        out, sal_map = encoder(x, token_budget=32, return_saliency_map=True)
        result1 = loss_fn(out, text, saliency_map=sal_map)

        # Backprop and step
        params = list(encoder.parameters()) + list(loss_fn.parameters())
        optimizer = torch.optim.SGD(params, lr=1e-2)
        optimizer.zero_grad()
        result1.total.backward()
        optimizer.step()

        # Second forward — loss should be finite and have moved.
        out2, sal_map2 = encoder(x, token_budget=32, return_saliency_map=True)
        result2 = loss_fn(out2, text, saliency_map=sal_map2)
        assert torch.isfinite(result2.total)
        assert not torch.allclose(result1.total, result2.total)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
