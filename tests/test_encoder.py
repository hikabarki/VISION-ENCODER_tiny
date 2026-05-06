"""Unit tests for architecture/encoder.py — the full integrated model.

This is THE test suite. If these all pass, we have a working encoder.
"""

from __future__ import annotations

import pytest
import torch

from alignment.output_contract import (
    LEVEL_DETAIL,
    LEVEL_GLOBAL,
    LEVEL_REGION,
    KeuralEncoderOutput,
)
from architecture.encoder import KeuralVisionEncoder
from keural_config import KeuralConfig, SpatialConfig, StemConfig


def _small_config() -> KeuralConfig:
    """A tiny but valid config for fast testing."""
    cfg = KeuralConfig(
        stem=StemConfig(channels=[16, 32, 48, 64]),
        spatial=SpatialConfig(embed_dim=64, num_heads=4, head_dim=16, depth=2),
    )
    # Disable gradient checkpointing for tests (eager forward is faster).
    cfg.training.gradient_checkpointing = False
    return cfg


class TestEncoderForward:
    def test_basic_forward_returns_contract(self):
        encoder = KeuralVisionEncoder(_small_config())
        x = torch.randn(2, 3, 224, 224)
        out = encoder(x, token_budget=64)
        assert isinstance(out, KeuralEncoderOutput)
        out.validate()

    def test_output_shapes_match_budget(self):
        encoder = KeuralVisionEncoder(_small_config())
        x = torch.randn(2, 3, 224, 224)
        for budget in (16, 32, 64, 128):
            out = encoder(x, token_budget=budget)
            assert out.tokens.shape == (2, budget, 64)
            assert out.attention_mask.shape == (2, budget)
            assert out.level_ids.shape == (2, budget)
            assert out.spatial_metadata.shape == (2, budget, 4)
            assert out.saliency_scores.shape == (2, budget)
            assert out.pooled.shape == (2, 64)
            assert out.token_budget_used.shape == (2,)
            assert (out.token_budget_used == budget).all()

    def test_output_levels_present(self):
        """All three levels (global, region, detail) should appear in output."""
        encoder = KeuralVisionEncoder(_small_config())
        x = torch.randn(1, 3, 224, 224)
        out = encoder(x, token_budget=128)
        levels = out.level_ids[0].unique().tolist()
        assert LEVEL_GLOBAL in levels
        assert LEVEL_REGION in levels
        assert LEVEL_DETAIL in levels

    def test_attention_mask_all_real(self):
        """The encoder always emits exactly token_budget real tokens."""
        encoder = KeuralVisionEncoder(_small_config())
        x = torch.randn(2, 3, 224, 224)
        out = encoder(x, token_budget=32)
        assert out.attention_mask.sum().item() == 2 * 32

    def test_default_budget_used_when_none(self):
        encoder = KeuralVisionEncoder(_small_config())
        x = torch.randn(1, 3, 224, 224)
        out = encoder(x, token_budget=None)
        # Default for our small config is 256
        assert out.tokens.shape[1] == 256

    @pytest.mark.parametrize("res", [224, 256, 320])
    def test_works_at_multiple_resolutions(self, res):
        encoder = KeuralVisionEncoder(_small_config())
        x = torch.randn(1, 3, res, res)
        out = encoder(x, token_budget=64)
        # Token count is independent of resolution — that's the ATB promise.
        assert out.tokens.shape[1] == 64
        # All metadata still in [0, 1]
        assert (out.spatial_metadata >= 0).all() and (out.spatial_metadata <= 1).all()


class TestEncoderEncode:
    def test_simple_mode_returns_tensor(self):
        encoder = KeuralVisionEncoder(_small_config())
        x = torch.randn(2, 3, 224, 224)
        tokens = encoder.encode(x, token_budget=64)
        assert isinstance(tokens, torch.Tensor)
        assert tokens.shape == (2, 64, 64)

    def test_simple_mode_consistent_with_full(self):
        encoder = KeuralVisionEncoder(_small_config())
        encoder.eval()
        x = torch.randn(1, 3, 224, 224)
        full_out = encoder(x, token_budget=32)
        simple_out = encoder.encode(x, token_budget=32)
        # Simple mode masks padding; here all tokens are real, so they match.
        assert torch.allclose(simple_out, full_out.tokens, atol=1e-5)


class TestEncoderGradientFlow:
    """The most important test set. End-to-end gradient must flow."""

    def test_grad_flow_pixels_to_loss(self):
        encoder = KeuralVisionEncoder(_small_config())
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        out = encoder(x, token_budget=32)
        loss = (
            out.tokens.mean()
            + out.pooled.mean()
            + out.saliency_scores.mean()
        )
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_all_params_get_grad_under_realistic_loss(self):
        """Under the realistic full loss (touching tokens, pooled, saliency),
        every parameter must get a non-zero gradient."""
        encoder = KeuralVisionEncoder(_small_config())
        x = torch.randn(1, 3, 224, 224)
        out = encoder(x, token_budget=32)
        loss = (
            out.tokens.mean()
            + out.pooled.mean()
            + out.saliency_scores.mean()
        )
        loss.backward()
        for name, p in encoder.named_parameters():
            assert p.grad is not None, f"No grad for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"
            # Stronger: every param should have NON-ZERO grad somewhere.
            assert p.grad.abs().sum() > 0, f"Zero grad for {name}"


class TestHCTSpecialization:
    """Verify the HCT specialization mechanisms are in place."""

    def test_level_embeddings_exist_and_differ(self):
        encoder = KeuralVisionEncoder(_small_config())
        # 3 distinct level embeddings
        assert encoder.level_embeddings.shape[0] == 3
        # They should differ at init (random trunc-normal).
        assert not torch.allclose(
            encoder.level_embeddings[0], encoder.level_embeddings[1]
        )
        assert not torch.allclose(
            encoder.level_embeddings[0], encoder.level_embeddings[2]
        )

    def test_separate_constructors(self):
        """Region and Detail constructors must be different modules."""
        encoder = KeuralVisionEncoder(_small_config())
        # The two constructors live inside the tokenizer.
        rc = encoder.tokenizer.region_constructor
        dc = encoder.tokenizer.detail_constructor
        assert rc is not dc
        # Their MLP linear weights should differ (different random inits).
        # Note: skip past the LayerNorm weights (which are init'd to all ones
        # in both modules and would falsely look "identical").
        rc_linear = next(p for p in rc.parameters() if p.dim() >= 2)
        dc_linear = next(p for p in dc.parameters() if p.dim() >= 2)
        assert rc_linear.shape == dc_linear.shape
        assert not torch.allclose(rc_linear, dc_linear)


class TestPoolToken:
    """The pool token is what makes `pooled` work — verify it's actually used."""

    def test_pool_token_is_learnable(self):
        encoder = KeuralVisionEncoder(_small_config())
        # Fill pool token with a known value
        with torch.no_grad():
            encoder.pool_token.fill_(0.5)
        x = torch.randn(1, 3, 224, 224)
        out = encoder(x, token_budget=32)
        # After forward, pooled is a function of the pool token's interactions.
        # It must not be exactly the zero vector — the residual stream
        # adds attention output.
        assert out.pooled.abs().sum() > 0


class TestEncoderDeterminism:
    def test_eval_mode_deterministic(self):
        encoder = KeuralVisionEncoder(_small_config())
        encoder.eval()
        x = torch.randn(1, 3, 224, 224)
        out1 = encoder(x, token_budget=32)
        out2 = encoder(x, token_budget=32)
        assert torch.allclose(out1.tokens, out2.tokens, atol=1e-5)
        assert torch.allclose(out1.pooled, out2.pooled, atol=1e-5)


class TestBatchedIndependence:
    """Different items in a batch should produce different outputs."""

    def test_different_images_give_different_outputs(self):
        encoder = KeuralVisionEncoder(_small_config())
        encoder.eval()
        x = torch.randn(2, 3, 224, 224)
        out = encoder(x, token_budget=32)
        # Pooled outputs for the two distinct images must differ.
        assert not torch.allclose(out.pooled[0], out.pooled[1], atol=1e-3)


class TestSizeProperty:
    """Confirm the model has a sensible parameter count."""

    def test_param_count_is_reasonable(self):
        encoder = KeuralVisionEncoder(_small_config())
        n = sum(p.numel() for p in encoder.parameters())
        # Tiny test config — should be in the millions but bounded.
        assert 100_000 < n < 5_000_000, f"Unexpected param count: {n:,}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
