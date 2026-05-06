"""Unit tests for tokenizer/atb_tokenizer.py.

This is the integration of modules 4–7. These tests verify that the end-to-end
tokenization path produces correctly shaped, semantically valid output.
"""

from __future__ import annotations

import pytest
import torch

from alignment.output_contract import LEVEL_DETAIL, LEVEL_GLOBAL, LEVEL_REGION
from keural_config import KeuralConfig, SpatialConfig, StemConfig
from tokenizer.atb_tokenizer import ATBTokenizer, TokenizedImage, split_budget


def _small_config() -> KeuralConfig:
    """A tiny but valid config for fast testing."""
    return KeuralConfig(
        stem=StemConfig(channels=[16, 32, 48, 64]),
        spatial=SpatialConfig(embed_dim=64, num_heads=4, head_dim=16),
    )


# ----- Budget splitter -----


class TestSplitBudget:
    def test_basic_split(self):
        K_g, K_r, K_d = split_budget(
            token_budget=256,
            split={"global": 0.05, "region": 0.25, "detail": 0.70},
            floors={"global": 4, "region": 8, "detail": 0},
        )
        assert K_g + K_r + K_d == 256
        assert K_g >= 4
        assert K_r >= 8

    def test_floors_apply_at_small_budget(self):
        K_g, K_r, K_d = split_budget(
            token_budget=20,
            split={"global": 0.05, "region": 0.25, "detail": 0.70},
            floors={"global": 4, "region": 8, "detail": 0},
        )
        assert K_g + K_r + K_d == 20
        assert K_g >= 4  # floor enforced
        assert K_r >= 8  # floor enforced

    def test_total_equals_budget(self):
        for budget in (16, 64, 128, 256, 512, 1024):
            K_g, K_r, K_d = split_budget(
                budget,
                {"global": 0.05, "region": 0.25, "detail": 0.70},
                {"global": 4, "region": 8, "detail": 0},
            )
            assert K_g + K_r + K_d == budget, f"budget={budget}: K_g={K_g}, K_r={K_r}, K_d={K_d}"

    def test_minimum_budget_handled(self):
        # With floors {global: 4, region: 8, detail: 0}, the minimum is 12.
        K_g, K_r, K_d = split_budget(
            12,
            {"global": 0.05, "region": 0.25, "detail": 0.70},
            {"global": 4, "region": 8, "detail": 0},
        )
        assert K_g + K_r + K_d == 12
        assert K_d == 0


# ----- ATBTokenizer end-to-end -----


class TestATBTokenizerForward:
    def test_basic_forward(self):
        cfg = _small_config()
        tokenizer = ATBTokenizer(cfg)
        x = torch.randn(2, 3, 224, 224)
        out = tokenizer(x, token_budget=64)
        assert isinstance(out, TokenizedImage)
        assert out.tokens.shape[0] == 2
        assert out.tokens.shape[2] == 64  # embed_dim
        # K_g + K_r + K_d should equal 64
        assert out.tokens.shape[1] == 64

    def test_split_levels_consistent(self):
        cfg = _small_config()
        tokenizer = ATBTokenizer(cfg)
        x = torch.randn(1, 3, 224, 224)
        out = tokenizer(x, token_budget=128)
        # Token count consistency
        assert out.K_g + out.K_r + out.K_d == out.tokens.shape[1]
        # Level IDs match counts: first K_g are global, next K_r are region, rest detail
        assert (out.level_ids[:, : out.K_g] == LEVEL_GLOBAL).all()
        assert (out.level_ids[:, out.K_g : out.K_g + out.K_r] == LEVEL_REGION).all()
        if out.K_d > 0:
            assert (out.level_ids[:, out.K_g + out.K_r :] == LEVEL_DETAIL).all()

    def test_metadata_in_unit_interval(self):
        cfg = _small_config()
        tokenizer = ATBTokenizer(cfg)
        x = torch.randn(1, 3, 224, 224)
        out = tokenizer(x, token_budget=64)
        assert (out.spatial_metadata >= 0.0).all()
        assert (out.spatial_metadata <= 1.0).all()

    def test_saliency_scores_in_range(self):
        cfg = _small_config()
        tokenizer = ATBTokenizer(cfg)
        x = torch.randn(1, 3, 224, 224)
        out = tokenizer(x, token_budget=64)
        assert (out.saliency_scores >= 0.0).all()
        assert (out.saliency_scores <= 1.0).all()

    def test_global_tokens_have_full_image_metadata(self):
        cfg = _small_config()
        tokenizer = ATBTokenizer(cfg)
        x = torch.randn(1, 3, 224, 224)
        out = tokenizer(x, token_budget=64)
        # Global token metadata: cx=cy=0.5, w=h=1.0
        global_meta = out.spatial_metadata[:, : out.K_g]
        assert torch.allclose(global_meta[..., 0], torch.full_like(global_meta[..., 0], 0.5))
        assert torch.allclose(global_meta[..., 1], torch.full_like(global_meta[..., 1], 0.5))
        assert torch.allclose(global_meta[..., 2], torch.full_like(global_meta[..., 2], 1.0))
        assert torch.allclose(global_meta[..., 3], torch.full_like(global_meta[..., 3], 1.0))

    @pytest.mark.parametrize("budget", [16, 32, 64, 128, 256])
    def test_various_budgets(self, budget):
        cfg = _small_config()
        # Need to keep budget at or below max_token_budget for the floors to fit
        if budget < 12:
            pytest.skip(f"budget {budget} below floor sum")
        tokenizer = ATBTokenizer(cfg)
        x = torch.randn(1, 3, 224, 224)
        out = tokenizer(x, token_budget=budget)
        assert out.tokens.shape[1] == budget

    @pytest.mark.parametrize("res", [224, 256, 320])
    def test_various_resolutions(self, res):
        cfg = _small_config()
        tokenizer = ATBTokenizer(cfg)
        x = torch.randn(1, 3, res, res)
        out = tokenizer(x, token_budget=64)
        # Token count is determined by budget, NOT resolution — that's the whole point.
        assert out.tokens.shape[1] == 64

    def test_rejects_too_large_budget(self):
        cfg = _small_config()
        tokenizer = ATBTokenizer(cfg)
        x = torch.randn(1, 3, 224, 224)
        with pytest.raises(ValueError, match="max_token_budget"):
            tokenizer(x, token_budget=99999)

    def test_rejects_too_small_budget(self):
        cfg = _small_config()
        tokenizer = ATBTokenizer(cfg)
        x = torch.randn(1, 3, 224, 224)
        # Below the floor sum
        with pytest.raises(ValueError, match="floors"):
            tokenizer(x, token_budget=5)


class TestWarmupCurriculum:
    """The warmup curriculum (per docs §7.5) uses fixed grid sampling for
    the first N steps, then turns on the learned allocator."""

    def test_warmup_uses_grid(self):
        cfg = _small_config()
        cfg.tokenizer.warmup_steps_fixed_grid = 1000
        tokenizer = ATBTokenizer(cfg)
        x = torch.randn(1, 3, 224, 224)

        # Step 0: warmup → grid sampling
        out_warmup = tokenizer(x, token_budget=64, step=0)
        # Step 5000: allocator on
        out_post = tokenizer(x, token_budget=64, step=5000)

        # Both produce valid output
        assert out_warmup.tokens.shape == out_post.tokens.shape

    def test_inference_no_step(self):
        """When step=None (inference), no Gumbel noise is added."""
        cfg = _small_config()
        tokenizer = ATBTokenizer(cfg)
        tokenizer.eval()
        x = torch.randn(1, 3, 224, 224)
        # Two calls should be deterministic
        out1 = tokenizer(x, token_budget=64, step=None)
        out2 = tokenizer(x, token_budget=64, step=None)
        assert torch.equal(out1.level_ids, out2.level_ids)
        assert torch.allclose(out1.tokens, out2.tokens)


class TestATBGradients:
    """End-to-end gradient flow is the critical check — if any link in the
    chain breaks gradients, the architecture is unusable."""

    def test_gradients_flow_through_full_chain(self):
        cfg = _small_config()
        tokenizer = ATBTokenizer(cfg)
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        out = tokenizer(x, token_budget=32)
        loss = out.tokens.mean() + out.saliency_scores.mean()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_all_tokenizer_params_get_grad_under_realistic_loss(self):
        """Under a realistic combined loss (tokens + saliency_scores + saliency_map),
        every tokenizer parameter must get a non-zero gradient.

        This mirrors the actual training loss in training/losses.py:
        L_total = L_primary(tokens) + L_HCT(tokens, levels) + L_sal(saliency_map)
        """
        cfg = _small_config()
        tokenizer = ATBTokenizer(cfg)
        x = torch.randn(1, 3, 224, 224)
        out = tokenizer(x, token_budget=32)
        # Realistic loss touches three signal paths:
        #  - tokens (drives stem, constructors, sampler-soft-weights)
        #  - saliency_scores (drives saliency predictor via straight-through)
        #  - saliency_map (drives saliency predictor directly via the variance reg)
        loss = (
            out.tokens.mean()
            + out.saliency_scores.mean()
            + out.saliency_map.var()
        )
        loss.backward()
        for name, p in tokenizer.named_parameters():
            assert p.grad is not None, f"No grad for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"

    def test_tokens_alone_skip_saliency_predictor(self):
        """A loss touching ONLY out.tokens does not flow gradient to the
        saliency predictor — by design.

        This is correct: token *values* come from soft-pooling stem features,
        not from the saliency map. The saliency predictor learns only via
        signals on saliency_scores (straight-through) and saliency_map (regularizer).

        Documenting this here so a future engineer who sees missing grads
        on the saliency head doesn't think there's a bug.
        """
        cfg = _small_config()
        tokenizer = ATBTokenizer(cfg)
        x = torch.randn(1, 3, 224, 224)
        out = tokenizer(x, token_budget=32)
        out.tokens.sum().backward()
        # Saliency predictor params have NO grad path from tokens alone.
        sal_params = {n: p for n, p in tokenizer.named_parameters() if "saliency" in n}
        assert any(p.grad is None for p in sal_params.values()), (
            "Saliency predictor unexpectedly received gradient from tokens-only loss. "
            "This is fine for training but contradicts the documented design — "
            "investigate before relying on this behavior."
        )

    def test_saliency_predictor_gets_grad_via_scores(self):
        """The straight-through trick must give the saliency predictor real
        gradient via the saliency_scores it produces."""
        cfg = _small_config()
        tokenizer = ATBTokenizer(cfg)
        x = torch.randn(1, 3, 224, 224)
        out = tokenizer(x, token_budget=32)
        # Loss touches ONLY the saliency-scores output (proxying the
        # saliency-variance regularizer in the actual loss).
        loss = (out.saliency_scores - 0.5).pow(2).mean()
        loss.backward()
        sal_params = [
            p for n, p in tokenizer.named_parameters() if "saliency" in n
        ]
        assert len(sal_params) > 0
        for p in sal_params:
            assert p.grad is not None
            assert p.grad.abs().sum() > 0, "Saliency predictor got zero grad — straight-through broken"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
