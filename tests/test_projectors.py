"""Unit tests for alignment/projectors.py."""

from __future__ import annotations

import pytest
import torch

from alignment.output_contract import (
    LEVEL_DETAIL,
    LEVEL_GLOBAL,
    LEVEL_REGION,
    KeuralEncoderOutput,
    empty_output,
)
from alignment.projectors import (
    KeuralProjectorBase,
    LevelAwareProjector,
    MetadataProjector,
    SimpleProjector,
)


def _make_output(B: int = 2, N: int = 16, D: int = 64) -> KeuralEncoderOutput:
    """Build a realistic KeuralEncoderOutput for testing."""
    out = empty_output(B, N, D)
    # All tokens real
    out.attention_mask[:] = 1
    # Mixed levels
    out.level_ids[:, :2] = LEVEL_GLOBAL
    out.level_ids[:, 2:6] = LEVEL_REGION
    out.level_ids[:, 6:] = LEVEL_DETAIL
    # Random tokens
    torch.manual_seed(0)
    out.tokens = torch.randn(B, N, D)
    out.pooled = torch.randn(B, D)
    out.spatial_metadata = torch.rand(B, N, 4)
    out.saliency_scores = torch.rand(B, N)
    out.token_budget_used[:] = N
    return out


class TestSimpleProjector:
    def test_basic_forward(self):
        proj = SimpleProjector(encoder_dim=64, llm_dim=128)
        out = _make_output(B=2, N=16, D=64)
        y = proj(out)
        assert y.shape == (2, 16, 128)

    def test_ignores_level_ids(self):
        """SimpleProjector must produce identical output for two outputs that
        differ ONLY in level_ids — by design (it doesn't read level_ids)."""
        proj = SimpleProjector(encoder_dim=64, llm_dim=64)
        proj.eval()
        out1 = _make_output()
        out2 = _make_output()
        out2.level_ids = torch.zeros_like(out2.level_ids)  # all GLOBAL now
        y1 = proj(out1)
        y2 = proj(out2)
        assert torch.allclose(y1, y2)

    def test_gradients_flow(self):
        proj = SimpleProjector(encoder_dim=32, llm_dim=64)
        out = _make_output(B=1, N=8, D=32)
        out.tokens.requires_grad_(True)
        y = proj(out)
        y.sum().backward()
        assert out.tokens.grad is not None


class TestLevelAwareProjector:
    def test_basic_forward(self):
        proj = LevelAwareProjector(encoder_dim=64, llm_dim=128)
        out = _make_output()
        y = proj(out)
        assert y.shape == (2, 16, 128)

    def test_level_bias_init_to_zero(self):
        """At init, level_embed should be zero so the projector behaves
        exactly like SimpleProjector. This makes pretraining stable."""
        proj = LevelAwareProjector(encoder_dim=32, llm_dim=64)
        assert (proj.level_embed.weight == 0).all()

    def test_level_bias_distinguishes_levels_after_training(self):
        """Once level_embed is non-trivial, tokens at different levels
        but same content should produce different projections."""
        proj = LevelAwareProjector(encoder_dim=32, llm_dim=64)
        # Manually set level embeddings to known distinct values.
        with torch.no_grad():
            proj.level_embed.weight[1].fill_(1.0)  # LEVEL_GLOBAL slot
            proj.level_embed.weight[2].fill_(-1.0)  # LEVEL_REGION slot
            proj.level_embed.weight[3].fill_(2.0)  # LEVEL_DETAIL slot

        out = _make_output(B=1, N=3, D=32)
        # Same token value at all 3 positions
        out.tokens[0, :] = torch.zeros(32)
        out.level_ids[0, 0] = LEVEL_GLOBAL
        out.level_ids[0, 1] = LEVEL_REGION
        out.level_ids[0, 2] = LEVEL_DETAIL

        y = proj(out)
        # Position 0 has +1 bias, position 1 has -1 bias, position 2 has +2 bias.
        # The pure-token projection is the same for all 3, so the differences
        # come entirely from the level bias.
        assert torch.allclose(y[0, 0] - y[0, 1], torch.full((64,), 2.0), atol=1e-5)
        assert torch.allclose(y[0, 2] - y[0, 0], torch.full((64,), 1.0), atol=1e-5)

    def test_padding_sentinel_handled(self):
        """level_ids contains -1 for padding. The projector must not crash."""
        proj = LevelAwareProjector(encoder_dim=32, llm_dim=64)
        out = _make_output(B=1, N=4, D=32)
        out.level_ids[0, 2:] = -1  # mark last two as padding sentinel
        y = proj(out)
        assert y.shape == (1, 4, 64)
        assert torch.isfinite(y).all()


class TestMetadataProjector:
    def test_basic_forward(self):
        proj = MetadataProjector(encoder_dim=64, llm_dim=128)
        out = _make_output()
        y = proj(out)
        assert y.shape == (2, 16, 128)

    def test_metadata_layer_zero_at_init(self):
        """At init, the metadata MLP's second layer should be zero so the
        projector starts identically to LevelAwareProjector."""
        proj = MetadataProjector(encoder_dim=32, llm_dim=64)
        assert (proj.meta_fc2.weight == 0).all()
        assert (proj.meta_fc2.bias == 0).all()

    def test_init_matches_level_aware(self):
        """Because metadata layer is zero at init, MetadataProjector must
        produce IDENTICAL output to LevelAwareProjector with the same other weights."""
        torch.manual_seed(0)
        meta_proj = MetadataProjector(encoder_dim=32, llm_dim=64)

        # Build a LevelAwareProjector and copy the shared params from MetadataProjector.
        level_proj = LevelAwareProjector(encoder_dim=32, llm_dim=64)
        with torch.no_grad():
            level_proj.fc1.weight.copy_(meta_proj.fc1.weight)
            level_proj.fc1.bias.copy_(meta_proj.fc1.bias)
            level_proj.fc2.weight.copy_(meta_proj.fc2.weight)
            level_proj.fc2.bias.copy_(meta_proj.fc2.bias)
            level_proj.level_embed.weight.copy_(meta_proj.level_embed.weight)

        out = _make_output(B=2, N=16, D=32)  # D must match encoder_dim
        y_meta = meta_proj(out)
        y_level = level_proj(out)
        assert torch.allclose(y_meta, y_level, atol=1e-5)

    def test_metadata_affects_output_after_training(self):
        proj = MetadataProjector(encoder_dim=32, llm_dim=64)
        # Manually fill in metadata weights so meta_proj is non-trivial.
        with torch.no_grad():
            proj.meta_fc2.weight.normal_(0, 0.1)
            proj.meta_fc2.bias.normal_(0, 0.1)

        out_a = _make_output(B=1, N=2, D=32)
        out_a.tokens[0, :] = torch.zeros(32)
        out_a.level_ids[0, :] = LEVEL_REGION
        # Distinct metadata
        out_a.spatial_metadata[0, 0] = torch.tensor([0.1, 0.1, 0.05, 0.05])
        out_a.spatial_metadata[0, 1] = torch.tensor([0.9, 0.9, 0.50, 0.50])
        out_a.saliency_scores[0, 0] = 0.2
        out_a.saliency_scores[0, 1] = 0.9

        y = proj(out_a)
        # Same tokens, same level, but different metadata → different projections.
        assert not torch.allclose(y[0, 0], y[0, 1])

    def test_gradients_flow(self):
        proj = MetadataProjector(encoder_dim=32, llm_dim=64)
        # Fill in metadata weights so they're not zero.
        with torch.no_grad():
            proj.meta_fc2.weight.normal_(0, 0.1)

        out = _make_output(B=1, N=4, D=32)
        out.tokens.requires_grad_(True)
        out.spatial_metadata.requires_grad_(True)
        y = proj(out)
        y.sum().backward()
        assert out.tokens.grad is not None
        assert out.spatial_metadata.grad is not None


class TestProjectorEnd2End:
    """The headline use case: feed a real encoder output into a projector
    and feed the result to a fake LLM. Verify shapes and gradients."""

    def test_full_chain_with_simple_projector(self):
        """Encoder → Projector → fake LLM. Shapes must align."""
        from architecture.encoder import KeuralVisionEncoder
        from keural_config import KeuralConfig, SpatialConfig, StemConfig

        cfg = KeuralConfig(
            stem=StemConfig(channels=[16, 32, 48, 64]),
            spatial=SpatialConfig(embed_dim=64, num_heads=4, head_dim=16, depth=2),
        )
        cfg.training.gradient_checkpointing = False
        encoder = KeuralVisionEncoder(cfg)
        projector = SimpleProjector(encoder_dim=64, llm_dim=128)

        x = torch.randn(2, 3, 224, 224)
        encoder_out = encoder(x, token_budget=32)
        llm_input = projector(encoder_out)
        assert llm_input.shape == (2, 32, 128)

    def test_full_chain_with_level_aware_projector(self):
        from architecture.encoder import KeuralVisionEncoder
        from keural_config import KeuralConfig, SpatialConfig, StemConfig

        cfg = KeuralConfig(
            stem=StemConfig(channels=[16, 32, 48, 64]),
            spatial=SpatialConfig(embed_dim=64, num_heads=4, head_dim=16, depth=2),
        )
        cfg.training.gradient_checkpointing = False
        encoder = KeuralVisionEncoder(cfg)
        projector = LevelAwareProjector(encoder_dim=64, llm_dim=128)

        x = torch.randn(1, 3, 224, 224)
        encoder_out = encoder(x, token_budget=32)
        llm_input = projector(encoder_out)
        assert llm_input.shape == (1, 32, 128)
        # Backprop check
        llm_input.sum().backward()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
