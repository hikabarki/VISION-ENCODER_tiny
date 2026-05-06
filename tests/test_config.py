"""Unit tests for keural_config.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from keural_config import (
    DataConfig,
    KeuralConfig,
    LossConfig,
    SaliencyConfig,
    SpatialConfig,
    StemConfig,
    TokenizerConfig,
    TrainingConfig,
)


class TestStemConfig:
    def test_defaults(self):
        c = StemConfig()
        assert len(c.channels) == 4
        assert c.kernel_size == 3
        assert c.activation == "gelu"

    def test_wrong_channel_count_raises(self):
        with pytest.raises(ValueError, match="length 4"):
            StemConfig(channels=[64, 128, 256])

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="activation"):
            StemConfig(activation="tanh")


class TestTokenizerConfig:
    def test_defaults_valid(self):
        c = TokenizerConfig()
        assert c.default_token_budget == 256
        assert sum(c.budget_split.values()) == pytest.approx(1.0)

    def test_split_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            TokenizerConfig(
                budget_split={"global": 0.1, "region": 0.2, "detail": 0.3}
            )

    def test_split_must_have_correct_keys(self):
        with pytest.raises(ValueError, match="keys"):
            TokenizerConfig(
                budget_split={"global": 0.5, "region": 0.5, "extra": 0.0}
            )

    def test_default_budget_cannot_exceed_max(self):
        with pytest.raises(ValueError, match="default_token_budget"):
            TokenizerConfig(default_token_budget=2000, max_token_budget=1024)

    def test_temperature_must_anneal_down(self):
        with pytest.raises(ValueError, match="anneal"):
            TokenizerConfig(
                gumbel_temperature_start=0.1, gumbel_temperature_end=1.0
            )


class TestSpatialConfig:
    def test_defaults(self):
        c = SpatialConfig()
        # embed_dim must equal num_heads * head_dim
        assert c.embed_dim == c.num_heads * c.head_dim

    def test_dim_mismatch_raises(self):
        with pytest.raises(ValueError, match="num_heads"):
            SpatialConfig(embed_dim=512, num_heads=7, head_dim=64)

    def test_zero_depth_raises(self):
        with pytest.raises(ValueError, match="depth"):
            SpatialConfig(depth=0)


class TestLossConfig:
    def test_defaults(self):
        c = LossConfig()
        assert c.primary == "sigmoid_siglip"

    def test_invalid_primary_raises(self):
        with pytest.raises(ValueError, match="primary"):
            LossConfig(primary="cosface")


class TestTrainingConfig:
    def test_defaults(self):
        c = TrainingConfig()
        assert c.precision == "bfloat16"

    def test_invalid_precision_raises(self):
        with pytest.raises(ValueError, match="precision"):
            TrainingConfig(precision="fp8")

    def test_unsupported_optimizer_raises(self):
        with pytest.raises(ValueError, match="adamw"):
            TrainingConfig(optimizer="sgd")

    def test_betas_must_have_length_2(self):
        with pytest.raises(ValueError, match="betas"):
            TrainingConfig(betas=[0.9, 0.95, 0.999])


class TestDataConfig:
    def test_defaults(self):
        c = DataConfig()
        assert c.dataset_name == "cc3m"

    def test_buckets_must_be_pairs(self):
        with pytest.raises(ValueError, match="\\[H, W\\]"):
            DataConfig(image_buckets=[[224, 224, 224]])

    def test_buckets_must_be_divisible_by_16(self):
        with pytest.raises(ValueError, match="divisible by 16"):
            DataConfig(image_buckets=[[100, 100]])


class TestKeuralConfigCrossValidation:
    def test_default_construction_is_valid(self):
        c = KeuralConfig()
        # Defaults: stem ends in 512, embed_dim=512 — match.
        assert c.stem.channels[-1] == c.spatial.embed_dim

    def test_stem_channel_must_match_embed_dim(self):
        with pytest.raises(ValueError, match="must equal"):
            KeuralConfig(
                stem=StemConfig(channels=[64, 128, 256, 384]),
                spatial=SpatialConfig(embed_dim=512, num_heads=8, head_dim=64),
            )

    def test_consistent_small_config_works(self):
        # All consistent: stem ends in 384, embed_dim=384, 6*64=384.
        c = KeuralConfig(
            stem=StemConfig(channels=[64, 128, 256, 384]),
            spatial=SpatialConfig(embed_dim=384, num_heads=6, head_dim=64),
        )
        assert c.spatial.embed_dim == 384


class TestSerialization:
    def test_round_trip_through_dict(self):
        original = KeuralConfig()
        d = original.to_dict()
        reconstructed = KeuralConfig.from_dict(d)
        assert reconstructed.to_dict() == original.to_dict()

    def test_round_trip_through_yaml(self):
        original = KeuralConfig(name="test-roundtrip")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            original.save_yaml(path)
            assert path.exists()
            reconstructed = KeuralConfig.load_yaml(path)
        assert reconstructed.name == "test-roundtrip"
        assert reconstructed.to_dict() == original.to_dict()

    def test_partial_dict_uses_defaults(self):
        # A YAML with just a name should still produce a valid config.
        c = KeuralConfig.from_dict({"name": "minimal"})
        assert c.name == "minimal"
        assert c.spatial.embed_dim == 512  # default

    def test_yaml_loads_validates_invariants(self):
        # If a YAML file violates a cross-config invariant, loading must raise.
        bad = {
            "stem": {"channels": [64, 128, 256, 384]},
            "spatial": {"embed_dim": 512, "num_heads": 8, "head_dim": 64},
        }
        with pytest.raises(ValueError, match="must equal"):
            KeuralConfig.from_dict(bad)


class TestPOCConfigFile:
    """The shipped POC config must load and pass all validations."""

    def test_poc_config_loads(self):
        path = Path(__file__).parent.parent / "configs" / "keural_tiny_poc.yaml"
        assert path.exists(), f"POC config not found at {path}"
        cfg = KeuralConfig.load_yaml(path)
        assert cfg.name == "keural-tiny-poc"
        # Verify the small-model invariants
        assert cfg.spatial.embed_dim == 384
        assert cfg.stem.channels[-1] == 384
        assert cfg.spatial.num_heads * cfg.spatial.head_dim == 384


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
