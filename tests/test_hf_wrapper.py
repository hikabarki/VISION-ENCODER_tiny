"""Unit tests for alignment/hf_wrapper.py.

Per docs/04_revisions.md §5 (Revision C), HF integration is validated on
Day 1 — these tests prove the save/load and forward-shape contracts work
on a randomly-initialized skeleton, before any real architecture is wired in.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModel

from alignment.hf_wrapper import KeuralVisionConfig, KeuralVisionModel
from alignment.output_contract import KeuralEncoderOutput, NUM_LEVELS
from keural_config import KeuralConfig, SpatialConfig, StemConfig


# ----- Config wrapping -----


class TestKeuralVisionConfig:
    def test_default_construction(self):
        cfg = KeuralVisionConfig()
        assert cfg.model_type == "keural_vision"
        assert isinstance(cfg.keural_config, KeuralConfig)

    def test_mirrored_fields_match_structured(self):
        cfg = KeuralVisionConfig()
        assert cfg.embed_dim == cfg.keural_config.spatial.embed_dim
        assert cfg.depth == cfg.keural_config.spatial.depth
        assert cfg.num_heads == cfg.keural_config.spatial.num_heads

    def test_to_dict_includes_keural_config(self):
        cfg = KeuralVisionConfig()
        d = cfg.to_dict()
        assert "keural_config" in d
        assert d["keural_config"]["spatial"]["embed_dim"] == 512


# ----- Model construction -----


class TestKeuralVisionModelConstruction:
    def test_default_init(self):
        cfg = KeuralVisionConfig()
        model = KeuralVisionModel(cfg)
        # Has at least one parameter (the placeholder)
        assert any(True for _ in model.parameters())

    def test_param_count_in_reasonable_range(self):
        cfg = KeuralVisionConfig()
        model = KeuralVisionModel(cfg)
        n = sum(p.numel() for p in model.parameters())
        # Default config is the ~50-80M POC config — large but bounded.
        # The bound is loose: accept anywhere from 30M to 150M.
        assert 30_000_000 < n < 150_000_000, (
            f"Default config param count {n:,} outside expected range"
        )


# ----- Forward signature -----


class TestForward:
    def _make_small_model(self) -> KeuralVisionModel:
        # Smallest sensible config to keep tests fast
        kc = KeuralConfig(
            stem=StemConfig(channels=[32, 64, 96, 128]),
            spatial=SpatialConfig(embed_dim=128, num_heads=4, head_dim=32),
        )
        cfg = KeuralVisionConfig(keural_config=kc.to_dict())
        return KeuralVisionModel(cfg)

    def test_returns_keural_encoder_output(self):
        model = self._make_small_model()
        x = torch.randn(2, 3, 224, 224)
        out = model(x, token_budget=64)
        assert isinstance(out, KeuralEncoderOutput)
        out.validate()

    def test_output_shapes(self):
        model = self._make_small_model()
        x = torch.randn(3, 3, 224, 224)
        out = model(x, token_budget=128)
        assert out.tokens.shape == (3, 128, 128)
        assert out.attention_mask.shape == (3, 128)
        assert out.pooled.shape == (3, 128)
        assert out.spatial_metadata.shape == (3, 128, 4)

    def test_token_budget_respected(self):
        model = self._make_small_model()
        x = torch.randn(1, 3, 224, 224)
        for budget in (32, 64, 128, 256):
            out = model(x, token_budget=budget)
            assert out.tokens.shape[1] == budget, (
                f"budget={budget} but got tokens.shape[1]={out.tokens.shape[1]}"
            )

    def test_token_budget_none_uses_default(self):
        model = self._make_small_model()
        x = torch.randn(1, 3, 224, 224)
        out = model(x, token_budget=None)
        # Default for default config at 224x224 is 256
        assert out.tokens.shape[1] == 256

    def test_token_budget_auto_scales_with_resolution(self):
        model = self._make_small_model()
        # 512x512 has 4x the pixels of 256x256, so budget should scale up
        x_low = torch.randn(1, 3, 224, 224)
        x_high = torch.randn(1, 3, 512, 512)
        out_low = model(x_low, token_budget=None)
        out_high = model(x_high, token_budget=None)
        assert out_high.tokens.shape[1] > out_low.tokens.shape[1]

    def test_simple_mode_encode(self):
        model = self._make_small_model()
        x = torch.randn(2, 3, 224, 224)
        tokens = model.encode(x, token_budget=64)
        # encode() returns just the tensor, not the dataclass
        assert isinstance(tokens, torch.Tensor)
        assert tokens.shape == (2, 64, 128)

    def test_level_ids_are_valid(self):
        model = self._make_small_model()
        x = torch.randn(1, 3, 224, 224)
        out = model(x, token_budget=64)
        # All non-padding level_ids must be in {0, 1, 2}
        real_levels = out.level_ids[out.attention_mask.bool()]
        assert real_levels.min() >= 0
        assert real_levels.max() < NUM_LEVELS


# ----- Save / load round-trip -----


class TestSaveLoad:
    def test_save_pretrained_creates_files(self):
        model = KeuralVisionModel(KeuralVisionConfig())
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            files = list(Path(tmp).iterdir())
            file_names = {f.name for f in files}
            # HF always writes config.json and a weights file
            assert "config.json" in file_names
            assert any(name.endswith(".safetensors") or name.endswith(".bin") for name in file_names)

    def test_save_load_round_trip_via_class(self):
        model = KeuralVisionModel(KeuralVisionConfig())
        # Set the pool token to something distinctive so we can test round-trip.
        with torch.no_grad():
            model.encoder.pool_token.fill_(0.42)

        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            reloaded = KeuralVisionModel.from_pretrained(tmp)

        # Pool token round-trips
        assert torch.allclose(
            reloaded.encoder.pool_token,
            torch.full_like(reloaded.encoder.pool_token, 0.42),
        )

    def test_save_load_via_automodel(self):
        """The headline requirement: AutoModel.from_pretrained() must work."""
        # Use a small custom config to make sure custom configs round-trip too
        kc = KeuralConfig(
            stem=StemConfig(channels=[32, 64, 96, 128]),
            spatial=SpatialConfig(embed_dim=128, num_heads=4, head_dim=32),
        )
        cfg = KeuralVisionConfig(keural_config=kc.to_dict())
        model = KeuralVisionModel(cfg)

        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            # The headline call:
            reloaded_cfg = AutoConfig.from_pretrained(tmp)
            reloaded = AutoModel.from_pretrained(tmp)

        assert isinstance(reloaded_cfg, KeuralVisionConfig)
        assert isinstance(reloaded, KeuralVisionModel)
        assert reloaded_cfg.embed_dim == 128
        assert reloaded_cfg.keural_config.spatial.embed_dim == 128

    def test_loaded_model_runs_forward(self):
        kc = KeuralConfig(
            stem=StemConfig(channels=[32, 64, 96, 128]),
            spatial=SpatialConfig(embed_dim=128, num_heads=4, head_dim=32),
        )
        cfg = KeuralVisionConfig(keural_config=kc.to_dict())
        model = KeuralVisionModel(cfg)

        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            reloaded = AutoModel.from_pretrained(tmp)

        # Reloaded model must produce a valid output
        x = torch.randn(1, 3, 224, 224)
        out = reloaded(x, token_budget=32)
        out.validate()
        assert out.tokens.shape == (1, 32, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
