"""Unit tests for training/train.py.

The headline test is the end-to-end smoke test: configure → train for a
few steps → produce a checkpoint that can be reloaded.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from keural_config import KeuralConfig, SpatialConfig, StemConfig
from training.train import (
    constant_lr,
    cosine_warmup_lr,
    get_lr_schedule,
    linear_warmup_lr,
    train,
)


# ----- LR schedules -----


class TestLRSchedules:
    def test_cosine_warmup(self):
        # Step 0 in warmup
        lr = cosine_warmup_lr(step=0, peak_lr=1e-3, warmup_steps=10, max_steps=100)
        assert lr <= 1e-3 / 10 + 1e-9

        # End of warmup
        lr = cosine_warmup_lr(step=10, peak_lr=1e-3, warmup_steps=10, max_steps=100)
        # Should be at or near peak
        assert abs(lr - 1e-3) < 1e-4

        # Mid-decay
        lr_mid = cosine_warmup_lr(step=55, peak_lr=1e-3, warmup_steps=10, max_steps=100)
        assert 1e-4 < lr_mid < 1e-3

        # End of training: still positive (above min_lr_ratio * peak)
        lr_end = cosine_warmup_lr(step=99, peak_lr=1e-3, warmup_steps=10, max_steps=100)
        assert lr_end > 0

    def test_constant_schedule(self):
        for s in (0, 50, 99):
            assert constant_lr(s, 1e-3) == 1e-3

    def test_linear_warmup(self):
        assert linear_warmup_lr(0, 1e-3, 10, 100) <= 1e-3 / 10 + 1e-9
        assert abs(linear_warmup_lr(10, 1e-3, 10, 100) - 1e-3) < 1e-4
        assert linear_warmup_lr(50, 1e-3, 10, 100) == 1e-3

    def test_get_lr_schedule(self):
        f = get_lr_schedule("cosine_with_warmup")
        assert f is cosine_warmup_lr

    def test_get_lr_schedule_invalid(self):
        with pytest.raises(ValueError):
            get_lr_schedule("not_a_schedule")


# ----- End-to-end smoke test -----


def _make_smoke_config() -> KeuralConfig:
    """A tiny config that runs fast enough for a CI smoke test."""
    cfg = KeuralConfig(
        stem=StemConfig(channels=[16, 32, 48, 64]),
        spatial=SpatialConfig(embed_dim=64, num_heads=4, head_dim=16, depth=2),
    )
    cfg.training.precision = "float32"  # disable AMP for CPU-friendly testing
    cfg.training.gradient_checkpointing = False
    cfg.training.batch_size_per_device = 4
    cfg.training.gradient_accumulation = 1
    cfg.training.lr = 1e-3
    cfg.training.warmup_steps = 2
    cfg.training.max_steps = 5
    cfg.training.log_every = 1
    cfg.training.save_every = 100  # Don't save mid-test
    cfg.tokenizer.warmup_steps_fixed_grid = 1  # Quickly through warmup
    cfg.data.image_buckets = [[224, 224]]
    return cfg


class TestEnd2EndTraining:
    def test_smoke_run_completes(self):
        """Train for 5 steps on synthetic data — must complete without errors."""
        cfg = _make_smoke_config()
        with tempfile.TemporaryDirectory() as tmp:
            result = train(
                config=cfg,
                output_dir=tmp,
                synthetic=True,
                synthetic_size=20,
                max_steps_override=5,
            )
            assert result["steps_completed"] == 5
            assert result["final_loss"] is not None
            # Final dir exists with HF-format files
            final_dir = Path(result["final_dir"])
            assert final_dir.exists()
            assert (final_dir / "config.json").exists()
            # Either safetensors or pytorch bin
            files = list(final_dir.iterdir())
            assert any(f.suffix in (".safetensors", ".bin") for f in files), (
                f"No weights file in {final_dir}: {[f.name for f in files]}"
            )

    def test_training_log_written(self):
        cfg = _make_smoke_config()
        with tempfile.TemporaryDirectory() as tmp:
            train(
                config=cfg,
                output_dir=tmp,
                synthetic=True,
                synthetic_size=20,
                max_steps_override=3,
            )
            log_path = Path(tmp) / "training_log.jsonl"
            assert log_path.exists()
            lines = log_path.read_text().strip().splitlines()
            assert len(lines) > 0
            # Each line is valid JSON with the loss components
            for line in lines:
                d = json.loads(line)
                assert "step" in d
                assert "loss/total" in d
                assert "lr" in d

    def test_checkpoint_can_be_reloaded(self):
        """The final checkpoint must be loadable via KeuralVisionModel.from_pretrained."""
        cfg = _make_smoke_config()
        with tempfile.TemporaryDirectory() as tmp:
            result = train(
                config=cfg,
                output_dir=tmp,
                synthetic=True,
                synthetic_size=12,
                max_steps_override=3,
            )
            from alignment.hf_wrapper import KeuralVisionModel

            reloaded = KeuralVisionModel.from_pretrained(result["final_dir"])
            # Run a forward pass on the reloaded model
            import torch

            x = torch.randn(1, 3, 224, 224)
            out = reloaded(x, token_budget=32)
            assert out.tokens.shape == (1, 32, 64)

    def test_loss_is_finite_throughout(self):
        """Across 5 steps, loss must remain finite — catches NaN explosions."""
        cfg = _make_smoke_config()
        with tempfile.TemporaryDirectory() as tmp:
            train(
                config=cfg,
                output_dir=tmp,
                synthetic=True,
                synthetic_size=20,
                max_steps_override=5,
            )
            log = (Path(tmp) / "training_log.jsonl").read_text().strip().splitlines()
            for line in log:
                d = json.loads(line)
                import math

                assert math.isfinite(d["loss/total"]), f"Non-finite loss at step {d['step']}"

    def test_real_data_branch_raises_clearly(self):
        """When synthetic=False without a real loader, the user gets a clear error."""
        cfg = _make_smoke_config()
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(NotImplementedError, match="CC3M"):
                train(
                    config=cfg,
                    output_dir=tmp,
                    synthetic=False,
                    max_steps_override=1,
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
