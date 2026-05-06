"""Unit tests for alignment/output_contract.py."""

from __future__ import annotations

import pytest
import torch

from alignment.output_contract import (
    LEVEL_DETAIL,
    LEVEL_GLOBAL,
    LEVEL_NAMES,
    LEVEL_REGION,
    NUM_LEVELS,
    KeuralEncoderOutput,
    empty_output,
)


# ----- Constants and module-level invariants -----


class TestLevelConstants:
    """Level constants are part of the public contract — verify they don't shift."""

    def test_level_values_are_stable(self):
        assert LEVEL_GLOBAL == 0
        assert LEVEL_REGION == 1
        assert LEVEL_DETAIL == 2

    def test_num_levels_matches_constants(self):
        assert NUM_LEVELS == 3

    def test_level_names_are_complete(self):
        for level in (LEVEL_GLOBAL, LEVEL_REGION, LEVEL_DETAIL):
            assert level in LEVEL_NAMES
        assert LEVEL_NAMES[LEVEL_GLOBAL] == "global"
        assert LEVEL_NAMES[LEVEL_REGION] == "region"
        assert LEVEL_NAMES[LEVEL_DETAIL] == "detail"


# ----- empty_output factory -----


class TestEmptyOutput:
    def test_default_construction(self):
        out = empty_output(batch_size=2, max_tokens=16, embed_dim=64)
        out.validate()
        assert out.tokens.shape == (2, 16, 64)
        assert out.attention_mask.shape == (2, 16)
        assert out.level_ids.shape == (2, 16)
        assert out.spatial_metadata.shape == (2, 16, 4)
        assert out.saliency_scores.shape == (2, 16)
        assert out.pooled.shape == (2, 64)
        assert out.token_budget_used.shape == (2,)

    def test_padding_sentinels(self):
        out = empty_output(batch_size=1, max_tokens=8, embed_dim=32)
        # Empty output represents all-padding state
        assert (out.attention_mask == 0).all(), "attention_mask should be all zero"
        assert (out.level_ids == -1).all(), "level_ids should be sentinel -1"
        assert out.num_real_tokens().item() == 0

    def test_dtype_propagation(self):
        out = empty_output(2, 8, 16, dtype=torch.float16)
        assert out.tokens.dtype == torch.float16
        assert out.spatial_metadata.dtype == torch.float16
        assert out.saliency_scores.dtype == torch.float16
        assert out.pooled.dtype == torch.float16
        # Integer fields stay int64 regardless of dtype arg
        assert out.attention_mask.dtype == torch.int64
        assert out.level_ids.dtype == torch.int64
        assert out.token_budget_used.dtype == torch.int64


# ----- Validate method -----


class TestValidate:
    def _make_valid(self, B: int = 2, N: int = 8, D: int = 16) -> KeuralEncoderOutput:
        out = empty_output(B, N, D)
        # Mark first 4 tokens of each batch item as real, alternating levels
        out.attention_mask[:, :4] = 1
        out.level_ids[:, 0] = LEVEL_GLOBAL
        out.level_ids[:, 1] = LEVEL_REGION
        out.level_ids[:, 2] = LEVEL_REGION
        out.level_ids[:, 3] = LEVEL_DETAIL
        out.spatial_metadata[:, :4] = 0.5  # any value in [0, 1] is fine
        out.saliency_scores[:, :4] = 0.7
        out.token_budget_used[:] = 4
        return out

    def test_valid_output_passes(self):
        self._make_valid().validate()  # Should not raise

    def test_wrong_attention_mask_shape_raises(self):
        out = self._make_valid()
        out.attention_mask = torch.zeros(2, 999, dtype=torch.int64)
        with pytest.raises(AssertionError, match="attention_mask"):
            out.validate()

    def test_wrong_pooled_shape_raises(self):
        out = self._make_valid()
        out.pooled = torch.zeros(2, 999)
        with pytest.raises(AssertionError, match="pooled"):
            out.validate()

    def test_wrong_spatial_metadata_dim_raises(self):
        out = self._make_valid()
        # 3 instead of 4 — a real bug we want to catch
        out.spatial_metadata = torch.zeros(2, 8, 3)
        with pytest.raises(AssertionError, match="spatial_metadata"):
            out.validate()

    def test_invalid_level_id_raises(self):
        out = self._make_valid()
        out.level_ids[0, 0] = 99  # invalid level
        with pytest.raises(AssertionError, match="level_ids"):
            out.validate()


# ----- Convenience accessors -----


class TestAccessors:
    def _make_realistic(self) -> KeuralEncoderOutput:
        """A 2-item batch with mixed real/padding tokens at multiple levels."""
        B, N, D = 2, 10, 8
        out = empty_output(B, N, D)
        # Item 0: 6 real tokens — 1 global, 2 region, 3 detail
        out.attention_mask[0, :6] = 1
        out.level_ids[0, 0] = LEVEL_GLOBAL
        out.level_ids[0, 1] = LEVEL_REGION
        out.level_ids[0, 2] = LEVEL_REGION
        out.level_ids[0, 3] = LEVEL_DETAIL
        out.level_ids[0, 4] = LEVEL_DETAIL
        out.level_ids[0, 5] = LEVEL_DETAIL
        # Distinct token values for identifiability
        out.tokens[0, :6] = torch.arange(6).float().unsqueeze(-1).expand(6, D)
        # Item 1: 4 real tokens — 1 global, 1 region, 2 detail
        out.attention_mask[1, :4] = 1
        out.level_ids[1, 0] = LEVEL_GLOBAL
        out.level_ids[1, 1] = LEVEL_REGION
        out.level_ids[1, 2] = LEVEL_DETAIL
        out.level_ids[1, 3] = LEVEL_DETAIL
        out.tokens[1, :4] = torch.arange(100, 104).float().unsqueeze(-1).expand(4, D)
        out.token_budget_used[0] = 6
        out.token_budget_used[1] = 4
        return out

    def test_num_real_tokens(self):
        out = self._make_realistic()
        nr = out.num_real_tokens()
        assert nr.tolist() == [6, 4]

    def test_get_level_mask_global(self):
        out = self._make_realistic()
        mask = out.get_level_mask(LEVEL_GLOBAL)
        assert mask.shape == (2, 10)
        # Item 0: only position 0 is global
        assert mask[0].tolist() == [True, False, False, False, False, False, False, False, False, False]
        # Item 1: only position 0 is global
        assert mask[1].tolist() == [True, False, False, False, False, False, False, False, False, False]

    def test_get_level_mask_excludes_padding(self):
        """Padding has level_id=-1, so it should never appear in any level mask."""
        out = self._make_realistic()
        for level in (LEVEL_GLOBAL, LEVEL_REGION, LEVEL_DETAIL):
            mask = out.get_level_mask(level)
            # All True positions must be within the real-token range
            assert ((~mask) | out.attention_mask.bool()).all()

    def test_get_level_mask_invalid_level_raises(self):
        out = self._make_realistic()
        with pytest.raises(ValueError, match="level"):
            out.get_level_mask(99)

    def test_get_tokens_for_level_detail(self):
        out = self._make_realistic()
        tokens, batch_idx = out.get_tokens_for_level(LEVEL_DETAIL)
        # Item 0 contributes 3 detail tokens, item 1 contributes 2 — total 5
        assert tokens.shape == (5, 8)
        assert batch_idx.tolist() == [0, 0, 0, 1, 1]
        # Item 0 detail tokens are token positions 3, 4, 5, with values 3, 4, 5
        # (because we set tokens[0, i] = i for i in range(6))
        assert tokens[0, 0].item() == 3.0
        assert tokens[1, 0].item() == 4.0
        assert tokens[2, 0].item() == 5.0
        # Item 1 detail tokens are positions 2, 3, with values 102, 103
        assert tokens[3, 0].item() == 102.0
        assert tokens[4, 0].item() == 103.0


# ----- Tensor-mutation methods (.to, .detach) -----


class TestTensorMethods:
    def test_to_dtype(self):
        out = empty_output(2, 8, 16, dtype=torch.float32)
        moved = out.to(dtype=torch.float16)
        assert moved.tokens.dtype == torch.float16
        assert moved.pooled.dtype == torch.float16
        # Integer fields stay int64 — to() with a float dtype should not affect them
        assert moved.attention_mask.dtype == torch.int64
        assert moved.level_ids.dtype == torch.int64

    def test_to_returns_new_object(self):
        out = empty_output(1, 4, 8)
        moved = out.to(dtype=torch.float64)
        assert moved is not out
        # Original is unchanged
        assert out.tokens.dtype == torch.float32

    def test_detach(self):
        out = empty_output(1, 4, 8)
        out.tokens.requires_grad_(True)
        # Run an op so the tensor has grad_fn
        new_tokens = out.tokens * 2
        out.tokens = new_tokens
        assert out.tokens.requires_grad

        detached = out.detach()
        assert not detached.tokens.requires_grad


# ----- Stability / public API contract -----


class TestStabilityContract:
    """These tests guard against accidental contract breakage.

    If a future PR removes a field or changes its semantics, these tests
    fail and the change is forced to be deliberate (with a docs update).
    """

    def test_required_fields_present(self):
        """Every documented field must exist in the dataclass."""
        out = empty_output(1, 1, 1)
        for required in (
            "tokens",
            "attention_mask",
            "level_ids",
            "spatial_metadata",
            "saliency_scores",
            "pooled",
            "token_budget_used",
        ):
            assert hasattr(out, required), f"Missing required field: {required}"

    def test_spatial_metadata_has_four_columns(self):
        """4 = (cx, cy, w, h). Don't change this without a docs revision."""
        out = empty_output(1, 1, 1)
        assert out.spatial_metadata.shape[-1] == 4

    def test_padding_sentinel_for_level_ids_is_negative_one(self):
        """Documented sentinel; downstream consumers may rely on it."""
        out = empty_output(1, 1, 1)
        assert out.level_ids.item() == -1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
