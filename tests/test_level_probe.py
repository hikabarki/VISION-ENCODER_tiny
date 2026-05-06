"""Unit tests for evaluation/level_probe.py."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from data.datasets import (
    BucketedSampler,
    FrozenTextEncoder,
    SyntheticPairDataset,
    build_collate_fn,
)
from evaluation.level_probe import (
    LevelProbeResult,
    evaluate_level_probe,
    render_probe_table,
)


def _setup():
    from alignment.hf_wrapper import KeuralVisionConfig, KeuralVisionModel
    from keural_config import KeuralConfig, SpatialConfig, StemConfig

    kc = KeuralConfig(
        stem=StemConfig(channels=[16, 32, 48, 64]),
        spatial=SpatialConfig(embed_dim=64, num_heads=4, head_dim=16, depth=2),
    )
    kc.training.gradient_checkpointing = False
    cfg = KeuralVisionConfig(keural_config=kc.to_dict())
    model = KeuralVisionModel(cfg)
    text_enc = FrozenTextEncoder(
        model_name="nonexistent/use-fallback", output_dim=64
    )
    ds = SyntheticPairDataset(num_samples=16, image_buckets=[(224, 224)])
    sampler = BucketedSampler(
        ds.bucket_assignments, batch_size=4, shuffle=False, drop_last=False
    )
    collate = build_collate_fn([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=0)
    return model, text_enc, loader


class TestLevelProbe:
    def test_basic_run(self):
        model, text_enc, loader = _setup()
        results = evaluate_level_probe(
            model=model, loader=loader, text_encoder=text_enc, token_budget=32
        )
        # 4 results: pooled, global, region, detail
        assert len(results) == 4
        names = {r.level_name for r in results}
        assert names == {"pooled", "global", "region", "detail"}

    def test_recall_in_range(self):
        model, text_enc, loader = _setup()
        results = evaluate_level_probe(
            model=model, loader=loader, text_encoder=text_enc, token_budget=32
        )
        for r in results:
            assert 0.0 <= r.recall_at_1 <= 1.0
            assert 0.0 <= r.recall_at_5 <= 1.0
            assert r.recall_at_5 >= r.recall_at_1

    def test_num_pairs_consistent(self):
        model, text_enc, loader = _setup()
        results = evaluate_level_probe(
            model=model, loader=loader, text_encoder=text_enc, token_budget=32
        )
        # All results should report the same num_pairs (they used the same dataset)
        n_pairs = [r.num_pairs for r in results]
        assert len(set(n_pairs)) == 1

    def test_render_table(self):
        results = [
            LevelProbeResult(
                level_name="pooled", level_id=-1, recall_at_1=0.5, recall_at_5=0.8, num_pairs=100
            ),
            LevelProbeResult(
                level_name="global", level_id=0, recall_at_1=0.3, recall_at_5=0.6, num_pairs=100
            ),
        ]
        table = render_probe_table(results)
        assert "pooled" in table
        assert "global" in table
        assert "0.5" in table


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
