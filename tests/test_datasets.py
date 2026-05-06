"""Unit tests for data/datasets.py."""

from __future__ import annotations

from collections import Counter

import pytest
import torch
from torch.utils.data import DataLoader

from data.datasets import (
    BucketedSampler,
    FrozenTextEncoder,
    SyntheticPairDataset,
    build_collate_fn,
)


# ----- SyntheticPairDataset -----


class TestSyntheticPairDataset:
    def test_basic_construction(self):
        ds = SyntheticPairDataset(num_samples=10, image_buckets=[(224, 224)])
        assert len(ds) == 10

    def test_getitem_shapes(self):
        ds = SyntheticPairDataset(
            num_samples=10, image_buckets=[(224, 224), (256, 256)]
        )
        item = ds[0]
        H, W = ds.image_buckets[item.bucket_idx]
        assert item.pixel_values.shape == (3, H, W)
        assert isinstance(item.caption, str)

    def test_deterministic_across_construction(self):
        """Same seed → same items."""
        ds1 = SyntheticPairDataset(num_samples=5, image_buckets=[(224, 224)], seed=42)
        ds2 = SyntheticPairDataset(num_samples=5, image_buckets=[(224, 224)], seed=42)
        for i in range(5):
            assert torch.equal(ds1[i].pixel_values, ds2[i].pixel_values)
            assert ds1[i].caption == ds2[i].caption

    def test_different_indices_different_images(self):
        ds = SyntheticPairDataset(num_samples=3, image_buckets=[(224, 224)])
        a = ds[0].pixel_values
        b = ds[1].pixel_values
        assert not torch.equal(a, b)

    def test_buckets_assigned_consistently(self):
        ds = SyntheticPairDataset(
            num_samples=20, image_buckets=[(224, 224), (256, 256), (320, 320)]
        )
        # Each item's bucket should be stable across reads.
        for i in range(20):
            b1 = ds[i].bucket_idx
            b2 = ds[i].bucket_idx
            assert b1 == b2

    def test_rejects_zero_samples(self):
        with pytest.raises(ValueError, match="num_samples"):
            SyntheticPairDataset(num_samples=0, image_buckets=[(224, 224)])

    def test_rejects_empty_buckets(self):
        with pytest.raises(ValueError, match="image_buckets"):
            SyntheticPairDataset(num_samples=5, image_buckets=[])

    def test_index_bounds(self):
        ds = SyntheticPairDataset(num_samples=3, image_buckets=[(224, 224)])
        with pytest.raises(IndexError):
            ds[5]


# ----- BucketedSampler -----


class TestBucketedSampler:
    def test_basic_iteration(self):
        # 10 samples in 2 buckets, batch_size 2
        bucket_assignments = [0, 0, 0, 1, 1, 0, 1, 0, 1, 1]  # bucket 0: 5, bucket 1: 5
        sampler = BucketedSampler(bucket_assignments, batch_size=2, shuffle=False)
        batches = list(sampler)
        # Each batch must have all-same bucket
        for batch in batches:
            buckets = {bucket_assignments[i] for i in batch}
            assert len(buckets) == 1, f"Batch {batch} mixed buckets {buckets}"

    def test_drop_last(self):
        bucket_assignments = [0, 0, 0, 0, 0]  # 5 samples, all bucket 0
        sampler = BucketedSampler(
            bucket_assignments, batch_size=2, shuffle=False, drop_last=True
        )
        batches = list(sampler)
        # 5 // 2 = 2 batches when dropping the partial
        assert len(batches) == 2

    def test_keep_last(self):
        bucket_assignments = [0, 0, 0, 0, 0]
        sampler = BucketedSampler(
            bucket_assignments, batch_size=2, shuffle=False, drop_last=False
        )
        batches = list(sampler)
        # 5 / 2 → 3 batches when keeping the partial
        assert len(batches) == 3

    def test_len_matches_iteration(self):
        bucket_assignments = [0, 0, 0, 1, 1, 1, 1, 0, 0]
        sampler = BucketedSampler(bucket_assignments, batch_size=2, drop_last=True)
        n_iter = len(list(sampler))
        assert len(sampler) == n_iter

    def test_shuffle_changes_order_across_epochs(self):
        bucket_assignments = [0] * 20
        sampler = BucketedSampler(
            bucket_assignments, batch_size=4, shuffle=True, seed=0
        )
        sampler.set_epoch(0)
        batches_0 = [tuple(b) for b in sampler]
        sampler.set_epoch(1)
        batches_1 = [tuple(b) for b in sampler]
        assert batches_0 != batches_1

    def test_all_indices_appear(self):
        bucket_assignments = [0, 1, 0, 1, 0, 1]
        sampler = BucketedSampler(
            bucket_assignments, batch_size=2, shuffle=False, drop_last=False
        )
        seen = []
        for batch in sampler:
            seen.extend(batch)
        assert sorted(seen) == sorted(range(len(bucket_assignments)))


# ----- collate_fn -----


class TestCollate:
    def test_basic_collate(self):
        ds = SyntheticPairDataset(num_samples=8, image_buckets=[(224, 224)])
        collate = build_collate_fn([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        batch_items = [ds[i] for i in range(4)]
        result = collate(batch_items)
        assert result["pixel_values"].shape == (4, 3, 224, 224)
        assert len(result["captions"]) == 4
        assert result["bucket_idx"] == 0

    def test_normalization_applied(self):
        ds = SyntheticPairDataset(num_samples=2, image_buckets=[(224, 224)])
        collate = build_collate_fn([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])  # identity
        result = collate([ds[0], ds[1]])
        # With identity mean/std, output should equal input float pixels
        assert torch.allclose(result["pixel_values"][0], ds[0].pixel_values.float())

    def test_bucket_mismatch_raises(self):
        ds = SyntheticPairDataset(
            num_samples=20, image_buckets=[(224, 224), (256, 256)]
        )
        # Find one item in each bucket
        b0 = next(ds[i] for i in range(20) if ds.bucket_assignments[i] == 0)
        b1 = next(ds[i] for i in range(20) if ds.bucket_assignments[i] == 1)
        collate = build_collate_fn([0.0] * 3, [1.0] * 3)
        with pytest.raises(ValueError, match="Bucket mismatch"):
            collate([b0, b1])

    def test_dataloader_integration(self):
        """Sampler + collate + DataLoader work together."""
        ds = SyntheticPairDataset(
            num_samples=20, image_buckets=[(224, 224), (256, 256)]
        )
        sampler = BucketedSampler(
            ds.bucket_assignments, batch_size=4, shuffle=False, drop_last=True
        )
        collate = build_collate_fn([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=0)
        for batch in loader:
            assert batch["pixel_values"].shape[0] == 4
            assert len(batch["captions"]) == 4
            assert batch["pixel_values"].shape[2] in (224, 256)


# ----- FrozenTextEncoder -----


class TestFrozenTextEncoder:
    def test_encoder_runs(self):
        # Use fallback by passing an invalid model name
        enc = FrozenTextEncoder(
            model_name="nonexistent/does-not-exist", output_dim=64
        )
        captions = ["hello world", "another caption"]
        out = enc(captions)
        assert out.shape == (2, 64)
        assert torch.isfinite(out).all()

    def test_distinct_captions_distinct_outputs(self):
        enc = FrozenTextEncoder(
            model_name="nonexistent/does-not-exist", output_dim=64
        )
        out_a = enc(["hello world"])
        out_b = enc(["completely different text"])
        assert not torch.allclose(out_a, out_b)

    def test_projection_is_trainable(self):
        enc = FrozenTextEncoder(
            model_name="nonexistent/does-not-exist", output_dim=64
        )
        # The projection layer's parameters must be trainable.
        proj_params = [p for p in enc.proj.parameters() if p.requires_grad]
        assert len(proj_params) > 0

    def test_pretrained_params_are_frozen(self):
        """If the real pretrained model loaded, its params must be frozen."""
        enc = FrozenTextEncoder(
            model_name="nonexistent/does-not-exist", output_dim=64
        )
        if not enc._fallback:
            # We're using the real pretrained model
            pretrained_params = [p for p in enc._model.parameters()]
            for p in pretrained_params:
                assert not p.requires_grad

    def test_gradient_through_projection(self):
        enc = FrozenTextEncoder(
            model_name="nonexistent/does-not-exist", output_dim=32
        )
        out = enc(["one", "two"])
        out.sum().backward()
        # The projection layer should have gradient.
        if isinstance(enc.proj, torch.nn.Linear):
            assert enc.proj.weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
