"""Unit tests for evaluation/budget_curve.py."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from data.datasets import FrozenTextEncoder
from evaluation.budget_curve import (
    BudgetCurvePoint,
    BudgetCurveResult,
    evaluate_budget_curve,
    render_ascii_chart,
)
from evaluation.imagenet_zeroshot import (
    DEFAULT_IMAGENET_CLASSES_TINY,
    DEFAULT_PROMPT_TEMPLATES,
    SyntheticLabeledDataset,
    synthetic_collate,
)


def _make_model_and_loader(num_classes: int = 5, num_samples: int = 16):
    from alignment.hf_wrapper import KeuralVisionConfig, KeuralVisionModel
    from keural_config import KeuralConfig, SpatialConfig, StemConfig

    kc = KeuralConfig(
        stem=StemConfig(channels=[16, 32, 48, 64]),
        spatial=SpatialConfig(embed_dim=64, num_heads=4, head_dim=16, depth=2),
    )
    kc.training.gradient_checkpointing = False
    cfg = KeuralVisionConfig(keural_config=kc.to_dict())
    model = KeuralVisionModel(cfg)
    ds = SyntheticLabeledDataset(num_samples=num_samples, num_classes=num_classes)
    loader = DataLoader(ds, batch_size=4, collate_fn=synthetic_collate, num_workers=0)
    return model, loader


class TestBudgetCurve:
    def test_basic_curve(self):
        model, loader = _make_model_and_loader()
        text_enc = FrozenTextEncoder(
            model_name="nonexistent/use-fallback", output_dim=64
        )
        result = evaluate_budget_curve(
            model=model,
            loader=loader,
            classes=DEFAULT_IMAGENET_CLASSES_TINY[:5],
            prompts=DEFAULT_PROMPT_TEMPLATES[:2],
            text_encoder=text_enc,
            budgets=[16, 32, 64],
            device="cpu",
        )
        assert isinstance(result, BudgetCurveResult)
        assert len(result.points) == 3
        assert result.num_images == 16

    def test_each_point_has_correct_budget(self):
        model, loader = _make_model_and_loader()
        text_enc = FrozenTextEncoder(
            model_name="nonexistent/use-fallback", output_dim=64
        )
        budgets = [16, 32, 64, 128]
        result = evaluate_budget_curve(
            model=model,
            loader=loader,
            classes=DEFAULT_IMAGENET_CLASSES_TINY[:5],
            prompts=DEFAULT_PROMPT_TEMPLATES[:2],
            text_encoder=text_enc,
            budgets=budgets,
            device="cpu",
        )
        for p, b in zip(result.points, budgets):
            assert p.token_budget == b

    def test_top5_geq_top1(self):
        model, loader = _make_model_and_loader(num_classes=10)
        text_enc = FrozenTextEncoder(
            model_name="nonexistent/use-fallback", output_dim=64
        )
        result = evaluate_budget_curve(
            model=model,
            loader=loader,
            classes=DEFAULT_IMAGENET_CLASSES_TINY[:10],
            prompts=DEFAULT_PROMPT_TEMPLATES[:1],
            text_encoder=text_enc,
            budgets=[32, 64],
            device="cpu",
        )
        for p in result.points:
            assert p.top5_acc >= p.top1_acc

    def test_empty_budgets_raises(self):
        model, loader = _make_model_and_loader()
        text_enc = FrozenTextEncoder(
            model_name="nonexistent/use-fallback", output_dim=64
        )
        with pytest.raises(ValueError, match="budgets"):
            evaluate_budget_curve(
                model=model,
                loader=loader,
                classes=DEFAULT_IMAGENET_CLASSES_TINY[:5],
                prompts=DEFAULT_PROMPT_TEMPLATES[:1],
                text_encoder=text_enc,
                budgets=[],
                device="cpu",
            )

    def test_to_json(self):
        result = BudgetCurveResult(
            points=[
                BudgetCurvePoint(token_budget=64, top1_acc=0.1, top5_acc=0.5),
                BudgetCurvePoint(token_budget=128, top1_acc=0.2, top5_acc=0.7),
            ],
            num_images=100,
            num_classes=10,
        )
        d = result.to_json()
        assert d["num_images"] == 100
        assert len(d["points"]) == 2
        assert d["points"][0]["token_budget"] == 64

    def test_render_ascii_chart(self):
        result = BudgetCurveResult(
            points=[
                BudgetCurvePoint(token_budget=64, top1_acc=0.10, top5_acc=0.5),
                BudgetCurvePoint(token_budget=128, top1_acc=0.20, top5_acc=0.7),
                BudgetCurvePoint(token_budget=256, top1_acc=0.40, top5_acc=0.9),
            ],
            num_images=100,
            num_classes=10,
        )
        chart = render_ascii_chart(result)
        # Smoke test: chart contains expected budgets
        assert "64" in chart
        assert "128" in chart
        assert "256" in chart


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
