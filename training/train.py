"""
Keural Vision Encoder — Training Loop.

This module provides the full pretraining loop. Highlights:

- Mixed-precision (bfloat16) via torch.amp
- Gradient accumulation for large effective batches on a single GPU
- AdamW + cosine LR schedule with linear warmup
- Per-step logging of losses and metrics
- Periodic checkpointing in HuggingFace `save_pretrained` format

Usage
-----
    python -m training.train --config configs/keural_tiny_poc.yaml

For a smoke test that doesn't need real data:

    python -m training.train --config configs/keural_tiny_poc.yaml --synthetic --max-steps 5

Reference: docs/02_release_plan.md §3.2 and docs/03_architecture_design.md §7-8.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import sys
import time
import traceback
import warnings
from dataclasses import asdict
from pathlib import Path

import psutil
import pynvml
import torch

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
import torch.nn as nn
from torch.utils.data import DataLoader

from alignment.hf_wrapper import KeuralVisionConfig, KeuralVisionModel
from data.datasets import (
    BucketedSampler,
    CC3MDataset,
    CombinedImageTextDataset,
    FrozenTextEncoder,
    SyntheticPairDataset,
    build_collate_fn,
)
from keural_config import KeuralConfig
from training.losses import KeuralLoss


# --------- System monitor ---------

_nvml_handle = None

def _init_nvml() -> None:
    global _nvml_handle
    try:
        pynvml.nvmlInit()
        _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        _nvml_handle = None


def get_system_stats() -> dict:
    """Collect GPU and CPU stats. Returns empty values gracefully on failure."""
    stats = {}
    # GPU stats via pynvml
    try:
        if _nvml_handle is not None:
            util = pynvml.nvmlDeviceGetUtilizationRates(_nvml_handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
            temp = pynvml.nvmlDeviceGetTemperature(_nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
            stats["gpu_util_pct"] = util.gpu
            stats["gpu_mem_used_mb"] = mem.used // (1024 * 1024)
            stats["gpu_mem_total_mb"] = mem.total // (1024 * 1024)
            stats["gpu_mem_pct"] = round(mem.used / mem.total * 100, 1)
            stats["gpu_temp_c"] = temp
    except Exception:
        pass
    # CPU + RAM stats via psutil
    try:
        stats["cpu_pct"] = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        stats["ram_used_gb"] = round(vm.used / (1024 ** 3), 2)
        stats["ram_total_gb"] = round(vm.total / (1024 ** 3), 2)
        stats["ram_pct"] = vm.percent
    except Exception:
        pass
    return stats


def _rotate_checkpoints(output_dir: Path, best_step: int | None = None) -> None:
    """Keep only the latest checkpoint and the best checkpoint. Delete all others."""
    ckpts = sorted(
        output_dir.glob("checkpoint-[0-9]*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not ckpts:
        return
    latest = ckpts[-1]
    for ckpt in ckpts:
        keep = ckpt == latest or (best_step is not None and ckpt.name == f"checkpoint-{best_step}")
        if not keep:
            shutil.rmtree(ckpt, ignore_errors=True)
            print(f"[train] Removed checkpoint: {ckpt.name}")


def _save_training_state(
    ckpt_dir: Path,
    step: int,
    optimizer: torch.optim.Optimizer,
    scaler,
) -> None:
    """Save optimizer state + step so training can resume exactly."""
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, ckpt_dir / "training_state.pt")


def _resume_from_checkpoint(
    output_dir: Path,
    model,
    optimizer: torch.optim.Optimizer,
    scaler,
) -> int:
    """Load the latest checkpoint if one exists. Returns the step to resume from."""
    ckpts = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
    )
    if not ckpts:
        return 0

    latest = ckpts[-1]
    resume_step = int(latest.name.split("-")[-1])

    print(f"[train] Resuming from checkpoint: {latest.name} (step {resume_step})")

    # Load model weights
    try:
        from alignment.hf_wrapper import KeuralVisionModel
        loaded = KeuralVisionModel.from_pretrained(latest)
        model.load_state_dict(loaded.state_dict())
        print(f"[train] Model weights loaded from {latest.name}")
    except Exception as e:
        print(f"[train] WARNING: Could not load model weights: {e}")
        return 0

    # Load optimizer + scaler state
    state_path = latest / "training_state.pt"
    if state_path.exists():
        try:
            state = torch.load(state_path, map_location="cpu")
            optimizer.load_state_dict(state["optimizer"])
            if scaler is not None and "scaler" in state:
                scaler.load_state_dict(state["scaler"])
            print(f"[train] Optimizer state loaded. Resuming at step {resume_step + 1}")
        except Exception as e:
            print(f"[train] WARNING: Could not load optimizer state: {e}. Starting optimizer fresh.")
    else:
        print(f"[train] No optimizer state found — model weights loaded but optimizer starts fresh.")

    return resume_step + 1


# --------- LR schedule ---------


def cosine_warmup_lr(
    step: int, peak_lr: float, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.05
) -> float:
    """Cosine LR schedule with linear warmup.

    Goes from 0 → peak_lr over `warmup_steps`, then cosine-decays from
    peak_lr → peak_lr * min_lr_ratio over the remaining steps.
    """
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, progress)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = peak_lr * min_lr_ratio
    return min_lr + (peak_lr - min_lr) * cosine


def constant_lr(step: int, peak_lr: float, *args, **kwargs) -> float:
    return peak_lr


def linear_warmup_lr(
    step: int, peak_lr: float, warmup_steps: int, max_steps: int, **kwargs
) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    return peak_lr


def get_lr_schedule(name: str):
    if name == "cosine_with_warmup":
        return cosine_warmup_lr
    if name == "constant":
        return constant_lr
    if name == "linear_with_warmup":
        return linear_warmup_lr
    raise ValueError(f"Unknown lr_schedule: {name}")


# --------- Build helpers ---------


def build_model(config: KeuralConfig) -> KeuralVisionModel:
    """Build a fresh KeuralVisionModel from a KeuralConfig."""
    hf_cfg = KeuralVisionConfig(keural_config=config.to_dict())
    model = KeuralVisionModel(hf_cfg)
    return model


def build_optimizer(model: nn.Module, loss_fn: nn.Module, config: KeuralConfig):
    """Build AdamW optimizer over both model and loss params (loss has
    learnable temperature/bias)."""
    if config.training.optimizer != "adamw":
        raise ValueError(f"Unsupported optimizer: {config.training.optimizer}")
    params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=config.training.lr,
        betas=tuple(config.training.betas),
        weight_decay=config.training.weight_decay,
    )
    return optimizer


def build_dataloader(
    config: KeuralConfig,
    synthetic: bool = False,
    synthetic_size: int = 1000,
    max_samples: int | None = None,
) -> DataLoader:
    """Build the training DataLoader.

    Synthetic mode uses SyntheticPairDataset (no data needed).
    Real mode supports:
      - dataset_name="cc3m"       → single CC3MDataset
      - dataset_name="cc12m"      → single CC3MDataset (cc12m paths)
      - dataset_name="cc3m+cc12m" → CombinedImageTextDataset

    For real mode, run build_filtered_index() once per source first to
    produce the filtered TSV files that CC3MDataset expects.
    """
    buckets = [tuple(b) for b in config.data.image_buckets]

    if synthetic:
        ds = SyntheticPairDataset(
            num_samples=synthetic_size, image_buckets=buckets, seed=config.training.seed
        )
        num_workers = 0
    else:
        name = config.data.dataset_name.lower()
        if name in ("cc3m", "cc12m"):
            ds = CC3MDataset(
                captions_path=config.data.captions_path,
                images_root=config.data.images_root,
                buckets=buckets,
                augment=True,
                max_samples=max_samples,
            )
        elif name == "cc3m+cc12m":
            if not config.data.extra_sources:
                raise ValueError(
                    "dataset_name='cc3m+cc12m' requires data.extra_sources to be set "
                    "with the second source's images_root and captions_path."
                )
            primary = CC3MDataset(
                captions_path=config.data.captions_path,
                images_root=config.data.images_root,
                buckets=buckets,
                augment=True,
                max_samples=max_samples,
            )
            extra_datasets = [
                CC3MDataset(
                    captions_path=src["captions_path"],
                    images_root=src["images_root"],
                    buckets=buckets,
                    augment=True,
                    max_samples=max_samples,
                )
                for src in config.data.extra_sources
            ]
            ds = CombinedImageTextDataset([primary] + extra_datasets)
        else:
            raise ValueError(
                f"Unknown dataset_name: '{config.data.dataset_name}'. "
                f"Supported: 'cc3m', 'cc12m', 'cc3m+cc12m'."
            )
        num_workers = config.data.num_workers

    sampler = BucketedSampler(
        ds.bucket_assignments,
        batch_size=config.training.batch_size_per_device,
        shuffle=True,
        seed=config.training.seed,
        drop_last=True,
        shard_keys=getattr(ds, "shard_keys", None),
    )
    collate = build_collate_fn(config.data.pixel_mean, config.data.pixel_std)
    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=1 if num_workers > 0 else None,
    )
    return loader


# --------- The training loop ---------


def train(
    config: KeuralConfig,
    output_dir: str | Path,
    synthetic: bool = False,
    synthetic_size: int = 1000,
    max_steps_override: int | None = None,
    max_samples: int | None = None,
    log_callback=None,
) -> dict:
    """Run the training loop.

    Parameters
    ----------
    config : KeuralConfig
    output_dir : path. Checkpoints and logs go here.
    synthetic : bool. If True, use SyntheticPairDataset (smoke-test mode).
    synthetic_size : int. Size of the synthetic dataset.
    max_steps_override : int or None. If given, override config.training.max_steps.
    log_callback : optional. A callable(dict) for streaming logs to wandb/tb.

    Returns
    -------
    Final summary dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Error log path (written on crash)
    error_log = output_dir / "error.log"

    # --- Init NVML for GPU monitoring
    _init_nvml()

    # --- Reproducibility
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    # --- Components
    model = build_model(config).to(device)
    loss_fn = KeuralLoss(config.loss).to(device)
    text_encoder = FrozenTextEncoder(
        model_name=config.data.text_encoder_name,
        output_dim=config.spatial.embed_dim,
        max_length=config.data.max_caption_length,
    ).to(device)
    optimizer = build_optimizer(model, loss_fn, config)
    # Also include trainable params from the text encoder's projection.
    optimizer.add_param_group({"params": [p for p in text_encoder.parameters() if p.requires_grad]})

    loader = build_dataloader(config, synthetic=synthetic, synthetic_size=synthetic_size, max_samples=max_samples)
    if len(loader.batch_sampler) == 0:
        raise ValueError(
            f"DataLoader produced 0 batches. This usually means your synthetic_size "
            f"({synthetic_size}) is smaller than batch_size_per_device "
            f"({config.training.batch_size_per_device}) with drop_last=True. "
            f"Either increase --synthetic-size or use --batch-size to lower the per-device batch."
        )

    # --- Mixed precision
    use_amp = config.training.precision in ("bfloat16", "float16") and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16 if config.training.precision == "bfloat16" else torch.float16
    )
    # GradScaler is needed for fp16 but not bf16. We handle both cases.
    needs_scaler = config.training.precision == "float16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if needs_scaler else None

    # --- Param count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # --- LR schedule
    schedule_fn = get_lr_schedule(config.training.lr_schedule)
    max_steps = max_steps_override if max_steps_override is not None else config.training.max_steps

    # --- Save the config alongside checkpoints
    with (output_dir / "training_config.json").open("w") as f:
        json.dump(asdict(config), f, indent=2)

    # --- Resume from latest checkpoint if one exists
    step = _resume_from_checkpoint(output_dir, model, optimizer, scaler)

    best_loss = float("inf")
    best_step = None

    accum_count = 0
    optimizer.zero_grad(set_to_none=True)
    loader_iter = iter(loader)
    last_log_time = time.time()
    training_start_time = time.time()
    log_lines: list[dict] = []

    model.train()
    loss_fn.train()
    text_encoder.train()  # only the projection moves; pretrained body stays frozen

    print(f"[train] Starting training. max_steps={max_steps}, "
          f"batch_per_device={config.training.batch_size_per_device}, "
          f"grad_accum={config.training.gradient_accumulation}")

    try:
        while step < max_steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            captions = batch["captions"]

            # Text embeddings (frozen body + trainable projection).
            text_embeds = text_encoder(captions)

            # Encoder forward + loss
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    encoder_out, sal_map = model.encoder(
                        pixel_values=pixel_values,
                        token_budget=None,
                        step=step,
                        return_saliency_map=True,
                    )
                    loss_out = loss_fn(encoder_out, text_embeds, saliency_map=sal_map)
                    loss = loss_out.total / config.training.gradient_accumulation
            else:
                encoder_out, sal_map = model.encoder(
                    pixel_values=pixel_values,
                    token_budget=None,
                    step=step,
                    return_saliency_map=True,
                )
                loss_out = loss_fn(encoder_out, text_embeds, saliency_map=sal_map)
                loss = loss_out.total / config.training.gradient_accumulation

            # Backward
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_count += 1
            if accum_count >= config.training.gradient_accumulation:
                # Step optimizer
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                accum_count = 0

                # Update LR for next step
                current_lr = schedule_fn(
                    step,
                    peak_lr=config.training.lr,
                    warmup_steps=config.training.warmup_steps,
                    max_steps=max_steps,
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr

                # Log
                if step % config.training.log_every == 0:
                    elapsed = time.time() - last_log_time
                    last_log_time = time.time()
                    sys_stats = get_system_stats()

                    # ETA calculation
                    elapsed_total = time.time() - training_start_time
                    steps_done = max(step, 1)
                    secs_per_step = elapsed_total / steps_done
                    steps_left = max_steps - step
                    eta_secs = int(secs_per_step * steps_left)
                    eta_h, eta_m = divmod(eta_secs // 60, 60)
                    eta_str = f"{eta_h}h {eta_m:02d}m"
                    pct_done = step / max_steps * 100

                    log = {
                        "step": step,
                        "lr": current_lr,
                        "grad_norm": float(grad_norm),
                        "elapsed_s": elapsed,
                        "eta_secs": eta_secs,
                        "pct_done": round(pct_done, 2),
                        **loss_out.components,
                        **sys_stats,
                    }
                    log_lines.append(log)
                    gpu_line = (
                        f"gpu={sys_stats.get('gpu_util_pct', '?')}% "
                        f"mem={sys_stats.get('gpu_mem_used_mb', '?')}/"
                        f"{sys_stats.get('gpu_mem_total_mb', '?')}MB "
                        f"temp={sys_stats.get('gpu_temp_c', '?')}°C "
                        f"cpu={sys_stats.get('cpu_pct', '?')}% "
                        f"ram={sys_stats.get('ram_used_gb', '?')}/"
                        f"{sys_stats.get('ram_total_gb', '?')}GB"
                    )
                    line = (
                        f"[train] step={step:6d}/{max_steps} ({pct_done:.1f}%) "
                        f"ETA={eta_str} "
                        f"lr={current_lr:.2e} "
                        f"loss={log['loss/total']:.4f} "
                        f"primary={log['loss/primary']:.4f} "
                        f"hct={log['loss/hct']:.4f} "
                        f"sal={log['loss/saliency']:.4f} "
                        f"grad_norm={float(grad_norm):.2e} "
                        f"| {gpu_line} "
                        f"({elapsed:.1f}s)"
                    )
                    print(line)
                    if log_callback is not None:
                        log_callback(log)

                # Periodic checkpoint
                if step > 0 and step % config.training.save_every == 0:
                    ckpt_dir = output_dir / f"checkpoint-{step}"
                    model.save_pretrained(ckpt_dir)
                    _save_training_state(ckpt_dir, step, optimizer, scaler)
                    current_loss = loss_out.total.item()
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_step = step
                        print(f"[train] New best checkpoint at step {step} (loss={best_loss:.4f})")
                    print(f"[train] Saved checkpoint to {ckpt_dir}")
                    _rotate_checkpoints(output_dir, best_step=best_step)

                step += 1

        # Final save
        final_dir = output_dir / "final"
        model.save_pretrained(final_dir)
        print(f"[train] Final model saved to {final_dir}")

    except KeyboardInterrupt:
        reason = "Training stopped by user (KeyboardInterrupt / SIGINT)"
        print(f"\n[train] {reason}")
        _write_error_log(error_log, reason, log_lines, step)
        _flush_log(output_dir, log_lines)
        raise

    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc()
        print(f"\n[train] FATAL ERROR — {reason}")
        print(tb)
        _write_error_log(error_log, reason, log_lines, step, tb)
        _flush_log(output_dir, log_lines)
        raise

    else:
        # Clean exit — remove any stale error log
        if error_log.exists():
            error_log.unlink()

    # Write the log JSONL
    _flush_log(output_dir, log_lines)

    return {
        "steps_completed": step,
        "final_loss": log_lines[-1]["loss/total"] if log_lines else None,
        "final_dir": str(final_dir),
    }


def _write_error_log(
    path: Path,
    reason: str,
    log_lines: list[dict],
    step: int,
    tb: str = "",
) -> None:
    with path.open("w") as f:
        f.write(f"=== Training stopped at step {step} ===\n")
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Reason: {reason}\n")
        if tb:
            f.write(f"\nTraceback:\n{tb}\n")
        if log_lines:
            last = log_lines[-1]
            f.write(f"\nLast logged step: {last.get('step')}\n")
            f.write(f"Last loss: {last.get('loss/total')}\n")
            f.write(f"Last lr: {last.get('lr')}\n")


def _flush_log(output_dir: Path, log_lines: list[dict]) -> None:
    with (output_dir / "training_log.jsonl").open("w") as f:
        for entry in log_lines:
            f.write(json.dumps(entry) + "\n")


# --------- CLI ---------


def main() -> None:
    parser = argparse.ArgumentParser(description="Keural Vision Encoder training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default="./outputs/run", help="Output directory")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic dataset for smoke-testing the training loop",
    )
    parser.add_argument(
        "--synthetic-size", type=int, default=1000, help="Size of synthetic dataset"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override config.training.max_steps. Useful for smoke tests.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit dataset to first N samples per source. Speeds up smoke tests with real data.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override config.training.batch_size_per_device. Useful for smoke tests.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Override config.training.gradient_accumulation. Useful for smoke tests.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["bfloat16", "float16", "float32"],
        help="Override config.training.precision.",
    )
    args = parser.parse_args()

    config = KeuralConfig.load_yaml(args.config)
    # Apply overrides
    if args.batch_size is not None:
        config.training.batch_size_per_device = args.batch_size
    if args.grad_accum is not None:
        config.training.gradient_accumulation = args.grad_accum
    if args.precision is not None:
        config.training.precision = args.precision
    summary = train(
        config=config,
        output_dir=args.output_dir,
        synthetic=args.synthetic,
        synthetic_size=args.synthetic_size,
        max_steps_override=args.max_steps,
        max_samples=args.max_samples,
    )
    print("[train] Summary:", summary)


if __name__ == "__main__":
    main()
