"""
Filter index TSV files using quality criteria comparable to LAION/DataComp.

Applies four filters in two passes:

Pass 1 — TSV-only (fast, no image reading):
  1. Min caption words >= 5
  2. Min image size 200x200
  3. Max aspect ratio 3.0
  4. English-only captions (langdetect)

Pass 2 — CLIP score (reads images from tar shards, GPU-accelerated):
  5. CLIP cosine similarity >= clip_threshold (default 0.25)

Usage:
    python scripts/filter_index.py \\
        --index data/indexes/cc3m_wds_train.tsv \\
        --images-root data/cc3m \\
        --output data/indexes/cc3m_wds_train_filtered.tsv

    # Both datasets in one run:
    python scripts/filter_index.py \\
        --index data/indexes/cc3m_wds_train.tsv data/indexes/cc12m_wds_train.tsv \\
        --images-root data/cc3m data/cc12m \\
        --output data/indexes/cc3m_wds_train_filtered.tsv data/indexes/cc12m_wds_train_filtered.tsv

    # Skip CLIP scoring (fast, use only cheap filters):
    python scripts/filter_index.py --index ... --images-root ... --output ... --no-clip
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pass 1: cheap TSV-only filters
# ---------------------------------------------------------------------------

def _is_english(text: str) -> bool:
    try:
        from langdetect import detect, LangDetectException
        return detect(text) == "en"
    except Exception:
        return True  # keep on failure


def pass1_filter(
    rows: list[tuple],
    min_words: int,
    min_size: int,
    max_aspect: float,
    lang_filter: bool,
) -> tuple[list[tuple], dict]:
    kept = []
    stats = defaultdict(int)
    stats["total"] = len(rows)

    for row in rows:
        caption, shard, sid, w, h = row
        stats["seen"] += 1

        if len(caption.split()) < min_words:
            stats["drop_caption_words"] += 1
            continue

        if w < min_size or h < min_size:
            stats["drop_size"] += 1
            continue

        ratio = max(w, h) / min(w, h)
        if ratio > max_aspect:
            stats["drop_aspect"] += 1
            continue

        if lang_filter and not _is_english(caption):
            stats["drop_language"] += 1
            continue

        kept.append(row)
        stats["kept_pass1"] += 1

    return kept, dict(stats)


# ---------------------------------------------------------------------------
# Pass 2: CLIP score filter
# ---------------------------------------------------------------------------

def _load_clip(model_name: str, device: torch.device):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    return model, preprocess, tokenizer


def _open_tar_cached(cache: dict, shard_path: Path) -> tarfile.TarFile | None:
    key = str(shard_path)
    if key not in cache:
        try:
            cache[key] = tarfile.open(shard_path, "r")
        except Exception:
            cache[key] = None
    return cache[key]


def _load_image_from_tar(tf: tarfile.TarFile, sample_id: str) -> Image.Image | None:
    for ext in ("jpg", "jpeg", "png", "webp"):
        try:
            member = tf.getmember(f"{sample_id}.{ext}")
            raw = tf.extractfile(member)
            if raw is None:
                continue
            return Image.open(io.BytesIO(raw.read())).convert("RGB")
        except KeyError:
            continue
        except Exception:
            continue
    return None


@torch.no_grad()
def pass2_clip_filter(
    rows: list[tuple],
    images_root: Path,
    clip_threshold: float,
    model_name: str,
    batch_size: int,
    device: torch.device,
) -> tuple[list[tuple], dict]:
    import open_clip

    log.info("Loading CLIP model '%s'...", model_name)
    model, preprocess, tokenizer = _load_clip(model_name, device)

    kept = []
    stats = defaultdict(int)
    tar_cache: dict[str, tarfile.TarFile | None] = {}

    total = len(rows)
    start = time.time()

    for batch_start in range(0, total, batch_size):
        batch = rows[batch_start: batch_start + batch_size]

        images, valid_rows, skipped = [], [], []
        for row in batch:
            caption, shard, sid, w, h = row
            shard_path = images_root / shard
            tf = _open_tar_cached(tar_cache, shard_path)
            if tf is None:
                skipped.append(row)
                stats["drop_missing_shard"] += 1
                continue
            img = _load_image_from_tar(tf, sid)
            if img is None:
                stats["drop_corrupt_image"] += 1
                continue
            images.append(preprocess(img))
            valid_rows.append(row)

        if not images:
            continue

        img_tensor = torch.stack(images).to(device)
        captions_list = [r[0] for r in valid_rows]
        tokens = tokenizer(captions_list).to(device)

        img_feats = model.encode_image(img_tensor)
        txt_feats = model.encode_text(tokens)

        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
        scores = (img_feats * txt_feats).sum(dim=-1)  # (B,)

        for row, score in zip(valid_rows, scores.cpu().tolist()):
            stats["scored"] += 1
            if score >= clip_threshold:
                kept.append(row)
                stats["kept_pass2"] += 1
            else:
                stats["drop_clip_score"] += 1

        done = batch_start + len(batch)
        elapsed = time.time() - start
        rate = done / elapsed
        eta = (total - done) / rate if rate > 0 else 0
        log.info(
            "CLIP scoring: %d/%d  kept=%d  score_drop=%d  eta=%.0fs",
            done, total, stats["kept_pass2"], stats["drop_clip_score"], eta,
        )

    for tf in tar_cache.values():
        if tf is not None:
            try:
                tf.close()
            except Exception:
                pass

    return kept, dict(stats)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def filter_one(
    index_path: Path,
    images_root: Path,
    output_path: Path,
    *,
    min_words: int,
    min_size: int,
    max_aspect: float,
    lang_filter: bool,
    clip_threshold: float | None,
    clip_model: str,
    clip_batch_size: int,
    device: torch.device,
) -> None:
    log.info("=== Filtering %s ===", index_path.name)

    # Read index
    rows = []
    with index_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue
            caption, shard, sid, w_str, h_str = row[:5]
            try:
                w, h = int(w_str), int(h_str)
            except ValueError:
                continue
            rows.append((caption, shard, sid, w, h))

    log.info("Loaded %d rows from index", len(rows))

    # Pass 1
    log.info("Pass 1: cheap filters (min_words=%d, min_size=%d, max_aspect=%.1f, lang=%s)",
             min_words, min_size, max_aspect, lang_filter)
    rows, stats1 = pass1_filter(rows, min_words, min_size, max_aspect, lang_filter)
    log.info("Pass 1 done: kept %d / %d  stats=%s",
             stats1.get("kept_pass1", 0), stats1.get("total", 0), stats1)

    # Pass 2 (optional)
    stats2 = {}
    if clip_threshold is not None and len(rows) > 0:
        log.info("Pass 2: CLIP score >= %.2f  (model=%s, batch=%d, device=%s)",
                 clip_threshold, clip_model, clip_batch_size, device)
        rows, stats2 = pass2_clip_filter(
            rows, images_root, clip_threshold, clip_model, clip_batch_size, device
        )
        log.info("Pass 2 done: kept %d  stats=%s", len(rows), stats2)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        for row in rows:
            writer.writerow(row)

    log.info("Written %d samples → %s", len(rows), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter dataset index TSVs")
    parser.add_argument("--index", nargs="+", required=True,
                        help="Input index TSV file(s)")
    parser.add_argument("--images-root", nargs="+", required=True,
                        help="Image shard root directory for each index (same order)")
    parser.add_argument("--output", nargs="+", required=True,
                        help="Output filtered TSV file(s) (same order as --index)")
    parser.add_argument("--min-words", type=int, default=5,
                        help="Minimum caption word count (default: 5)")
    parser.add_argument("--min-size", type=int, default=200,
                        help="Minimum image dimension in px (default: 200)")
    parser.add_argument("--max-aspect", type=float, default=3.0,
                        help="Maximum aspect ratio long/short (default: 3.0)")
    parser.add_argument("--no-lang-filter", action="store_true",
                        help="Skip English-only language filter")
    parser.add_argument("--clip-threshold", type=float, default=0.25,
                        help="CLIP cosine similarity threshold (default: 0.25). "
                             "Use --no-clip to skip CLIP scoring entirely.")
    parser.add_argument("--no-clip", action="store_true",
                        help="Skip CLIP score filtering (fast mode)")
    parser.add_argument("--clip-model", type=str, default="ViT-B-32",
                        help="open_clip model name (default: ViT-B-32)")
    parser.add_argument("--clip-batch-size", type=int, default=256,
                        help="Batch size for CLIP scoring (default: 256)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for CLIP (default: cuda if available)")
    args = parser.parse_args()

    if len(args.index) != len(args.images_root):
        parser.error("--index and --images-root must have the same number of entries")
    if len(args.index) != len(args.output):
        parser.error("--index and --output must have the same number of entries")

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("Device: %s", device)

    clip_threshold = None if args.no_clip else args.clip_threshold

    for idx_path, img_root, out_path in zip(args.index, args.images_root, args.output):
        filter_one(
            index_path=Path(idx_path),
            images_root=Path(img_root),
            output_path=Path(out_path),
            min_words=args.min_words,
            min_size=args.min_size,
            max_aspect=args.max_aspect,
            lang_filter=not args.no_lang_filter,
            clip_threshold=clip_threshold,
            clip_model=args.clip_model,
            clip_batch_size=args.clip_batch_size,
            device=device,
        )

    log.info("All done.")


if __name__ == "__main__":
    main()
