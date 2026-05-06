# Keural Vision Encoder — Dataset & Pipeline Reference

## Datasets

### CC3M (Conceptual Captions 3M)
| Property | Value |
|---|---|
| Location | `poc/data/cc3m/` |
| Format | WebDataset tar shards |
| Shards | 592 (`cc3m-train-0000.tar` → `cc3m-train-0591.tar`) |
| Samples | 2,919,397 |
| Disk size | ~262 GB |
| Index | `poc/data/indexes/cc3m_wds_train.tsv` |

### CC12M (Conceptual Captions 12M)
| Property | Value |
|---|---|
| Location | `poc/data/cc12m/` |
| Format | WebDataset tar shards |
| Shards | 790 (`cc12m-train-0000.tar` → `cc12m-train-0789.tar`) |
| Samples | 3,980,041 |
| Disk size | ~401 GB |
| Index | `poc/data/indexes/cc12m_wds_train.tsv` |
| Note | Partial — full CC12M (2176 shards) is on `/mnt/ssd/data/cc12m/` |

**Combined total: ~6.9M samples**

---

## Shard Format (WebDataset)

Each tar shard contains groups of three files per sample:

```
cc3m-train-0000.tar
├── 000000000.jpg    ← image
├── 000000000.txt    ← caption
├── 000000000.json   ← metadata (width, height, url, etc.)
├── 000000001.jpg
├── 000000001.txt
├── 000000001.json
└── ...
```

---

## Index Format (TSV)

Each index file has **5 tab-separated columns**, no header:

```
caption    shard_filename          sample_id    width    height
A river... cc3m-train-0000.tar    000000000    852      480
```

Indexes are built once by `build_index_from_shards()` in `datasets.py`. They enable true random access and shuffling without streaming the tars sequentially.

---

## Data Pipeline (`data/datasets.py`)

### Classes

**`CC3MDataset`**
- Loads the index TSV at startup
- Random-access reads images from tar shards on demand
- Per-worker LRU cache of open tar file handles (max 16)
- 8-attempt fallback on corrupt images; returns black tensor if all fail
- Assigns each sample to a resolution bucket based on aspect ratio

**`CombinedImageTextDataset`**
- Concatenates multiple `CC3MDataset` instances (CC3M + CC12M)
- Single unified index across all sources

**`BucketedSampler`**
- Groups samples by resolution bucket so all items in a batch share the same spatial size
- Shuffles within each bucket every epoch
- Avoids padding waste from mixed-resolution batches

**`FrozenTextEncoder`**
- Wraps `sentence-transformers/all-MiniLM-L6-v2` (frozen weights)
- Adds a trainable linear projection to the model's embedding dimension
- Uses `.clone()` to exit `torch.inference_mode()` before the projection layer

**`SyntheticPairDataset`**
- Generates random tensors — no data files needed
- Used for smoke-testing the training loop: `--synthetic --max-steps 5`

**`build_collate_fn`**
- Stacks pixel tensors and text embeddings into batched tensors
- Returns a dict with keys: `pixel_values`, `text_embeddings`

**`build_index_from_shards`**
- Scans all `.tar` files in a directory
- Extracts image dimensions via PIL
- Writes the 5-column TSV index

### Image Augmentation

| Split | Pipeline |
|---|---|
| Train | `RandomResizedCrop` → `RandomHorizontalFlip` → `ColorJitter` → `ToTensor` → `Normalize` |
| Eval | `Resize` → `CenterCrop` → `ToTensor` → `Normalize` |

Normalization uses ImageNet stats: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`

### Resolution Buckets

Defined in `configs/keural_tiny_poc.yaml`. Each batch contains images from exactly one bucket:

| Bucket | Resolution |
|---|---|
| Small | 224 × 224 |
| Medium | 256 × 256 |
| Large | 320 × 320 |

---

## Config (`configs/keural_tiny_poc.yaml` — data section)

```yaml
data:
  dataset_name: cc3m+cc12m
  images_root: data/cc3m
  captions_path: data/indexes/cc3m_wds_train.tsv
  extra_sources:
    - images_root: data/cc12m
      captions_path: data/indexes/cc12m_wds_train.tsv
  image_buckets:
    - [224, 224]
    - [256, 256]
    - [320, 320]
  pixel_mean: [0.485, 0.456, 0.406]
  pixel_std: [0.229, 0.224, 0.225]
  max_caption_length: 64
  num_workers: 2
  text_encoder_name: sentence-transformers/all-MiniLM-L6-v2
```

---

## Rebuilding the Index

If you add new shards, rebuild the index:

```python
from data.datasets import build_index_from_shards
from pathlib import Path

build_index_from_shards(
    shards_root=Path("data/cc3m"),
    output_tsv=Path("data/indexes/cc3m_wds_train.tsv"),
)
```

---

## Key Design Decisions

- **Random access over streaming**: Using a pre-built index instead of WebDataset streaming gives proper shuffling, which matters for contrastive training.
- **Tar handle LRU cache**: Each DataLoader worker caches up to 16 open tar handles to avoid reopening files on every sample, while bounding file descriptor usage.
- **Bucketed batching**: Avoids wasting VRAM on padding when images have different aspect ratios.
- **num_workers=2**: Higher worker counts (e.g. 8) cause OOM because each worker pickles the full 6.9M-entry sample index into shared memory.
