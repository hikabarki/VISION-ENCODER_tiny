"""
Keural Vision Encoder — Configuration system.

Every model dimension, ratio, and hyperparameter lives here. No magic
numbers in the model code. Configs round-trip through YAML for full
reproducibility.

Design principle
----------------
Each config dataclass corresponds to a logical sub-system of the model:
  - StemConfig: the conv stem
  - SaliencyConfig: the saliency predictor
  - TokenizerConfig: the ATB tokenizer (allocator + sampling + token construction)
  - SpatialConfig: the transformer encoder over tokens
  - LossConfig: the loss formulation
  - TrainingConfig: optimizer, scheduler, batch size, etc.
  - DataConfig: dataset paths and preprocessing

KeuralConfig is the root container. It is the only object the model needs.

Reference: docs/03_architecture_design.md §8 and docs/04_revisions.md §11.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ----- Sub-configs -----


@dataclass
class StemConfig:
    """Configuration for the variable-resolution conv stem.

    The stem reduces input spatial size by 16x via 4 strided conv layers.
    Output channels at the final layer must equal the model's embed_dim.

    Attributes
    ----------
    channels
        Channel counts at each stem layer. Length must be 4. Final value
        must equal SpatialConfig.embed_dim. Stride is fixed at 2 per layer.
    kernel_size
        Conv kernel size for all stem layers. Default 3.
    activation
        Activation between layers. One of {"gelu", "relu", "silu"}.
    """

    channels: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    kernel_size: int = 3
    activation: str = "gelu"

    def __post_init__(self) -> None:
        if len(self.channels) != 4:
            raise ValueError(
                f"StemConfig.channels must have length 4 (got {len(self.channels)}). "
                f"The stem is fixed at 4 strided layers."
            )
        if self.activation not in {"gelu", "relu", "silu"}:
            raise ValueError(
                f"StemConfig.activation must be one of gelu/relu/silu, got {self.activation}"
            )


@dataclass
class SaliencyConfig:
    """Configuration for the saliency predictor head.

    Two small conv layers on top of stem features producing a (B, 1, h, w)
    sigmoid-activated map.

    Attributes
    ----------
    hidden_channels
        Hidden dim of the first 3x3 conv in the saliency head.
    """

    hidden_channels: int = 128


@dataclass
class TokenizerConfig:
    """Configuration for the ATB tokenizer (allocator + sampler + constructor).

    Attributes
    ----------
    default_token_budget
        Default number of tokens to allocate when forward() is called
        without an explicit budget. See docs/04_revisions.md §3 (Revision A).
    max_token_budget
        Hard upper bound on token budget. Used for output buffer allocation.
    budget_split
        Default ratios for {global, region, detail}. Must sum to 1.0.
    budget_floors
        Minimum tokens per level regardless of budget. Prevents degenerate
        allocations at small budgets.
    saliency_smooth_kernel
        Gaussian smoothing kernel size per level. Larger for region (broad
        receptive fields), smaller for detail.
    nms_radius
        Spatial radius (in stem-grid cells) over which selecting one region
        suppresses nearby saliency for subsequent picks within the same level.
    gumbel_temperature_start
        Starting temperature for Gumbel-noise sampling. High temperature
        means more random exploration.
    gumbel_temperature_end
        Final annealed Gumbel temperature.
    temperature_anneal_steps
        Number of training steps over which temperature anneals from start
        to end (after warmup).
    warmup_steps_fixed_grid
        Number of initial training steps using fixed-grid tokenization. The
        allocator is disabled during warmup to give the spatial encoder
        stable gradients.
    roi_align_size
        RoIAlign output size for region/detail tokens (e.g., 2 means 2x2
        pooled features per region).
    """

    default_token_budget: int = 256
    max_token_budget: int = 1024
    budget_split: dict[str, float] = field(
        default_factory=lambda: {"global": 0.05, "region": 0.25, "detail": 0.70}
    )
    budget_floors: dict[str, int] = field(
        default_factory=lambda: {"global": 4, "region": 8, "detail": 0}
    )
    saliency_smooth_kernel: dict[str, int] = field(
        default_factory=lambda: {"region": 5, "detail": 3}
    )
    nms_radius: dict[str, int] = field(
        default_factory=lambda: {"region": 2, "detail": 1}
    )
    gumbel_temperature_start: float = 1.0
    gumbel_temperature_end: float = 0.1
    temperature_anneal_steps: int = 10000
    warmup_steps_fixed_grid: int = 5000
    roi_align_size: int = 2

    def __post_init__(self) -> None:
        # Budget split must sum to ~1.0
        total = sum(self.budget_split.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"TokenizerConfig.budget_split must sum to 1.0 (got {total:.4f}). "
                f"split={self.budget_split}"
            )
        # Validate keys
        required = {"global", "region", "detail"}
        if set(self.budget_split.keys()) != required:
            raise ValueError(
                f"TokenizerConfig.budget_split must have exactly keys "
                f"{required}, got {set(self.budget_split.keys())}"
            )
        if set(self.budget_floors.keys()) != required:
            raise ValueError(
                f"TokenizerConfig.budget_floors must have exactly keys "
                f"{required}, got {set(self.budget_floors.keys())}"
            )
        if self.default_token_budget > self.max_token_budget:
            raise ValueError(
                f"default_token_budget ({self.default_token_budget}) > "
                f"max_token_budget ({self.max_token_budget})"
            )
        if self.gumbel_temperature_end >= self.gumbel_temperature_start:
            raise ValueError(
                f"Gumbel temperature must anneal *down* "
                f"(start {self.gumbel_temperature_start} -> end {self.gumbel_temperature_end})"
            )


@dataclass
class SpatialConfig:
    """Configuration for the spatial transformer encoder over tokens.

    Attributes
    ----------
    embed_dim
        Token embedding dim D. Fixed across the model. Must equal the last
        value of StemConfig.channels.
    depth
        Number of transformer blocks.
    num_heads
        Number of attention heads. Must divide embed_dim evenly.
    head_dim
        Per-head dim. embed_dim must equal num_heads * head_dim.
    mlp_ratio
        SwiGLU expansion ratio. Hidden dim = mlp_ratio * embed_dim.
    norm_eps
        Epsilon for RMSNorm.
    rope_base
        RoPE frequency base. Standard value is 10000.
    use_pool_token
        Whether to prepend a learnable [POOL] token. Required for
        single-vector outputs (KeuralEncoderOutput.pooled).
    """

    embed_dim: int = 512
    depth: int = 12
    num_heads: int = 8
    head_dim: int = 64
    mlp_ratio: int = 4
    norm_eps: float = 1e-6
    rope_base: float = 10000.0
    use_pool_token: bool = True

    def __post_init__(self) -> None:
        if self.num_heads * self.head_dim != self.embed_dim:
            raise ValueError(
                f"SpatialConfig: num_heads ({self.num_heads}) * head_dim ({self.head_dim}) "
                f"= {self.num_heads * self.head_dim}, must equal embed_dim ({self.embed_dim})"
            )
        if self.depth < 1:
            raise ValueError(f"SpatialConfig.depth must be >= 1, got {self.depth}")


@dataclass
class LossConfig:
    """Configuration for the training loss.

    Attributes
    ----------
    primary
        Primary contrastive loss formulation. {"sigmoid_siglip", "infonce_clip"}.
    hct_alpha
        Weight on the masked-detail-prediction component within the HCT
        auxiliary loss.
    saliency_beta
        Weight on the saliency-variance regularizer within itself.
    lambda_hct
        Weight of L_HCT in the total loss.
    lambda_sal
        Weight of L_saliency in the total loss.
    learnable_temperature_init
        Initial value (in log space) of the learned temperature for the
        contrastive loss.
    learnable_bias_init
        Initial value of the learned bias for the sigmoid contrastive loss.
    """

    primary: str = "sigmoid_siglip"
    hct_alpha: float = 0.5
    saliency_beta: float = 0.01
    lambda_hct: float = 1.0
    lambda_sal: float = 1.0
    learnable_temperature_init: float = 2.30  # log(10)
    learnable_bias_init: float = -10.0

    def __post_init__(self) -> None:
        if self.primary not in {"sigmoid_siglip", "infonce_clip"}:
            raise ValueError(
                f"LossConfig.primary must be sigmoid_siglip or infonce_clip, "
                f"got {self.primary}"
            )


@dataclass
class TrainingConfig:
    """Configuration for the training loop.

    Attributes
    ----------
    precision
        Mixed precision mode. {"bfloat16", "float16", "float32"}.
    gradient_checkpointing
        Whether to use gradient checkpointing in transformer blocks. Trades
        compute for memory.
    optimizer
        Optimizer name. {"adamw"}. (Only AdamW supported in V1.)
    betas
        AdamW betas.
    weight_decay
        AdamW weight decay.
    lr
        Peak learning rate (after warmup, before cosine decay).
    lr_schedule
        Schedule name. {"cosine_with_warmup", "constant", "linear_with_warmup"}.
    warmup_steps
        Linear warmup steps from 0 to peak lr.
    max_steps
        Total training steps.
    batch_size_per_device
        Micro-batch size per GPU.
    gradient_accumulation
        Gradient accumulation steps. Effective batch = per-device * accum * world_size.
    log_every
        Step interval for loss/metrics logging.
    save_every
        Step interval for checkpoint saving.
    seed
        Random seed for reproducibility.
    """

    precision: str = "bfloat16"
    gradient_checkpointing: bool = True
    optimizer: str = "adamw"
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.05
    lr: float = 1.0e-3
    lr_schedule: str = "cosine_with_warmup"
    warmup_steps: int = 2000
    max_steps: int = 75000
    batch_size_per_device: int = 256
    gradient_accumulation: int = 4
    log_every: int = 50
    save_every: int = 5000
    seed: int = 42

    def __post_init__(self) -> None:
        if self.precision not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"Invalid precision: {self.precision}")
        if self.optimizer != "adamw":
            raise ValueError(f"V1 only supports adamw, got {self.optimizer}")
        if self.lr_schedule not in {
            "cosine_with_warmup",
            "constant",
            "linear_with_warmup",
        }:
            raise ValueError(f"Invalid lr_schedule: {self.lr_schedule}")
        if len(self.betas) != 2:
            raise ValueError(f"betas must have length 2, got {self.betas}")


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing.

    Attributes
    ----------
    dataset_name
        Name identifier. {"cc3m", "laion400m_subset", "synthetic"}.
    images_root
        Path to the root directory of downloaded images.
    captions_path
        Path to the captions file (TSV or JSONL).
    image_buckets
        List of (H, W) bucket sizes. Each batch uses exactly one bucket.
        Per docs/04_revisions.md §8, this is how variable resolution is
        handled in single-GPU POC training.
    pixel_mean, pixel_std
        Normalization statistics. Standard ImageNet values by default.
    max_caption_length
        Token cap for the text tokenizer.
    num_workers
        DataLoader workers.
    text_encoder_name
        Frozen text encoder used for the contrastive loss signal during
        the POC. Per docs/02_release_plan.md §5.3 this is the only
        pretrained-weights compromise in the POC.
    """

    dataset_name: str = "cc3m"
    images_root: str = "data/cc3m/images"
    captions_path: str = "data/cc3m/captions.tsv"
    image_buckets: list[list[int]] = field(
        default_factory=lambda: [[224, 224], [256, 256], [320, 320]]
    )
    pixel_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    pixel_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    max_caption_length: int = 64
    num_workers: int = 4
    text_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    extra_sources: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        for b in self.image_buckets:
            if len(b) != 2:
                raise ValueError(f"Each image_bucket must be [H, W], got {b}")
            if b[0] % 16 != 0 or b[1] % 16 != 0:
                raise ValueError(
                    f"Image bucket {b} must be divisible by 16 (the stem stride)."
                )


# ----- Root config -----


@dataclass
class KeuralConfig:
    """Top-level Keural Vision Encoder configuration.

    This is the only object the encoder needs. All sub-configs are validated
    on construction (via __post_init__ chains). Cross-config invariants
    (like stem channels matching embed_dim) are validated here.

    Attributes
    ----------
    name
        A human-readable identifier. Used for run naming and HF model card.
    stem
        StemConfig.
    saliency
        SaliencyConfig.
    tokenizer
        TokenizerConfig.
    spatial
        SpatialConfig.
    loss
        LossConfig.
    training
        TrainingConfig.
    data
        DataConfig.

    See Also
    --------
    docs/03_architecture_design.md §8 — initial values for all knobs.
    """

    name: str = "keural-tiny-poc"
    stem: StemConfig = field(default_factory=StemConfig)
    saliency: SaliencyConfig = field(default_factory=SaliencyConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self) -> None:
        # Cross-config invariant: stem's last channel count == embed_dim.
        # Without this, the projection from stem features to tokens would
        # need an extra layer and the design doc's "stem outputs match
        # token dim" property would not hold.
        if self.stem.channels[-1] != self.spatial.embed_dim:
            raise ValueError(
                f"StemConfig.channels[-1] ({self.stem.channels[-1]}) must equal "
                f"SpatialConfig.embed_dim ({self.spatial.embed_dim}). "
                f"Update one to match the other."
            )

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for YAML or JSON dumping."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KeuralConfig:
        """Construct from a plain dict (e.g., from YAML).

        Each sub-config is re-built from its sub-dict, so __post_init__
        validations run.
        """
        # Allow partial dicts — fall back to defaults for missing sub-configs.
        kwargs: dict[str, Any] = {}
        if "name" in data:
            kwargs["name"] = data["name"]
        if "stem" in data:
            kwargs["stem"] = StemConfig(**data["stem"])
        if "saliency" in data:
            kwargs["saliency"] = SaliencyConfig(**data["saliency"])
        if "tokenizer" in data:
            kwargs["tokenizer"] = TokenizerConfig(**data["tokenizer"])
        if "spatial" in data:
            kwargs["spatial"] = SpatialConfig(**data["spatial"])
        if "loss" in data:
            kwargs["loss"] = LossConfig(**data["loss"])
        if "training" in data:
            kwargs["training"] = TrainingConfig(**data["training"])
        if "data" in data:
            d = dict(data["data"])
            d.setdefault("extra_sources", [])
            kwargs["data"] = DataConfig(**d)
        return cls(**kwargs)

    def save_yaml(self, path: str | Path) -> None:
        """Write to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)

    @classmethod
    def load_yaml(cls, path: str | Path) -> KeuralConfig:
        """Read from a YAML file."""
        with Path(path).open("r") as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        return cls.from_dict(data)
