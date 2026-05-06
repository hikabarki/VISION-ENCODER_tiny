"""
Keural Vision Encoder — Training Losses.

Three losses combined into the total training objective:

  L_total = L_primary + lambda_hct * L_HCT + lambda_sal * L_saliency

1. **L_primary**: SigLIP-style sigmoid contrastive loss between image
   pooled embeddings and text embeddings. The single-vector retrieval signal.

2. **L_HCT (auxiliary)**: Two components forcing hierarchy specialization:
   - Global-text: sum-pool global tokens, contrast against caption embedding.
   - Detail-patch: masked-prediction on detail tokens (reconstruction).

3. **L_saliency (regularizer)**: Penalizes flat saliency maps to prevent
   the saliency predictor from collapsing to a constant.

Reference: docs/03_architecture_design.md §7.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from alignment.output_contract import LEVEL_GLOBAL, KeuralEncoderOutput
from keural_config import LossConfig


@dataclass
class LossOutput:
    """Bundle of loss components for logging.

    Attributes
    ----------
    total : Tensor (scalar)
        The total weighted loss. This is what `.backward()` is called on.
    primary : Tensor (scalar)
        L_primary value (unscaled — for logging).
    hct : Tensor (scalar)
        L_HCT value (unscaled — for logging).
    saliency : Tensor (scalar)
        L_saliency value (unscaled — for logging).
    components : dict[str, float]
        Convenient dict of scalars for logging frameworks that prefer dicts.
    """

    total: torch.Tensor
    primary: torch.Tensor
    hct: torch.Tensor
    saliency: torch.Tensor
    components: dict[str, float]


def sigmoid_siglip_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
) -> torch.Tensor:
    """SigLIP-style sigmoid contrastive loss.

    For each (image_i, text_j) pair, the target is +1 if i==j, -1 otherwise.
    The loss is sum over all pairs of -log sigmoid(target * (logit_scale * sim(i,j) + logit_bias)).

    Stable at small batches because there's no in-batch softmax that scales
    with batch size. This is why we chose it over InfoNCE for single-GPU
    training.

    Parameters
    ----------
    image_embeds : Tensor of shape (B, D)
        L2-normalized image embeddings.
    text_embeds : Tensor of shape (B, D)
        L2-normalized text embeddings.
    logit_scale : Tensor (scalar). Learnable. Typically initialized to log(10).
        We use exp(logit_scale) as the temperature.
    logit_bias : Tensor (scalar). Learnable. Typically initialized to -10.

    Returns
    -------
    Scalar loss.
    """
    B = image_embeds.shape[0]
    device = image_embeds.device

    # Pairwise similarity (B, B)
    similarity = image_embeds @ text_embeds.T

    # Logits = scale * similarity + bias
    logits = similarity * logit_scale.exp() + logit_bias

    # Targets: +1 on the diagonal, -1 elsewhere
    targets = -torch.ones(B, B, device=device, dtype=logits.dtype)
    targets.fill_diagonal_(1.0)

    # Sigmoid loss: -log sigmoid(target * logits)
    # Equivalent to F.softplus(-target * logits)
    loss = F.softplus(-targets * logits)
    return loss.sum() / B


def hct_global_text_loss(
    encoder_output: KeuralEncoderOutput,
    text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
) -> torch.Tensor:
    """Global-text contrastive loss component of HCT.

    Sum-pool the GLOBAL-level tokens of each batch item, project them by
    L2-norm, and contrast against the text embeddings using the same
    sigmoid loss as L_primary. This forces global tokens to capture
    image-level semantics rather than drifting toward whatever the
    primary loss happens to optimize on the pooled token alone.

    Returns
    -------
    Scalar loss.
    """
    # Build per-batch sum of global tokens. Get the level mask first.
    global_mask = (encoder_output.level_ids == LEVEL_GLOBAL) & encoder_output.attention_mask.bool()
    # Mask tokens then sum over sequence
    masked_tokens = encoder_output.tokens * global_mask.unsqueeze(-1).to(encoder_output.tokens.dtype)
    global_sum = masked_tokens.sum(dim=1)  # (B, D)
    # Normalize by count to avoid scale-dependence on K_g
    counts = global_mask.sum(dim=1, keepdim=True).clamp(min=1).to(global_sum.dtype)
    global_pooled = global_sum / counts

    # L2 normalize for the contrastive loss
    image_embeds = F.normalize(global_pooled, dim=-1)
    text_normed = F.normalize(text_embeds, dim=-1)
    return sigmoid_siglip_loss(image_embeds, text_normed, logit_scale, logit_bias)


def saliency_variance_regularizer(saliency_map: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Penalize flat saliency maps.

    Computes spatial variance of the saliency map per batch item, then
    penalizes the inverse: low variance → high loss.

    L = beta * mean_b 1 / (var_b(saliency) + eps)

    With beta applied externally, this returns just the inverse-variance term.

    Parameters
    ----------
    saliency_map : Tensor of shape (B, 1, h, w)
    eps : float — added to the variance to bound the loss when variance is low.

    Returns
    -------
    Scalar loss.
    """
    B = saliency_map.shape[0]
    flat = saliency_map.flatten(1)  # (B, h*w)
    variance = flat.var(dim=1, unbiased=False)  # (B,)
    return (1.0 / (variance + eps)).mean()


class KeuralLoss(nn.Module):
    """The combined Keural training loss.

    Owns the learnable temperature and bias for the contrastive losses.
    Forward returns a LossOutput dataclass with both the total loss
    (for backprop) and per-component values (for logging).

    Parameters
    ----------
    config : LossConfig
    """

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config

        # Learnable temperature and bias for both primary and HCT contrastive losses.
        # Per the architecture spec, we share these across the two contrastive
        # signals — they are functionally the same loss against the same text
        # embeddings, just with different image features.
        self.logit_scale = nn.Parameter(
            torch.tensor(config.learnable_temperature_init, dtype=torch.float32)
        )
        self.logit_bias = nn.Parameter(
            torch.tensor(config.learnable_bias_init, dtype=torch.float32)
        )

    def forward(
        self,
        encoder_output: KeuralEncoderOutput,
        text_embeds: torch.Tensor,
        saliency_map: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute the combined loss.

        Parameters
        ----------
        encoder_output
            Output of a Keural Vision Encoder forward pass.
        text_embeds
            Shape (B, text_dim). One text embedding per batch item.
            Must already match the encoder's pooled vector dim. The data
            pipeline / projection layer handles dim-matching.
        saliency_map
            Shape (B, 1, h, w). The raw saliency map from the tokenizer.
            If None, the saliency regularizer is set to zero.

        Returns
        -------
        LossOutput.
        """
        if encoder_output.pooled.shape[0] != text_embeds.shape[0]:
            raise ValueError(
                f"Batch size mismatch: encoder pooled {encoder_output.pooled.shape[0]} "
                f"vs text {text_embeds.shape[0]}"
            )
        if encoder_output.pooled.shape[-1] != text_embeds.shape[-1]:
            raise ValueError(
                f"Embedding dim mismatch: pooled D={encoder_output.pooled.shape[-1]} "
                f"vs text D={text_embeds.shape[-1]}. Use a projection layer to align them."
            )

        # L2 normalize for contrastive
        image_embeds = F.normalize(encoder_output.pooled, dim=-1)
        text_normed = F.normalize(text_embeds, dim=-1)

        # 1) Primary contrastive (SigLIP-style on pooled vectors)
        primary = sigmoid_siglip_loss(
            image_embeds, text_normed, self.logit_scale, self.logit_bias
        )

        # 2) HCT auxiliary: global-text contrastive on global-token mean.
        # We don't have a self-supervised detail-patch reconstruction in
        # V1 (would require an additional head and target features).
        # The architecture doc described this as alpha-weighted with a
        # patch reconstruction term; we ship V1 with global-text only and
        # document the deferred component. Detail-patch is a clean addition
        # in V0.2.
        hct = hct_global_text_loss(
            encoder_output, text_embeds, self.logit_scale, self.logit_bias
        )

        # 3) Saliency variance regularizer
        if saliency_map is not None:
            saliency = saliency_variance_regularizer(saliency_map)
        else:
            saliency = torch.tensor(0.0, device=primary.device, dtype=primary.dtype)

        # Combine
        total = (
            primary
            + self.config.lambda_hct * hct
            + self.config.lambda_sal * self.config.saliency_beta * saliency
        )

        components = {
            "loss/total": total.detach().item(),
            "loss/primary": primary.detach().item(),
            "loss/hct": hct.detach().item(),
            "loss/saliency": saliency.detach().item(),
            "loss/logit_scale": self.logit_scale.detach().item(),
            "loss/logit_bias": self.logit_bias.detach().item(),
        }

        return LossOutput(
            total=total,
            primary=primary,
            hct=hct,
            saliency=saliency,
            components=components,
        )
