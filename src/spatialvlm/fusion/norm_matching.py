"""RMS norm matching: scale vision tokens to match text token magnitude.

Background:
    In LLaVA-style models, vision token L₂ norms are 10–100× larger than text token
    norms. This magnitude imbalance suppresses RoPE positional information and causes
    the LLM to treat visual inputs as noise. The "Beyond Semantics" paper (2024)
    identifies this as a primary source of spatial reasoning failure.

    This module corrects the imbalance by maintaining a running EMA of the text token
    RMS norm and scaling vision tokens to match it. Zero learnable parameters —
    scaling factors are derived from the data distribution.

Hypothesis H3b: this module adds measurable spatial task performance beyond gated
    cross-attention alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNormMatching(nn.Module):
    """Scale vision tokens to match text token RMS norm magnitude.

    Tracks an exponential moving average (EMA) of text token RMS norms across
    training batches. At inference, uses the stored EMA (no text tokens required).

    Zero learnable parameters — state is in register_buffer.

    Parameters
    ----------
    ema_momentum : float
        EMA decay factor. 0.99 = slow-moving average (stable against batch noise).
    eps : float
        Numerical stability epsilon to avoid division by zero.
    """

    def __init__(self, ema_momentum: float = 0.99, eps: float = 1e-6) -> None:
        super().__init__()
        self.ema_momentum = ema_momentum
        self.eps = eps
        # Initialised to 1.0: no-op scaling until first text batch is seen
        self.register_buffer("text_rms_ema", torch.ones(1))

    def forward(
        self,
        vision_tokens: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Scale vision tokens to text-token magnitude.

        Parameters
        ----------
        vision_tokens : Tensor[B, N, D]
            Vision tokens to rescale (output of SVA or projector).
        text_tokens : Tensor[B, T, D] | None
            Text tokens used to update the EMA during training.
            If None (inference), uses the stored EMA value unchanged.

        Returns
        -------
        scaled_vision : Tensor[B, N, D]
            Vision tokens scaled to match text RMS norm.
            Same dtype as input; computation done in float32 for stability.
        """
        original_dtype = vision_tokens.dtype

        if text_tokens is not None and self.training:
            # RMS norm per text token: sqrt(mean(x², dim=-1)) → [B, T]
            text_rms_per_token = text_tokens.float().pow(2).mean(dim=-1).sqrt()  # [B, T]
            # Mean RMS over the batch
            batch_text_rms = text_rms_per_token.mean()  # scalar

            # EMA update (in-place to preserve registered buffer for state_dict/device)
            self.text_rms_ema.mul_(self.ema_momentum).add_(
                (1.0 - self.ema_momentum) * batch_text_rms.detach()
            )

        # RMS of the current vision tokens (mean over all tokens and batch)
        vision_rms = vision_tokens.float().pow(2).mean(dim=-1).sqrt().mean()  # scalar

        # Scale factor: text magnitude / vision magnitude
        scale = self.text_rms_ema / (vision_rms + self.eps)

        return (vision_tokens.float() * scale).to(original_dtype)
