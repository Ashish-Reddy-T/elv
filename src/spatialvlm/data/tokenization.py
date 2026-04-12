"""Tokenization utilities: build input_ids with spatial placeholders and labels.

Two formats supported:
  - **prealign**: ``[BOS] <instruction> [SPATIAL×1369] <caption> [EOS]``
  - **sft**:      ``[BOS] <system> <instruction> [SPATIAL×1369] <reasoning+action> [EOS]``

Spatial placeholder tokens are replaced at runtime by the DeepStack embedding
hook in ``backbone/rope_patch.py``.  Their token IDs must occupy exactly
``num_spatial_tokens`` contiguous positions so that ``spatial_start_idx``
points to the first one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

# Default spatial placeholder — we reuse Qwen3-VL's image-pad token so the
# tokenizer already knows it.  Override via ``SpatialTokenizerConfig`` if
# needed (e.g. when using a non-Qwen tokenizer).
DEFAULT_SPATIAL_PLACEHOLDER = "<|image_pad|>"

SYSTEM_PROMPT = (
    "You are a spatial navigation assistant. Given an observation of an "
    "indoor scene and an instruction, reason about the 3D layout and "
    "produce an action."
)


@dataclass
class SpatialTokenizerConfig:
    """Knobs for the tokenization helpers."""

    num_spatial_tokens: int = 1369
    spatial_placeholder: str = DEFAULT_SPATIAL_PLACEHOLDER
    system_prompt: str = SYSTEM_PROMPT
    max_length: int = 2048
    ignore_index: int = -100


def _resolve_placeholder_id(tokenizer: Any, placeholder: str) -> int:
    """Get the token ID for the spatial placeholder, adding it if absent."""
    vocab: dict[str, int] = tokenizer.get_vocab()
    if placeholder in vocab:
        return vocab[placeholder]
    # Fallback: use the pad token
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        return pad_id
    # Last resort: use token ID 0
    return 0


def build_input_ids(
    tokenizer: Any,
    instruction: str,
    target: str | None = None,
    config: SpatialTokenizerConfig | None = None,
) -> dict[str, torch.Tensor | int]:
    """Build input_ids, labels, attention_mask, and spatial_start_idx.

    Parameters
    ----------
    tokenizer
        A HuggingFace-style tokenizer with ``encode`` / ``__call__``.
    instruction : str
        The navigation instruction or question.
    target : str | None
        Target response (for SFT).  When ``None`` the labels are all
        ``ignore_index`` (pre-alignment captioning loss is applied
        to the full sequence externally).
    config : SpatialTokenizerConfig | None
        Override defaults.

    Returns
    -------
    dict with keys:
        input_ids          Tensor[seq_len]   (long)
        labels             Tensor[seq_len]   (long, -100 on non-target)
        attention_mask     Tensor[seq_len]   (long, 1 everywhere)
        spatial_start_idx  int
    """
    if config is None:
        config = SpatialTokenizerConfig()

    placeholder_id = _resolve_placeholder_id(tokenizer, config.spatial_placeholder)

    # --- Prefix: system + instruction ---
    prefix_text = f"{config.system_prompt}\n\nInstruction: {instruction}\n\nObservation:"
    prefix_ids: list[int] = tokenizer.encode(prefix_text, add_special_tokens=True)

    # --- Spatial placeholder block ---
    spatial_ids = [placeholder_id] * config.num_spatial_tokens
    spatial_start_idx = len(prefix_ids)

    # --- Suffix / target ---
    if target is not None:
        suffix_text = f"\n\n{target}"
        # encode without BOS — it's a continuation
        suffix_ids: list[int] = tokenizer.encode(suffix_text, add_special_tokens=False)
    else:
        suffix_ids = []

    # Add EOS
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        suffix_ids.append(eos_id)

    # --- Assemble ---
    all_ids = prefix_ids + spatial_ids + suffix_ids

    # Truncate to max_length
    if len(all_ids) > config.max_length:
        all_ids = all_ids[: config.max_length]

    seq_len = len(all_ids)
    input_ids = torch.tensor(all_ids, dtype=torch.long)
    attention_mask = torch.ones(seq_len, dtype=torch.long)

    # --- Labels ---
    labels = torch.full((seq_len,), config.ignore_index, dtype=torch.long)
    if target is not None:
        # Only the suffix (target response + EOS) contributes to loss
        target_start = spatial_start_idx + config.num_spatial_tokens
        labels[target_start:] = input_ids[target_start:]
    else:
        # Prealign: autoregressive loss on entire sequence (except spatial block)
        labels[:] = input_ids
        labels[spatial_start_idx : spatial_start_idx + config.num_spatial_tokens] = (
            config.ignore_index
        )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "spatial_start_idx": spatial_start_idx,
    }
