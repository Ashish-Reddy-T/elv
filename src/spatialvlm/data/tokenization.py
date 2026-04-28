"""Tokenization utilities: build input_ids with spatial placeholders and labels.

Three formats, controlled by whether ``target`` is passed to ``build_input_ids``:

  - **prealign (no caption)**:  ``Observation: [SPATIAL×N] \\n\\nInstruction: {text} [EOS]``
    Spatial tokens come first so every instruction token attends to them.
    Loss on instruction + EOS only.

  - **prealign (with caption)** / **sft**:
    ``{system} Instruction: {text} Observation: [SPATIAL×N] \\n\\n{target} [EOS]``
    ``target`` is the MP3D scene caption (prealign) or reasoning+action (SFT).
    Loss on target + EOS only.

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
        Target response (for SFT).  When ``None``, uses prealign format:
        spatial tokens come first so that the instruction prediction is
        conditioned on the spatial context (rich gradient signal to projectors).
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
    eos_id = getattr(tokenizer, "eos_token_id", None)

    if target is not None:
        # --- SFT format: [system + instruction] [spatial×N] [target] [EOS] ---
        # Loss on target + EOS only.
        prefix_text = f"{config.system_prompt}\n\nInstruction: {instruction}\n\nObservation:"
        prefix_ids: list[int] = tokenizer.encode(prefix_text, add_special_tokens=True)
        spatial_ids = [placeholder_id] * config.num_spatial_tokens
        spatial_start_idx = len(prefix_ids)
        suffix_text = f"\n\n{target}"
        suffix_ids: list[int] = tokenizer.encode(suffix_text, add_special_tokens=False)
        if eos_id is not None:
            suffix_ids.append(eos_id)

        all_ids = prefix_ids + spatial_ids + suffix_ids
        if len(all_ids) > config.max_length:
            all_ids = all_ids[: config.max_length]
        seq_len = len(all_ids)
        input_ids = torch.tensor(all_ids, dtype=torch.long)
        labels = torch.full((seq_len,), config.ignore_index, dtype=torch.long)
        target_start = spatial_start_idx + config.num_spatial_tokens
        labels[target_start:] = input_ids[target_start:]

    else:
        # --- Prealign format: Observation: [spatial×N] [instruction] [EOS] ---
        # Spatial tokens come BEFORE the instruction so that every instruction
        # token's prediction is conditioned on the spatial context.  Loss is
        # computed on the instruction + EOS only, giving a rich gradient signal
        # back through the spatial projectors.
        prefix_text = "Observation:"
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=True)
        spatial_ids = [placeholder_id] * config.num_spatial_tokens
        spatial_start_idx = len(prefix_ids)
        suffix_text = f"\n\nInstruction: {instruction}"
        suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
        if eos_id is not None:
            suffix_ids.append(eos_id)

        all_ids = prefix_ids + spatial_ids + suffix_ids
        if len(all_ids) > config.max_length:
            all_ids = all_ids[: config.max_length]
        seq_len = len(all_ids)
        input_ids = torch.tensor(all_ids, dtype=torch.long)
        labels = torch.full((seq_len,), config.ignore_index, dtype=torch.long)
        target_start = spatial_start_idx + config.num_spatial_tokens
        # Only predict instruction + EOS (the part conditioned on spatial tokens)
        labels[target_start:] = input_ids[target_start:]

    attention_mask = torch.ones(seq_len, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "spatial_start_idx": spatial_start_idx,
    }
