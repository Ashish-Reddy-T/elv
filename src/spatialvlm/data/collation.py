"""Batch collation: convert cached frames into model-ready tensors.

The collator takes a list of frame dicts (from ``CachedFrameDataset``) and
produces a single dict consumable by ``SpatialVLM.forward()``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from spatialvlm.data.tokenization import SpatialTokenizerConfig, build_input_ids
from spatialvlm.utils.camera import CameraIntrinsics

# Default Habitat intrinsics for 518×518 renders (used when frame omits them)
_DEFAULT_INTRINSICS = CameraIntrinsics(
    fx=256.0, fy=256.0, cx=259.0, cy=259.0, width=518, height=518
)


class SpatialVLMCollator:
    """Collates cached frames into a model-ready batch.

    Parameters
    ----------
    tokenizer
        HuggingFace tokenizer (e.g. ``AutoTokenizer.from_pretrained(...)``).
    stage : str
        ``"prealign"`` or ``"sft"``.  Controls whether ``target`` is used.
    tok_config : SpatialTokenizerConfig | None
        Override tokenization defaults.
    siglip_size : tuple[int, int]
        Resize for SigLIP input.
    dinov2_size : tuple[int, int]
        Resize for DINOv2 input.
    """

    def __init__(
        self,
        tokenizer: Any,
        stage: str = "sft",
        tok_config: SpatialTokenizerConfig | None = None,
        siglip_size: tuple[int, int] = (384, 384),
        dinov2_size: tuple[int, int] = (518, 518),
    ) -> None:
        self.tokenizer = tokenizer
        self.stage = stage
        self.tok_config = tok_config or SpatialTokenizerConfig()
        self.siglip_size = siglip_size
        self.dinov2_size = dinov2_size

    def __call__(self, frames: list[dict[str, Any]]) -> dict[str, Any]:
        siglip_pixels_list: list[torch.Tensor] = []
        dinov2_pixels_list: list[torch.Tensor] = []
        depth_list: list[torch.Tensor] = []
        input_ids_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        attn_mask_list: list[torch.Tensor] = []
        spatial_start_idx: int | None = None

        for frame in frames:
            rgb = frame["rgb"]  # [3, H, W] float
            depth = frame["depth"]  # [H, W] float, metric metres

            # Resize RGB for each encoder
            rgb_4d = rgb.unsqueeze(0)  # [1, 3, H, W]
            siglip_px = F.interpolate(
                rgb_4d, size=self.siglip_size, mode="bilinear", align_corners=False
            )
            dinov2_px = F.interpolate(
                rgb_4d, size=self.dinov2_size, mode="bilinear", align_corners=False
            )
            siglip_pixels_list.append(siglip_px.squeeze(0))  # [3, 384, 384]
            dinov2_pixels_list.append(dinov2_px.squeeze(0))  # [3, 518, 518]

            # Depth — resize to 518×518 if needed (nearest to preserve metric values)
            if depth.shape != (518, 518):
                depth = (
                    F.interpolate(depth.unsqueeze(0).unsqueeze(0), size=(518, 518), mode="nearest")
                    .squeeze(0)
                    .squeeze(0)
                )
            depth_list.append(depth)

            # Tokenize
            instruction = frame.get("instruction", "Navigate the environment.")
            target = frame.get("target") if self.stage == "sft" else None
            tok_out = build_input_ids(
                self.tokenizer,
                instruction=instruction,
                target=target,
                config=self.tok_config,
            )
            input_ids_list.append(tok_out["input_ids"])
            labels_list.append(tok_out["labels"])
            attn_mask_list.append(tok_out["attention_mask"])

            if spatial_start_idx is None:
                spatial_start_idx = tok_out["spatial_start_idx"]

        # Pad sequences to uniform length
        max_len = max(ids.shape[0] for ids in input_ids_list)
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0

        for i in range(len(input_ids_list)):
            pad_len = max_len - input_ids_list[i].shape[0]
            if pad_len > 0:
                input_ids_list[i] = F.pad(input_ids_list[i], (0, pad_len), value=pad_id)
                labels_list[i] = F.pad(
                    labels_list[i], (0, pad_len), value=self.tok_config.ignore_index
                )
                attn_mask_list[i] = F.pad(attn_mask_list[i], (0, pad_len), value=0)

        # Build intrinsics (shared per batch — Habitat uses same sensor config)
        intr = frames[0].get("intrinsics")
        if intr is not None and isinstance(intr, dict):
            intrinsics = CameraIntrinsics(**intr)
        else:
            intrinsics = _DEFAULT_INTRINSICS

        return {
            "siglip_pixels": torch.stack(siglip_pixels_list),  # [B, 3, 384, 384]
            "dinov2_pixels": torch.stack(dinov2_pixels_list),  # [B, 3, 518, 518]
            "depth": torch.stack(depth_list),  # [B, 518, 518]
            "intrinsics": intrinsics,
            "input_ids": torch.stack(input_ids_list),  # [B, seq_len]
            "labels": torch.stack(labels_list),  # [B, seq_len]
            "attention_mask": torch.stack(attn_mask_list),  # [B, seq_len]
            "spatial_start_idx": spatial_start_idx or 0,
        }
