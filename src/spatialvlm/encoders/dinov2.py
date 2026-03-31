"""DINOv2-L/14 encoder with multi-layer feature extraction.

Architecture notes:
  - Model: facebook/dinov2-large  (⚠ VERIFY model ID at runtime)
  - Input resolution: 518×518 → patch_size=14 → 37×37 = 1369 patches (exact)
  - DINOv2 DOES have a CLS token prepended at position 0 — must be stripped
  - Extract intermediate features at layers {8, 16, 24} (evenly-spaced thirds)
  - Channel-concatenate 3 × hidden_size → [B, 1369, 3072]
  - All encoder parameters are frozen
  - NO spatial pooling — full 1369 tokens preserved (hypothesis H1d)

Resolution choice (518px):
  518 / 14 = 37.0 exactly. DINOv2's default input is 224px (16 patches/side),
  but 518px is our choice to ensure pixel-perfect alignment with the depth map
  and GATr geometric branch (both at 518×518).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


def _find_dinov2_encoder_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the transformer encoder layer list in a DINOv2 model.

    Returns
    -------
    layers : nn.ModuleList

    Raises
    ------
    RuntimeError
    """
    candidate_paths = [
        ["encoder", "layer"],   # Dinov2Model standard path
        ["encoder", "layers"],
        ["model", "encoder", "layer"],
    ]
    for path in candidate_paths:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if isinstance(obj, nn.ModuleList):
                return obj
        except AttributeError:
            continue

    raise RuntimeError(
        f"Cannot find encoder layers in {type(model).__name__}. "
        "Expected path like encoder.layer. "
        "Check the model with: print(model)"
    )


class DINOv2Encoder(nn.Module):
    """DINOv2-L encoder with multi-layer feature extraction.

    Extracts features from intermediate transformer layers, strips the CLS
    token, and channel-concatenates the patch tokens.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID. ⚠ VERIFY before use.
    extract_layers : list of int
        1-indexed layer numbers to extract (e.g. [8, 16, 24]).
    device : torch.device
    """

    def __init__(
        self,
        model_id: str = "facebook/dinov2-large",
        extract_layers: list[int] | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        if extract_layers is None:
            extract_layers = [8, 16, 24]

        if device is None:
            device = torch.device("cpu")

        # Load model and verify architecture from config
        cfg = AutoConfig.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        model = model.to(device)

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()

        # Introspect config — ⚠ all values from cfg, never hardcoded
        self._hidden_size: int = int(cfg.hidden_size)         # ⚠ verified
        self._num_layers: int = int(cfg.num_hidden_layers)    # ⚠ verified
        self._patch_size: int = int(cfg.patch_size)           # ⚠ verified

        # Validate extract_layers
        for layer_idx in extract_layers:
            if not (1 <= layer_idx <= self._num_layers):
                raise ValueError(
                    f"extract_layer {layer_idx} is out of range "
                    f"[1, {self._num_layers}] for {model_id}"
                )

        self._extract_layers = sorted(extract_layers)
        self._model = model
        self._device = device

        # Our design choice: 518px input → 37×37 = 1369 patches (no pooling)
        # The patch count is validated during the first forward pass
        self._image_size: int = 518
        self._n_patches: int = (self._image_size // self._patch_size) ** 2

        # Register forward hooks
        self._hook_outputs: dict[int, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

        encoder_layers = _find_dinov2_encoder_layers(self._model)

        for layer_1idx in self._extract_layers:
            layer_0idx = layer_1idx - 1  # 0-indexed
            hook = encoder_layers[layer_0idx].register_forward_hook(
                self._make_hook(layer_0idx)
            )
            self._hooks.append(hook)

    def _make_hook(self, layer_0idx: int):
        """Create a forward hook that stores hidden states for the given layer."""
        def hook(module: nn.Module, input: tuple, output) -> None:  # noqa: ARG001
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            self._hook_outputs[layer_0idx] = hidden
        return hook

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract multi-layer features from DINOv2-L.

        Parameters
        ----------
        pixel_values : Tensor[B, 3, H, W]
            Preprocessed images. H=W=518 for our design (37×37 patch grid).

        Returns
        -------
        features : Tensor[B, n_patches, n_layers × hidden_size]
            CLS-stripped, channel-concatenated features.
            For default config: [B, 1369, 3072]  (1369 = 37², 3072 = 3 × 1024)
        """
        self._hook_outputs.clear()

        with torch.no_grad():
            _ = self._model(pixel_values=pixel_values)

        features = []
        for layer_1idx in self._extract_layers:
            layer_0idx = layer_1idx - 1
            h = self._hook_outputs[layer_0idx]  # [B, T, hidden_size]

            # DINOv2 prepends CLS token at position 0 → T = n_patches + 1
            # Strip it: keep only patch tokens
            if h.shape[1] == self._n_patches + 1:
                h = h[:, 1:, :]    # [B, n_patches, hidden_size]
            elif h.shape[1] == self._n_patches:
                pass               # no CLS — use as-is (defensive)
            else:
                raise RuntimeError(
                    f"Unexpected token count {h.shape[1]} at layer {layer_1idx}. "
                    f"Expected {self._n_patches} (no CLS) or "
                    f"{self._n_patches + 1} (with CLS). "
                    f"Image size {pixel_values.shape[-2]}×{pixel_values.shape[-1]}, "
                    f"patch_size {self._patch_size}."
                )

            features.append(h)  # [B, n_patches, hidden_size]

        return torch.cat(features, dim=-1)  # [B, n_patches, n_layers × hidden_size]

    def __del__(self) -> None:
        for hook in self._hooks:
            hook.remove()

    @property
    def out_dim(self) -> int:
        """Output channel dimension = n_extract_layers × hidden_size."""
        return len(self._extract_layers) * self._hidden_size

    @property
    def n_patches(self) -> int:
        """Number of spatial patch tokens (no CLS)."""
        return self._n_patches
