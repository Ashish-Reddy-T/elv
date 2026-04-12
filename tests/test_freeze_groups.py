"""Tests for `set_trainable_by_groups` and the `trainable_groups` config path.

The freeze groups give the caller a named lever for staged training
(e.g. GATr-first → DINO-second → all-combined) without touching the
default training pipeline. Each group resolves to one or more
parameter-name substrings defined in `FREEZE_GROUP_PATTERNS`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from spatialvlm.training.prealign import PrealignConfig, PrealignmentTrainer
from spatialvlm.training.sft import (
    FREEZE_GROUP_PATTERNS,
    SFTConfig,
    SFTTrainer,
    set_trainable_by_groups,
)


class _MockSpatialVLM(nn.Module):
    """Parameter-name layout that mirrors `SpatialVLM` in `spatialvlm/model.py`."""

    def __init__(self, vocab_size: int = 17, hidden_dim: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)

        self.siglip_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.dinov2_encoder = nn.Linear(hidden_dim, hidden_dim)

        self.siglip_projector = nn.Linear(hidden_dim, hidden_dim)
        self.dinov2_projector = nn.Linear(hidden_dim, hidden_dim)

        # Mimic `self.gatr = GATrWrapper(...)` which itself owns `.projector`
        # and `.gatr` (the blocks) — so `gatr.*` is the correct group match.
        self.gatr = nn.Sequential(nn.Linear(hidden_dim, hidden_dim))

        self.sva = nn.Linear(hidden_dim, hidden_dim)
        self.norm_matching = nn.LayerNorm(hidden_dim)

        # Approximation of a peft LoRA adapter on the LLM backbone.
        self.backbone = nn.Module()
        self.backbone.lora_A = nn.Parameter(torch.randn(hidden_dim, 4))
        self.backbone.lora_B = nn.Parameter(torch.randn(4, hidden_dim))
        self.backbone.base_weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        x = self.embed(input_ids)
        x = self.siglip_projector(x) + self.dinov2_projector(x)
        x = self.gatr(x) + self.sva(x)
        x = x + (x @ self.backbone.lora_A @ self.backbone.lora_B)
        logits = self.lm_head(x)
        return SimpleNamespace(logits=logits)


def _grad_param_names(model: nn.Module) -> set[str]:
    return {name for name, p in model.named_parameters() if p.requires_grad}


def test_siglip_proj_group_only_unfreezes_siglip_projector():
    model = _MockSpatialVLM()
    touched = set_trainable_by_groups(model, {"siglip_proj"})
    grad_names = _grad_param_names(model)

    assert set(touched) == grad_names
    assert all("siglip_projector." in n for n in grad_names), grad_names
    assert all("dinov2_projector." not in n for n in grad_names)


def test_dino_proj_group_only_unfreezes_dinov2_projector():
    model = _MockSpatialVLM()
    set_trainable_by_groups(model, {"dino_proj"})
    grad_names = _grad_param_names(model)

    assert all("dinov2_projector." in n for n in grad_names), grad_names
    assert all("siglip_projector." not in n for n in grad_names)
    assert all("gatr." not in n for n in grad_names)


def test_gatr_group_covers_gatr_submodules():
    model = _MockSpatialVLM()
    set_trainable_by_groups(model, {"gatr"})
    grad_names = _grad_param_names(model)

    assert any(n.startswith("gatr.") for n in grad_names), grad_names
    assert all("gatr." in n.lower() for n in grad_names)


def test_sva_group_only_unfreezes_sva():
    model = _MockSpatialVLM()
    set_trainable_by_groups(model, {"sva"})
    grad_names = _grad_param_names(model)

    assert all("sva." in n for n in grad_names), grad_names


def test_lora_group_matches_peft_adapter_names():
    model = _MockSpatialVLM()
    set_trainable_by_groups(model, {"lora"})
    grad_names = _grad_param_names(model)

    assert {"backbone.lora_A", "backbone.lora_B"}.issubset(grad_names)
    assert "backbone.base_weight" not in grad_names
    assert "lm_head.weight" not in grad_names


def test_multiple_groups_are_unioned():
    model = _MockSpatialVLM()
    set_trainable_by_groups(model, {"gatr", "sva"})
    grad_names = _grad_param_names(model)

    assert any(n.startswith("gatr.") for n in grad_names)
    assert any(n.startswith("sva.") for n in grad_names)
    assert all(("gatr." in n) or ("sva." in n) for n in grad_names)


def test_all_groups_union_leaves_expected_params_frozen():
    """Union of every group should still leave embed/encoder/base weights frozen."""
    model = _MockSpatialVLM()
    set_trainable_by_groups(model, set(FREEZE_GROUP_PATTERNS.keys()))
    grad_names = _grad_param_names(model)
    frozen_names = {n for n, p in model.named_parameters() if not p.requires_grad}

    # Encoders and embeddings are frozen regardless of group choice.
    assert "embed.weight" in frozen_names
    assert "siglip_encoder.weight" in frozen_names
    assert "dinov2_encoder.weight" in frozen_names
    assert "backbone.base_weight" in frozen_names
    # The union should have unfrozen projectors, gatr, sva, and lora.
    assert any(n.startswith("siglip_projector.") for n in grad_names)
    assert any(n.startswith("dinov2_projector.") for n in grad_names)
    assert any(n.startswith("gatr.") for n in grad_names)
    assert any(n.startswith("sva.") for n in grad_names)
    assert "backbone.lora_A" in grad_names


def test_unknown_group_raises():
    model = _MockSpatialVLM()
    with pytest.raises(ValueError, match="Unknown freeze group"):
        set_trainable_by_groups(model, {"encoder_heads"})


def test_sft_trainer_uses_groups_when_set():
    model = _MockSpatialVLM()
    config = SFTConfig(trainable_groups=("gatr", "sva"), learning_rate=1e-3)
    trainer = SFTTrainer(model=model, config=config)

    grad_names = _grad_param_names(model)
    assert set(trainer.trainable_parameter_names) == grad_names
    assert all(("gatr." in n) or ("sva." in n) for n in grad_names)

    out = trainer.step(
        {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "labels": torch.tensor([[1, 2, 3], [4, 5, -100]]),
        }
    )
    assert out.loss > 0
    assert out.trainable_params == trainer.trainable_parameter_count()


def test_sft_trainer_keyword_path_unchanged_when_groups_is_none():
    """Default path (no `trainable_groups`) must match the old behaviour."""
    model = _MockSpatialVLM()
    config = SFTConfig(
        trainable_keywords=("siglip_projector", "gatr"),
        learning_rate=1e-3,
    )
    trainer = SFTTrainer(model=model, config=config)
    grad_names = _grad_param_names(model)

    assert set(trainer.trainable_parameter_names) == grad_names
    assert any(n.startswith("siglip_projector.") for n in grad_names)
    assert any(n.startswith("gatr.") for n in grad_names)
    # Keyword-path selection must NOT touch SVA.
    assert all("sva." not in n for n in grad_names)


def test_sft_trainer_raises_on_empty_group_match():
    class _Empty(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.unrelated = nn.Linear(4, 4)

        def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(logits=torch.zeros(1, 1, 1))

    with pytest.raises(ValueError, match="No trainable parameters"):
        SFTTrainer(model=_Empty(), config=SFTConfig(trainable_groups=("gatr",)))


def test_prealign_trainer_uses_groups_when_set():
    model = _MockSpatialVLM()
    config = PrealignConfig(trainable_groups=("gatr",), learning_rate=1e-3)
    trainer = PrealignmentTrainer(model=model, config=config)

    grad_names = _grad_param_names(model)
    assert set(trainer.trainable_parameter_names) == grad_names
    assert all("gatr." in n for n in grad_names)
    assert all("projector." not in n for n in grad_names)


def test_prealign_trainer_keyword_path_unchanged_when_groups_is_none():
    model = _MockSpatialVLM()
    config = PrealignConfig(projector_keywords=("projector",), learning_rate=1e-3)
    trainer = PrealignmentTrainer(model=model, config=config)
    grad_names = _grad_param_names(model)

    assert set(trainer.trainable_parameter_names) == grad_names
    # Both siglip_projector and dinov2_projector match "projector".
    assert any("siglip_projector" in n for n in grad_names)
    assert any("dinov2_projector" in n for n in grad_names)
    assert all("gatr" not in n for n in grad_names)
