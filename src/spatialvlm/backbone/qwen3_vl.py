"""Qwen3-VL backbone wrapper with LoRA and PEFT #2880 workaround support."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForImageTextToText


def _extract_mrope_section(config: Any) -> list[int]:
    """Extract M-RoPE section from a HF config object."""
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        return []
    if isinstance(rope_scaling, dict):
        section = rope_scaling.get("mrope_section")
    else:
        section = getattr(rope_scaling, "mrope_section", None)
    if section is None:
        return []
    return [int(x) for x in section]


def _resolve_text_config(config: Any) -> Any:
    """Return the text config object for multimodal configs, or config itself for text-only."""
    text_cfg = getattr(config, "text_config", None)
    return text_cfg if text_cfg is not None else config


def _call_loader(
    loader: Callable[..., Any],
    model_id: str,
    **kwargs: Any,
) -> Any:
    """Call loader with supported kwargs only."""
    try:
        signature = inspect.signature(loader)
    except (TypeError, ValueError):
        return loader(model_id, **kwargs)

    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return loader(model_id, **kwargs)

    supported_kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return loader(model_id, **supported_kwargs)


@dataclass
class Qwen3BackboneStats:
    """Small stats container for debug/verification."""

    trainable_params: int
    total_params: int
    peft_2880_modules_touched: int
    peft_2880_params_touched: int


class Qwen3VLBackbone(nn.Module):
    """Wrapper around Qwen3-VL with optional LoRA and PEFT #2880 workaround."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        lora_rank: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.0,
        lora_target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
        enable_lora: bool = True,
        freeze_base_model: bool = True,
        apply_peft_2880_workaround: bool = True,
        device: torch.device | None = None,
        torch_dtype: torch.dtype | None = None,
        config: Any | None = None,
        model: nn.Module | None = None,
        config_loader: Callable[[str], Any] = AutoConfig.from_pretrained,
        model_loader: Callable[..., nn.Module] = AutoModelForImageTextToText.from_pretrained,
        peft_model_factory: Callable[[nn.Module, Any], nn.Module] | None = None,
        lora_config_factory: Callable[..., Any] | None = None,
        task_type_causal_lm: Any | None = None,
        lazy_load: bool = False,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")

        self.model_id = model_id
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = tuple(lora_target_modules)
        self._device = device
        self._torch_dtype = torch_dtype
        self._local_files_only = local_files_only
        self._model_loader = model_loader
        self._freeze_base_model = freeze_base_model
        self._enable_lora = enable_lora
        self._apply_peft_2880_workaround = apply_peft_2880_workaround
        self._peft_model_factory = peft_model_factory
        self._lora_config_factory = lora_config_factory
        self._task_type_causal_lm = task_type_causal_lm

        self.config = (
            config
            if config is not None
            else _call_loader(config_loader, model_id, local_files_only=local_files_only)
        )
        text_cfg = _resolve_text_config(self.config)
        self.hidden_size = int(getattr(text_cfg, "hidden_size"))
        self.num_hidden_layers = int(getattr(text_cfg, "num_hidden_layers"))
        self.num_attention_heads = int(getattr(text_cfg, "num_attention_heads"))
        self.num_key_value_heads = int(
            getattr(text_cfg, "num_key_value_heads", self.num_attention_heads)
        )
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.mrope_section = _extract_mrope_section(text_cfg)
        self.rotary_pairs = sum(self.mrope_section) if self.mrope_section else (self.head_dim // 2)
        self.model: nn.Module | None = None
        self._stats: Qwen3BackboneStats | None = None

        if model is not None:
            self._initialize_loaded_model(model)
        elif not lazy_load:
            self.load_model()

    def freeze_all_parameters(self) -> None:
        """Freeze all current model parameters."""
        if self.model is None:
            return
        for p in self.model.parameters():
            p.requires_grad_(False)

    def enable_peft_2880_workaround(
        self,
        vision_keywords: Sequence[str] = ("vision", "visual", "vit", "image_tower", "vision_tower"),
        qkv_keywords: Sequence[str] = ("q_proj", "k_proj", "v_proj"),
    ) -> tuple[int, int]:
        """Set `requires_grad=True` on vision QKV modules to avoid PEFT bug #2880 behavior."""
        if self.model is None:
            return (0, 0)

        modules_touched = 0
        params_touched = 0
        for module_name, module in self.model.named_modules():
            if not isinstance(module, nn.Module):
                continue
            lower_name = module_name.lower()
            if not any(k in lower_name for k in vision_keywords):
                continue
            if not any(k in lower_name for k in qkv_keywords):
                continue

            module_had_change = False
            for p in module.parameters(recurse=False):
                if not p.requires_grad:
                    p.requires_grad_(True)
                    params_touched += p.numel()
                    module_had_change = True
            if module_had_change:
                modules_touched += 1

        return modules_touched, params_touched

    def _count_trainable_params(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _count_total_params(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())

    @property
    def is_model_loaded(self) -> bool:
        return self.model is not None

    def _initialize_loaded_model(self, model: nn.Module) -> None:
        model = model.to(self._device)
        self.model = model

        if self._freeze_base_model:
            self.freeze_all_parameters()

        if self._enable_lora:
            peft_model_factory = self._peft_model_factory
            lora_config_factory = self._lora_config_factory
            task_type_causal_lm = self._task_type_causal_lm
            if (
                peft_model_factory is None
                or lora_config_factory is None
                or task_type_causal_lm is None
            ):
                from peft import LoraConfig, TaskType, get_peft_model

                peft_model_factory = get_peft_model
                lora_config_factory = LoraConfig
                task_type_causal_lm = TaskType.CAUSAL_LM

            lora_cfg = lora_config_factory(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=list(self.lora_target_modules),
                bias="none",
                task_type=task_type_causal_lm,
            )
            self.model = peft_model_factory(self.model, lora_cfg)

        modules_touched = 0
        params_touched = 0
        if self._apply_peft_2880_workaround:
            modules_touched, params_touched = self.enable_peft_2880_workaround()

        self._stats = Qwen3BackboneStats(
            trainable_params=self._count_trainable_params(),
            total_params=self._count_total_params(),
            peft_2880_modules_touched=modules_touched,
            peft_2880_params_touched=params_touched,
        )
        self.to(self._device)

    def load_model(self) -> None:
        """Load backbone weights when lazy initialization is enabled."""
        if self.model is not None:
            return

        loader_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": self._local_files_only,
        }
        if self._torch_dtype is not None:
            loader_kwargs["torch_dtype"] = self._torch_dtype
        model = _call_loader(self._model_loader, self.model_id, **loader_kwargs)
        self._initialize_loaded_model(model)

    @property
    def stats(self) -> Qwen3BackboneStats:
        if self._stats is None:
            return Qwen3BackboneStats(
                trainable_params=0,
                total_params=0,
                peft_2880_modules_touched=0,
                peft_2880_params_touched=0,
            )
        return self._stats

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate forward pass to wrapped model."""
        self.load_model()
        if self.model is None:
            raise RuntimeError("Qwen3 model is unavailable. Call load_model() before forward().")
        return self.model(*args, **kwargs)
