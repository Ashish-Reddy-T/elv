"""Tests for Qwen3-VL backbone wrapper (no large-model downloads)."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from spatialvlm.backbone.qwen3_vl import Qwen3VLBackbone


class DummyVisionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = nn.Module()
        self.visual.block = DummyVisionBlock()
        self.lm_head = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(x)


def make_dummy_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=4096,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        rope_scaling={"mrope_section": [24, 20, 20]},
    )


class TestQwen3VLBackbone:
    def test_runtime_config_introspection(self):
        wrapper = Qwen3VLBackbone(
            model_id="dummy/model",
            enable_lora=False,
            apply_peft_2880_workaround=False,
            config=make_dummy_config(),
            model=DummyModel(),
        )
        assert wrapper.hidden_size == 4096
        assert wrapper.num_hidden_layers == 36
        assert wrapper.num_attention_heads == 32
        assert wrapper.num_key_value_heads == 8
        assert wrapper.head_dim == 128
        assert wrapper.mrope_section == [24, 20, 20]
        assert wrapper.rotary_pairs == 64

    def test_freeze_then_peft_2880_workaround_touches_vision_qkv(self):
        wrapper = Qwen3VLBackbone(
            model_id="dummy/model",
            enable_lora=False,
            freeze_base_model=True,
            apply_peft_2880_workaround=True,
            config=make_dummy_config(),
            model=DummyModel(),
        )
        stats = wrapper.stats
        assert stats.peft_2880_modules_touched >= 3
        assert stats.peft_2880_params_touched > 0

        # Vision QKV modules should be trainable after workaround.
        qkv = wrapper.model.visual.block
        assert qkv.q_proj.weight.requires_grad
        assert qkv.k_proj.weight.requires_grad
        assert qkv.v_proj.weight.requires_grad

    def test_lora_path_uses_injected_factories(self):
        called = {"factory": False}

        def fake_lora_config_factory(**kwargs):
            return kwargs

        def fake_peft_factory(model: nn.Module, lora_cfg):
            called["factory"] = True
            model._fake_lora_cfg = lora_cfg
            return model

        wrapper = Qwen3VLBackbone(
            model_id="dummy/model",
            enable_lora=True,
            freeze_base_model=False,
            apply_peft_2880_workaround=False,
            config=make_dummy_config(),
            model=DummyModel(),
            peft_model_factory=fake_peft_factory,
            lora_config_factory=fake_lora_config_factory,
            task_type_causal_lm="CAUSAL_LM",
        )
        assert called["factory"]
        assert wrapper.model._fake_lora_cfg["r"] == 32
        assert wrapper.model._fake_lora_cfg["lora_alpha"] == 64
        assert wrapper.model._fake_lora_cfg["target_modules"] == [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]

    def test_forward_delegates_to_model(self):
        wrapper = Qwen3VLBackbone(
            model_id="dummy/model",
            enable_lora=False,
            apply_peft_2880_workaround=False,
            config=make_dummy_config(),
            model=DummyModel(),
        )
        x = torch.randn(2, 4, 8)
        out = wrapper(x)
        assert out.shape == (2, 4, 8)

    def test_lazy_load_defers_model_loader_until_forward(self):
        called = {"config": 0, "model": 0, "loader_kwargs": None}

        def fake_config_loader(model_id: str, **kwargs):
            called["config"] += 1
            return make_dummy_config()

        def fake_model_loader(model_id: str, **kwargs):
            called["model"] += 1
            called["loader_kwargs"] = kwargs
            return DummyModel()

        wrapper = Qwen3VLBackbone(
            model_id="dummy/model",
            enable_lora=False,
            apply_peft_2880_workaround=False,
            lazy_load=True,
            local_files_only=True,
            config_loader=fake_config_loader,
            model_loader=fake_model_loader,
        )
        assert called["config"] == 1
        assert called["model"] == 0
        assert not wrapper.is_model_loaded
        assert wrapper.stats.total_params == 0

        out = wrapper(torch.randn(1, 2, 8))
        assert out.shape == (1, 2, 8)
        assert called["model"] == 1
        assert called["loader_kwargs"] == {
            "trust_remote_code": True,
            "local_files_only": True,
        }
        assert wrapper.is_model_loaded
