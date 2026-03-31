#!/usr/bin/env python3
"""Phase-0 environment verification helper for local/HPC execution."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class CheckResult:
    name: str
    status: str
    message: str
    details: dict[str, Any]


def _ok(name: str, message: str, **details: Any) -> CheckResult:
    return CheckResult(name=name, status="ok", message=message, details=details)


def _fail(name: str, message: str, **details: Any) -> CheckResult:
    return CheckResult(name=name, status="fail", message=message, details=details)


def check_habitat() -> CheckResult:
    try:
        import habitat
    except Exception as exc:  # pragma: no cover
        return _fail("habitat", f"Habitat import failed: {exc}")
    return _ok(
        "habitat",
        "Habitat import succeeded.",
        version=getattr(habitat, "__version__", "unknown"),
    )


def check_qwen(load_model: bool) -> CheckResult:
    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    try:
        from transformers import AutoConfig, AutoModelForImageTextToText

        cfg = AutoConfig.from_pretrained(model_id)
        details: dict[str, Any] = {
            "model_type": getattr(cfg, "model_type", "unknown"),
            "hidden_size": getattr(getattr(cfg, "text_config", cfg), "hidden_size", None),
        }
        if load_model:
            _ = AutoModelForImageTextToText.from_pretrained(model_id, trust_remote_code=True)
            details["model_loaded"] = True
        return _ok("qwen3_vl", "Qwen3-VL config check succeeded.", **details)
    except Exception as exc:  # pragma: no cover
        return _fail("qwen3_vl", f"Qwen3-VL check failed: {exc}", model_id=model_id)


def check_dinov2(load_model: bool) -> CheckResult:
    model_id = "facebook/dinov2-large"
    try:
        from transformers import AutoConfig, AutoModel

        cfg = AutoConfig.from_pretrained(model_id)
        details: dict[str, Any] = {
            "model_type": getattr(cfg, "model_type", "unknown"),
            "hidden_size": getattr(cfg, "hidden_size", None),
        }
        if load_model:
            _ = AutoModel.from_pretrained(model_id)
            details["model_loaded"] = True
        return _ok("dinov2", "DINOv2 config check succeeded.", **details)
    except Exception as exc:  # pragma: no cover
        return _fail("dinov2", f"DINOv2 check failed: {exc}", model_id=model_id)


def check_wandb() -> CheckResult:
    try:
        import wandb
    except Exception as exc:  # pragma: no cover
        return _fail("wandb", f"wandb import failed: {exc}")

    key_present = bool(os.environ.get("WANDB_API_KEY"))
    return _ok(
        "wandb",
        "wandb import succeeded.",
        version=getattr(wandb, "__version__", "unknown"),
        api_key_present=key_present,
    )


def check_gpu() -> CheckResult:
    if not torch.cuda.is_available():
        return _fail("gpu", "CUDA is not available.", device_count=0)

    infos: list[dict[str, Any]] = []
    max_mem_gb = 0.0
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        mem_gb = props.total_memory / (1024**3)
        infos.append({"index": idx, "name": props.name, "total_memory_gb": round(mem_gb, 2)})
        max_mem_gb = max(max_mem_gb, mem_gb)

    return _ok(
        "gpu",
        "CUDA devices detected.",
        device_count=torch.cuda.device_count(),
        devices=infos,
        supports_lora_18gb=max_mem_gb >= 18.0,
        supports_full_40gb=max_mem_gb >= 40.0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase0/env_check.json"),
        help="JSON output report path.",
    )
    parser.add_argument(
        "--load-models",
        action="store_true",
        help="Load full Qwen3-VL and DINOv2 weights (slow, memory-heavy).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with nonzero code when any check fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks = [
        check_habitat(),
        check_qwen(load_model=args.load_models),
        check_dinov2(load_model=args.load_models),
        check_wandb(),
        check_gpu(),
    ]
    payload = {
        "checks": [c.__dict__ for c in checks],
        "all_ok": all(c.status == "ok" for c in checks),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for c in checks:
        print(f"[{c.status.upper()}] {c.name}: {c.message}")
    print(f"Report written to {args.output}")

    if args.strict and not payload["all_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
