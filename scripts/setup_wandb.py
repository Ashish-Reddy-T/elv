#!/usr/bin/env python3
"""Set up and verify wandb integration for SpatialVLM runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="wandb project name.")
    parser.add_argument("--entity", default=None, help="wandb entity/team (optional).")
    parser.add_argument(
        "--mode",
        default="online",
        choices=("online", "offline", "disabled"),
        help="wandb execution mode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase0/wandb_check.json"),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when setup/check fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {
        "project": args.project,
        "entity": args.entity,
        "mode": args.mode,
        "status": "ok",
        "message": "",
        "details": {},
    }

    try:
        import wandb
    except Exception as exc:  # pragma: no cover
        report["status"] = "fail"
        report["message"] = f"wandb import failed: {exc}"
        _write_report(args.output, report)
        if args.strict:
            raise SystemExit(1)
        print(report["message"])
        return

    api_key_present = bool(os.environ.get("WANDB_API_KEY"))
    report["details"]["wandb_version"] = getattr(wandb, "__version__", "unknown")
    report["details"]["api_key_present"] = api_key_present

    if args.mode == "online" and not api_key_present:
        report["status"] = "fail"
        report["message"] = "WANDB_API_KEY missing for online mode."
        _write_report(args.output, report)
        if args.strict:
            raise SystemExit(1)
        print(report["message"])
        return

    os.environ["WANDB_MODE"] = args.mode
    run = wandb.init(project=args.project, entity=args.entity, reinit="finish_previous")
    run.log({"setup/check": 1})
    run.finish()

    report["message"] = "wandb init/log/finish succeeded."
    _write_report(args.output, report)
    print(report["message"])


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()
