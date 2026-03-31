#!/usr/bin/env python3
"""Materialize and validate the Phase 9 ablation run matrix.

This script does not fabricate benchmark results. It builds a canonical run-plan artifact and can
merge externally produced scores (from real training/eval jobs) into a single JSON report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from spatialvlm.eval.phase9 import missing_phase9_runs, phase9_coverage_complete, phase9_run_specs


def build_phase9_report(
    config_path: Path,
    scores_path: Path | None,
) -> dict[str, Any]:
    cfg = OmegaConf.load(config_path)
    specs = phase9_run_specs()
    scores: dict[str, Any] = {}
    if scores_path is not None and scores_path.exists():
        scores = json.loads(scores_path.read_text(encoding="utf-8"))

    runs: dict[str, Any] = {}
    for spec in specs:
        row = {
            "run_id": spec.run_id,
            "title": spec.title,
            "hypotheses": list(spec.hypotheses),
            "config_overrides": spec.config_overrides,
            "status": "pending",
            "score": None,
            "delta_vs_full_model": None,
            "notes": "",
        }
        if spec.run_id in scores:
            merged = dict(scores[spec.run_id])
            row["status"] = merged.get("status", "completed")
            row["score"] = merged.get("score")
            row["delta_vs_full_model"] = merged.get("delta_vs_full_model")
            row["notes"] = merged.get("notes", "")
        runs[spec.run_id] = row

    return {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "runs": runs,
        "coverage_complete": phase9_coverage_complete(runs),
        "missing_runs": missing_phase9_runs(runs),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval.yaml"),
        help="Evaluation config used for the run plan.",
    )
    parser.add_argument(
        "--scores-json",
        type=Path,
        default=None,
        help="Optional JSON file with real run results keyed by run_id.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase9/ablation_results.json"),
        help="Output JSON report path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_phase9_report(config_path=args.config, scores_path=args.scores_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if report["coverage_complete"]:
        print(f"Phase 9 report written to {args.output} (coverage complete).")
    else:
        missing = ", ".join(report["missing_runs"])
        print(f"Phase 9 report written to {args.output} (missing runs: {missing}).")


if __name__ == "__main__":
    main()
