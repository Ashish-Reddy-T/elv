"""Utilities for generating paper-ready assets from ablation outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _fmt(value: Any) -> str:
    if value is None:
        return "TBD"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def load_phase9_results(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Phase 9 results file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = payload.get("runs")
    if not isinstance(runs, dict):
        raise ValueError("Phase 9 JSON must contain top-level `runs` object.")
    return runs


def render_ablation_table_tex(runs: dict[str, Any]) -> str:
    header = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Ablation & Score & $\\Delta$ vs Full \\\\",
        "\\midrule",
    ]
    body: list[str] = []
    for run_id, row in runs.items():
        title = row.get("title", run_id).replace("_", "\\_")
        score = _fmt(row.get("score"))
        delta = _fmt(row.get("delta_vs_full_model"))
        body.append(f"{title} & {score} & {delta} \\\\")
    footer = ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(header + body + footer) + "\n"


def render_main_results_table_tex(runs: dict[str, Any]) -> str:
    full = runs.get("full-model", {})
    score = _fmt(full.get("score"))
    return (
        "\\begin{tabular}{lc}\n"
        "\\toprule\n"
        "Model & Composite Score \\\\\n"
        "\\midrule\n"
        f"SpatialVLM (Full) & {score} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def write_permutation_csv(runs: dict[str, Any], output_path: Path) -> None:
    row = runs.get("permutation-test", {})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "relative_drop",
                "baseline_relative_drop",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model": "SpatialVLM",
                "relative_drop": row.get("relative_drop", ""),
                "baseline_relative_drop": row.get("baseline_relative_drop", ""),
            }
        )


def write_paper_assets(runs: dict[str, Any], paper_dir: Path) -> None:
    tables_dir = paper_dir / "tables"
    figures_dir = paper_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    (tables_dir / "ablation_results.tex").write_text(
        render_ablation_table_tex(runs),
        encoding="utf-8",
    )
    (tables_dir / "main_results.tex").write_text(
        render_main_results_table_tex(runs),
        encoding="utf-8",
    )
    write_permutation_csv(runs, figures_dir / "permutation_curve.csv")
