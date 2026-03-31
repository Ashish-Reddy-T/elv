#!/usr/bin/env python3
"""Generate paper-ready tables and CSV assets from Phase 9 results."""

from __future__ import annotations

import argparse
from pathlib import Path

from spatialvlm.eval.paper_assets import load_phase9_results, write_paper_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase9-results",
        type=Path,
        default=Path("artifacts/phase9/ablation_results.json"),
        help="Phase 9 JSON file from run_phase9_ablations.py.",
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("paper"),
        help="Paper directory where tables/figures should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = load_phase9_results(args.phase9_results)
    write_paper_assets(runs, args.paper_dir)
    print(f"Paper assets generated in {args.paper_dir}")


if __name__ == "__main__":
    main()
