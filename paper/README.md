# Paper Scaffold

## Generate Phase 9 artifact

```bash
python scripts/run_phase9_ablations.py \
  --config configs/eval.yaml \
  --output artifacts/phase9/ablation_results.json
```

## Generate paper tables/figure CSV

```bash
python scripts/generate_paper_assets.py \
  --phase9-results artifacts/phase9/ablation_results.json \
  --paper-dir paper
```

This updates:
- `paper/tables/main_results.tex`
- `paper/tables/ablation_results.tex`
- `paper/figures/permutation_curve.csv`
