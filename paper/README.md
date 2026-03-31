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

## Useful validation commands

```bash
make lint
make test-unit
pytest tests/test_rewards.py tests/test_phase9.py tests/test_paper_assets.py -v --tb=short
```
