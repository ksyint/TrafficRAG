# Paired grounding comparisons

Use two prediction files with identical query IDs and targets to compare pipeline settings on the same recordings.

```bash
python traffic.py report --predictions outputs/candidate/predictions.jsonl --compare outputs/baseline/predictions.jsonl --output reports/comparison
```

`paired.csv` contains per-query temporal IoU before and after the change, its difference, and event-presence decisions. Target disagreements stop the comparison. `metrics.json` includes the mean temporal-IoU change. Keep each run's `run.json` alongside these outputs to retain its settings.
