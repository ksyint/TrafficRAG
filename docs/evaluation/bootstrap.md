# Recording-level confidence intervals

Bootstrap resampling operates on recording groups, retaining all query records from a selected video in each draw. This preserves the dependence between repeated queries from one recording.

```bash
python traffic.py report --predictions outputs/red_light/predictions.jsonl --output reports/red_light --bootstrap 1000 --seed 42
```

The report stores percentile intervals for event F1, precision, recall, accuracy, and mean temporal IoU. It records the confidence level, repetition count, and resampling field. Keep stable video IDs when preparing the evaluation manifest.
