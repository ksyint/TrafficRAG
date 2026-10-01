# Event presence metrics

A non-null prediction indicates a detected event. A non-null target indicates a reviewed positive recording. Presence reporting includes TP, FP, TN, FN, precision, recall, specificity, accuracy, and F1.

```bash
python traffic.py report --predictions outputs/red_light/predictions.jsonl --output reports/red_light
```

For separate ground truth, add `--targets data/red_light/targets.jsonl`. Joining requires identical IDs and checks carried-over domain and recording fields. Duplicate or missing IDs stop evaluation. Presence metrics use every joined query.
