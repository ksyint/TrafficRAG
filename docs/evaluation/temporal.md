# Temporal localization metrics

Temporal IoU is interval intersection length divided by union length. Mean temporal IoU uses positive ground-truth recordings and assigns zero to missed detections.

```bash
python traffic.py report --predictions outputs/red_light/predictions.jsonl --output reports/red_light --iou-thresholds 0.3 0.5 0.7
```

Recall at each temporal threshold uses the same positive-video denominator. Start and end MAE and signed bias use detected positive recordings. The report records both denominators so the boundary measurements can be interpreted alongside event recall.
