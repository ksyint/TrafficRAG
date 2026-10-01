# Inspecting retrieved evidence

Batch results retain candidate captions, motion probabilities, semantic scores, neighbor identities, and the refinement crop. Reports aggregate candidate counts and memory usage without loading model weights.

```bash
python traffic.py report --predictions outputs/red_light/predictions.jsonl --output reports/red_light
python traffic.py cache --root outputs/red_light/cache --namespace captions
```

`metrics.json` includes cache hits, misses, writes, total stage time, and frequently retrieved memory IDs. `confidence.csv` evaluates the saved selected scores at a grid of thresholds. Use a held-out validation partition to choose thresholds, then retain those settings for test reporting.
