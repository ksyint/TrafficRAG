# Semantic verification

Each proposed interval receives a Qwen caption. ModernBERT retrieves the configured number of nearest reviewed memory captions. The actual returned neighbor count is used when memory has fewer entries than requested.

The selected candidate maximizes the weighted motion and semantic score. The default semantic contribution is 0.4 and the motion contribution is 0.6. Results retain each neighbor ID, caption, label, and similarity.

```bash
python traffic.py report --predictions outputs/red_light/predictions.jsonl --output reports/red_light
```

The report summarizes retrieval calls, candidate counts, neighbor labels, and frequently used memory entries.
