# Batch grounding and continuation

Prepare one whole-video query row per intended target event, with a stable `id`, source `video_id`, and nullable `target` interval.

```bash
python traffic.py batch --manifest data/red_light/queries.jsonl --motion-checkpoint outputs/motion/red_light/last.pt --kb data/red_light/kb.npz --domain red-light --output outputs/red_light --device cuda
python traffic.py batch --manifest data/red_light/queries.jsonl --motion-checkpoint outputs/motion/red_light/last.pt --kb data/red_light/kb.npz --domain red-light --output outputs/red_light --device cuda --resume
```

The runner finishes the motion pass before loading Qwen and ModernBERT. Atomic per-query JSON files make completed results durable. Resume validates the manifest, checkpoints, memory, and pipeline identity before skipping completed IDs. Final rows follow manifest order in `predictions.jsonl`.
