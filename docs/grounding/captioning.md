# Captioning and temporal generation

Qwen3-VL receives uniformly sampled crop frames, decoded presentation times, and a domain-specific event criterion. Caption generation uses deterministic decoding.

```bash
python traffic.py build-kb --input data/red_light/kb.jsonl --domain red-light --output data/red_light/kb.npz --vlm Qwen/Qwen3-VL-8B-Instruct --device cuda
```

Use the same caption model and domain for memory construction and query inference. Refinement prompts include the candidate description and the retrieved memory captions with their reviewed labels. Returned crop-relative boundaries are validated and converted to source-video seconds.
