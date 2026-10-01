# ModernBERT caption retrieval

Frozen ModernBERT encodes captions using attention-mask mean pooling. The memory stores embeddings, binary labels, captions, and stable segment IDs. Retrieval normalizes vectors and ranks by cosine similarity with stable ordering for ties.

```bash
python traffic.py build-kb --input data/red_light/kb.jsonl --domain red-light --output data/red_light/kb.npz --text-encoder answerdotai/ModernBERT-base --device cuda
```

Keep motion-training and query recording IDs out of the memory. `--exclude_ids data/red_light/train_query_ids.txt` checks these identities before embedding. The neighboring positive-label fraction is combined with motion confidence by the pipeline configuration.
