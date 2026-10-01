# Stage artifact cache

The artifact cache stores motion evidence as JSON, captions as JSON, text embeddings as NumPy arrays, and refined intervals as JSON. A SHA-256 key selects a two-character shard directory.

```bash
python traffic.py cache --root outputs/red_light/cache --namespace motion
python traffic.py cache --root outputs/red_light/cache --namespace embeddings
```

Motion keys include the trained checkpoint checksum, video identity, domain, and proposal settings. Language keys include resolved model identities, sampling settings, text, crop, and domain. Refined intervals are stored inside an object so an explicit no-violation result can be distinguished from a cache miss. Writes replace completed temporary files atomically.
