# Recording and query identities

Keep `video_id` stable across all segments of one source recording. The manifest reader rejects one ID referring to multiple paths, one path using multiple video IDs, and inconsistent group assignments.

Provide explicit `id` values for query records that will be compared across model runs. Automatic interval IDs use the video ID and manifest line number. Exporting a canonical manifest persists those IDs.

```bash
python traffic.py manifest --manifest data/queries.jsonl --output data/queries.canonical.jsonl
```

Manifest fingerprints include normalized absolute paths, labels, targets, and auxiliary fields. Batch run identity additionally includes video sizes and modification timestamps. Keep the manifest and source files stable when resuming a batch.
