# Segment annotations

Create one JSONL row per reviewed interval. `video` resolves relative to the manifest, `video_id` identifies the complete recording, and `id` identifies the interval. Labels are integer zero for safe events and one for violations.

```bash
python traffic.py manifest --manifest data/annotations.jsonl --domain red-light --output data/canonical.jsonl
```

Both `start` and `end` use source-video seconds and satisfy `0 <= start < end`. Use the same `group` for recordings from one drive or driver when those recordings must stay together. Annotation records and whole-video query records share the schema in [video-record.json](../../schemas/video-record.json). The JSONL files under `examples/manifests` demonstrate field names. Point their video paths at acquired recordings before processing.
