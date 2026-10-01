# Runtime packages

The root `traffic.py` dispatches explicit data, training, grounding, and reporting operations.

| Branch | Responsibility |
| --- | --- |
| `trafficrag/models/motion.py` | Video decoding, frame transforms, VideoMAE and CUDA motion training |
| `trafficrag/models/backends.py` | Qwen caption/refinement and ModernBERT retrieval |
| `trafficrag/grounding.py` | Proposal, semantic verification and refinement |
| `trafficrag/runner.py` | Two-pass model lifecycle and resumable query execution |
| `trafficrag/data` | Record validation, group partitions, atomic stage artifacts and model fingerprints |
| `trafficrag/experiments` | Presence and temporal metrics, paired reports, and motion/grounding recipes |
| `configs/retrieval` | Neighbor-count and fusion groups, with threshold sweeps under `k10/f04` and representative recipes at each level |
| `examples` | Annotation, memory, and query examples alongside their record and output schemas |

Caches and prediction records are separate from downloaded foundation-model weights. Each batch records the trained motion checkpoint and memory checksum in its run identity.
