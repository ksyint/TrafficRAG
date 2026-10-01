# Runtime packages

The root `traffic.py` dispatches explicit data, training, grounding, and reporting operations.

| Branch | Responsibility |
| --- | --- |
| `trafficrag/models/video` | Video decoding, frame transforms, VideoMAE and CUDA motion training |
| `trafficrag/models/multimodal` | Qwen caption/refinement and ModernBERT retrieval |
| `trafficrag/data/manifests` | Record validation, stable identities and group partitions |
| `trafficrag/data/cache` | Atomic stage artifacts and resolved model fingerprints |
| `trafficrag/pipeline/temporal` | Proposal, semantic verification and refinement |
| `trafficrag/pipeline/batch` | Two-pass model lifecycle and resumable query execution |
| `trafficrag/evaluation/events` | Presence, temporal and paired metrics |
| `trafficrag/evaluation/reports` | Ground-truth joins and CSV/JSON outputs |
| `trafficrag/experiments/catalog` | Paired motion/grounding recipes |

Caches and prediction records are separate from downloaded foundation-model weights. Each batch records the trained motion checkpoint and memory checksum in its run identity.
