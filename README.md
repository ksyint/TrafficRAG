# TrafficRAG

**TrafficRAG: Temporal Grounding for Traffic Violations via Retrieval-Augmented Generation**
Soo Yong Kim, Joonyoung Kim, Jaewon Lee, Seong Wook Lee, Tae-San Eom, Jeonghun Chae. WACV Workshops 2026, pp. 66–74.

[Paper](https://openaccess.thecvf.com/content/WACV2026W/RWS/html/Kim_TrafficRAG_Temporal_Grounding_for_Traffic_Violations_via_Retrieval-Augmented_Generation_WACVW_2026_paper.html)

A configurable traffic-grounding system built from motion proposals, caption retrieval, semantic verification, and crop-relative VLM refinement.

## Three-stage inference

1. **Proposal:** score overlapping 2-second segments at 1-second stride, apply Gaussian smoothing, and group consecutive probabilities above the motion threshold.
2. **Verification:** caption each candidate, retrieve cosine-nearest caption embeddings, and combine their positive-label ratio with motion confidence.
3. **Refinement:** choose the strongest candidate, pad it by `base_padding + adaptive_padding * (1 - motion_confidence)`, and translate the backend's crop-relative prediction to video time.

The `TrafficRAG.Config` dataclass composes three stage configurations. `trafficrag/systems/grounding/` implements the operators; `trafficrag/backends/` registers model-output providers. `MotionSystem` separately manages binary encoder optimization, checkpointing, and segment scoring.

## Install and prepare encoders

Use a CUDA-enabled PyTorch build. Neural runners accept `cuda` or `cuda:N`; caption retrieval and timestamp arithmetic use host NumPy arrays.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-models.txt
```

Train one binary encoder per domain: `red-light`, `blind-spot-left`, or `blind-spot-right`. An NPZ training set contains binary `labels` of shape `N` and either cached `features` (`N,D`) or video `pixels` (`N,T,C,H,W`).

```bash
python train.py --data data/red_light_features.npz --domain red-light \
  --output outputs/red_light --device cuda
python train.py --data data/red_light_pixels.npz --encoder /path/to/compatible-videomae \
  --domain red-light --device cuda --output outputs/red_light_full
```

The encoder adapter accepts a Hugging Face VideoMAE model with `pixel_values` input and `last_hidden_state` output. Use pixel normalization, frame sampling, resolution, and positional embeddings matching the checkpoint. The provided training configuration selects 50 epochs, batch size 1 and learning rate `1e-6`.

## Build caption memory

Prepare held-out segments separated from the motion-training and query videos. Each JSONL row contains an ID, caption, binary label, and an optional precomputed embedding:

```json
{"id":"kb_clip_001", "caption":"The vehicle stops before the line at a red signal.", "label":0, "embedding":[0.1,0.9]}
```

```bash
python build_kb.py --input data/kb.jsonl --output data/kb.npz --exclude_ids data/train_query_ids.txt
python build_kb.py --input data/kb_captions.jsonl --encoder /path/to/modernbert \
  --output data/kb.npz --device cuda
```

Use the same captioning prompt, text encoder and pooling for memory entries and queries. The ModernBERT adapter uses attention-masked mean pooling. `--exclude_ids` checks exact source-ID separation.

## Ground a video

The recorded backend accepts externally computed captions, embeddings and local timestamps in a query manifest:

```json
{
  "duration":2.0,
  "motion_scores":[0.9],
  "captions":[{"start":0.0,"end":2.0,"caption":"A vehicle crosses at red.","embedding":[1.0,0.0]}],
  "refinements":[{"start":0.0,"end":2.0,"local_interval":[0.4,1.2]}]
}
```

Provide one exact caption record for each candidate interval and one refinement record for the selected crop.

```bash
python inference.py --kb data/kb.npz --query data/query.json \
  --output outputs/prediction.json --device cuda
```

For CUDA motion scoring, replace `motion_scores` with `motion_data`, a `.npy` feature/pixel array containing one row per segment, and pass `--motion_checkpoint outputs/red_light/last.pt`. Relative paths resolve from the manifest directory.

A live backend is selected with `--backend my_backend:create_backend`. The factory receives the query manifest with its `device` field set to the selected CUDA device and returns an object implementing:

```python
caption(interval) -> str
embed(caption) -> numpy.ndarray
ground(crop, candidate) -> (local_start, local_end) or None
```

The backend handles video decoding, CUDA VLM invocation, and crop-relative prediction. The candidate object carries captions, retrieved IDs/labels/similarities, and motion/fusion scores. Final boundaries are validated against the crop and video duration.

## Paired experiment recipes

The catalog contains **240 JSON recipes**, each pairing a grounding configuration with the motion-training configuration used for that experiment:

| Recipe coordinate | Values |
| --- | --- |
| Retrieved neighbors | 1, 3, 5, 10, 20 |
| Semantic fusion weight | 0.2, 0.4, 0.6 |
| Motion threshold | 0.2, 0.3, 0.4, 0.5 |
| Base crop padding | 0.25, 0.50 seconds |
| Binary model learning rate | `1e-6`, `5e-6` |

`train.py --profile` consumes the recipe's `motion` section. `inference.py --profile` consumes its `pipeline` section. Each recipe is parsed into the native dataclass configurations.

```bash
python -m trafficrag.experiments
python -m trafficrag.experiments --validate-all
python train.py --profile configs/catalog/traffic/retrieval/k10/f04/m03/p050/lr1e6.json \
  --data data/red_light_features.npz --domain red-light --device cuda --output outputs/recipe_motion
python inference.py --profile configs/catalog/traffic/retrieval/k10/f04/m03/p050/lr1e6.json \
  --kb data/kb.npz --query data/query.json --device cuda --output outputs/recipe_grounding.json
python inference.py --profile configs/catalog/traffic/retrieval/k10/f04/m03/p050/lr1e6.json --dry-run
python -m trafficrag.experiments.build
```

Dry-run and catalog validation inspect settings without loading encoders or running inference. `configs/default.yaml` remains available for directly configuring the three grounding stages.

## Evaluation

Create JSONL rows with `prediction` and `target`, each `[start,end]` or `null`:

```bash
python eval.py --data data/predictions.jsonl
```

Classification F1 measures event presence. Temporal IoU averages over positive ground-truth videos and assigns zero to missed detections. Prediction JSON includes smoothed motion scores, candidate retrieval evidence, the selected crop and final video timestamps.
