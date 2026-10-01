# TrafficRAG

**TrafficRAG: Temporal Grounding for Traffic Violations via Retrieval-Augmented Generation**
Soo Yong Kim, Joonyoung Kim, Jaewon Lee, Seong Wook Lee, Tae-San Eom, Jeonghun Chae. WACV Workshops 2026, pp. 66–74.

[Paper](https://openaccess.thecvf.com/content/WACV2026W/RWS/html/Kim_TrafficRAG_Temporal_Grounding_for_Traffic_Violations_via_Retrieval-Augmented_Generation_WACVW_2026_paper.html)

Independent implementation of the three-stage traffic grounding algorithm. The repository includes binary motion training, caption knowledge-base construction, cosine retrieval, semantic verification, adaptive crop selection and local-to-global timestamp conversion. The original proprietary dataset, trained encoder checkpoints and VLM outputs are not included; no benchmark reproduction is claimed.

## Pipeline

1. Split the video into overlapping 2-second windows with 1-second stride. Smooth binary motion probabilities and group consecutive scores strictly greater than 0.3.
2. Caption each candidate, encode its text, retrieve the top 10 KB entries, and compute their positive-label ratio. Fuse with motion using `0.4 * semantic + 0.6 * motion`.
3. Select the highest scoring candidate. Add `0.5 + (1 - mean_motion)` seconds of padding on each side, clipped to the video. A grounding backend returns crop-relative timestamps, which are validated and converted to the original timeline.

`configs/default.yaml` controls smoothing bandwidth, retrieval count and temporal parameters. Retrieval uses cosine-normalized text features and divides by the actual returned neighbor count if the KB has fewer than K entries. The last full window covers a fractional-duration tail. Candidate intervals use the full extent of their contiguous segment run.

## Installation and CPU smoke

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --epochs 5 --batch_size 16 --output outputs/motion
python inference.py --smoke --output outputs/smoke.json
python -m pytest -q
```

The training smoke performs binary cross-entropy optimization on generated features. The inference smoke uses explicitly synthetic caption/embedding/grounding callbacks to verify algorithmic execution. It does not run VideoMAE or a VLM.

## Motion models

Train a separate binary model for each domain (`red-light`, `blind-spot-left`, `blind-spot-right`). NPZ data contains either `features` (`N,D`) for head-only training, or `pixels` (`N,T,C,H,W`) with `--encoder` for full compatible VideoMAE fine-tuning, plus `labels` (`N`, 0/1).

```bash
python train.py --data data/red_light_features.npz --domain red-light --output outputs/red_light
pip install -r requirements-models.txt
python train.py --data data/red_light_pixels.npz --encoder /path/to/compatible-videomae \
  --domain red-light --device cuda --output outputs/red_light_full
```

The optional adapter expects a Hugging Face model exposing `pixel_values` and `last_hidden_state`. Supply/convert your VideoMAE V2 checkpoint accordingly. Pixel normalization, sampling and resize must match that checkpoint. The paper uses 30 frames at 350×350; a backbone's positional embeddings and tubelet layout must support those dimensions. The adapter does not silently resize learned positional embeddings. The provided training configuration uses 50 epochs, batch 1 and LR 1e-6. Domain-specific data and matching checkpoint conversion are external requirements.

## Knowledge base

Prepare held-out annotated segments, separate from motion training and query videos. Each JSONL record has `id`, `caption`, `label`, and optionally `embedding`:

```json
{"id":"kb_clip_001", "caption":"The vehicle stops before the line at a red signal.", "label":0, "embedding":[0.1,0.9]}
```

```bash
python build_kb.py --input data/kb.jsonl --output data/kb.npz --exclude_ids data/train_query_ids.txt
python build_kb.py --input data/kb_captions.jsonl --encoder /path/to/modernbert --output data/kb.npz
```

The optional frozen ModernBERT adapter uses attention-masked mean pooling. Apply the **same encoder and pooling** to KB and query captions. Use the same captioning VLM/prompt across both sets. `--exclude_ids` checks exact video-ID overlap; source-level separation remains part of dataset preparation.

## Inference with real outputs

The default backend replays previously computed model outputs from a JSON manifest. Matching is exact: each candidate must have one caption/embedding entry and the selected crop must have one refinement. This is useful for auditing and repeatable evaluation of expensive VLM calls.

```json
{
  "duration":2.0,
  "motion_scores":[0.9],
  "captions":[{"start":0.0,"end":2.0,"caption":"A vehicle crosses at red.","embedding":[1.0,0.0]}],
  "refinements":[{"start":0.0,"end":2.0,"local_interval":[0.4,1.2]}]
}
```

```bash
python inference.py --kb data/kb.npz --query data/query.json --output outputs/prediction.json
```

To score a trained motion model, replace `motion_scores` with `motion_data` (a `.npy` feature/pixel array, one row per window) and pass `--motion_checkpoint outputs/red_light/last.pt`. Relative paths resolve from the query manifest directory.

For live captioning and grounding, pass `--backend my_backend:create_backend`. This imports your factory with the query manifest and expects an object implementing:

```python
caption(interval) -> str
embed(caption) -> numpy.ndarray  # D-dimensional KB-compatible embedding
ground(crop, candidate) -> (local_start, local_end) or None
```

The backend owns video decoding and VLM invocation; `Interval` values are in seconds. `candidate` contains caption, retrieval neighbors and fused/motion scores. The core validates final boundaries and exposes intermediate scores in its JSON output. A VLM returning no event may return `None`. No remote model endpoint is called automatically.

## Evaluation

Create JSONL rows with `prediction` and `target` as `[start,end]` or `null`:

```bash
python eval.py --data data/predictions.jsonl
```

Classification F1 uses event presence. Mean temporal IoU averages over all positive ground-truth videos, assigning zero to missed detections. This explicit denominator makes offline experiments comparable within this repository. Tests cover thresholding, window grouping, cosine retrieval, semantic selection, motion-dependent padding, empty proposals and timestamp validation.

## Citation

```bibtex
@InProceedings{Kim_2026_WACV,
  author = {Kim, Soo Yong and Kim, Joonyoung and Lee, Jaewon and Lee, Seong Wook and Eom, Tae-San and Chae, Jeonghun},
  title = {TrafficRAG: Temporal Grounding for Traffic Violations via Retrieval-Augmented Generation},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
  month = {March},
  year = {2026},
  pages = {66--74}
}
```
