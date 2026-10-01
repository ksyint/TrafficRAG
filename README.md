# TrafficRAG

[Paper](https://openaccess.thecvf.com/content/WACV2026W/RWS/html/Kim_TrafficRAG_Temporal_Grounding_for_Traffic_Violations_via_Retrieval-Augmented_Generation_WACVW_2026_paper.html)

**TrafficRAG: Temporal Grounding for Traffic Violations via Retrieval-Augmented Generation**
WACV Workshops 2026, pp. 66–74.

A raw-video pipeline using **VideoMAE V2 Small** for motion proposals, **Qwen3-VL-8B-Instruct** for captioning and temporal refinement, and frozen **ModernBERT-base** for caption retrieval. Model weights download automatically when their stage first runs.

## Install

Use Python 3.10+ with CUDA-enabled PyTorch:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model runners require `cuda` or `cuda:N`. Video decoding uses PyAV. VideoMAE optimization and Qwen generation use CUDA BF16. The Qwen model is placed entirely on the selected GPU. Provision GPU memory for its approximately 9-billion-parameter checkpoint, ModernBERT, and video activations. Inference releases the motion model before allocating Qwen.

## Automatic and local weight loading

| Runtime stage | Published initialization | Loader |
| --- | --- | --- |
| Binary motion encoder | [OpenGVLab/VideoMAE2](https://huggingface.co/OpenGVLab/VideoMAE2), `distill/vit_s_k710_dl_from_giant.pth` | `hf_hub_download`, strict ViT-S parameter loading |
| Captions and boundary refinement | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | `Qwen3VLForConditionalGeneration` and `AutoProcessor` |
| Caption vectors | [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) | frozen `AutoModel`, attention-mask mean pooling |

The VideoMAE release is the official Kinetics-710 distilled **Small** checkpoint: 384-dimensional features, 12 transformer layers, six heads, tubelet size two, and spatial patch size 16. Its released classification head is replaced with one learned binary head per traffic domain. Every backbone parameter loads strictly. Fine-tuning uses 30 uniformly sampled frames at 350×350, sinusoidal token positions, ImageNet normalization, and activation checkpointing.

`--cache-dir weights/cache` redirects the Hugging Face cache. Otherwise it uses `~/.cache/huggingface/hub/`. The corresponding cache directories are `models--OpenGVLab--VideoMAE2`, `models--Qwen--Qwen3-VL-8B-Instruct`, and `models--answerdotai--ModernBERT-base`. For named local destinations:

```bash
hf download OpenGVLab/VideoMAE2 distill/vit_s_k710_dl_from_giant.pth --local-dir weights/videomaev2
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir weights/qwen3-vl
hf download answerdotai/ModernBERT-base --local-dir weights/modernbert
```

Use `traffic.py train --weights weights/videomaev2/distill/vit_s_k710_dl_from_giant.pth --offline`, and pass `--vlm weights/qwen3-vl --text-encoder weights/modernbert --offline` to KB construction and inference. The motion checkpoint produced by training contains the entire fine-tuned encoder and binary head, so inference restores it without downloading its foundation weights again. The downloadable VideoMAE weights provide action-recognition initialization. The traffic-domain checkpoint is trained on the annotated segments supplied below.

## Prepare dashcam videos and annotations

Start with acquired dashcam recordings and human-reviewed event intervals. Use a separate annotation manifest for each domain. The criteria are stop-line crossing during a red signal (`red-light`), or a pedestrian/road user on a collision trajectory from the named side (`blind-spot-left` / `blind-spot-right`). Mark positive intervals and visually similar safe intervals with label 0. Include both labels in each domain's training and held-out data.

```text
data/
  videos/
    dashcam_001.mp4
    dashcam_002.mp4
  annotations_red_light.jsonl
  red_light/
    motion_train.jsonl
    validation.jsonl
    kb.jsonl
    train_query_ids.txt
```

Each annotation uses source-video seconds and a stable `video_id`. Reuse that ID for every segment from the same recording:

```json
{"video_id":"dashcam_001","video":"videos/dashcam_001.mp4","start":1.0,"end":3.0,"label":1,"domain":"red-light"}
{"video_id":"dashcam_001","video":"videos/dashcam_001.mp4","start":5.0,"end":7.0,"label":0,"domain":"red-light"}
```

Split before motion training or KB construction. The helper groups all segments by `video_id`, checks path/ID consistency, and emits separate training, validation/query, and KB manifests:

```bash
python traffic.py prepare --annotations data/annotations_red_light.jsonl \
  --output data/red_light --validation-fraction 0.15 --kb-fraction 0.15 --seed 42
python traffic.py train --data data/red_light/motion_train.jsonl --domain red-light \
  --output outputs/red_light --device cuda
python traffic.py build-kb --input data/red_light/kb.jsonl --domain red-light \
  --exclude_ids data/red_light/train_query_ids.txt --output data/kb_red_light.npz --device cuda
```

Keep recordings from the same drive/driver in one partition when preparing the source manifests. Use validation videos for grounding evaluation. Their positive annotation boundaries supply `target` intervals for `traffic.py evaluate`. Preserve multiple-event annotations when preparing the ground truth and choose the intended target event for each single-interval query.

Native preprocessing decodes actual presentation timestamps, uniformly samples each interval, resizes RGB frames to 350×350, and applies ImageNet mean/std before VideoMAE. Qwen separately samples its crop with timestamp metadata and its own processor. No extracted frame folder is required. For codec conversion, install FFmpeg and run:

```bash
ffmpeg -i dashcam.mov -map 0:v:0 -an -c:v libx264 -pix_fmt yuv420p data/videos/dashcam.mp4
```

## Train domain-specific motion models

Prepare JSONL with a video path, interval in seconds, binary label, and optional domain. Relative paths resolve from the manifest directory:

```json
{"video":"videos/clip_001.mp4","start":1.0,"end":3.0,"label":1,"domain":"red-light"}
{"video":"videos/clip_002.mp4","start":0.0,"end":2.0,"label":0,"domain":"red-light"}
```

```bash
python traffic.py train --data data/motion_train.jsonl --domain red-light \
  --cache-dir weights/cache --output outputs/red_light --device cuda
python traffic.py train --data data/motion_train.jsonl --domain blind-spot-left \
  --cache-dir weights/cache --output outputs/blind_left --device cuda
python traffic.py train --data data/motion_train.jsonl --domain blind-spot-right \
  --cache-dir weights/cache --output outputs/blind_right --device cuda
```

Each run fine-tunes the full released VideoMAE backbone with binary cross-entropy. The default schedule is 50 epochs, batch size one, and learning rate `1e-6`. `MotionSystem` manages training, strict checkpoint restoration, and raw-video segment scoring. Each output directory receives `last.pt` and `metrics.json`.

## Build caption memory

Use held-out segments, including hard negatives, separated from motion-training and query videos. Build one memory per domain:

```json
{"id":"kb_001_segment0","video_id":"kb_001","video":"videos/kb_001.mp4","start":0.0,"end":2.0,"label":0}
{"id":"kb_002_segment0","video_id":"kb_002","video":"videos/kb_002.mp4","start":2.0,"end":4.0,"label":1}
```

```bash
python traffic.py build-kb --input data/kb_red_light.jsonl --domain red-light \
  --output data/kb_red_light.npz --exclude_ids data/train_query_ids.txt \
  --cache-dir weights/cache --device cuda
```

Qwen captions each segment. Frozen ModernBERT embeds those captions with attention-masked mean pooling. Optional `caption` fields reuse previously generated captions. `--exclude_ids` checks `video_id`, falling back to segment `id` when absent. The output includes the memory NPZ and a `.models.json` record of the domain, checkpoint paths, pooling, and similarity. Use the same text-encoder path/ID and caption model for KB and queries.

## Ground a raw video

```bash
python traffic.py infer --video data/query.mp4 --domain red-light \
  --motion_checkpoint outputs/red_light/last.pt --kb data/kb_red_light.npz \
  --cache-dir weights/cache --output outputs/prediction.json --device cuda
```

1. VideoMAE scores overlapping two-second windows at one-second stride. Gaussian smoothing and threshold `0.3` produce contiguous candidate intervals.
2. Qwen captions each candidate. ModernBERT retrieves ten nearest KB captions. The positive-label ratio contributes `0.4` and motion confidence contributes `0.6` to verification.
3. The strongest candidate is padded by `0.5 + (1 - motion_confidence)` seconds per side. Qwen inspects this crop and returns JSON boundaries relative to the crop. Timestamp metadata preserves decoded frame times, and validated local boundaries are converted to full-video seconds.

The same `VideoLanguageSystem` and caption prompt serve KB construction and query inference. The output JSON includes raw stage evidence, smoothed scores, retrieved IDs, selected crop, and final interval. A VLM response declaring no violation produces a null interval.

`--query recorded.json` remains available to inspect previously captured motion/caption/refinement outputs. Raw-video inference uses the real model stages by default through `--video`.

## Paired experiment recipes

The **240 JSON recipes** pair motion optimization with grounding settings:

| Coordinate | Values |
| --- | --- |
| Retrieved neighbors | 1, 3, 5, 10, 20 |
| Semantic fusion weight | 0.2, 0.4, 0.6 |
| Motion threshold | 0.2, 0.3, 0.4, 0.5 |
| Base crop padding | 0.25, 0.50 seconds |
| Binary-model learning rate | `1e-6`, `5e-6` |

`traffic.py train --profile` applies the motion section to full VideoMAE fine-tuning. `traffic.py infer --profile` applies the pipeline section to model-backed raw-video inference. Both deserialize native dataclass configurations.

```bash
python traffic.py train --profile configs/catalog/traffic/retrieval/k10/f04/m03/p050/lr1e6.json \
  --data data/motion_train.jsonl --domain red-light --device cuda --output outputs/recipe_motion
python traffic.py infer --profile configs/catalog/traffic/retrieval/k10/f04/m03/p050/lr1e6.json \
  --video data/query.mp4 --motion_checkpoint outputs/recipe_motion/last.pt \
  --kb data/kb_red_light.npz --device cuda --output outputs/recipe_grounding.json
python traffic.py infer --profile configs/catalog/traffic/retrieval/k10/f04/m03/p050/lr1e6.json --dry-run
python traffic.py recipes --validate-all
python traffic.py build-recipes
```

Dry-run and catalog validation inspect settings without loading models. `TrafficRAG.Config` composes proposal, verification, and refinement stages in `trafficrag/pipeline/temporal/grounding.py`. Video decoding and motion optimization share `trafficrag/models/video/motion.py`, while `trafficrag/models/multimodal/backends.py` owns captioning, embedding, and recorded evidence. `traffic.py` exposes each operation as a subcommand.

## Evaluate intervals

Provide JSONL rows containing `prediction` and `target`, each `[start,end]` or `null`:

```bash
python traffic.py evaluate --data data/predictions.jsonl
```

Classification F1 measures event presence. Temporal IoU averages over positive ground-truth videos and assigns zero to missed detections.

## Manifest batches and benchmark reports

The data and runtime packages support stable query identities, drive-group partitions, reusable stage artifacts, and resumable CUDA grounding.

```bash
python traffic.py manifest --manifest data/red_light/queries.jsonl --domain red-light
python traffic.py batch --manifest data/red_light/queries.jsonl --motion-checkpoint outputs/red_light/last.pt --kb data/kb_red_light.npz --domain red-light --output outputs/red_light_batch --device cuda
python traffic.py batch --manifest data/red_light/queries.jsonl --motion-checkpoint outputs/red_light/last.pt --kb data/kb_red_light.npz --domain red-light --output outputs/red_light_batch --device cuda --resume
python traffic.py report --predictions outputs/red_light_batch/predictions.jsonl --output reports/red_light --bootstrap 1000
```

The batch runner completes motion scoring before allocating Qwen and ModernBERT. It persists completed records individually and checks the input/model identity when resumed. Retrieved captions and reviewed labels are included in the refinement prompt. Reports retain event-presence metrics, temporal IoU, boundary errors, recording-level intervals, and retrieval evidence.

[Workflow guides](docs/index.md) cover manifest preparation, model caches, domain recipes, and paired evaluation. [Runtime packages](docs/architecture.md) maps the nested source directories to those stages.
