# Workflow guides

## Overview

- [Runtime packages](architecture.md)

## Data

- [Segment annotations](data/annotations.md)
- [Recording and query identities](data/identities.md)
- [Group-preserving partitions](data/partitions.md)
- [Stage artifact cache](data/stage-cache.md)
- [Video timestamps and frame preparation](data/video-decoding.md)

## Evaluation

- [Recording-level confidence intervals](evaluation/bootstrap.md)
- [Inspecting retrieved evidence](evaluation/evidence.md)
- [Paired grounding comparisons](evaluation/paired.md)
- [Event presence metrics](evaluation/presence.md)
- [Temporal localization metrics](evaluation/temporal.md)
- [Red-light events](evaluation/domain-red_light.md)
- [Left-side blind-spot events](evaluation/domain-blind_left.md)
- [Right-side blind-spot events](evaluation/domain-blind_right.md)

## Models

- [Captioning and temporal generation](grounding/captioning.md)
- [VideoMAE motion lifecycle](grounding/motion.md)
- [Local pretrained artifacts](grounding/offline.md)
- [ModernBERT caption retrieval](grounding/retrieval.md)

## Pipeline

- [Batch grounding and continuation](grounding/batch-resume.md)
- [Motion proposals](grounding/proposals.md)
- [Adaptive boundary refinement](grounding/refinement.md)
- [Semantic verification](grounding/verification.md)

## Preparing motion windows and pretrained models

Keep source recordings in separate train, validation, memory, and test partitions. Each full-video JSONL row needs `id`, `video_id`, `video`, `domain`, `group`, and `target`. The target is `[start_seconds, end_seconds]` for a violation and `null` otherwise. Paths are resolved relative to the manifest. Recording groups keep related clips together.

```bash
python traffic.py audit-videos --manifests data/train.jsonl data/validation.jsonl data/memory.jsonl data/test.jsonl --names train validation memory test --output outputs/data-audit.json --require-disjoint
python traffic.py motion-windows --manifest data/train.jsonl --domain red-light --output data/motion-train --window 2 --stride 1
python traffic.py motion-windows --manifest data/validation.jsonl --domain red-light --output data/motion-validation --window 2 --stride 1
python traffic.py prepare-models --output weights --sha256
```

A window is positive when at least half of its duration overlaps the annotated event. Zero-overlap windows are negative. Intermediate windows are recorded separately and omitted by default. `--positive-overlap`, `--negative-overlap`, and `--keep-ambiguous` select another labeling policy. Video audits inspect presentation timestamps, segment boundaries, source identities, and optional file hashes through `--scan-frames` and `--content-hashes`.

`prepare-models` downloads complete pretrained artifacts into `weights/motion`, `weights/vlm`, and `weights/text_encoder`. It checks the Transformer weight index and all referenced shards, writes a file inventory, and optionally records SHA256 digests. Runtime model loading also downloads missing pretrained files into the Hugging Face cache. If downloads are unavailable on the execution host, prepare these directories on another host and copy them intact.

For manual preparation, copy `distill/vit_s_k710_dl_from_giant.pth` from [VideoMAE2](https://huggingface.co/OpenGVLab/VideoMAE2/tree/main/distill) into `weights/motion/distill`. Copy the configuration, tokenizer, processor files, weight index, and every referenced tensor shard from [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/main) into `weights/vlm`, and from [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base/tree/main) into `weights/text_encoder`. Use the local paths with `--offline` in training, memory construction, and grounding. The domain-trained motion checkpoint is produced by the training command below.

## Motion training and held-out evaluation

```bash
python traffic.py fit-motion --train-manifest data/motion-train/segments.jsonl --validation-manifest data/motion-validation/segments.jsonl --domain red-light --output outputs/motion --weights weights/motion --offline --accumulation 4
python traffic.py fit-motion --train-manifest data/motion-train/segments.jsonl --validation-manifest data/motion-validation/segments.jsonl --domain red-light --output outputs/motion --weights weights/motion --offline --accumulation 4 --resume outputs/motion/last.pt
python traffic.py threshold --validation outputs/motion/validation-best.json --output outputs/threshold.json --metric f1
python traffic.py evaluate-motion --checkpoint outputs/motion/best.pt --manifest data/motion-test/segments.jsonl --output outputs/motion-test.json --threshold-selection outputs/threshold.json
```

`fit-motion` uses CUDA mixed precision, sample-weighted gradient accumulation, and held-out validation. The default objective is binary cross entropy. Focal loss, class-weighted loss, and balanced sampling are explicit options. Choose either weighted loss or balanced sampling when adjusting class balance. The validation metric chooses `best.pt`, while `last.pt` stores model, optimizer, scaler, random generator, sampler, and history state. Resume in the original output directory using the same data and training settings. A resumed model reads its trained weights directly from the checkpoint.

The threshold command uses saved validation probabilities for segment classification. `evaluate-motion` applies the selected segment threshold and reports per-video metrics and recording-group bootstrap intervals. Full-video motion proposals retain the threshold and smoothing policy in the grounding configuration. `evaluate-motion --split validation` can create the validation probability artifact from an existing checkpoint.

## Caption memory construction and retrieval

Memory rows contain an annotated `start`, `end`, and binary `label`. Add `caption` to use an existing description, or let Qwen3-VL generate the description. Caption-only JSONL inputs for `--captions` and `--queries` use `id`, `video_id`, `group`, `domain`, `caption`, `label`, `start`, and `end`.

```bash
python traffic.py memory-build --manifest data/memory.jsonl --output outputs/memory --vlm weights/vlm --text-encoder weights/text_encoder --offline --exclude-manifests data/test.jsonl
python traffic.py memory-audit --memory outputs/memory --output outputs/memory-audit.json --ks 1 3 5 10
python traffic.py memory-subset --memory outputs/memory --output outputs/memory-small --maximum-rows 1000 --maximum-per-video 20 --balanced
python traffic.py memory-merge --inputs outputs/memory-a outputs/memory-b --output outputs/memory-merged
python traffic.py retrieval-query --memory outputs/memory --queries data/query-captions.jsonl --output outputs/query-report.json --offline
python traffic.py memory-export --memory outputs/memory --output outputs/memory.npz
```

Caption generation resumes with `memory-build --resume`. The journal checks source identities, model configuration, and prompts before continuing. Text embeddings use a content-addressed cache, and completed memories contain normalized float32 embedding shards with checksums. `memory-export` creates the NPZ file and model metadata consumed by `batch` and `infer`.

Retrieval audits and query evaluation exclude entries with the same recording group or video identity. Query evaluation also excludes the same segment ID. Reports include TopK label precision, reciprocal rank, vote classification, neighbor diversity, and difficult query pairs. Memory subsets alternate among source videos before applying an optional balanced class budget. Merge accepts compatible encoders and domains, coalesces identical entries, and rejects conflicting IDs or embeddings.

## Grounding inspection

```bash
python traffic.py report-detailed --predictions outputs/grounding/predictions.jsonl --output outputs/grounding/detailed.json
python traffic.py audit-frames --video data/videos/example.mp4 --start 2 --end 6 --frames 30 --output outputs/frame-sampling.json
python traffic.py audit-responses --responses outputs/recorded-responses.jsonl --output outputs/response-audit.json
```

The detailed grounding report measures proposal recall, motion-only and verified candidate selection, crop coverage, final boundary error, and the change caused by refinement. Every positive record receives a stage outcome at temporal IoU 0.5. Retrieved labels are checked against the recorded semantic vote. Response audit inputs contain `id`, `duration`, and raw `response` text, with timestamps relative to the video crop. Frame audits record the actual presentation timestamps supplied to the models.
