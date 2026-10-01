"""Motion data preparation, validation training, and detailed grounding reports."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from trafficrag.data.records import DOMAINS, VideoManifest
from trafficrag.data.decoding import inspect_video, manifest_video_report, sample_report
from trafficrag.data.windows import WindowLabeling, annotation_windows, window_summary
from trafficrag.data.leakage import source_summary
from trafficrag.models.motion import MotionSystem, VideoMAEEncoder, VideoSource
from trafficrag.models.responses import response_audit
from trafficrag.experiments.recipes import read_settings
from trafficrag.training.engine import MotionTrainer, TrainingOptions
from trafficrag.evaluation.classification import classification_summary
from trafficrag.evaluation.temporal import (
    temporal_report,
    proposal_report,
    duration_groups,
    refinement_change,
    stage_attribution,
    retrieval_agreement,
)
from trafficrag.evaluation.curves import select_threshold, bootstrap_binary, bootstrap_temporal
from trafficrag.retrieval.storage import atomic_json


def motion_windows_cli():
    parser = argparse.ArgumentParser(
        description="Create motion segment labels from full-video event boundaries."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--domain", choices=DOMAINS)
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--stride", type=float, default=1.0)
    parser.add_argument("--positive-overlap", type=float, default=0.5)
    parser.add_argument("--negative-overlap", type=float, default=0.0)
    parser.add_argument("--keep-ambiguous", action="store_true")
    args = parser.parse_args()
    manifest = VideoManifest.read(args.manifest, args.domain, require_target=True)
    settings = WindowLabeling(
        args.window, args.stride, args.positive_overlap, args.negative_overlap, args.keep_ambiguous
    )
    durations = {}

    def duration_of(path):
        key = str(path)
        if key not in durations:
            durations[key] = VideoSource(path).duration
        return durations[key]

    segments, omitted = annotation_windows(manifest, duration_of, settings)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    segments.write(output / "segments.jsonl")
    report = {
        "settings": asdict(settings),
        "summary": window_summary(segments),
        "ambiguous_windows": omitted,
    }
    atomic_json(output / "windows.json", report)
    print(json.dumps(report["summary"], indent=2))


def video_audit_cli():
    parser = argparse.ArgumentParser(
        description="Inspect source-video timing and train/memory/evaluation overlap."
    )
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scan-frames", action="store_true")
    parser.add_argument("--content-hashes", action="store_true")
    parser.add_argument("--require-disjoint", action="store_true")
    args = parser.parse_args()
    names = args.names or [Path(path).stem for path in args.manifests]
    if len(names) != len(args.manifests) or len(set(names)) != len(names):
        raise ValueError("Each manifest needs a distinct partition name")
    partitions = {name: VideoManifest.read(path) for name, path in zip(names, args.manifests)}
    report = source_summary(partitions, args.content_hashes)
    report["videos"] = {
        name: manifest_video_report(manifest, args.scan_frames)
        for name, manifest in partitions.items()
    }
    atomic_json(args.output, report)
    print(
        json.dumps({"partitions": report["partitions"], "disjoint": report["disjoint"]}, indent=2)
    )
    if args.require_disjoint and not report["disjoint"]:
        raise ValueError("Source video audit found overlapping partitions")


def motion_fit_cli():
    parser = argparse.ArgumentParser(
        description="Train and resume a domain-specific VideoMAE model with held-out validation."
    )
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights")
    parser.add_argument("--cache-dir")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=350)
    parser.add_argument("--precision", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--weighted-loss", action="store_true")
    parser.add_argument("--loss", choices=("bce", "focal"), default="bce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--metric", choices=("f1", "accuracy", "balanced_accuracy"), default="f1")
    parser.add_argument("--patience", type=int, default=0)
    args = parser.parse_args()
    config = MotionSystem.Config(**read_settings(args.config))
    encoder = VideoMAEEncoder.Config(
        weights=args.weights or "",
        cache_dir=args.cache_dir or "",
        offline=args.offline,
        frames=args.frames,
        image_size=args.image_size,
    )
    system = MotionSystem(config, args.domain, encoder, args.device)
    options = TrainingOptions(
        precision=args.precision,
        accumulation=args.accumulation,
        workers=args.workers,
        balanced_sampling=args.balanced_sampling,
        weighted_loss=args.weighted_loss,
        loss=args.loss,
        focal_gamma=args.focal_gamma,
        checkpoint_metric=args.metric,
        patience=args.patience,
    )
    trainer = MotionTrainer(
        system, args.train_manifest, args.validation_manifest, args.output, options
    )
    print(json.dumps(trainer.run(args.resume), indent=2))


def read_grounding_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Grounding predictions are empty")
    seen = set()
    for row in rows:
        identity = row.get("id")
        if identity is None or identity in seen:
            raise ValueError("Detailed grounding reports require unique record IDs")
        seen.add(identity)
        if "target" not in row:
            raise ValueError("Detailed grounding reports require target interval or null")
    return rows


def grounding_report_cli():
    parser = argparse.ArgumentParser(
        description="Analyze event detection, proposal recall, and boundary refinement."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = read_grounding_rows(args.predictions)
    probabilities = [float(row.get("prediction", row.get("interval")) is not None) for row in rows]
    labels = [int(row["target"] is not None) for row in rows]
    groups = [row.get("group", row.get("video_id", row["id"])) for row in rows]
    report = {
        "classification": classification_summary(probabilities, labels),
        "temporal": temporal_report(rows),
        "proposal": proposal_report(rows),
        "duration_groups": duration_groups(rows),
        "refinement": refinement_change(rows),
        "stage_attribution": stage_attribution(rows),
        "retrieval_agreement": retrieval_agreement(rows),
        "classification_intervals": bootstrap_binary(
            probabilities, labels, groups, args.bootstrap, args.seed
        ),
        "temporal_interval": bootstrap_temporal(rows, args.bootstrap, args.seed),
    }
    domains = sorted({row.get("domain", "unspecified") for row in rows})
    report["domains"] = {
        domain: temporal_report([row for row in rows if row.get("domain", "unspecified") == domain])
        for domain in domains
    }
    atomic_json(args.output, report)
    print(json.dumps({"videos": len(rows), "output": args.output}, indent=2))


def threshold_cli():
    parser = argparse.ArgumentParser(
        description="Select a motion threshold from saved validation probabilities."
    )
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--metric", choices=("f1", "accuracy", "balanced_accuracy", "mcc", "recall"), default="f1"
    )
    parser.add_argument("--maximum-fpr", type=float)
    args = parser.parse_args()
    values = json.loads(Path(args.validation).read_text())
    if values.get("split") != "validation":
        raise ValueError("Threshold selection requires a labeled validation artifact")
    report = select_threshold(
        values["probabilities"], values["labels"], args.metric, args.maximum_fpr
    )
    report["source"] = str(Path(args.validation).resolve())
    atomic_json(args.output, report)
    print(json.dumps(report["selected"], indent=2))


def response_audit_cli():
    parser = argparse.ArgumentParser(
        description="Check recorded VLM grounding responses against their crop durations."
    )
    parser.add_argument("--responses", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line) for line in Path(args.responses).read_text().splitlines() if line.strip()
    ]
    report = response_audit(rows)
    atomic_json(args.output, report)
    print(
        json.dumps(
            {"records": report["records"], "valid_fraction": report["valid_fraction"]}, indent=2
        )
    )


def frame_audit_cli():
    parser = argparse.ArgumentParser(
        description="Record the actual presentation timestamps used for a video crop."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = VideoSource(args.video)
    report = sample_report(source, args.start, args.end or source.duration, args.frames)
    report["video"] = inspect_video(args.video)
    atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "frames": report["requested_frames"],
                "unique_timestamps": report["unique_timestamps"],
            },
            indent=2,
        )
    )


def prepare_models_cli():
    from trafficrag.models.embeddings import prepare_pretrained_models

    parser = argparse.ArgumentParser(
        description="Prepare complete local VideoMAE, Qwen3-VL, and ModernBERT artifacts."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--vlm", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--text-encoder", default="answerdotai/ModernBERT-base")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--sha256", action="store_true")
    args = parser.parse_args()
    prepare_pretrained_models(
        args.output,
        args.vlm,
        args.text_encoder,
        args.revision,
        args.cache_dir,
        args.offline,
        args.sha256,
    )


def evaluate_motion_cli():
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from trafficrag.models.motion import AnnotatedSegments
    from trafficrag.training.objectives import precision_dtype

    parser = argparse.ArgumentParser(
        description="Evaluate a trained CUDA motion model on labeled video segments."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--threshold-selection")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("Motion evaluation needs a positive batch size and nonnegative workers")
    system = MotionSystem.restore(args.checkpoint, args.device)
    manifest = VideoManifest.read(args.manifest, system.domain)
    dataset = AnnotatedSegments(args.manifest, system.encoder_cfg, system.domain)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    probabilities = []
    targets = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=precision_dtype(args.precision)):
        for pixels, labels in loader:
            logits = system.model(pixels.to(system.device, non_blocking=True))
            if not torch.isfinite(logits).all():
                raise RuntimeError("Motion checkpoint produced nonfinite evaluation logits")
            probabilities.extend(logits.float().sigmoid().cpu().tolist())
            targets.extend(labels.long().tolist())
    expected = [row.label for row in manifest]
    if targets != expected:
        raise ValueError("Motion evaluation data order differs from its manifest")
    threshold = 0.5
    selection = None
    if args.threshold_selection:
        from trafficrag.evaluation.curves import apply_selected_threshold

        selection = json.loads(Path(args.threshold_selection).read_text())
        threshold = float(selection["selected"]["threshold"])
        decisions = apply_selected_threshold(probabilities, selection)
    else:
        decisions = [int(value >= threshold) for value in probabilities]
    groups = [row.group for row in manifest]
    report = {
        "split": args.split,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest_digest": manifest.digest(include_sources=True),
        "probabilities": probabilities,
        "labels": targets,
        "predictions": decisions,
        "ids": [row.identifier for row in manifest],
        "video_ids": [row.video_id for row in manifest],
        "groups": groups,
        "metrics": classification_summary(probabilities, targets, threshold),
        "intervals": bootstrap_binary(
            probabilities, targets, groups, args.bootstrap, args.seed, threshold
        ),
        "threshold_selection": selection,
        "per_video": {},
    }
    values = np.asarray(probabilities)
    labels = np.asarray(targets)
    for video in sorted({row.video_id for row in manifest}):
        positions = [index for index, row in enumerate(manifest) if row.video_id == video]
        report["per_video"][video] = classification_summary(
            values[positions], labels[positions], threshold
        )
    atomic_json(args.output, report)
    print(json.dumps(report["metrics"], indent=2))
