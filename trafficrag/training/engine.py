"""Validation-driven CUDA VideoMAE optimization with resumable epochs."""

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from trafficrag.models.motion import AnnotatedSegments
from trafficrag.data.records import VideoManifest
from trafficrag.data.leakage import require_disjoint
from trafficrag.data.sampling import SegmentSampler, class_weights
from trafficrag.training.objectives import (
    MotionLossConfig,
    MotionAccumulator,
    motion_loss,
    accumulation_block,
    precision_dtype,
    update_optimizer,
)
from trafficrag.training.snapshots import save_motion_snapshot, restore_motion_snapshot
from trafficrag.retrieval.storage import atomic_json


@dataclass
class TrainingOptions:
    precision: str = "bf16"
    accumulation: int = 1
    workers: int = 0
    gradient_clip: float = 1.0
    balanced_sampling: bool = False
    weighted_loss: bool = False
    loss: str = "bce"
    focal_gamma: float = 2.0
    checkpoint_metric: str = "f1"
    patience: int = 0

    def __post_init__(self):
        precision_dtype(self.precision)
        if (
            self.accumulation < 1
            or self.workers < 0
            or self.gradient_clip <= 0
            or self.patience < 0
        ):
            raise ValueError("Invalid motion execution options")
        if self.checkpoint_metric not in ("f1", "accuracy", "balanced_accuracy"):
            raise ValueError("Unsupported motion checkpoint metric")
        if self.balanced_sampling and self.weighted_loss:
            raise ValueError(
                "Choose balanced sampling or class-weighted loss to avoid weighting the same imbalance twice"
            )


class MotionTrainer:
    def __init__(self, system, training, validation, output, options=None):
        self.system = system
        self.options = options or TrainingOptions()
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        train_manifest = VideoManifest.read(training, system.domain)
        validation_manifest = VideoManifest.read(validation, system.domain)
        self.partition_report = require_disjoint(
            {"train": train_manifest, "validation": validation_manifest}
        )
        self.train_digest = train_manifest.digest(include_sources=True)
        self.validation_digest = validation_manifest.digest(include_sources=True)
        training_data = AnnotatedSegments(training, system.encoder_cfg, system.domain)
        validation_data = AnnotatedSegments(validation, system.encoder_cfg, system.domain)
        sampler = SegmentSampler(
            training_data.rows, system.cfg.seed, self.options.balanced_sampling
        )
        self.train_loader = DataLoader(
            training_data,
            batch_size=system.cfg.batch_size,
            sampler=sampler,
            num_workers=self.options.workers,
            pin_memory=True,
            generator=torch.Generator().manual_seed(system.cfg.seed),
        )
        self.validation_loader = DataLoader(
            validation_data,
            batch_size=system.cfg.batch_size,
            shuffle=False,
            num_workers=self.options.workers,
            pin_memory=True,
        )
        weight = (
            class_weights(training_data.rows)["positive_weight"]
            if self.options.weighted_loss
            else 1.0
        )
        self.objective = MotionLossConfig(self.options.loss, weight, self.options.focal_gamma)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.options.precision == "fp16")
        self.dtype = precision_dtype(self.options.precision)

    def configure(self, resume=False):
        seed = self.system.cfg.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.system.configure(initialize=not resume)

    def epoch(self, loader, training=False):
        model, optimizer = self.system.model, self.system.optimizer
        model.train(training)
        if training:
            optimizer.zero_grad(set_to_none=True)
        meter = MotionAccumulator()
        updates, skipped, gradients = 0, 0, []
        torch.cuda.synchronize(self.system.device)
        torch.cuda.reset_peak_memory_stats(self.system.device)
        started = time.perf_counter()
        for index, (pixels, labels) in enumerate(loader):
            pixels = pixels.to(self.system.device, non_blocking=True)
            labels = labels.to(self.system.device, non_blocking=True)
            with torch.set_grad_enabled(training), torch.autocast("cuda", dtype=self.dtype):
                logits = model(pixels)
                loss = motion_loss(logits, labels, self.objective)
            meter.update(loss, logits, labels)
            if training:
                block_batches, boundary = accumulation_block(
                    index, len(loader), self.options.accumulation
                )
                block_start = index // self.options.accumulation * self.options.accumulation
                remaining = len(loader.sampler) - block_start * loader.batch_size
                block_examples = min(block_batches * loader.batch_size, remaining)
                self.scaler.scale(loss * len(labels) / block_examples).backward()
                if boundary:
                    success, gradient = update_optimizer(
                        model, optimizer, self.scaler, self.options.gradient_clip
                    )
                    updates += int(success)
                    skipped += int(not success)
                    if success:
                        gradients.append(gradient)
        torch.cuda.synchronize(self.system.device)
        elapsed = time.perf_counter() - started
        report = meter.result()
        report.update(
            seconds=elapsed,
            samples_per_second=meter.count / max(elapsed, 1e-9),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(self.system.device),
            optimizer_updates=updates,
            skipped_updates=skipped,
            mean_gradient_norm=float(np.mean(gradients)) if gradients else None,
        )
        return report, meter

    def run(self, resume=None):
        if resume and Path(resume).resolve().parent != self.output.resolve():
            raise ValueError("Resume motion training in the original checkpoint output directory")
        self.configure(resume=bool(resume))
        epoch, best, history = 0, float("-inf"), []
        if resume:
            epoch, best, history = restore_motion_snapshot(resume, self)
        elif (self.output / "last.pt").exists():
            raise FileExistsError("Motion run exists, resume it or select another output directory")
        if epoch >= self.system.cfg.epochs:
            raise ValueError("Motion checkpoint already completed the requested epoch budget")
        atomic_json(self.output / "partitions.json", self.partition_report)
        stale = 0
        for previous in reversed(history):
            if previous["validation"][self.options.checkpoint_metric] == best:
                break
            stale += 1
        if self.options.patience and stale >= self.options.patience:
            return {
                "epochs": len(history),
                "best_validation": best,
                "metric": self.options.checkpoint_metric,
                "early_stopped": True,
            }
        for current in range(epoch, self.system.cfg.epochs):
            self.train_loader.sampler.set_epoch(current)
            train, _ = self.epoch(self.train_loader, training=True)
            validation, predictions = self.epoch(self.validation_loader)
            score = validation[self.options.checkpoint_metric]
            improved = score > best
            best = max(best, score)
            stale = 0 if improved else stale + 1
            record = {"epoch": current + 1, "train": train, "validation": validation}
            history.append(record)
            if improved:
                save_motion_snapshot(self.output / "best.pt", self, current + 1, best, history)
                atomic_json(
                    self.output / "validation-best.json",
                    {
                        "epoch": current + 1,
                        "split": "validation",
                        "probabilities": predictions.probabilities,
                        "labels": predictions.labels,
                        "ids": [
                            row.get("id", str(index))
                            for index, row in enumerate(self.validation_loader.dataset.rows)
                        ],
                        "video_ids": [
                            row.get("video_id", row["video"])
                            for row in self.validation_loader.dataset.rows
                        ],
                    },
                )
            save_motion_snapshot(self.output / "last.pt", self, current + 1, best, history)
            atomic_json(self.output / "history.json", history)
            print(json.dumps(record, allow_nan=False), flush=True)
            if self.options.patience and stale >= self.options.patience:
                break
        return {
            "epochs": len(history),
            "best_validation": best,
            "metric": self.options.checkpoint_metric,
        }
