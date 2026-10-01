"""Motion loss accounting and validation probabilities for binary segments."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MotionLossConfig:
    name: str = "bce"
    positive_weight: float = 1.0
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0

    def __post_init__(self):
        if self.name not in ("bce", "focal"):
            raise ValueError("Motion objectives support bce or focal")
        if self.positive_weight <= 0 or self.focal_gamma < 0 or not 0 <= self.label_smoothing < 0.5:
            raise ValueError("Invalid motion loss weights or smoothing")
        if not np.isfinite([self.positive_weight, self.focal_gamma, self.label_smoothing]).all():
            raise ValueError("Motion objective parameters must be finite")


def motion_loss(logits, labels, config=None):
    config = config or MotionLossConfig()
    if logits.shape != labels.shape or logits.ndim != 1:
        raise ValueError("Motion logits and labels must be aligned vectors")
    if not torch.isfinite(logits).all() or not torch.isin(labels, labels.new_tensor([0, 1])).all():
        raise ValueError("Motion logits must be finite and labels binary")
    logits, labels = logits.float(), labels.float()
    smoothed = labels * (1 - 2 * config.label_smoothing) + config.label_smoothing
    losses = F.binary_cross_entropy_with_logits(
        logits,
        smoothed,
        pos_weight=logits.new_tensor(config.positive_weight),
        reduction="none",
    )
    if config.name == "focal":
        probabilities = logits.sigmoid()
        correct_probability = torch.where(labels.bool(), probabilities, 1 - probabilities)
        losses = losses * (1 - correct_probability).pow(config.focal_gamma)
    return losses.mean()


class MotionAccumulator:
    def __init__(self):
        self.loss_sum = 0.0
        self.count = 0
        self.probabilities = []
        self.labels = []

    def update(self, loss, logits, labels):
        if not torch.isfinite(loss.detach()):
            raise RuntimeError("Nonfinite motion objective")
        self.loss_sum += float(loss.detach()) * len(labels)
        self.count += len(labels)
        self.probabilities.extend(logits.detach().float().sigmoid().cpu().tolist())
        self.labels.extend(labels.detach().long().cpu().tolist())

    def result(self, threshold=0.5):
        from trafficrag.evaluation.classification import binary_report

        if not self.count:
            raise ValueError("Cannot summarize an empty motion epoch")
        return {
            "loss": self.loss_sum / self.count,
            **binary_report(self.probabilities, self.labels, threshold),
        }


def accumulation_block(index, batches, steps):
    if min(batches, steps) < 1 or not 0 <= index < batches:
        raise ValueError("Invalid gradient accumulation indices")
    start = index // steps * steps
    length = min(steps, batches - start)
    return length, index + 1 == start + length


def precision_dtype(name):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError("Motion training precision must be bf16 or fp16")


def update_optimizer(model, optimizer, scaler, clip=1.0):
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    if not scaler.is_enabled() and not torch.isfinite(norm):
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("Motion gradients became nonfinite")
    previous = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    success = scaler.get_scale() >= previous
    optimizer.zero_grad(set_to_none=True)
    return success, float(norm)
