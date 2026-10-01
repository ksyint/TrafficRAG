"""Binary motion and violation detection metrics on saved probabilities."""

import numpy as np


def validate_binary(probabilities, labels):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels)
    if probabilities.ndim != 1 or labels.shape != probabilities.shape or not len(labels):
        raise ValueError("Binary probabilities and labels must be aligned nonempty vectors")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite values in [0,1]")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Binary evaluation labels must be zero or one")
    return probabilities, labels.astype(np.int64)


def binary_report(probabilities, labels, threshold=0.5):
    probabilities, labels = validate_binary(probabilities, labels)
    if not 0 <= threshold <= 1:
        raise ValueError("Decision threshold must lie in [0,1]")
    predicted = probabilities >= threshold
    positive = labels == 1
    tp = int((predicted & positive).sum())
    fp = int((predicted & ~positive).sum())
    fn = int((~predicted & positive).sum())
    tn = int((~predicted & ~positive).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    denominator = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    present_recalls = ([recall] if tp + fn else []) + ([specificity] if tn + fp else [])
    return {
        "samples": len(labels),
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": 1 - specificity if tn + fp else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": float(np.mean(present_recalls)),
        "mcc": float((tp * tn - fp * fn) / denominator) if denominator else 0.0,
        "brier": float(np.square(probabilities - labels).mean()),
        "negative_log_likelihood": float(
            -(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)).mean()
        ),
    }


def rank_auc(probabilities, labels):
    probabilities, labels = validate_binary(probabilities, labels)
    positives, negatives = int(labels.sum()), int((labels == 0).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(probabilities, kind="stable")
    rank_sum = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and probabilities[order[end]] == probabilities[order[start]]:
            end += 1
        mean_rank = (start + 1 + end) / 2
        rank_sum += mean_rank * labels[order[start:end]].sum()
        start = end
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision(probabilities, labels):
    probabilities, labels = validate_binary(probabilities, labels)
    if not labels.sum():
        return None
    order = np.argsort(-probabilities, kind="stable")
    sorted_labels = labels[order]
    sorted_scores = probabilities[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)) + 1, len(order)]
    positives = np.cumsum(sorted_labels)[ends - 1]
    recall = positives / labels.sum()
    precision = positives / ends
    return float(np.sum(np.diff(np.r_[0, recall]) * precision))


def calibration_bins(probabilities, labels, bins=10):
    probabilities, labels = validate_binary(probabilities, labels)
    if bins < 1:
        raise ValueError("Calibration bin count must be positive")
    assignments = np.minimum((probabilities * bins).astype(int), bins - 1)
    rows = []
    total = 0.0
    for index in range(bins):
        selected = assignments == index
        count = int(selected.sum())
        average = float(probabilities[selected].mean()) if count else None
        observed = float(labels[selected].mean()) if count else None
        if count:
            total += count * abs(average - observed)
        rows.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": count,
                "probability": average,
                "positive_rate": observed,
            }
        )
    return {"ece": total / len(labels), "bins": rows}


def classification_summary(probabilities, labels, threshold=0.5):
    return {
        **binary_report(probabilities, labels, threshold),
        "auroc": rank_auc(probabilities, labels),
        "average_precision": average_precision(probabilities, labels),
        "calibration": calibration_bins(probabilities, labels),
    }
