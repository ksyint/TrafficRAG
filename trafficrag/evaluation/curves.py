"""Validation threshold selection and video-group bootstrap confidence intervals."""

from collections import defaultdict

import numpy as np

from trafficrag.evaluation.classification import validate_binary, binary_report
from trafficrag.evaluation.temporal import boundary_measurements


def threshold_curve(probabilities, labels, thresholds=None):
    probabilities, labels = validate_binary(probabilities, labels)
    if thresholds is None:
        thresholds = np.unique(np.r_[0.0, probabilities, 1.0])
    return [binary_report(probabilities, labels, float(threshold)) for threshold in thresholds]


def select_threshold(probabilities, labels, metric="f1", maximum_false_positive_rate=None):
    if metric not in ("f1", "accuracy", "balanced_accuracy", "mcc", "recall"):
        raise ValueError("Unsupported threshold selection metric")
    rows = threshold_curve(probabilities, labels)
    eligible = rows
    if maximum_false_positive_rate is not None:
        if not 0 <= maximum_false_positive_rate <= 1:
            raise ValueError("False-positive-rate budget must lie in [0,1]")
        eligible = [
            row
            for row in rows
            if row["false_positive_rate"] is not None
            and row["false_positive_rate"] <= maximum_false_positive_rate
        ]
    if not eligible:
        raise ValueError("No threshold meets the selected validation constraint")
    best = max(eligible, key=lambda row: (row[metric], row["threshold"]))
    return {"metric": metric, "selected": best, "curve": rows}


def grouped_indices(groups):
    units = defaultdict(list)
    for index, group in enumerate(groups):
        units[str(group)].append(index)
    if not units:
        raise ValueError("Grouped evaluation requires source identities")
    return [np.asarray(indices) for _, indices in sorted(units.items())]


def percentile(values, confidence=0.95):
    if not 0 < confidence < 1:
        raise ValueError("Confidence must lie in (0,1)")
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Confidence intervals require finite resampled estimates")
    tail = (1 - confidence) / 2
    return {
        "lower": float(np.quantile(values, tail)),
        "upper": float(np.quantile(values, 1 - tail)),
        "confidence": confidence,
    }


def bootstrap_binary(probabilities, labels, groups, repeats=2000, seed=42, threshold=0.5):
    probabilities, labels = validate_binary(probabilities, labels)
    if len(groups) != len(labels) or repeats < 1:
        raise ValueError(
            "Bootstrap groups must align with predictions and repeats must be positive"
        )
    units = grouped_indices(groups)
    generator = np.random.default_rng(seed)
    distributions = {name: [] for name in ("f1", "accuracy", "recall", "precision")}
    for _ in range(repeats):
        indices = np.concatenate(
            [units[index] for index in generator.integers(0, len(units), len(units))]
        )
        report = binary_report(probabilities[indices], labels[indices], threshold)
        for name, values in distributions.items():
            values.append(report[name])
    estimate = binary_report(probabilities, labels, threshold)
    return {
        "groups": len(units),
        "repeats": repeats,
        "metrics": {
            name: {"estimate": estimate[name], **percentile(values)}
            for name, values in distributions.items()
        },
    }


def bootstrap_temporal(records, repeats=2000, seed=42):
    positives = []
    groups = []
    for row in records:
        if row["target"] is None:
            continue
        positives.append(
            boundary_measurements(row.get("prediction", row.get("interval")), row["target"])["iou"]
        )
        groups.append(row.get("group", row.get("video_id", row.get("id"))))
    if not positives:
        return None
    if repeats < 1 or any(group is None for group in groups):
        raise ValueError("Temporal bootstrap requires video identities and positive repeats")
    values = np.asarray(positives)
    units = grouped_indices(groups)
    generator = np.random.default_rng(seed)
    samples = []
    for _ in range(repeats):
        indices = np.concatenate(
            [units[index] for index in generator.integers(0, len(units), len(units))]
        )
        samples.append(float(values[indices].mean()))
    return {
        "mean_iou": float(values.mean()),
        "videos": len(values),
        "groups": len(units),
        **percentile(samples),
    }


def apply_selected_threshold(probabilities, selection):
    values = np.asarray(probabilities, dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Threshold inputs must be probabilities")
    threshold = selection["selected"]["threshold"]
    return (values >= threshold).astype(int).tolist()
