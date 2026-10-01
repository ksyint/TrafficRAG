"""Temporal boundary errors, proposal coverage, and final grounding recall."""

import numpy as np

from trafficrag.grounding import Interval, temporal_iou
from trafficrag.data.records import finite_interval


def interval(value):
    normalized = finite_interval(value)
    return None if normalized is None else Interval(*normalized)


def boundary_measurements(prediction, target):
    prediction, target = interval(prediction), interval(target)
    values = {
        "iou": temporal_iou(prediction, target),
        "predicted": prediction is not None,
        "positive": target is not None,
    }
    if prediction is None or target is None:
        return values
    intersection = max(0.0, min(prediction.end, target.end) - max(prediction.start, target.start))
    prediction_length = prediction.end - prediction.start
    target_length = target.end - target.start
    values.update(
        start_error=prediction.start - target.start,
        end_error=prediction.end - target.end,
        absolute_start_error=abs(prediction.start - target.start),
        absolute_end_error=abs(prediction.end - target.end),
        duration_error=prediction_length - target_length,
        target_coverage=intersection / target_length,
        prediction_precision=intersection / prediction_length,
        center_error=(prediction.start + prediction.end - target.start - target.end) / 2,
    )
    return values


def temporal_report(records, thresholds=(0.3, 0.5, 0.7)):
    rows = [
        boundary_measurements(row.get("prediction", row.get("interval")), row["target"])
        for row in records
    ]
    positives = [row for row in rows if row["positive"]]
    matched = [row for row in positives if row["predicted"]]
    report = {
        "videos": len(rows),
        "positive_videos": len(positives),
        "localized_positive_videos": len(matched),
        "positive_video_mean_iou": float(np.mean([row["iou"] for row in positives]))
        if positives
        else None,
        "localized_positive_mean_iou": float(np.mean([row["iou"] for row in matched]))
        if matched
        else None,
        "recall": {},
    }
    for threshold in thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError("Temporal recall thresholds must lie in [0,1]")
        report["recall"][str(threshold)] = (
            sum(row["predicted"] and row["iou"] >= threshold for row in positives) / len(positives)
            if positives
            else None
        )
    for name in (
        "start_error",
        "end_error",
        "absolute_start_error",
        "absolute_end_error",
        "duration_error",
        "target_coverage",
        "prediction_precision",
        "center_error",
    ):
        values = [row[name] for row in matched]
        report[name] = (
            {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p90": float(np.quantile(values, 0.9)),
            }
            if values
            else None
        )
    return report


def proposal_report(records, thresholds=(0.3, 0.5, 0.7)):
    rows = []
    for record in records:
        target = interval(record["target"])
        proposals = [interval(row.get("interval", row)) for row in record.get("candidates", [])]
        values = [temporal_iou(proposal, target) for proposal in proposals]
        selected = record.get("selected_score")
        rows.append(
            {
                "id": record.get("id"),
                "positive": target is not None,
                "proposals": len(proposals),
                "maximum_iou": max(values, default=0.0),
                "selected_score": selected,
                "proposal_seconds": sum(proposal.end - proposal.start for proposal in proposals),
            }
        )
    positive = [row for row in rows if row["positive"]]
    return {
        "records": rows,
        "mean_proposals": float(np.mean([row["proposals"] for row in rows])) if rows else None,
        "positive_proposal_recall": {
            str(threshold): sum(
                row["proposals"] > 0 and row["maximum_iou"] >= threshold for row in positive
            )
            / len(positive)
            if positive
            else None
            for threshold in thresholds
        },
        "mean_oracle_iou": float(np.mean([row["maximum_iou"] for row in positive]))
        if positive
        else None,
    }


def duration_groups(records, edges=(0, 1, 3, 5, float("inf"))):
    reports = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = []
        for row in records:
            target = interval(row["target"])
            if target is not None and low <= target.end - target.start < high:
                selected.append(row)
        reports.append(
            {
                "minimum_seconds": low,
                "maximum_seconds": high if np.isfinite(high) else None,
                **temporal_report(selected),
            }
        )
    return reports


def refinement_change(records):
    rows = []
    for row in records:
        target = interval(row["target"])
        candidates = row.get("candidates", [])
        if target is None or not candidates:
            continue
        chosen = max(candidates, key=lambda value: value["verified_score"])
        before = temporal_iou(interval(chosen["interval"]), target)
        after = temporal_iou(interval(row.get("prediction", row.get("interval"))), target)
        rows.append(
            {"id": row.get("id"), "before": before, "after": after, "difference": after - before}
        )
    return {
        "records": rows,
        "mean_iou_change": float(np.mean([row["difference"] for row in rows])) if rows else None,
    }


def stage_attribution(records, overlap_threshold=0.5):
    if not 0 < overlap_threshold <= 1:
        raise ValueError("Stage attribution requires an overlap threshold in (0,1]")
    details = []
    for row in records:
        target = interval(row["target"])
        final = interval(row.get("prediction", row.get("interval")))
        candidates = row.get("candidates", [])
        candidate_intervals = [interval(candidate["interval"]) for candidate in candidates]
        if any(value is None for value in candidate_intervals):
            raise ValueError("A motion proposal cannot have a null interval")
        oracle_values = [temporal_iou(value, target) for value in candidate_intervals]
        selected_index = None
        motion_index = None
        if candidates:
            scores = np.asarray(
                [candidate["verified_score"] for candidate in candidates], dtype=float
            )
            motion = np.asarray(
                [candidate["motion_score"] for candidate in candidates], dtype=float
            )
            if not np.isfinite(scores).all() or not np.isfinite(motion).all():
                raise ValueError("Stage attribution requires finite proposal scores")
            selected_index = int(scores.argmax())
            motion_index = int(motion.argmax())
        selected_iou = oracle_values[selected_index] if selected_index is not None else 0.0
        motion_iou = oracle_values[motion_index] if motion_index is not None else 0.0
        final_iou = temporal_iou(final, target)
        crop = interval(row.get("crop"))
        crop_coverage = None
        if crop is not None and target is not None:
            overlap = max(0.0, min(crop.end, target.end) - max(crop.start, target.start))
            crop_coverage = overlap / (target.end - target.start)
        oracle_iou = max(oracle_values, default=0.0)
        if target is None:
            outcome = "correct_rejection" if final is None else "false_positive"
        elif final is not None and final_iou >= overlap_threshold:
            outcome = "localized"
        elif not candidates:
            outcome = "no_motion_proposal"
        elif oracle_iou < overlap_threshold:
            outcome = "proposal_coverage"
        elif selected_iou < overlap_threshold:
            outcome = "candidate_selection"
        elif crop_coverage is not None and crop_coverage < overlap_threshold:
            outcome = "crop_coverage"
        elif final is None:
            outcome = "refinement_rejection"
        else:
            outcome = "refinement_boundaries"
        details.append(
            {
                "id": row["id"],
                "domain": row.get("domain"),
                "positive": target is not None,
                "outcome": outcome,
                "proposals": len(candidates),
                "motion_selected_index": motion_index,
                "verified_selected_index": selected_index,
                "selection_changed": motion_index != selected_index,
                "motion_selected_iou": motion_iou,
                "verified_selected_iou": selected_iou,
                "proposal_oracle_iou": oracle_iou,
                "crop_target_coverage": crop_coverage,
                "final_iou": final_iou,
                "semantic_iou_change": selected_iou - motion_iou,
                "refinement_iou_change": final_iou - selected_iou,
            }
        )
    positives = [row for row in details if row["positive"]]
    counts = {}
    for row in details:
        counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
    fields = ("motion_selected_iou", "verified_selected_iou", "proposal_oracle_iou", "final_iou")
    return {
        "threshold": overlap_threshold,
        "outcomes": counts,
        "positive_video_means": {
            key: float(np.mean([row[key] for row in positives])) if positives else None
            for key in fields
        },
        "semantic_selection_changes": sum(row["selection_changed"] for row in details),
        "records": details,
    }


def retrieval_agreement(records):
    candidates = []
    for row in records:
        target = interval(row["target"])
        for index, candidate in enumerate(row.get("candidates", [])):
            neighbors = candidate.get("neighbors") or []
            if not neighbors:
                continue
            labels = np.asarray([neighbor["label"] for neighbor in neighbors], dtype=float)
            similarities = np.asarray(
                [neighbor["similarity"] for neighbor in neighbors], dtype=float
            )
            if not np.isin(labels, (0, 1)).all() or not np.isfinite(similarities).all():
                raise ValueError(
                    "Retrieved evidence requires binary labels and finite cosine scores"
                )
            semantic = float(candidate["semantic_score"])
            expected = float(labels.mean())
            if not np.isclose(semantic, expected, atol=1e-6):
                raise ValueError("Recorded semantic vote differs from the retrieved label fraction")
            proposal = interval(candidate["interval"])
            overlap = temporal_iou(proposal, target)
            candidates.append(
                {
                    "id": row["id"],
                    "candidate": index,
                    "target_iou": overlap,
                    "semantic_vote": semantic,
                    "motion_probability": float(candidate["motion_score"]),
                    "verified_probability": float(candidate["verified_score"]),
                    "neighbor_count": len(neighbors),
                    "unique_neighbor_ids": len({str(neighbor["id"]) for neighbor in neighbors}),
                    "nearest_similarity": float(similarities[0]),
                    "mean_similarity": float(similarities.mean()),
                    "unanimous_vote": bool(labels.min() == labels.max()),
                }
            )
    groups = {}
    for name, selected in (
        ("overlapping", [row for row in candidates if row["target_iou"] > 0]),
        ("nonoverlapping", [row for row in candidates if row["target_iou"] == 0]),
    ):
        groups[name] = {
            "candidates": len(selected),
            "mean_semantic_vote": float(np.mean([row["semantic_vote"] for row in selected]))
            if selected
            else None,
            "mean_motion_probability": float(
                np.mean([row["motion_probability"] for row in selected])
            )
            if selected
            else None,
            "unanimous_votes": sum(row["unanimous_vote"] for row in selected),
        }
    return {"groups": groups, "candidates": candidates}
