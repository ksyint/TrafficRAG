"""Video-level event classification and temporal localization metrics."""

import math
from collections import defaultdict

import numpy as np

from trafficrag.pipeline.grounding.data.records import finite_interval


def temporal_overlap(left, right):
    left, right = finite_interval(left), finite_interval(right)
    if left is None or right is None:
        return 0.0
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union


def event_record(row):
    if 'prediction' not in row or 'target' not in row:
        raise ValueError('Evaluation requires prediction and target fields')
    prediction = finite_interval(row['prediction'])
    target = finite_interval(row['target'])
    confidence = row.get('selected_score')
    if confidence is not None:
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Selected confidence must lie in [0,1]')
    result = dict(row, prediction=prediction, target=target)
    result.update(
        iou=temporal_overlap(prediction, target),
        target_positive=target is not None,
        predicted_positive=prediction is not None,
        confidence=confidence,
    )
    if target is not None and prediction is not None:
        result['start_error'] = prediction[0] - target[0]
        result['end_error'] = prediction[1] - target[1]
    return result


def classification(rows):
    counts = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    for row in rows:
        actual, predicted = row['target_positive'], row['predicted_positive']
        name = ('tp' if predicted else 'fn') if actual else ('fp' if predicted else 'tn')
        counts[name] += 1
    tp, fp, tn, fn = (counts[name] for name in ('tp', 'fp', 'tn', 'fn'))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return dict(
        counts,
        precision=precision,
        recall=recall,
        specificity=specificity,
        accuracy=(tp + tn) / max(1, tp + fp + tn + fn),
        f1=2 * tp / max(1, 2 * tp + fp + fn),
    )


def localization(rows, thresholds=(0.3, 0.5, 0.7)):
    positive = [row for row in rows if row['target_positive']]
    matched = [row for row in positive if row['predicted_positive']]
    count = len(positive)
    result = {
        'positive_videos': count,
        'detected_positive_videos': len(matched),
        'mean_tiou': sum(row['iou'] for row in positive) / max(1, count),
    }
    for threshold in thresholds:
        if not 0 < threshold <= 1:
            raise ValueError('Temporal IoU thresholds must lie in (0,1]')
        key = f'recall_at_tiou_{threshold:g}'
        result[key] = sum(row['iou'] >= threshold for row in positive) / max(1, count)
    if matched:
        for boundary in ('start', 'end'):
            errors = np.asarray([row[boundary + '_error'] for row in matched])
            result[boundary + '_mae'] = float(np.abs(errors).mean())
            result[boundary + '_bias'] = float(errors.mean())
    else:
        result.update(start_mae=None, end_mae=None, start_bias=None, end_bias=None)
    return result


def score_events(records, thresholds=(0.3, 0.5, 0.7)):
    rows = [event_record(record) for record in records]
    if not rows:
        raise ValueError('At least one evaluated video is required')
    return {'samples': len(rows), **classification(rows), **localization(rows, thresholds)}


def grouped_scores(records, field='domain', thresholds=(0.3, 0.5, 0.7)):
    groups = defaultdict(list)
    for row in records:
        if field not in row:
            raise ValueError(f'Missing grouping field: {field}')
        groups[str(row[field])].append(row)
    return {name: score_events(rows, thresholds) for name, rows in sorted(groups.items())}


def bootstrap_events(records, repeats=1000, seed=42, confidence=0.95, group='video_id'):
    if repeats < 1 or not 0 < confidence < 1:
        raise ValueError('Use positive repeats and confidence in (0,1)')
    clusters = defaultdict(list)
    for index, row in enumerate(records):
        clusters[str(row.get(group, row.get('id', index)))].append(row)
    if not clusters:
        raise ValueError('Cannot bootstrap an empty benchmark')
    keys = sorted(clusters)
    generator = np.random.default_rng(seed)
    values = defaultdict(list)
    for _ in range(repeats):
        selected = generator.integers(0, len(keys), size=len(keys))
        sample = [row for index in selected for row in clusters[keys[int(index)]]]
        scores = score_events(sample)
        for name in ('f1', 'precision', 'recall', 'mean_tiou', 'accuracy'):
            values[name].append(scores[name])
    tail = (1 - confidence) / 2
    return {
        name: {
            'lower': float(np.quantile(samples, tail)),
            'upper': float(np.quantile(samples, 1 - tail)),
            'confidence': confidence,
            'replicates': repeats,
            'resampling_group': group,
        }
        for name, samples in values.items()
    }


def confidence_curve(records, thresholds=None):
    rows = [event_record(row) for row in records]
    if any(row['predicted_positive'] and row['confidence'] is None for row in rows):
        raise ValueError(
            'Every positive prediction needs its selected_score for confidence analysis'
        )
    thresholds = list(thresholds) if thresholds is not None else np.linspace(0, 1, 21)
    result = []
    for threshold in thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError('Confidence thresholds must lie in [0,1]')
        selected = []
        for row in rows:
            keep = row['confidence'] is not None and row['confidence'] >= threshold
            selected.append(dict(row, prediction=row['prediction'] if keep else None))
        result.append(dict(threshold=float(threshold), **score_events(selected)))
    return result


def compare_paired(left, right):
    def indexed(rows):
        result = {}
        for row in rows:
            identifier = str(row['id'])
            if identifier in result:
                raise ValueError(f'Duplicate evaluation ID: {identifier}')
            result[identifier] = row
        return result

    before, after = indexed(left), indexed(right)
    if before.keys() != after.keys():
        raise ValueError('Paired evaluation requires identical record IDs')
    changes = []
    for key in sorted(before):
        first, second = event_record(before[key]), event_record(after[key])
        if first['target'] != second['target']:
            raise ValueError(f'Ground truth differs for record {key}')
        changes.append(
            {
                'id': key,
                'tiou_before': first['iou'],
                'tiou_after': second['iou'],
                'tiou_change': second['iou'] - first['iou'],
                'presence_before': first['predicted_positive'],
                'presence_after': second['predicted_positive'],
            }
        )
    return changes
