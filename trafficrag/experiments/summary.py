"""Joined prediction reports, retrieval evidence summaries, and paired comparisons."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from trafficrag.experiments.metrics import (
    bootstrap_events,
    compare_paired,
    confidence_curve,
    grouped_scores,
    score_events,
)


def read_predictions(path):
    path = Path(path)
    records = []
    with path.open(encoding='utf-8') as stream:
        for index, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{index}: expected a JSON object')
            if 'prediction' not in row and 'interval' in row:
                row['prediction'] = row['interval']
            records.append(row)
    if not records:
        raise ValueError(f'No predictions in {path}')
    return records


def join_targets(predictions, targets):
    lookup = {}
    for row in targets:
        key = str(row['id'])
        if key in lookup:
            raise ValueError(f'Duplicate target ID: {key}')
        lookup[key] = row
    merged, seen = [], set()
    for row in predictions:
        key = str(row['id'])
        if key in seen or key not in lookup:
            raise ValueError(f'Duplicate or unknown prediction ID: {key}')
        seen.add(key)
        target = lookup[key]
        if 'target' not in target:
            raise ValueError(f'Missing target interval for {key}')
        if 'target' in row and row['target'] != target['target']:
            raise ValueError(f'Prediction file carries different ground truth for {key}')
        result = dict(row, target=target['target'])
        for field in ('domain', 'video_id', 'group'):
            if field in target:
                if field in row and row[field] != target[field]:
                    raise ValueError(f'{field} mismatch for {key}')
                result[field] = target[field]
        merged.append(result)
    if seen != lookup.keys():
        raise ValueError('Predictions do not cover every target ID')
    return merged


def retrieval_report(rows):
    identifiers = Counter()
    similarities = []
    neighbor_counts = []
    label_counts = Counter()
    candidates_per_video = []
    for row in rows:
        candidates = row.get('candidates', [])
        candidates_per_video.append(len(candidates))
        for candidate in candidates:
            neighbors = candidate.get('neighbors') or []
            neighbor_counts.append(len(neighbors))
            for neighbor in neighbors:
                identifiers[str(neighbor['id'])] += 1
                label_counts[str(neighbor['label'])] += 1
                similarities.append(float(neighbor['similarity']))
    return {
        'videos': len(rows),
        'candidate_count': sum(candidates_per_video),
        'videos_without_candidates': sum(value == 0 for value in candidates_per_video),
        'retrieval_calls': len(neighbor_counts),
        'mean_neighbors': sum(neighbor_counts) / max(1, len(neighbor_counts)),
        'mean_similarity': sum(similarities) / max(1, len(similarities)),
        'neighbor_labels': dict(label_counts),
        'unique_memory_entries': len(identifiers),
        'most_retrieved': [{'id': key, 'uses': value} for key, value in identifiers.most_common(20)],
    }


def stage_statistics(rows):
    elapsed = [float(row['elapsed_seconds']) for row in rows if 'elapsed_seconds' in row]
    stores = defaultdict(Counter)
    for row in rows:
        for cache in row.get('cache', []):
            for key in ('hits', 'misses', 'writes'):
                stores[cache['namespace']][key] += int(cache[key])
    return {
        'timed_records': len(elapsed),
        'total_seconds': sum(elapsed),
        'mean_seconds': sum(elapsed) / len(elapsed) if elapsed else None,
        'cache': {name: dict(value) for name, value in stores.items()},
    }


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


class BenchmarkReport:
    def __init__(self, rows, thresholds=(0.3, 0.5, 0.7)):
        self.rows = list(rows)
        self.thresholds = tuple(thresholds)
        identifiers = [str(row['id']) for row in self.rows if 'id' in row]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError('Benchmark prediction IDs must be unique')
        self.summary = score_events(self.rows, self.thresholds)

    def export(self, output, bootstrap=0, seed=42, paired=None):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        report = {
            'format': 'traffic-benchmark-v1',
            'overall': self.summary,
            'retrieval': retrieval_report(self.rows),
            'runtime': stage_statistics(self.rows),
        }
        if all('domain' in row for row in self.rows):
            report['domains'] = grouped_scores(self.rows, 'domain', self.thresholds)
            write_csv(
                output / 'domains.csv',
                [dict(domain=key, **values) for key, values in report['domains'].items()],
            )
        if bootstrap:
            report['intervals'] = bootstrap_events(self.rows, bootstrap, seed)
        if all(row.get('prediction') is None or row.get('selected_score') is not None for row in self.rows):
            write_csv(output / 'confidence.csv', confidence_curve(self.rows))
        if paired is not None:
            comparison = compare_paired(paired, self.rows)
            write_csv(output / 'paired.csv', comparison)
            report['paired_mean_tiou_change'] = sum(row['tiou_change'] for row in comparison) / len(comparison)
        (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        write_csv(output / 'overall.csv', [self.summary])
        return report


def report_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--targets')
    parser.add_argument('--output', required=True)
    parser.add_argument('--iou-thresholds', nargs='+', type=float, default=(0.3, 0.5, 0.7))
    parser.add_argument('--bootstrap', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--compare')
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error('--bootstrap must be nonnegative')
    rows = read_predictions(args.predictions)
    if args.targets:
        rows = join_targets(rows, read_predictions(args.targets))
    paired = read_predictions(args.compare) if args.compare else None
    if paired is not None and args.targets:
        paired = join_targets(paired, read_predictions(args.targets))
    report = BenchmarkReport(rows, args.iou_thresholds).export(args.output, args.bootstrap, args.seed, paired)
    print(json.dumps(report, indent=2))
