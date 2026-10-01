"""Group-preserving train, query, and caption-memory partitions."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from trafficrag.pipeline.grounding.data.records import VideoManifest


class PartitionPlan:
    def __init__(self, manifest, fractions=(0.7, 0.15, 0.15), seed=42):
        if len(fractions) != 3 or abs(sum(fractions) - 1) > 1e-8:
            raise ValueError('Train, query, and KB fractions must sum to one')
        if any(value <= 0 for value in fractions):
            raise ValueError('All three partition fractions must be positive')
        self.manifest = manifest
        self.fractions = tuple(float(value) for value in fractions)
        self.seed = int(seed)
        self.names = ('train', 'query', 'kb')
        self.assignment = {}

    def allocate(self):
        groups = defaultdict(list)
        for record in self.manifest:
            groups[record.group].append(record)
        if len(groups) < len(self.names):
            raise ValueError('At least three independent recording groups are needed')

        def order(group):
            digest = hashlib.sha256(f'{self.seed}:{group}'.encode()).hexdigest()
            return -len(groups[group]), digest

        counts = {name: 0 for name in self.names}
        targets = dict(zip(self.names, [len(self.manifest) * value for value in self.fractions]))
        labels = {name: defaultdict(int) for name in self.names}
        for index, group in enumerate(sorted(groups, key=order)):
            rows = groups[group]
            if index < len(self.names):
                selected = self.names[index]
            else:
                selected = max(
                    self.names, key=lambda name: (targets[name] - counts[name]) / targets[name]
                )
            self.assignment[group] = selected
            counts[selected] += len(rows)
            for row in rows:
                labels[selected][str(row.label)] += 1
        return self

    def manifests(self):
        if not self.assignment:
            self.allocate()
        result = {}
        for name in self.names:
            rows = [row for row in self.manifest if self.assignment[row.group] == name]
            result[name] = VideoManifest(rows, self.manifest.path)
        validate_disjoint(result)
        return result

    def export(self, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        parts = self.manifests()
        paths = {'train': 'motion_train.jsonl', 'query': 'validation.jsonl', 'kb': 'kb.jsonl'}
        for name, manifest in parts.items():
            manifest.write(output / paths[name])
        excluded = sorted({row.video_id for name in ('train', 'query') for row in parts[name]})
        (output / 'train_query_ids.txt').write_text(''.join(value + '\n' for value in excluded))
        report = {
            'format': 'traffic-partitions-v1',
            'seed': self.seed,
            'fractions': dict(zip(self.names, self.fractions)),
            'source_digest': self.manifest.digest(),
            'assignment': self.assignment,
            'partitions': {name: value.summary() for name, value in parts.items()},
        }
        (output / 'partitions.json').write_text(json.dumps(report, indent=2) + '\n')
        return report


def validate_disjoint(manifests):
    owners = {'group': {}, 'video_id': {}, 'video': {}}
    overlaps = []
    for name, manifest in manifests.items():
        for record in manifest:
            for field, index in owners.items():
                value = str(getattr(record, field))
                previous = index.setdefault(value, name)
                if previous != name:
                    overlaps.append(
                        {'field': field, 'value': value, 'partitions': [previous, name]}
                    )
    if overlaps:
        raise ValueError('Partition overlap: ' + json.dumps(overlaps[:10]))
    return {name: manifest.summary() for name, manifest in manifests.items()}


def leakage_report(paths):
    manifests = {name: VideoManifest.read(path) for name, path in paths.items()}
    summary = validate_disjoint(manifests)
    for name, manifest in manifests.items():
        labels = {record.label for record in manifest if record.label is not None}
        summary[name]['both_binary_labels'] = labels == {0, 1}
    return summary


def balance_rows(manifest):
    counts = defaultdict(lambda: defaultdict(int))
    for record in manifest:
        counts[record.domain][str(record.label)] += 1
    result = []
    for domain, values in sorted(counts.items()):
        total = sum(values.values())
        for label, count in sorted(values.items()):
            result.append(
                {'domain': domain, 'label': label, 'count': count, 'fraction': count / total}
            )
    return result


def partitions_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest')
    parser.add_argument('--output', default='data/partitions')
    parser.add_argument('--fractions', nargs=3, type=float, default=(0.7, 0.15, 0.15))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train')
    parser.add_argument('--query')
    parser.add_argument('--kb')
    args = parser.parse_args()
    if args.manifest:
        source = VideoManifest.read(args.manifest)
        report = PartitionPlan(source, args.fractions, args.seed).export(args.output)
        report['source_balance'] = balance_rows(source)
    else:
        paths = {
            name: getattr(args, name) for name in ('train', 'query', 'kb') if getattr(args, name)
        }
        if len(paths) < 2:
            parser.error('Supply --manifest or at least two named partitions')
        report = leakage_report(paths)
    print(json.dumps(report, indent=2))
