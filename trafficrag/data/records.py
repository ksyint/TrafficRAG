"""Validated annotation and query records with stable source-video identities."""

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

DOMAINS = ('red-light', 'blind-spot-left', 'blind-spot-right')


def finite_interval(value, duration=None):
    if value is None:
        return None
    if isinstance(value, dict):
        value = (value['start'], value['end'])
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError('An event interval contains exactly two timestamps')
    start, end = map(float, value)
    if not math.isfinite(start + end) or start < 0 or end <= start:
        raise ValueError('Event timestamps require finite 0 <= start < end')
    if duration is not None and end > duration + 1e-6:
        raise ValueError('Event interval exceeds the video duration')
    return [start, end]


@dataclass(frozen=True)
class VideoRecord:
    identifier: str
    video_id: str
    video: Path
    domain: str
    target: object = None
    group: str = ''
    label: object = None
    start: object = None
    end: object = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row, root, index, domain=None, require_files=True):
        if not isinstance(row, dict):
            raise TypeError('Manifest rows must be JSON objects')
        video = (Path(root) / str(row['video'])).resolve()
        if require_files and not video.is_file():
            raise FileNotFoundError(video)
        source = str(row.get('video_id', '')).strip()
        if not source:
            raise ValueError('Every record needs a stable video_id')
        identity = str(row.get('id', f'{source}:{index}')).strip()
        if not identity:
            raise ValueError('Record IDs must be nonempty')
        selected = row.get('domain', domain)
        if selected not in DOMAINS:
            raise ValueError(f'Unknown traffic domain: {selected}')
        if domain is not None and selected != domain:
            raise ValueError('Manifest domain differs from the selected run')
        label = row.get('label')
        if label is not None and (type(label) is not int or label not in (0, 1)):
            raise ValueError('Segment labels must be integer zero or one')
        start, end = row.get('start'), row.get('end')
        if (start is None) != (end is None):
            raise ValueError('Segment records require both start and end')
        if start is not None:
            start, end = finite_interval([start, end])
        target = finite_interval(row.get('target'))
        known = {'id', 'video_id', 'video', 'domain', 'target', 'group', 'label', 'start', 'end'}
        return cls(
            identity,
            source,
            video,
            selected,
            target,
            str(row.get('group', source)),
            label,
            start,
            end,
            {key: value for key, value in row.items() if key not in known},
        )

    def as_dict(self):
        result = dict(self.metadata)
        result.update(
            id=self.identifier,
            video_id=self.video_id,
            video=str(self.video),
            domain=self.domain,
            group=self.group,
            target=self.target,
        )
        if self.label is not None:
            result['label'] = self.label
        if self.start is not None:
            result.update(start=self.start, end=self.end)
        return result

    def source_stamp(self):
        status = self.video.stat()
        return {'path': str(self.video), 'bytes': status.st_size, 'mtime_ns': status.st_mtime_ns}


class VideoManifest:
    def __init__(self, records, path=None):
        self.records = tuple(records)
        self.path = Path(path).resolve() if path else None
        if not self.records:
            raise ValueError('Manifest must contain at least one record')
        identities, paths, groups = set(), {}, {}
        reverse_paths = {}
        for record in self.records:
            if record.identifier in identities:
                raise ValueError(f'Duplicate record ID: {record.identifier}')
            identities.add(record.identifier)
            if paths.setdefault(record.video_id, record.video) != record.video:
                raise ValueError(f'video_id refers to multiple files: {record.video_id}')
            if reverse_paths.setdefault(record.video, record.video_id) != record.video_id:
                raise ValueError(f'One recording has multiple video IDs: {record.video}')
            if groups.setdefault(record.video_id, record.group) != record.group:
                raise ValueError('All segments of one recording must share its group')
        self.by_id = {record.identifier: record for record in self.records}

    @classmethod
    def read(cls, path, domain=None, require_files=True, require_target=False):
        path = Path(path).resolve()
        records = []
        with path.open(encoding='utf-8') as stream:
            for index, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if require_target and 'target' not in row:
                        raise ValueError('Benchmark queries require an explicit target interval or null')
                    record = VideoRecord.from_dict(row, path.parent, index, domain, require_files)
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f'{path}:{index}: {error}') from error
                records.append(record)
        return cls(records, path)

    def __iter__(self):
        return iter(self.records)

    def __len__(self):
        return len(self.records)

    def digest(self, include_sources=False):
        values = []
        for record in self.records:
            value = record.as_dict()
            if include_sources:
                value['source_stamp'] = record.source_stamp()
            values.append(value)
        raw = json.dumps(values, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        return hashlib.sha256(raw.encode()).hexdigest()

    def select(self, identifiers):
        requested = set(identifiers)
        missing = requested - self.by_id.keys()
        if missing:
            raise KeyError(f'Unknown record IDs: {sorted(missing)}')
        return VideoManifest([row for row in self if row.identifier in requested], self.path)

    def summary(self):
        return {
            'records': len(self),
            'videos': len({row.video_id for row in self}),
            'groups': len({row.group for row in self}),
            'domains': dict(Counter(row.domain for row in self)),
            'labels': dict(Counter(str(row.label) for row in self)),
            'positive_targets': sum(row.target is not None for row in self),
            'fingerprint': self.digest(),
        }

    def write(self, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + '.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            for record in self:
                stream.write(json.dumps(record.as_dict(), ensure_ascii=False) + '\n')
        temporary.replace(destination)
        return destination


def manifest_cli():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--domain', choices=DOMAINS)
    parser.add_argument('--output')
    parser.add_argument('--skip-file-check', action='store_true')
    args = parser.parse_args()
    manifest = VideoManifest.read(args.manifest, args.domain, not args.skip_file_check)
    if args.output:
        manifest.write(args.output)
    print(json.dumps(manifest.summary(), indent=2))
