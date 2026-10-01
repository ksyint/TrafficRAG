"""Fingerprint-addressed stage artifacts with atomic JSON and NumPy storage."""

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


def digest(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def file_stamp(path, content=False):
    path = Path(path).resolve()
    stat = path.stat()
    result = {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    if content:
        checksum = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                checksum.update(block)
        result['sha256'] = checksum.hexdigest()
    return result


class ArtifactStore:
    def __init__(self, root, namespace):
        if not namespace or any(
            c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in namespace
        ):
            raise ValueError(
                'Cache namespace uses lowercase letters, digits, hyphens, or underscores'
            )
        self.root = Path(root) / namespace
        self.namespace = namespace
        self.hits = self.misses = self.writes = 0

    def path(self, key, suffix):
        identity = digest(key)
        return self.root / identity[:2] / (identity + suffix)

    def read_json(self, key):
        path = self.path(key, '.json')
        if not path.is_file():
            self.misses += 1
            return None
        record = json.loads(path.read_text())
        if record.get('key') != key or record.get('format') != 'traffic-cache-v1':
            raise ValueError(f'Cache identity differs at {path}')
        self.hits += 1
        return record['value']

    def write_json(self, key, value):
        path = self.path(key, '.json')
        payload = json.dumps(
            {'format': 'traffic-cache-v1', 'key': key, 'value': value},
            ensure_ascii=False,
            allow_nan=False,
        )
        self._atomic(path, payload.encode())
        self.writes += 1
        return value

    def read_array(self, key):
        path = self.path(key, '.npy')
        if not path.is_file():
            self.misses += 1
            return None
        with path.open('rb') as stream:
            value = np.load(stream, allow_pickle=False)
        if not np.isfinite(value).all():
            raise ValueError(f'Nonfinite cached embedding at {path}')
        self.hits += 1
        return value

    def write_array(self, key, value):
        value = np.asarray(value, dtype=np.float32)
        if not np.isfinite(value).all():
            raise ValueError('Cached vectors must be finite')
        path = self.path(key, '.npy')
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            try:
                np.save(stream, value, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        temporary.replace(path)
        self.writes += 1
        return value

    @staticmethod
    def _atomic(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        temporary.replace(path)

    def summary(self):
        return {
            'namespace': self.namespace,
            'hits': self.hits,
            'misses': self.misses,
            'writes': self.writes,
        }

    def inventory(self):
        rows = []
        for path in sorted(self.root.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.npy'):
                rows.append(
                    {'artifact': str(path.relative_to(self.root)), 'bytes': path.stat().st_size}
                )
        return {
            'namespace': self.namespace,
            'artifacts': rows,
            'total_bytes': sum(row['bytes'] for row in rows),
        }


class CachedVideoBackend:
    def __init__(self, backend, root, video_path, model_identity=None):
        self.backend = backend
        self.video = file_stamp(video_path)
        self.identity = model_identity or asdict(backend.system.cfg)
        self.identity = json.loads(json.dumps(self.identity, sort_keys=True))
        self.domain = backend.domain
        self.captions = ArtifactStore(root, 'captions')
        self.embeddings = ArtifactStore(root, 'embeddings')
        self.boundaries = ArtifactStore(root, 'boundaries')

    def _key(self, interval):
        return {
            'video': self.video,
            'model': self.identity,
            'domain': self.domain,
            'interval': [float(interval.start), float(interval.end)],
        }

    def caption(self, interval):
        key = self._key(interval)
        value = self.captions.read_json(key)
        if value is None:
            value = self.captions.write_json(key, self.backend.caption(interval))
        return value

    def embed(self, caption):
        key = {'text': caption, 'model': self.identity, 'pooling': 'masked-mean-v1'}
        vector = self.embeddings.read_array(key)
        if vector is None:
            vector = self.embeddings.write_array(key, self.backend.embed(caption))
        return vector

    def ground(self, interval, candidate):
        references = [
            {'caption': row.get('caption', ''), 'label': row['label']}
            for row in candidate.neighbors or []
        ]
        key = dict(self._key(interval), caption=candidate.caption, references=references)
        value = self.boundaries.read_json(key)
        if value is None:
            prediction = self.backend.ground(interval, candidate)
            value = self.boundaries.write_json(key, {'interval': prediction})
        return value['interval']

    def summary(self):
        return [store.summary() for store in (self.captions, self.embeddings, self.boundaries)]


def cache_cli():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument(
        '--namespace', choices=('motion', 'captions', 'embeddings', 'boundaries'), required=True
    )
    args = parser.parse_args()
    print(json.dumps(ArtifactStore(args.root, args.namespace).inventory(), indent=2))
