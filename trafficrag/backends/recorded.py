import numpy as np

from .registry import register_backend


@register_backend("recorded")
class RecordedBackend:
    """Replay externally computed caption/embedding/grounding outputs exactly.

    Interval matching is strict; no nearest-caption substitution is performed.
    """
    def __init__(self, records):
        self.captions = records['captions']
        self.refinements = records['refinements']

    @staticmethod
    def matches(row, interval):
        return abs(row['start'] - interval.start) < 1e-5 and abs(row['end'] - interval.end) < 1e-5

    def caption(self, interval):
        rows = [row for row in self.captions if self.matches(row, interval)]
        if len(rows) != 1:
            raise ValueError(f'Exactly one caption record is required for candidate {interval}.')
        return rows[0]['caption']

    def embed(self, text):
        rows = [row for row in self.captions if row['caption'] == text]
        if not rows:
            raise ValueError('Missing caption embedding.')
        vectors = np.asarray([row['embedding'] for row in rows])
        if not np.allclose(vectors, vectors[:1]):
            raise ValueError('Repeated caption has inconsistent embeddings.')
        return vectors[0]

    def ground(self, crop, candidate):
        rows = [row for row in self.refinements if self.matches(row, crop)]
        if len(rows) != 1:
            raise ValueError(f'Exactly one refinement record is required for crop {crop}.')
        return rows[0]['local_interval']
