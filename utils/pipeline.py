from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class Interval:
    start: float
    end: float

    def __post_init__(self):
        if not math.isfinite(self.start) or not math.isfinite(self.end) or self.start < 0 or self.end <= self.start:
            raise ValueError('Intervals need finite timestamps and 0 <= start < end.')


@dataclass
class Candidate:
    interval: Interval
    segment_indices: list
    motion_score: float
    semantic_score: float = 0.0
    verified_score: float = 0.0
    caption: str = ''
    neighbors: object = None


def segment_video(duration, window=2.0, stride=1.0):
    if not all(math.isfinite(v) and v > 0 for v in (duration, window, stride)) or stride > window:
        raise ValueError('Positive duration/window and 0 < stride <= window are required.')
    if duration <= window:
        return [Interval(0, duration)]
    starts = np.arange(0, duration - window + 1e-9, stride).tolist()
    # Include a final full window for durations not divisible by the stride.
    if starts[-1] + window < duration - 1e-9:
        starts.append(duration - window)
    return [Interval(float(t), float(t + window)) for t in starts]


def propose_candidates(segments, scores, threshold=0.3, sigma=1.0):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(segments),) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('One finite motion probability in [0,1] is required per segment.')
    if sigma < 0 or not 0 <= threshold <= 1:
        raise ValueError('Invalid smoothing bandwidth or motion threshold.')
    smooth = gaussian_filter1d(scores, sigma=sigma, mode='nearest') if sigma else scores.copy()
    active = np.flatnonzero(smooth > threshold)
    runs = np.split(active, np.flatnonzero(np.diff(active) > 1) + 1) if len(active) else []
    candidates = []
    for run in runs:
        interval = Interval(segments[run[0]].start, segments[run[-1]].end)
        candidates.append(Candidate(interval, run.tolist(), float(smooth[run].mean())))
    return candidates, smooth


class KnowledgeBase:
    def __init__(self, embeddings, labels, captions=None, ids=None):
        vectors = np.asarray(embeddings, dtype=np.float64)
        raw_labels = np.asarray(labels)
        if not np.isin(raw_labels, [0, 1]).all():
            raise ValueError('Knowledge base labels must be binary integers.')
        self.labels = raw_labels.astype(np.int64)
        if vectors.ndim != 2 or len(vectors) == 0 or not np.isfinite(vectors).all():
            raise ValueError('KB embeddings must be a finite nonempty N,D matrix.')
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0) or self.labels.shape != (len(vectors),) or not np.isin(self.labels, [0, 1]).all():
            raise ValueError('KB embeddings need nonzero norms and one binary label each.')
        self.embeddings = vectors / norms
        self.captions = captions if captions is not None else [''] * len(vectors)
        self.ids = ids if ids is not None else [str(i) for i in range(len(vectors))]
        if len(self.captions) != len(vectors) or len(self.ids) != len(vectors):
            raise ValueError('KB metadata length mismatch.')

    def retrieve(self, embedding, k=10):
        query = np.asarray(embedding, dtype=np.float64)
        if query.shape != (self.embeddings.shape[1],) or not np.isfinite(query).all() or np.linalg.norm(query) == 0:
            raise ValueError('Query embedding shape/value mismatch.')
        if k < 1:
            raise ValueError('K must be positive.')
        similarities = self.embeddings @ (query / np.linalg.norm(query))
        order = np.argsort(-similarities, kind='stable')[:min(k, len(similarities))]
        return order, similarities[order], float(self.labels[order].mean())


def temporal_iou(a, b):
    if a is None or b is None:
        return 0.0
    intersection = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return intersection / union


class TrafficRAG:
    """Algorithm 1 with injectable caption, text embedding, and VLM grounding."""
    def __init__(self, kb, motion_threshold=0.3, semantic_weight=0.4, topk=10,
                 smoothing_sigma=1.0, base_padding=0.5, adaptive_padding=1.0,
                 window=2.0, stride=1.0):
        if not 0 <= semantic_weight <= 1 or min(base_padding, adaptive_padding) < 0:
            raise ValueError('Invalid fusion weight or padding.')
        self.kb = kb
        self.motion_threshold, self.semantic_weight, self.topk = motion_threshold, semantic_weight, topk
        self.smoothing_sigma = smoothing_sigma
        self.base_padding, self.adaptive_padding = base_padding, adaptive_padding
        self.window, self.stride = window, stride

    def __call__(self, duration, motion_scores, caption, embed, ground):
        segments = segment_video(duration, self.window, self.stride)
        candidates, smoothed = propose_candidates(segments, motion_scores, self.motion_threshold, self.smoothing_sigma)
        result = {'interval': None, 'candidates': [], 'smoothed_scores': smoothed.tolist(), 'crop': None}
        if not candidates:
            return result
        for candidate in candidates:
            candidate.caption = caption(candidate.interval)
            neighbors, similarities, semantic_score = self.kb.retrieve(embed(candidate.caption), self.topk)
            candidate.semantic_score = semantic_score
            candidate.verified_score = self.semantic_weight * semantic_score + (1 - self.semantic_weight) * candidate.motion_score
            candidate.neighbors = [{'id': self.kb.ids[int(i)], 'similarity': float(s), 'label': int(self.kb.labels[i])}
                                   for i, s in zip(neighbors, similarities)]
        best = max(candidates, key=lambda candidate: candidate.verified_score)
        padding = self.base_padding + self.adaptive_padding * (1 - best.motion_score)
        crop = Interval(max(0, best.interval.start - padding), min(duration, best.interval.end + padding))
        local = ground(crop, best)
        if local is not None:
            if len(local) != 2 or not all(math.isfinite(float(v)) for v in local):
                raise ValueError('Grounder must return two finite crop-relative timestamps or None.')
            start, end = map(float, local)
            if start < 0 or end <= start or end > crop.end - crop.start + 1e-6:
                raise ValueError('VLM timestamps must lie inside the crop and satisfy start < end.')
            result['interval'] = asdict(Interval(crop.start + start, min(duration, crop.start + end)))
        result.update({'candidates': [asdict(c) for c in candidates], 'crop': asdict(crop),
                       'selected_score': best.verified_score})
        return result
