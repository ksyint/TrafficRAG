"""Motion proposal, caption retrieval, and temporal boundary refinement."""

import math
import numpy as np
from dataclasses import dataclass, asdict, field
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class Interval:
    start: float
    end: float

    def __post_init__(self):
        if (
            not math.isfinite(self.start)
            or not math.isfinite(self.end)
            or self.start < 0
            or self.end <= self.start
        ):
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


@dataclass
class ProposalConfig:
    threshold: float = 0.3
    sigma: float = 1.0
    window: float = 2.0
    stride: float = 1.0

    def __post_init__(self):
        if not 0 <= self.threshold <= 1 or self.sigma < 0:
            raise ValueError('Proposal threshold must lie in [0,1] and smoothing sigma must be nonnegative.')
        if not all(math.isfinite(value) for value in (self.threshold, self.sigma, self.window, self.stride)):
            raise ValueError('Proposal parameters must be finite.')
        if not 0 < self.stride <= self.window:
            raise ValueError('Proposal windows require 0 < stride <= window.')


class MotionProposal:
    def __init__(self, config):
        self.cfg = config

    def __call__(self, duration, scores):
        segments = segment_video(duration, self.cfg.window, self.cfg.stride)
        return propose_candidates(segments, scores, self.cfg.threshold, self.cfg.sigma)


@dataclass
class VerificationConfig:
    semantic_weight: float = 0.4
    topk: int = 10

    def __post_init__(self):
        if not 0 <= self.semantic_weight <= 1 or self.topk < 1:
            raise ValueError('Verification needs 0 <= semantic_weight <= 1 and topk >= 1.')


class SemanticVerification:
    def __init__(self, config, knowledge_base):
        self.cfg = config
        self.knowledge_base = knowledge_base

    def __call__(self, candidates, caption, embed):
        for candidate in candidates:
            candidate.caption = caption(candidate.interval)
            neighbors, similarities, ratio = self.knowledge_base.retrieve(
                embed(candidate.caption), self.cfg.topk
            )
            candidate.semantic_score = ratio
            weight = self.cfg.semantic_weight
            candidate.verified_score = weight * ratio + (1 - weight) * candidate.motion_score
            candidate.neighbors = [
                dict(
                    id=self.knowledge_base.ids[int(i)],
                    similarity=float(similarity),
                    label=int(self.knowledge_base.labels[i]),
                    caption=self.knowledge_base.captions[int(i)],
                )
                for i, similarity in zip(neighbors, similarities)
            ]
        return max(candidates, key=lambda candidate: candidate.verified_score)


@dataclass
class RefinementConfig:
    base_padding: float = 0.5
    adaptive_padding: float = 1.0

    def __post_init__(self):
        if min(self.base_padding, self.adaptive_padding) < 0:
            raise ValueError('Padding must be nonnegative.')


class BoundaryRefinement:
    def __init__(self, config):
        self.cfg = config

    def crop(self, candidate, duration):
        padding = self.cfg.base_padding + self.cfg.adaptive_padding * (1 - candidate.motion_score)
        return Interval(
            max(0, candidate.interval.start - padding),
            min(duration, candidate.interval.end + padding),
        )

    def __call__(self, candidate, duration, ground):
        crop = self.crop(candidate, duration)
        local = ground(crop, candidate)
        if local is None:
            return crop, None
        if len(local) != 2 or not all(math.isfinite(float(v)) for v in local):
            raise ValueError('Grounder must return two finite crop-relative timestamps or None.')
        start, end = map(float, local)
        if start < 0 or end <= start or end > crop.end - crop.start + 1e-6:
            raise ValueError('VLM timestamps must lie inside the crop and satisfy start < end.')
        return crop, Interval(crop.start + start, min(duration, crop.start + end))


class TrafficRAG:
    """Configurable proposal -> verification -> refinement system."""

    @dataclass
    class Config:
        proposal: ProposalConfig = field(default_factory=ProposalConfig)
        verification: VerificationConfig = field(default_factory=VerificationConfig)
        refinement: RefinementConfig = field(default_factory=RefinementConfig)

        @classmethod
        def from_dict(cls, values):
            values = dict(values)
            nested = {'proposal', 'verification', 'refinement'}
            if set(values) <= nested:
                return cls(
                    ProposalConfig(**values.get('proposal', {})),
                    VerificationConfig(**values.get('verification', {})),
                    RefinementConfig(**values.get('refinement', {})),
                )
            # Load earlier flat configurations and keyword construction transparently.
            mapping = {
                'motion_threshold': ('proposal', 'threshold'),
                'smoothing_sigma': ('proposal', 'sigma'),
                'window': ('proposal', 'window'),
                'stride': ('proposal', 'stride'),
                'semantic_weight': ('verification', 'semantic_weight'),
                'topk': ('verification', 'topk'),
                'base_padding': ('refinement', 'base_padding'),
                'adaptive_padding': ('refinement', 'adaptive_padding'),
            }
            grouped = {key: {} for key in nested}
            for key, value in values.items():
                if key not in mapping:
                    raise ValueError(f'Unknown TrafficRAG setting: {key}')
                group, setting = mapping[key]
                grouped[group][setting] = value
            return cls.from_dict(grouped)

    def __init__(self, knowledge_base, config=None, **settings):
        if config is not None and settings:
            raise ValueError('Provide a Config object or keyword settings, not both.')
        self.cfg = config or self.Config.from_dict(settings)
        self.knowledge_base = knowledge_base
        self.configure()

    def configure(self):
        self.propose = MotionProposal(self.cfg.proposal)
        self.verify = SemanticVerification(self.cfg.verification, self.knowledge_base)
        self.refine = BoundaryRefinement(self.cfg.refinement)

    def __call__(self, duration, motion_scores, caption, embed, ground):
        candidates, smoothed = self.propose(duration, motion_scores)
        result = {
            'interval': None,
            'candidates': [],
            'smoothed_scores': smoothed.tolist(),
            'crop': None,
        }
        if not candidates:
            return result
        best = self.verify(candidates, caption, embed)
        crop, interval = self.refine(best, duration, ground)
        result.update(
            interval=asdict(interval) if interval is not None else None,
            candidates=[asdict(candidate) for candidate in candidates],
            crop=asdict(crop),
            selected_score=best.verified_score,
        )
        return result


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
        if (
            query.shape != (self.embeddings.shape[1],)
            or not np.isfinite(query).all()
            or np.linalg.norm(query) == 0
        ):
            raise ValueError('Query embedding shape/value mismatch.')
        if k < 1:
            raise ValueError('K must be positive.')
        similarities = self.embeddings @ (query / np.linalg.norm(query))
        order = np.argsort(-similarities, kind='stable')[: min(k, len(similarities))]
        return order, similarities[order], float(self.labels[order].mean())


def load_kb(path):
    with np.load(path, allow_pickle=False) as data:
        return KnowledgeBase(
            data['embeddings'], data['labels'], data['captions'].tolist(), data['ids'].tolist()
        )


def temporal_iou(a, b):
    if a is None or b is None:
        return 0.0
    intersection = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return intersection / union
