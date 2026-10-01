from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .types import Candidate, Interval


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
