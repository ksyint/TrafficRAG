from dataclasses import dataclass
import math

from .types import Interval


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
        return Interval(max(0, candidate.interval.start - padding), min(duration, candidate.interval.end + padding))

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
