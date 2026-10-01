from dataclasses import dataclass
import math


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
