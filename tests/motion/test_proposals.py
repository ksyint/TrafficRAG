import numpy as np
import pytest

from trafficrag.systems.grounding import Interval, TrafficRAG
from trafficrag.systems.grounding.proposal import propose_candidates, segment_video
from trafficrag.knowledge import KnowledgeBase
from trafficrag.metrics import temporal_iou


def test_segments_cover_tail_without_exceeding_duration():
    segments = segment_video(5.5)
    assert segments[0] == Interval(0, 2)
    assert segments[-1] == Interval(3.5, 5.5)
    assert segment_video(0.7) == [Interval(0, 0.7)]

def test_strict_threshold_and_run_to_timestamp_mapping():
    segments = segment_video(6)
    candidates, scores = propose_candidates(segments, [0.3, 0.9, 0.8, 0.1, 0.7], sigma=0)
    assert [c.interval for c in candidates] == [Interval(1, 4), Interval(4, 6)]
    assert candidates[0].motion_score == pytest.approx(0.85)
