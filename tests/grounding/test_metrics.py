import numpy as np
import pytest

from trafficrag.systems.grounding import Interval, TrafficRAG
from trafficrag.systems.grounding.proposal import propose_candidates, segment_video
from trafficrag.knowledge import KnowledgeBase
from trafficrag.metrics import temporal_iou


def test_temporal_iou_and_missed_positive():
    assert temporal_iou(Interval(1, 3), Interval(2, 4)) == pytest.approx(1 / 3)
    assert temporal_iou(None, Interval(2, 4)) == 0
