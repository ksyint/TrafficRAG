import numpy as np
import pytest

from trafficrag.systems.grounding import Interval, TrafficRAG
from trafficrag.systems.grounding.proposal import propose_candidates, segment_video
from trafficrag.knowledge import KnowledgeBase
from trafficrag.metrics import temporal_iou


def test_retrieval_uses_cosine_and_actual_neighbor_count():
    kb = KnowledgeBase([[100, 0], [1, 1], [0, 3]], [1, 0, 0])
    ids, _, score = kb.retrieve([2, 0], 2)
    assert ids.tolist() == [0, 1]
    assert score == 0.5
    assert kb.retrieve([2, 0], 10)[2] == pytest.approx(1 / 3)
