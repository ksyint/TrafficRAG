import numpy as np
import pytest

from trafficrag.systems.grounding import Interval, TrafficRAG
from trafficrag.systems.grounding.proposal import propose_candidates, segment_video
from trafficrag.knowledge import KnowledgeBase
from trafficrag.metrics import temporal_iou


def test_semantics_selects_candidate_and_padding_uses_motion():
    kb = KnowledgeBase([[1, 0], [0, 1]], [1, 0])
    pipeline = TrafficRAG(kb, topk=1, smoothing_sigma=0)
    seen = {}
    def caption(interval):
        return 'true' if interval.start >= 4 else 'negative'
    def embed(text):
        return [1, 0] if text == 'true' else [0, 1]
    def ground(crop, candidate):
        seen['crop'] = crop
        return (0.2, 0.8)
    result = pipeline(8, [0.9, 0.1, 0.1, 0.1, 0.6, 0.1, 0.1], caption, embed, ground)
    # First score=.54. Second=.4 + .6*.6=.76, so second must win.
    assert result['selected_score'] == pytest.approx(0.76)
    assert seen['crop'].start == pytest.approx(4 - (0.5 + 0.4))
    assert result['interval']['start'] == pytest.approx(3.3)

def test_empty_proposal_never_calls_models():
    kb = KnowledgeBase([[1]], [1])
    def forbidden(*args):
        raise AssertionError('Backend should not be invoked.')
    result = TrafficRAG(kb)(2, [0.1], forbidden, forbidden, forbidden)
    assert result['interval'] is None

def test_invalid_local_grounding_is_rejected():
    kb = KnowledgeBase([[1]], [1])
    with pytest.raises(ValueError, match='inside the crop'):
        TrafficRAG(kb)(2, [0.9], lambda interval: 'x', lambda text: [1], lambda crop, candidate: (-1, 3))
