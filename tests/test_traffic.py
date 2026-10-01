"""Temporal proposal and retrieval invariants."""

import pytest
from trafficrag.grounding import (
    Interval,
    TrafficRAG,
    propose_candidates,
    segment_video,
    KnowledgeBase,
    temporal_iou,
)


def test_temporal_iou_and_missed_positive():
    assert temporal_iou(Interval(1, 3), Interval(2, 4)) == pytest.approx(1 / 3)
    assert temporal_iou(None, Interval(2, 4)) == 0


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


def test_retrieval_uses_cosine_and_actual_neighbor_count():
    kb = KnowledgeBase([[100, 0], [1, 1], [0, 3]], [1, 0, 0])
    ids, _, score = kb.retrieve([2, 0], 2)
    assert ids.tolist() == [0, 1]
    assert score == 0.5
    assert kb.retrieve([2, 0], 10)[2] == pytest.approx(1 / 3)
