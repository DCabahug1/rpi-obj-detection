from callout import AHEAD, LEFT, RIGHT, CalloutPlanner, ObjectHistory, direction_from_cx
from distance import FAR, MEDIUM, NEAR
from helpers import det

PRIORITY = ["person", "stairs", "bicycle", "car", "pole", "post", "chair", "table"]


def planner(**kw) -> CalloutPlanner:
    return CalloutPlanner(priority=PRIORITY, rate_limit_s=4.0, left_max=0.38, right_min=0.62,
                          approach_window=5, approach_min_growth=0.04, **kw)


def test_direction():
    assert direction_from_cx(0.1, 0.38, 0.62) == LEFT
    assert direction_from_cx(0.5, 0.38, 0.62) == AHEAD
    assert direction_from_cx(0.9, 0.38, 0.62) == RIGHT


def test_phrasing():
    p = planner()
    assert p.phrase("chair", NEAR, LEFT) == "chair, close, left"
    assert p.phrase("person", NEAR, AHEAD) == "person, ahead"
    assert p.phrase("car", MEDIUM, RIGHT) == "car, medium, right"


def test_near_object_is_spoken():
    p = planner()
    c, _ = p.plan([det("chair", 0.2, 0.7)], [NEAR], now=0.0)
    assert c is not None and c.text == "chair, close, left"


def test_far_and_still_medium_are_silent():
    p = planner()
    c, _ = p.plan([det("chair", 0.5, 0.2)], [FAR], now=0.0)
    assert c is None
    for i in range(10):  # medium but not growing
        c, _ = p.plan([det("chair", 0.5, 0.4)], [MEDIUM], now=float(i))
        assert c is None


def test_medium_approaching_is_spoken():
    p = planner()
    heights = [0.30, 0.32, 0.34, 0.37, 0.40]
    results = [p.plan([det("chair", 0.5, h)], [MEDIUM], now=i * 0.2)[0] for i, h in enumerate(heights)]
    assert all(r is None for r in results[:-1])
    assert results[-1] is not None and results[-1].text == "chair, medium, ahead"


def test_priority_person_beats_chair():
    p = planner()
    c, _ = p.plan([det("chair", 0.2, 0.9), det("person", 0.8, 0.6)], [NEAR, NEAR], now=0.0)
    assert c.label == "person" and c.direction == RIGHT


def test_unlisted_class_ranks_last_but_still_spoken():
    p = planner()
    c, _ = p.plan([det("plant", 0.5, 0.8)], [NEAR], now=0.0)
    assert c.text == "plant, ahead"
    assert c.priority_rank == len(PRIORITY)


def test_rate_limit_same_class_same_direction():
    p = planner()
    d = [det("chair", 0.2, 0.7)]
    assert p.plan(d, [NEAR], now=0.0)[0] is not None
    assert p.plan(d, [NEAR], now=1.0)[0] is None
    assert p.plan(d, [NEAR], now=3.9)[0] is None
    assert p.plan(d, [NEAR], now=4.0)[0] is not None


def test_rate_limit_allows_different_direction():
    p = planner()
    assert p.plan([det("chair", 0.2, 0.7)], [NEAR], now=0.0)[0].direction == LEFT
    assert p.plan([det("chair", 0.8, 0.7)], [NEAR], now=0.5)[0].direction == RIGHT


def test_rate_limited_top_falls_through_to_next():
    p = planner()
    both = [det("person", 0.5, 0.7), det("chair", 0.2, 0.7)]
    first, _ = p.plan(both, [NEAR, NEAR], now=0.0)
    assert first.label == "person"
    second, _ = p.plan(both, [NEAR, NEAR], now=0.5)
    assert second.label == "chair"


def test_history_resets_when_object_disappears():
    h = ObjectHistory(window=3, min_growth_frac=0.04)
    for v in (0.3, 0.35):
        h.update("chair", AHEAD, v)
    h.forget_missing([])
    h.update("chair", AHEAD, 0.4)
    assert not h.is_approaching("chair", AHEAD)


def test_escalation_to_near_bypasses_rate_limit_once():
    p = planner()
    heights = [0.30, 0.32, 0.34, 0.37, 0.40]
    for i, h in enumerate(heights):
        c, _ = p.plan([det("person", 0.5, h)], [MEDIUM], now=i * 0.2)
    assert c.text == "person, medium, ahead"
    c, _ = p.plan([det("person", 0.5, 0.6)], [NEAR], now=1.5)
    assert c is not None and c.text == "person, ahead"
    c, _ = p.plan([det("person", 0.5, 0.7)], [NEAR], now=2.0)   # near again: rate limited
    assert c is None
