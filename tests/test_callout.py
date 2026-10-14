from callout import AHEAD, LEFT, RIGHT, CalloutPlanner, ObjectHistory, direction_from_cx
from distance import FAR, MEDIUM, NEAR
from helpers import obs
from obstacle import LOW, NORMAL, OVERHEAD

PRIORITY = ["person", "stairs", "bicycle", "car", "pole", "post", "object", "chair", "table"]


def planner(**kw) -> CalloutPlanner:
    kw.setdefault("min_stable_frames", 1)   # most tests look at single frames
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
    assert p.phrase("object", NEAR, LEFT, "step over") == "object, close, left, step over"
    assert p.phrase("object", NEAR, AHEAD, "step over") == "object, ahead, step over"
    assert p.phrase("sign", MEDIUM, RIGHT, "duck") == "sign, medium, right, duck"


def test_near_object_is_spoken():
    p = planner()
    c, _ = p.plan([obs("chair", 0.2, 0.7)], [NEAR], now=0.0)
    assert c is not None and c.text == "chair, close, left"


def test_far_and_still_medium_are_silent():
    p = planner()
    c, _ = p.plan([obs("chair", 0.5, 0.2)], [FAR], now=0.0)
    assert c is None
    for i in range(10):  # medium but not growing
        c, _ = p.plan([obs("chair", 0.5, 0.4)], [MEDIUM], now=float(i))
        assert c is None


def test_medium_approaching_is_spoken():
    p = planner()
    heights = [0.30, 0.32, 0.34, 0.37, 0.40]
    results = [p.plan([obs("chair", 0.5, h)], [MEDIUM], now=i * 0.2)[0] for i, h in enumerate(heights)]
    assert all(r is None for r in results[:-1])
    assert results[-1] is not None and results[-1].text == "chair, medium, ahead"


def test_priority_person_beats_chair():
    p = planner()
    c, _ = p.plan([obs("chair", 0.2, 0.9), obs("person", 0.8, 0.6)], [NEAR, NEAR], now=0.0)
    assert c.label == "person" and c.direction == RIGHT


def test_unlisted_known_class_ranks_last_but_still_spoken():
    p = planner()
    c, _ = p.plan([obs("plant", 0.5, 0.8)], [NEAR], now=0.0)
    assert c.text == "plant, ahead"
    assert c.priority_rank == len(PRIORITY)


def test_rate_limit_same_class_same_direction():
    p = planner()
    d = [obs("chair", 0.2, 0.7)]
    assert p.plan(d, [NEAR], now=0.0)[0] is not None
    assert p.plan(d, [NEAR], now=1.0)[0] is None
    assert p.plan(d, [NEAR], now=3.9)[0] is None
    assert p.plan(d, [NEAR], now=4.0)[0] is not None


def test_rate_limit_allows_different_direction():
    p = planner()
    assert p.plan([obs("chair", 0.2, 0.7)], [NEAR], now=0.0)[0].direction == LEFT
    assert p.plan([obs("chair", 0.8, 0.7)], [NEAR], now=0.5)[0].direction == RIGHT


def test_rate_limited_top_falls_through_to_next():
    p = planner()
    both = [obs("person", 0.5, 0.7), obs("chair", 0.2, 0.7)]
    first, _ = p.plan(both, [NEAR, NEAR], now=0.0)
    assert first.label == "person"
    second, _ = p.plan(both, [NEAR, NEAR], now=0.5)
    assert second.label == "chair"


def test_history_resets_when_object_disappears():
    h = ObjectHistory(window=3, min_growth_frac=0.04)
    for v in (0.3, 0.35):
        h.update(("chair", AHEAD), v)
    h.forget_missing([])
    h.update(("chair", AHEAD), 0.4)
    assert not h.is_approaching(("chair", AHEAD))


def test_escalation_to_near_bypasses_rate_limit_once():
    p = planner()
    heights = [0.30, 0.32, 0.34, 0.37, 0.40]
    for i, h in enumerate(heights):
        c, _ = p.plan([obs("person", 0.5, h)], [MEDIUM], now=i * 0.2)
    assert c.text == "person, medium, ahead"
    c, _ = p.plan([obs("person", 0.5, 0.6)], [NEAR], now=1.5)
    assert c is not None and c.text == "person, ahead"
    c, _ = p.plan([obs("person", 0.5, 0.7)], [NEAR], now=2.0)   # near again: rate limited
    assert c is None


# --- object abstraction -----------------------------------------------------------------------

def test_ambiguous_object_at_body_height_is_silent():
    p = planner()
    c, cands = p.plan([obs("backpack", 0.5, 0.8, zone=NORMAL, known=False)], [NEAR], now=0.0)
    assert c is None and cands[0].obs.label == "object"


def test_ambiguous_low_object_ahead_says_step_over():
    p = planner()
    c, _ = p.plan([obs("backpack", 0.5, 0.15, zone=LOW, known=False, y2_frac=0.95)], [NEAR], now=0.0)
    assert c.text == "object, ahead, step over"
    assert c.zone == LOW and c.action == "step over"


def test_ambiguous_low_object_to_the_side_is_described_not_prescribed():
    p = planner()
    c, _ = p.plan([obs("backpack", 0.2, 0.15, zone=LOW, known=False, y2_frac=0.95)], [NEAR], now=0.0)
    assert c.text == "object, close, left, low"
    p2 = planner(prescriptive_only_ahead=False)
    c, _ = p2.plan([obs("backpack", 0.2, 0.15, zone=LOW, known=False, y2_frac=0.95)], [NEAR], now=0.0)
    assert c.text == "object, close, left, step over"


def test_ambiguous_low_object_that_is_not_steppable_is_described():
    p = planner()
    big = obs("suitcase", 0.5, 0.15, zone=LOW, known=False, y2_frac=0.95, steppable=False, w_frac=0.6)
    c, _ = p.plan([big], [NEAR], now=0.0)
    assert c.text == "object, ahead, low"


def test_overhead_object_says_duck_ahead_and_overhead_to_the_side():
    p = planner()
    c, _ = p.plan([obs("umbrella", 0.5, 0.3, zone=OVERHEAD, known=False, y2_frac=0.3)], [NEAR], now=0.0)
    assert c.text == "object, ahead, duck"
    c, _ = p.plan([obs("umbrella", 0.8, 0.3, zone=OVERHEAD, known=False, y2_frac=0.3)], [NEAR], now=0.0)
    assert c.text == "object, close, right, overhead"


def test_known_class_low_but_not_steppable_gets_no_suffix():
    # the reported plant case: known class, low zone, not steppable -> plain callout
    p = planner()
    plant = obs("plant", 0.2, 0.45, zone=LOW, steppable=False, truncated=True)
    c, _ = p.plan([plant], [NEAR], now=0.0)
    assert c.text == "plant, close, left"


def test_known_steppable_class_ahead_gets_step_over():
    p = planner()
    c, _ = p.plan([obs("ball", 0.5, 0.10, zone=LOW, y2_frac=0.95)], [NEAR], now=0.0)
    assert c.text == "ball, ahead, step over"


def test_zoned_obstacle_waits_for_stable_frames_before_speaking():
    p = planner(min_stable_frames=3)
    o = obs("backpack", 0.5, 0.15, zone=LOW, known=False, y2_frac=0.95)
    assert p.plan([o], [NEAR], now=0.0)[0] is None
    assert p.plan([o], [NEAR], now=0.1)[0] is None
    c, _ = p.plan([o], [NEAR], now=0.2)
    assert c is not None and c.text == "object, ahead, step over"
    # normal-zone objects are not delayed
    assert planner(min_stable_frames=3).plan([obs("chair", 0.5, 0.8)], [NEAR], now=0.0)[0] is not None


def test_low_object_approaches_by_sliding_down_the_frame():
    p = planner()
    bottoms = [0.70, 0.72, 0.74, 0.77, 0.80]   # bottom edge moving down; box height constant
    results = [p.plan([obs("ball", 0.5, 0.10, zone=LOW, known=False, y2_frac=b)], [MEDIUM], now=i * 0.2)[0]
               for i, b in enumerate(bottoms)]
    assert all(r is None for r in results[:-1])
    assert results[-1].text == "object, medium, ahead, step over"


def test_object_outranks_chair_but_not_person():
    p = planner()
    low = obs("ball", 0.2, 0.10, zone=LOW, known=False, y2_frac=0.95)
    c, _ = p.plan([obs("chair", 0.8, 0.9), low], [NEAR, NEAR], now=0.0)
    assert c.label == "object"
    c, _ = p.plan([obs("person", 0.8, 0.9), low], [NEAR, NEAR], now=10.0)
    assert c.label == "person"


def test_same_box_under_a_different_label_is_not_repeated():
    p = planner()
    bus = obs("bus", 0.5, 0.6, w_frac=0.3)
    truck = obs("truck", 0.5, 0.6, w_frac=0.3)      # identical box, different label
    assert p.plan([bus], [NEAR], now=0.0)[0].label == "bus"
    assert p.plan([truck], [NEAR], now=0.3)[0] is None
    assert p.plan([truck], [NEAR], now=4.1)[0].label == "truck"   # rate-limit window over


def test_same_box_dedupe_still_allows_escalation_to_near():
    p = planner()
    heights = [0.30, 0.32, 0.34, 0.37, 0.40]
    for i, h in enumerate(heights):
        c, _ = p.plan([obs("bus", 0.5, h, w_frac=0.3)], [MEDIUM], now=i * 0.2)
    assert c.text == "bus, medium, ahead"
    # same object now read as "truck" and near: new information, spoken
    c, _ = p.plan([obs("truck", 0.5, 0.6, w_frac=0.3)], [NEAR], now=1.5)
    assert c is not None and c.text == "truck, ahead"


def test_different_boxes_same_label_are_both_spoken_in_different_directions():
    p = planner()
    assert p.plan([obs("chair", 0.2, 0.7)], [NEAR], now=0.0)[0].direction == LEFT
    assert p.plan([obs("chair", 0.8, 0.7)], [NEAR], now=0.1)[0].direction == RIGHT
