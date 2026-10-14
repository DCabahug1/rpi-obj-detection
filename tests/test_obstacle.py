from helpers import det
from obstacle import LOW, NORMAL, OVERHEAD, ObstacleClassifier

H = 240


def clf() -> ObstacleClassifier:
    return ObstacleClassifier(vocabulary=["person", "chair", "table", "bus", "plant"],
                              never_step_over=["chair", "table", "plant", "bus", "cat"])


def test_known_class_keeps_its_name():
    o = clf().classify_one(det("chair", 0.5, 0.7), H)
    assert o.known and o.label == "chair" and o.zone == NORMAL and o.action is None


def test_unknown_class_becomes_ambiguous_word():
    o = clf().classify_one(det("suitcase", 0.5, 0.7), H)
    assert not o.known and o.label == "object"


def test_person_is_always_known():
    assert ObstacleClassifier(vocabulary=[]).classify_one(det("person", 0.5, 0.9), H).known


def test_small_floor_object_is_low_and_steppable():
    # small box whose bottom edge is near the frame bottom: top at 0.80, bottom at 0.95
    o = clf().classify_one(det("backpack", 0.5, 0.15, y2_frac=0.95), H)
    assert o.zone == LOW and o.steppable and o.action == "step over" and o.position == "low"
    assert o.label == "object" and not o.truncated


def test_hanging_object_is_overhead_duck():
    # box from the top edge down to 0.30 of the frame
    o = clf().classify_one(det("umbrella", 0.5, 0.30, y2_frac=0.30), H)
    assert o.zone == OVERHEAD and o.action == "duck" and o.position == "overhead"


# --- the action is a separate, confidence-gated decision -----------------------------------------

def test_tall_plant_cut_off_in_the_corner_is_low_but_not_steppable():
    # the reported case: a tall plant at the bottom-left corner, box touching the left edge,
    # top just below mid frame. Low by position, but neither small nor fully visible.
    d = det("plant", 0.12, 0.45, y2_frac=1.0, w_frac=0.24)   # x1 = 0.0 -> truncated
    o = clf().classify_one(d, H)
    assert o.zone == LOW
    assert o.truncated and not o.steppable
    assert o.action is None and o.position == "low"


def test_big_low_object_is_not_steppable():
    # a flat suitcase: short but wide
    o = clf().classify_one(det("suitcase", 0.5, 0.15, y2_frac=0.95, w_frac=0.60), H)
    assert o.zone == LOW and not o.steppable and o.action is None


def test_never_step_over_class_even_when_small():
    o = clf().classify_one(det("cat", 0.5, 0.15, y2_frac=0.95), H)
    assert o.zone == LOW and not o.steppable and o.action is None and o.position == "low"


def test_object_touching_right_edge_is_truncated():
    o = clf().classify_one(det("backpack", 0.95, 0.15, y2_frac=0.95, w_frac=0.12), H)
    assert o.truncated and not o.steppable and o.action is None


def test_known_small_class_can_be_stepped_over_when_allowed():
    c = ObstacleClassifier(vocabulary=["ball"], never_step_over=[])
    o = c.classify_one(det("ball", 0.5, 0.10, y2_frac=0.95), H)
    assert o.known and o.steppable and o.action == "step over"


def test_mid_frame_small_object_is_normal():
    # top 0.50, bottom 0.60: not on the floor, not overhead
    o = clf().classify_one(det("bottle", 0.5, 0.10, y2_frac=0.60), H)
    assert o.zone == NORMAL and o.action is None


def test_near_person_is_normal_and_small_low_person_is_never_low():
    c = clf()
    assert c.classify_one(det("person", 0.5, 0.98), H).zone == NORMAL
    assert c.classify_one(det("person", 0.5, 0.15, y2_frac=0.95), H).zone == NORMAL


def test_from_config_builds_vocabulary_from_priority_and_class_map():
    cfg = {
        "callout": {"priority": ["person", "chair"], "ambiguous_word": "thing",
                    "step_over_word": "step", "duck_word": "duck"},
        "class_map": {"dining table": "table"},
        "zones": {"low_top_min": 0.5, "low_bottom_min": 0.8, "overhead_top_max": 0.1,
                  "overhead_bottom_max": 0.5, "exclude": ["person"]},
        "actions": {"low_word": "down", "step_over_max_h": 0.3, "step_over_max_w": 0.45,
                    "edge_margin_frac": 0.02, "never_step_over": ["horse"]},
    }
    c = ObstacleClassifier.from_config(cfg)
    assert c.classify_one(det("table", 0.5, 0.7), H).label == "table"
    assert c.classify_one(det("horse", 0.5, 0.7), H).label == "thing"
    low_horse = c.classify_one(det("horse", 0.5, 0.15, y2_frac=0.95), H)
    assert low_horse.action is None and low_horse.position == "down"
    assert c.classify_one(det("ball", 0.5, 0.15, y2_frac=0.95), H).action == "step"
