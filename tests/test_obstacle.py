from helpers import det
from obstacle import LOW, NORMAL, OVERHEAD, ObstacleClassifier

H = 240


def clf() -> ObstacleClassifier:
    return ObstacleClassifier(vocabulary=["person", "chair", "table", "bus"])


def test_known_class_keeps_its_name():
    o = clf().classify_one(det("chair", 0.5, 0.7), H)
    assert o.known and o.label == "chair" and o.zone == NORMAL and o.action is None


def test_unknown_class_becomes_ambiguous_word():
    o = clf().classify_one(det("suitcase", 0.5, 0.7), H)
    assert not o.known and o.label == "object"


def test_person_is_always_known():
    assert ObstacleClassifier(vocabulary=[]).classify_one(det("person", 0.5, 0.9), H).known


def test_floor_object_is_low_step_over():
    # small box whose bottom edge is near the frame bottom: top at 0.80, bottom at 0.95
    o = clf().classify_one(det("backpack", 0.5, 0.15, y2_frac=0.95), H)
    assert o.zone == LOW and o.action == "step over" and o.label == "object"


def test_hanging_object_is_overhead_duck():
    # box from the top edge down to 0.30 of the frame
    o = clf().classify_one(det("umbrella", 0.5, 0.30, y2_frac=0.30), H)
    assert o.zone == OVERHEAD and o.action == "duck"


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
    }
    c = ObstacleClassifier.from_config(cfg)
    assert c.classify_one(det("table", 0.5, 0.7), H).label == "table"
    assert c.classify_one(det("horse", 0.5, 0.7), H).label == "thing"
    assert c.classify_one(det("horse", 0.5, 0.15, y2_frac=0.95), H).action == "step"
