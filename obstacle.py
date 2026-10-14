"""Object abstraction: turns a raw Detection into an Obstacle the callout logic can reason about.

Two things happen here:

1. Name abstraction. Classes outside the spoken vocabulary (the priority list plus the class_map
   words) are announced with an ambiguous word such as "object" instead of their raw COCO name.
2. Vertical zone. From where the box sits in the frame of a chest-mounted camera:
     low      - sits on the floor close by (box top in the lower half, bottom near the frame edge)
     overhead - hangs into view from above (box top at the frame edge, bottom above mid frame)
     normal   - everything else
   low -> "step over", overhead -> "duck".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from detect import Detection

LOW, OVERHEAD, NORMAL = "low", "overhead", "normal"


@dataclass(frozen=True)
class Obstacle:
    det: Detection
    label: str              # spoken word: the class name, or the ambiguous word
    known: bool             # False when the class name was not in the vocabulary
    zone: str               # LOW | OVERHEAD | NORMAL
    action: Optional[str]   # "step over" | "duck" | None
    y1_frac: float          # box top as a fraction of frame height
    y2_frac: float          # box bottom as a fraction of frame height


class ObstacleClassifier:
    def __init__(self, vocabulary: Iterable[str], ambiguous_word: str = "object",
                 step_over_word: str = "step over", duck_word: str = "duck",
                 low_top_min: float = 0.50, low_bottom_min: float = 0.80,
                 overhead_top_max: float = 0.10, overhead_bottom_max: float = 0.50,
                 exclude: Iterable[str] = ("person",)):
        self.vocabulary = set(vocabulary) | {"person"}
        self.ambiguous_word = ambiguous_word
        self.step_over_word = step_over_word
        self.duck_word = duck_word
        self.low_top_min = low_top_min
        self.low_bottom_min = low_bottom_min
        self.overhead_top_max = overhead_top_max
        self.overhead_bottom_max = overhead_bottom_max
        self.exclude = set(exclude)

    @classmethod
    def from_config(cls, cfg: dict) -> "ObstacleClassifier":
        c, z = cfg["callout"], cfg.get("zones", {})
        vocab = set(c.get("priority", [])) | set(cfg.get("class_map", {}).values())
        return cls(
            vocabulary=vocab,
            ambiguous_word=c.get("ambiguous_word", "object"),
            step_over_word=c.get("step_over_word", "step over"),
            duck_word=c.get("duck_word", "duck"),
            low_top_min=float(z.get("low_top_min", 0.50)),
            low_bottom_min=float(z.get("low_bottom_min", 0.80)),
            overhead_top_max=float(z.get("overhead_top_max", 0.10)),
            overhead_bottom_max=float(z.get("overhead_bottom_max", 0.50)),
            exclude=z.get("exclude", ["person"]),
        )

    def zone_of(self, label: str, y1_frac: float, y2_frac: float) -> str:
        if label in self.exclude:
            return NORMAL
        if y1_frac >= self.low_top_min and y2_frac >= self.low_bottom_min:
            return LOW
        if y1_frac <= self.overhead_top_max and y2_frac <= self.overhead_bottom_max:
            return OVERHEAD
        return NORMAL

    def classify_one(self, det: Detection, frame_h: int) -> Obstacle:
        known = det.label in self.vocabulary
        label = det.label if known else self.ambiguous_word
        y1_frac = det.y1 / frame_h
        y2_frac = det.y2 / frame_h
        zone = self.zone_of(det.label, y1_frac, y2_frac)
        action = self.step_over_word if zone == LOW else self.duck_word if zone == OVERHEAD else None
        return Obstacle(det=det, label=label, known=known, zone=zone, action=action,
                        y1_frac=y1_frac, y2_frac=y2_frac)

    def classify(self, dets: Sequence[Detection], frame_h: int) -> list[Obstacle]:
        return [self.classify_one(d, frame_h) for d in dets]
