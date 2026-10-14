"""Object abstraction: turns a raw Detection into an Obstacle the callout logic can reason about.

Three things happen here:

1. Name abstraction. Classes outside the spoken vocabulary (the priority list plus the class_map
   words) are announced with an ambiguous word such as "object" instead of their raw COCO name.
2. Vertical zone, from where the box sits in the frame of a chest-mounted camera:
     low      - sits on the floor close by (box top in the lower half, bottom near the frame edge)
     overhead - hangs into view from above (box top at the frame edge, bottom above mid frame)
     normal   - everything else
3. Action. The zone only says WHERE the thing is. Whether the user should "step over" or "duck" is
   a separate decision that also needs the object to be small enough, of a class that can be
   stepped over, and fully in view. When that is not certain the obstacle only carries a
   descriptive position word ("low", "overhead") and the callout never prescribes a move.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from detect import Detection

LOW, OVERHEAD, NORMAL = "low", "overhead", "normal"


@dataclass(frozen=True)
class Obstacle:
    det: Detection
    label: str                 # spoken word: the class name, or the ambiguous word
    known: bool                # False when the class name was not in the vocabulary
    zone: str                  # LOW | OVERHEAD | NORMAL
    action: Optional[str]      # prescriptive: "step over" | "duck" | None (only when confident)
    position: Optional[str]    # descriptive: "low" | "overhead" | None
    steppable: bool            # small, fully visible, and of a class one can step over
    truncated: bool            # touches the left or right frame edge: size is not trustworthy
    y1_frac: float             # box top as a fraction of frame height
    y2_frac: float             # box bottom as a fraction of frame height


class ObstacleClassifier:
    def __init__(self, vocabulary: Iterable[str], ambiguous_word: str = "object",
                 step_over_word: str = "step over", duck_word: str = "duck",
                 low_word: str = "low", overhead_word: str = "overhead",
                 low_top_min: float = 0.50, low_bottom_min: float = 0.80,
                 overhead_top_max: float = 0.10, overhead_bottom_max: float = 0.50,
                 exclude: Iterable[str] = ("person",),
                 step_over_max_h: float = 0.30, step_over_max_w: float = 0.45,
                 edge_margin_frac: float = 0.02,
                 never_step_over: Iterable[str] = ()):
        self.vocabulary = set(vocabulary) | {"person"}
        self.ambiguous_word = ambiguous_word
        self.step_over_word = step_over_word
        self.duck_word = duck_word
        self.low_word = low_word
        self.overhead_word = overhead_word
        self.low_top_min = low_top_min
        self.low_bottom_min = low_bottom_min
        self.overhead_top_max = overhead_top_max
        self.overhead_bottom_max = overhead_bottom_max
        self.exclude = set(exclude)
        self.step_over_max_h = step_over_max_h
        self.step_over_max_w = step_over_max_w
        self.edge_margin = edge_margin_frac
        self.never_step_over = set(never_step_over) | {"person"}

    @classmethod
    def from_config(cls, cfg: dict) -> "ObstacleClassifier":
        c, z, a = cfg["callout"], cfg.get("zones", {}), cfg.get("actions", {})
        vocab = set(c.get("priority", [])) | set(cfg.get("class_map", {}).values())
        return cls(
            vocabulary=vocab,
            ambiguous_word=c.get("ambiguous_word", "object"),
            step_over_word=c.get("step_over_word", "step over"),
            duck_word=c.get("duck_word", "duck"),
            low_word=a.get("low_word", "low"),
            overhead_word=a.get("overhead_word", "overhead"),
            low_top_min=float(z.get("low_top_min", 0.50)),
            low_bottom_min=float(z.get("low_bottom_min", 0.80)),
            overhead_top_max=float(z.get("overhead_top_max", 0.10)),
            overhead_bottom_max=float(z.get("overhead_bottom_max", 0.50)),
            exclude=z.get("exclude", ["person"]),
            step_over_max_h=float(a.get("step_over_max_h", 0.30)),
            step_over_max_w=float(a.get("step_over_max_w", 0.45)),
            edge_margin_frac=float(a.get("edge_margin_frac", 0.02)),
            never_step_over=a.get("never_step_over", []),
        )

    # -- zone -------------------------------------------------------------------------------------

    def zone_of(self, label: str, y1_frac: float, y2_frac: float) -> str:
        if label in self.exclude:
            return NORMAL
        if y1_frac >= self.low_top_min and y2_frac >= self.low_bottom_min:
            return LOW
        if y1_frac <= self.overhead_top_max and y2_frac <= self.overhead_bottom_max:
            return OVERHEAD
        return NORMAL

    # -- action -----------------------------------------------------------------------------------

    def is_truncated(self, det: Detection) -> bool:
        """True when the box touches the left or right frame edge, i.e. the object continues
        out of view and its apparent size says nothing about its real size."""
        x1_frac = det.cx_frac - det.w_frac / 2
        x2_frac = det.cx_frac + det.w_frac / 2
        return x1_frac <= self.edge_margin or x2_frac >= 1.0 - self.edge_margin

    def is_steppable(self, det: Detection, zone: str, truncated: bool) -> bool:
        if zone != LOW or truncated:
            return False
        if det.raw_label in self.never_step_over or det.label in self.never_step_over:
            return False
        return det.h_frac <= self.step_over_max_h and det.w_frac <= self.step_over_max_w

    # -- classify ---------------------------------------------------------------------------------

    def classify_one(self, det: Detection, frame_h: int) -> Obstacle:
        known = det.label in self.vocabulary
        label = det.label if known else self.ambiguous_word
        y1_frac = det.y1 / frame_h
        y2_frac = det.y2 / frame_h
        zone = self.zone_of(det.label, y1_frac, y2_frac)
        truncated = self.is_truncated(det)
        steppable = self.is_steppable(det, zone, truncated)

        if zone == LOW:
            position = self.low_word
            action = self.step_over_word if steppable else None
        elif zone == OVERHEAD:
            position = self.overhead_word
            action = self.duck_word
        else:
            position = None
            action = None

        return Obstacle(det=det, label=label, known=known, zone=zone, action=action,
                        position=position, steppable=steppable, truncated=truncated,
                        y1_frac=y1_frac, y2_frac=y2_frac)

    def classify(self, dets: Sequence[Detection], frame_h: int) -> list[Obstacle]:
        return [self.classify_one(d, frame_h) for d in dets]
