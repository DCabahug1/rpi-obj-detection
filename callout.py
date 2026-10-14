"""Decides what to say: qualification, priority, rate limiting, and phrasing.

Works on Obstacles (see obstacle.py), so it knows each object's spoken label, whether that
label is a real class name or the ambiguous word, and whether the object is low (step over),
overhead (duck) or at body height.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Hashable, Iterable, Optional, Sequence, Tuple

from detect import Detection
from distance import FAR, MEDIUM, NEAR
from obstacle import LOW, NORMAL, Obstacle

LEFT, AHEAD, RIGHT = "left", "ahead", "right"


def box_iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union if union > 0 else 0.0


def direction_from_cx(cx_frac: float, left_max: float, right_min: float) -> str:
    if cx_frac < left_max:
        return LEFT
    if cx_frac > right_min:
        return RIGHT
    return AHEAD


@dataclass(frozen=True)
class Callout:
    text: str
    label: str
    distance: str
    direction: str
    priority_rank: int   # lower = more important
    is_near: bool
    zone: str = NORMAL
    action: Optional[str] = None


@dataclass(frozen=True)
class Candidate:
    obs: Obstacle
    distance: str
    direction: str
    approaching: bool
    stable: bool = True            # zoned obstacle seen in its zone for min_stable_frames
    suffix: Optional[str] = None   # what gets appended to the phrase: action or position word

    @property
    def det(self) -> Detection:
        return self.obs.det


class ObjectHistory:
    """Remembers a recent size metric per object key to detect approach (growth)."""

    def __init__(self, window: int = 5, min_growth_frac: float = 0.04):
        self.window = window
        self.min_growth = min_growth_frac
        self._hist: Dict[Hashable, Deque[float]] = {}

    def update(self, key: Hashable, value: float) -> None:
        dq = self._hist.get(key)
        if dq is None:
            dq = deque(maxlen=self.window)
            self._hist[key] = dq
        dq.append(value)

    def is_approaching(self, key: Hashable) -> bool:
        dq = self._hist.get(key)
        if not dq or len(dq) < self.window:
            return False
        first, last = dq[0], dq[-1]
        if last - first < self.min_growth:
            return False
        # Sustained growth, not a single jump: at least half the steps grow and at most one shrinks.
        steps = list(zip(dq, list(dq)[1:]))
        grows = sum(1 for a, b in steps if b > a)
        shrinks = sum(1 for a, b in steps if b < a)
        return grows * 2 >= len(steps) and shrinks <= 1

    def count(self, key: Hashable) -> int:
        """Consecutive frames this key has been seen (capped at the window)."""
        dq = self._hist.get(key)
        return len(dq) if dq else 0

    def forget_missing(self, seen: Iterable[Hashable]) -> None:
        """Drop history for objects not seen this frame so a reappearing object starts fresh."""
        seen = set(seen)
        for key in list(self._hist):
            if key not in seen:
                del self._hist[key]


class CalloutPlanner:
    def __init__(self, priority: Sequence[str], rate_limit_s: float = 4.0,
                 left_max: float = 0.38, right_min: float = 0.62,
                 approach_window: int = 5, approach_min_growth: float = 0.04,
                 near_word: str = "close", medium_word: str = "medium",
                 prescriptive_only_ahead: bool = True, min_stable_frames: int = 3,
                 same_box_iou: float = 0.5):
        self.priority = {name: i for i, name in enumerate(priority)}
        self.rate_limit_s = rate_limit_s
        self.left_max = left_max
        self.right_min = right_min
        self.history = ObjectHistory(approach_window, approach_min_growth)
        self.near_word = near_word
        self.medium_word = medium_word
        self.prescriptive_only_ahead = prescriptive_only_ahead
        # the history window caps how many consecutive frames we can count
        self.min_stable_frames = max(1, min(int(min_stable_frames), approach_window))
        self.same_box_iou = same_box_iou
        self._last_spoken: Dict[Tuple[str, str], Tuple[float, str]] = {}  # key -> (time, distance)
        self._recent_boxes: list[Tuple[float, Detection, str]] = []        # (time, box, distance)

    def rank(self, label: str) -> int:
        return self.priority.get(label, len(self.priority))

    def phrase(self, label: str, distance: str, direction: str, suffix: Optional[str] = None) -> str:
        if distance == NEAR and direction == AHEAD:
            text = f"{label}, {direction}"
        else:
            word = self.near_word if distance == NEAR else self.medium_word
            text = f"{label}, {word}, {direction}"
        return f"{text}, {suffix}" if suffix else text

    @staticmethod
    def speakable(obs: Obstacle) -> bool:
        """Known classes speak in any zone. Ambiguous ones only when there is something to do:
        a bag carried at chest height is not a hazard, the same bag on the floor is."""
        return obs.known or obs.zone != NORMAL

    def suffix_for(self, obs: Obstacle, direction: str) -> Optional[str]:
        """What to append to the phrase.

        Prescriptive ("step over", "duck") only when the obstacle qualifies for it AND it is in the
        walking line (ahead) when prescriptive_only_ahead is set. Otherwise ambiguous obstacles get
        the descriptive position word ("low", "overhead") so "object" still carries information;
        known classes off to the side or not steppable get nothing: "plant, close, left" is enough.
        """
        if obs.zone == NORMAL:
            return None
        if obs.action and (direction == AHEAD or not self.prescriptive_only_ahead):
            return obs.action
        return obs.position if not obs.known else None

    def plan(self, obstacles: Sequence[Obstacle], distances: Sequence[str], now: float
             ) -> Tuple[Optional[Callout], list[Candidate]]:
        """Return (callout or None, all candidates with their computed fields).

        `distances` is the per-obstacle bucket from the distance estimator(s).
        """
        cands: list[Candidate] = []
        seen: list[Hashable] = []
        for obs, dist in zip(obstacles, distances):
            direction = direction_from_cx(obs.det.cx_frac, self.left_max, self.right_min)
            key = (obs.label, obs.zone, direction)
            # Low objects get closer by sliding down the frame; everything else by growing taller.
            metric = obs.y2_frac if obs.zone == LOW else obs.det.h_frac
            self.history.update(key, metric)
            seen.append(key)
            # A low/overhead reading must hold for a few frames before we act on it; a single
            # frame at the bottom of the view is not enough to say "step over".
            stable = obs.zone == NORMAL or self.history.count(key) >= self.min_stable_frames
            cands.append(Candidate(obs, dist, direction, self.history.is_approaching(key),
                                   stable=stable, suffix=self.suffix_for(obs, direction)))
        self.history.forget_missing(seen)

        qualified = [c for c in cands if self.speakable(c.obs) and c.stable
                     and (c.distance == NEAR or (c.distance == MEDIUM and c.approaching))]
        if not qualified:
            return None, cands

        # person > stairs > ... ; ties broken by the bigger (closer) box
        qualified.sort(key=lambda c: (self.rank(c.obs.label), -c.det.h_frac))

        self._recent_boxes = [r for r in self._recent_boxes if now - r[0] < self.rate_limit_s]
        for c in qualified:
            key = (c.obs.label, c.direction)
            last = self._last_spoken.get(key)
            if last is not None and now - last[0] < self.rate_limit_s:
                # Rate limited, unless this is an escalation from medium to near: that is new
                # information the user needs right away, so it bypasses the limit once.
                escalated = c.distance == NEAR and last[1] != NEAR
                if not escalated:
                    continue
            # Same physical object under a different name (YOLO flipping bus/truck, chair/couch):
            # if a box overlapping this one was spoken recently, do not announce it again.
            same = [r for r in self._recent_boxes if box_iou(r[1], c.det) >= self.same_box_iou]
            if same and not (c.distance == NEAR and all(r[2] != NEAR for r in same)):
                continue
            self._last_spoken[key] = (now, c.distance)
            self._recent_boxes.append((now, c.det, c.distance))
            return Callout(
                text=self.phrase(c.obs.label, c.distance, c.direction, c.suffix),
                label=c.obs.label, distance=c.distance, direction=c.direction,
                priority_rank=self.rank(c.obs.label), is_near=(c.distance == NEAR),
                zone=c.obs.zone, action=c.suffix,
            ), cands
        return None, cands
