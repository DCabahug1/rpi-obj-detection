"""Decides what to say: qualification, priority, rate limiting, and phrasing."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Optional, Sequence, Tuple

from detect import Detection
from distance import FAR, MEDIUM, NEAR

LEFT, AHEAD, RIGHT = "left", "ahead", "right"


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


@dataclass(frozen=True)
class Candidate:
    det: Detection
    distance: str
    direction: str
    approaching: bool


class ObjectHistory:
    """Remembers recent box heights per (label, direction) to detect approach (growth)."""

    def __init__(self, window: int = 5, min_growth_frac: float = 0.04):
        self.window = window
        self.min_growth = min_growth_frac
        self._hist: Dict[Tuple[str, str], Deque[float]] = {}

    def update(self, label: str, direction: str, h_frac: float) -> None:
        key = (label, direction)
        dq = self._hist.get(key)
        if dq is None:
            dq = deque(maxlen=self.window)
            self._hist[key] = dq
        dq.append(h_frac)

    def is_approaching(self, label: str, direction: str) -> bool:
        dq = self._hist.get((label, direction))
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

    def forget_missing(self, seen: Iterable[Tuple[str, str]]) -> None:
        """Drop history for objects not seen this frame so a reappearing object starts fresh."""
        seen = set(seen)
        for key in list(self._hist):
            if key not in seen:
                del self._hist[key]


class CalloutPlanner:
    def __init__(self, priority: Sequence[str], rate_limit_s: float = 4.0,
                 left_max: float = 0.38, right_min: float = 0.62,
                 approach_window: int = 5, approach_min_growth: float = 0.04,
                 near_word: str = "close", medium_word: str = "medium"):
        self.priority = {name: i for i, name in enumerate(priority)}
        self.rate_limit_s = rate_limit_s
        self.left_max = left_max
        self.right_min = right_min
        self.history = ObjectHistory(approach_window, approach_min_growth)
        self.near_word = near_word
        self.medium_word = medium_word
        self._last_spoken: Dict[Tuple[str, str], Tuple[float, str]] = {}  # key -> (time, distance)

    def rank(self, label: str) -> int:
        return self.priority.get(label, len(self.priority))

    def phrase(self, label: str, distance: str, direction: str) -> str:
        if distance == NEAR and direction == AHEAD:
            return f"{label}, {direction}"
        word = self.near_word if distance == NEAR else self.medium_word
        return f"{label}, {word}, {direction}"

    def plan(self, dets: Sequence[Detection], distances: Sequence[str], now: float
             ) -> Tuple[Optional[Callout], list[Candidate]]:
        """Return (callout or None, all candidates with their computed fields).

        `distances` is the per-detection bucket from the distance estimator.
        """
        cands: list[Candidate] = []
        seen: list[Tuple[str, str]] = []
        for det, dist in zip(dets, distances):
            direction = direction_from_cx(det.cx_frac, self.left_max, self.right_min)
            self.history.update(det.label, direction, det.h_frac)
            seen.append((det.label, direction))
            approaching = self.history.is_approaching(det.label, direction)
            cands.append(Candidate(det, dist, direction, approaching))
        self.history.forget_missing(seen)

        qualified = [c for c in cands if c.distance == NEAR or (c.distance == MEDIUM and c.approaching)]
        if not qualified:
            return None, cands

        # person > stairs > ... ; ties broken by the bigger (closer) box
        qualified.sort(key=lambda c: (self.rank(c.det.label), -c.det.h_frac))

        for c in qualified:
            key = (c.det.label, c.direction)
            last = self._last_spoken.get(key)
            if last is not None and now - last[0] < self.rate_limit_s:
                # Rate limited, unless this is an escalation from medium to near: that is new
                # information the user needs right away, so it bypasses the limit once.
                escalated = c.distance == NEAR and last[1] != NEAR
                if not escalated:
                    continue
            self._last_spoken[key] = (now, c.distance)
            return Callout(
                text=self.phrase(c.det.label, c.distance, c.direction),
                label=c.det.label, distance=c.distance, direction=c.direction,
                priority_rank=self.rank(c.det.label), is_near=(c.distance == NEAR),
            ), cands
        return None, cands
