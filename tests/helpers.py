from detect import Detection
from obstacle import LOW, NORMAL, OVERHEAD, Obstacle


def det(label: str, cx_frac: float, h_frac: float, w: int = 320, h: int = 240,
        y2_frac: float = 1.0, w_frac: float = 0.0625) -> Detection:
    """Build a Detection with the given centre-x, height, bottom-edge and width fractions."""
    bh = int(round(h_frac * h))
    bw = int(round(w_frac * w))
    cx = int(cx_frac * w)
    y2 = int(round(y2_frac * h))
    return Detection(label=label, raw_label=label, conf=0.9,
                     x1=cx - bw // 2, y1=y2 - bh, x2=cx + bw // 2, y2=y2,
                     cx_frac=cx_frac, h_frac=h_frac, w_frac=w_frac)


_ACTION = {LOW: "step over", OVERHEAD: "duck", NORMAL: None}
_POSITION = {LOW: "low", OVERHEAD: "overhead", NORMAL: None}


def obs(label: str, cx_frac: float, h_frac: float, zone: str = NORMAL, known: bool = True,
        y2_frac: float = 1.0, spoken: str | None = None, steppable: bool | None = None,
        truncated: bool = False, w_frac: float = 0.0625) -> Obstacle:
    """Build an Obstacle directly (bypassing the classifier) for callout tests.

    By default a low obstacle is steppable and an overhead one duckable, as the old tests assume.
    """
    d = det(label, cx_frac, h_frac, y2_frac=y2_frac, w_frac=w_frac)
    if steppable is None:
        steppable = zone == LOW
    action = _ACTION[zone] if (zone == OVERHEAD or steppable) else None
    return Obstacle(det=d, label=spoken or (label if known else "object"), known=known, zone=zone,
                    action=action, position=_POSITION[zone], steppable=steppable,
                    truncated=truncated, y1_frac=d.y1 / 240, y2_frac=d.y2 / 240)
