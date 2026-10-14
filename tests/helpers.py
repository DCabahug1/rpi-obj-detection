from detect import Detection
from obstacle import LOW, NORMAL, OVERHEAD, Obstacle


def det(label: str, cx_frac: float, h_frac: float, w: int = 320, h: int = 240,
        y2_frac: float = 1.0) -> Detection:
    """Build a Detection with the given centre-x, height and bottom-edge fractions."""
    bh = int(round(h_frac * h))
    cx = int(cx_frac * w)
    y2 = int(round(y2_frac * h))
    return Detection(label=label, raw_label=label, conf=0.9,
                     x1=cx - 10, y1=y2 - bh, x2=cx + 10, y2=y2,
                     cx_frac=cx_frac, h_frac=h_frac)


_ACTION = {LOW: "step over", OVERHEAD: "duck", NORMAL: None}


def obs(label: str, cx_frac: float, h_frac: float, zone: str = NORMAL, known: bool = True,
        y2_frac: float = 1.0, spoken: str | None = None) -> Obstacle:
    """Build an Obstacle directly (bypassing the classifier) for callout tests."""
    d = det(label, cx_frac, h_frac, y2_frac=y2_frac)
    return Obstacle(det=d, label=spoken or (label if known else "object"), known=known, zone=zone,
                    action=_ACTION[zone], y1_frac=d.y1 / 240, y2_frac=d.y2 / 240)
