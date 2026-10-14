from detect import Detection


def det(label: str, cx_frac: float, h_frac: float, w: int = 320, h: int = 240) -> Detection:
    """Build a Detection with the given centre-x and height fractions."""
    bh = int(h_frac * h)
    cx = int(cx_frac * w)
    return Detection(label=label, raw_label=label, conf=0.9,
                     x1=cx - 10, y1=h - bh, x2=cx + 10, y2=h,
                     cx_frac=cx_frac, h_frac=h_frac)
