import numpy as np
import pytest

from distance import FAR, MEDIUM, NEAR, BoxHeightEstimator, MidasEstimator
from helpers import det


def test_buckets():
    e = BoxHeightEstimator(near_frac=0.55, medium_frac=0.30)
    assert e.bucket(0.9) == NEAR
    assert e.bucket(0.55) == NEAR
    assert e.bucket(0.4) == MEDIUM
    assert e.bucket(0.30) == MEDIUM
    assert e.bucket(0.1) == FAR


def test_bad_thresholds():
    with pytest.raises(ValueError):
        BoxHeightEstimator(near_frac=0.3, medium_frac=0.5)


def _fake_depth(frame):
    """Left half of the frame reads 'close' (high inverse depth), right half 'far'."""
    h, w = frame.shape[:2]
    d = np.zeros((h, w), np.float32)
    d[:, : w // 2] = 10.0
    d[:, w // 2 :] = 1.0
    return d


def test_midas_uses_box_median_and_runs_every_nth_frame():
    box = BoxHeightEstimator(0.55, 0.30)
    m = MidasEstimator(every_n_frames=3, near=0.7, medium=0.45, fallback=box, infer=_fake_depth)
    frame = np.zeros((240, 320, 3), np.uint8)
    left, right = det("chair", 0.2, 0.2), det("chair", 0.8, 0.9)
    for i in range(6):
        m.tick()
        # a tiny far-looking box on the left is near by depth; a huge box on the right is far
        assert m.estimate(frame, left) == NEAR
        assert m.estimate(frame, right) == FAR
    assert m.runs == 2  # frames 0 and 3 only


def test_midas_falls_back_for_degenerate_box():
    box = BoxHeightEstimator(0.55, 0.30)
    m = MidasEstimator(1, 0.7, 0.45, fallback=box, infer=_fake_depth)
    m.tick()
    frame = np.zeros((240, 320, 3), np.uint8)
    d = det("chair", 0.5, 0.6)
    bad = type(d)(**{**d.__dict__, "x1": 300, "x2": 300})
    assert m.estimate(frame, bad) == NEAR  # from box height 0.6


def test_bottom_edge_buckets():
    from distance import BottomEdgeEstimator

    e = BottomEdgeEstimator(near_bottom=0.90, medium_bottom=0.75)
    assert e.bucket(0.95) == NEAR
    assert e.bucket(0.90) == NEAR
    assert e.bucket(0.80) == MEDIUM
    assert e.bucket(0.60) == FAR
    frame = np.zeros((240, 320, 3), np.uint8)
    assert e.estimate(frame, det("ball", 0.5, 0.1, y2_frac=0.95)) == NEAR
    assert e.estimate(frame, det("ball", 0.5, 0.1, y2_frac=0.70)) == FAR
    with pytest.raises(ValueError):
        BottomEdgeEstimator(0.5, 0.9)
