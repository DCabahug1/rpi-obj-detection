"""Distance estimation: near / medium / far.

Default: bounding-box height as a fraction of frame height (no depth model).
Optional: MiDaS small (only when --depth is passed) runs every Nth frame and the
median relative depth inside the box replaces the box-height estimate.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from detect import Detection

log = logging.getLogger(__name__)

NEAR, MEDIUM, FAR = "near", "medium", "far"


class BoxHeightEstimator:
    def __init__(self, near_frac: float, medium_frac: float):
        if not 0 < medium_frac < near_frac <= 1.0:
            raise ValueError("need 0 < medium_frac < near_frac <= 1")
        self.near_frac = near_frac
        self.medium_frac = medium_frac

    def bucket(self, h_frac: float) -> str:
        if h_frac >= self.near_frac:
            return NEAR
        if h_frac >= self.medium_frac:
            return MEDIUM
        return FAR

    def estimate(self, frame_bgr: np.ndarray, det: Detection) -> str:
        return self.bucket(det.h_frac)

    def tick(self) -> None:
        """Called once per frame. No-op for the box estimator."""


class MidasEstimator:
    """MiDaS_small relative depth. Larger value = closer (MiDaS outputs inverse depth).

    Runs the network every `every_n_frames`; in between, the cached depth map is reused.
    The map is min-max normalised per frame to 0..1 so thresholds are scene-relative.
    Falls back to the box-height estimate for boxes with no valid depth pixels.
    """

    def __init__(self, every_n_frames: int, near: float, medium: float,
                 fallback: BoxHeightEstimator, device: str = "cpu", infer=None):
        self.every_n = max(1, int(every_n_frames))
        self.near = near
        self.medium = medium
        self.fallback = fallback
        self.device = device
        self._frame_idx = -1
        self._last_run = -1
        self._depth: Optional[np.ndarray] = None  # normalised 0..1, frame-sized
        self.runs = 0
        # `infer(frame_bgr) -> raw depth map (H, W)`; injectable for tests, else MiDaS via torch.hub.
        self._infer = infer or self._load_midas()

    def _load_midas(self):
        import torch  # lazy: only pulled in when --depth is used

        log.info("Loading MiDaS_small via torch.hub (first run downloads weights)")
        # MiDaS_small's hubconf loads its EfficientNet backbone from a second GitHub repo without
        # trust_repo, so torch.hub stops to ask on stdin (and raises EOFError when there is none).
        # Fetching that repo ourselves with trust_repo=True adds it to torch's trusted list first.
        torch.hub.help("rwightman/gen-efficientnet-pytorch", "tf_efficientnet_lite3", trust_repo=True)
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        model.to(self.device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform = transforms.small_transform

        def infer(frame_bgr: np.ndarray) -> np.ndarray:
            import cv2

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            inp = transform(rgb).to(self.device)
            with torch.no_grad():
                pred = model(inp)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False
                ).squeeze()
            return pred.cpu().numpy()

        return infer

    def tick(self) -> None:
        self._frame_idx += 1

    def _run(self, frame_bgr: np.ndarray) -> np.ndarray:
        self.runs += 1
        depth = np.asarray(self._infer(frame_bgr), dtype=np.float32)
        lo, hi = float(depth.min()), float(depth.max())
        if hi - lo < 1e-6:
            return np.zeros_like(depth)
        return (depth - lo) / (hi - lo)

    def estimate(self, frame_bgr: np.ndarray, det: Detection) -> str:
        due = self._depth is None or self._frame_idx % self.every_n == 0
        if due and self._last_run != self._frame_idx:
            self._depth = self._run(frame_bgr)
            self._last_run = self._frame_idx
        h, w = self._depth.shape
        x1, x2 = max(0, det.x1), min(w, det.x2)
        y1, y2 = max(0, det.y1), min(h, det.y2)
        if x2 <= x1 or y2 <= y1:
            return self.fallback.estimate(frame_bgr, det)
        med = float(np.median(self._depth[y1:y2, x1:x2]))
        log.debug("depth median %.2f for %s (box h=%.2f)", med, det.label, det.h_frac)
        if med >= self.near:
            return NEAR
        if med >= self.medium:
            return MEDIUM
        return FAR


class BottomEdgeEstimator:
    """Distance for *low* (floor) obstacles from the box's bottom edge.

    Box height says nothing useful about a shoe box on the floor: it stays small however close it
    is. With a chest camera the floor rises toward the horizon, so the closer a floor object, the
    lower its bottom edge sits in the frame. Thresholds are fractions of frame height.
    """

    def __init__(self, near_bottom: float, medium_bottom: float):
        if not 0 < medium_bottom < near_bottom <= 1.0:
            raise ValueError("need 0 < medium_bottom < near_bottom <= 1")
        self.near_bottom = near_bottom
        self.medium_bottom = medium_bottom

    def bucket(self, y2_frac: float) -> str:
        if y2_frac >= self.near_bottom:
            return NEAR
        if y2_frac >= self.medium_bottom:
            return MEDIUM
        return FAR

    def estimate(self, frame_bgr: np.ndarray, det: Detection) -> str:
        return self.bucket(det.y2 / frame_bgr.shape[0])

    def tick(self) -> None:
        """Called once per frame. No-op."""
