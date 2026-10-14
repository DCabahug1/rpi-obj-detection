"""Frame sources: webcam (OpenCV), Raspberry Pi camera (picamera2), or a video file for development.

Every source hands back BGR frames already resized to the configured capture size.
A small grabber thread keeps only the *latest* frame, so when the main loop is slower
than the camera, old frames are dropped instead of queued.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

if sys.platform == "win32":
    # OpenCV's Media Foundation webcam backend can take ~20 s to deliver its first frame when its
    # hardware transforms are enabled (22 s measured with a Logitech C270); with them off it is
    # ~1 s at the full 30 fps. Must be set before cv2 is first imported anywhere in the process.
    os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import numpy as np

log = logging.getLogger(__name__)


class FrameSource(ABC):
    """Abstract frame source. `read()` returns a BGR frame of (height, width) or None at end of stream."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    @abstractmethod
    def _grab(self) -> Optional[np.ndarray]:
        """Fetch one raw frame from the device. Blocking. Return None if the stream has ended."""

    def read(self) -> Optional[np.ndarray]:
        frame = self._grab()
        if frame is None:
            return None
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame

    def release(self) -> None:  # pragma: no cover - trivial
        pass


class WebcamSource(FrameSource):
    def __init__(self, width: int, height: int, index: int = 0):
        super().__init__(width, height)
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        log.info("Webcam %d opened at %dx%d", index,
                 int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def _grab(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
        self.cap.release()


class PicamSource(FrameSource):
    """Raspberry Pi camera via picamera2 (imported lazily so laptops don't need it)."""

    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        from picamera2 import Picamera2  # type: ignore

        self.cam = Picamera2()
        config = self.cam.create_video_configuration(main={"size": (width, height), "format": "RGB888"})
        self.cam.configure(config)
        self.cam.start()
        time.sleep(0.5)  # let exposure settle
        log.info("picamera2 started at %dx%d", width, height)

    def _grab(self) -> Optional[np.ndarray]:
        rgb = self.cam.capture_array("main")
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def release(self) -> None:
        self.cam.stop()


class VideoFileSource(FrameSource):
    """Development source: replays a video file at its native frame rate.

    With realtime=False the file is decoded as fast as possible and looped forever, which
    turns it into a throughput benchmark (stop with --max-frames or Ctrl-C).
    """

    def __init__(self, width: int, height: int, path: str, realtime: bool = True):
        super().__init__(width, height)
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.realtime = realtime
        self.loop = not realtime
        self._next_t = time.monotonic()

    def _grab(self) -> Optional[np.ndarray]:
        if self.realtime:
            now = time.monotonic()
            if now < self._next_t:
                time.sleep(self._next_t - now)
            self._next_t = max(self._next_t + 1.0 / self.fps, time.monotonic())
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
        self.cap.release()


class LatestFrameGrabber:
    """Runs a FrameSource in a background thread and exposes only the newest frame.

    This is the frame-skipping mechanism: if detection takes 200 ms and the camera
    produces 30 fps, the 5 intermediate frames are simply overwritten.
    """

    def __init__(self, source: FrameSource):
        self.source = source
        self._lock = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._ended = False
        self._thread = threading.Thread(target=self._run, name="frame-grabber", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._ended:
            frame = self.source.read()
            with self._lock:
                if frame is None:
                    self._ended = True
                else:
                    self._frame = frame
                    self._seq += 1
                self._lock.notify_all()

    def latest(self, last_seq: int, timeout: float = 1.0) -> tuple[Optional[np.ndarray], int]:
        """Block until a frame newer than `last_seq` exists (or the stream ends). Returns (frame, seq)."""
        with self._lock:
            deadline = time.monotonic() + timeout
            while self._seq == last_seq and not self._ended:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, last_seq
                self._lock.wait(remaining)
            if self._ended and self._seq == last_seq:
                return None, last_seq
            return self._frame, self._seq

    @property
    def ended(self) -> bool:
        return self._ended

    def stop(self) -> None:
        self._ended = True
        self._thread.join(timeout=2.0)
        self.source.release()


def make_source(kind: str, width: int, height: int, camera_index: int = 0,
                video_path: Optional[str] = None, video_realtime: bool = True) -> FrameSource:
    if kind == "webcam":
        return WebcamSource(width, height, camera_index)
    if kind == "picam":
        return PicamSource(width, height)
    if kind == "video":
        if not video_path:
            raise ValueError("--source video requires --video PATH")
        return VideoFileSource(width, height, video_path, realtime=video_realtime)
    raise ValueError(f"Unknown source {kind!r}")
