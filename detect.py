"""YOLOv8n wrapper: runs the model on CPU and returns plain-word detections."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    label: str          # plain spoken word, after class_map
    raw_label: str      # original COCO name
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int
    cx_frac: float      # box centre x as a fraction of frame width (0 left .. 1 right)
    h_frac: float       # box height as a fraction of frame height


def map_label(raw: str, class_map: dict[str, str]) -> str:
    return class_map.get(raw, raw)


class Detector:
    def __init__(self, model_path: str, imgsz: int = 320, conf: float = 0.4, iou: float = 0.5,
                 device: str = "cpu", class_map: dict[str, str] | None = None,
                 ignore_classes: Iterable[str] = ()):
        from ultralytics import YOLO  # heavy import, keep it local

        log.info("Loading model %s (imgsz=%d, device=%s)", model_path, imgsz, device)
        self.model = YOLO(model_path, task="detect")
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.class_map = class_map or {}
        self.ignore = set(ignore_classes)
        self.names = self.model.names  # id -> COCO name

    def warmup(self, width: int, height: int) -> None:
        self.detect(np.zeros((height, width, 3), dtype=np.uint8))

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        results = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                     device=self.device, verbose=False)
        out: list[Detection] = []
        if not results:
            return out
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            raw = self.names.get(int(k), str(k)) if isinstance(self.names, dict) else self.names[int(k)]
            if raw in self.ignore:
                continue
            out.append(Detection(
                label=map_label(raw, self.class_map), raw_label=raw, conf=float(c),
                x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                cx_frac=float((x1 + x2) / 2.0 / w), h_frac=float((y2 - y1) / h),
            ))
        return out
