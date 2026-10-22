#!/usr/bin/env python3
"""Vision Pin: chest camera -> YOLOv8n -> spoken obstacle callouts.

    python main.py --source webcam --show          # laptop
    python main.py --source picam                  # Raspberry Pi 5
    python main.py --source video --video clip.mp4 # development without a camera
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

from callout import Callout, CalloutPlanner, Candidate
from capture import LatestFrameGrabber, make_source
from detect import Detector
from distance import BottomEdgeEstimator, BoxHeightEstimator, NEAR
from obstacle import LOW, NORMAL, ObstacleClassifier

log = logging.getLogger("visionpin")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spoken obstacle callouts from a chest camera")
    p.add_argument("--source", choices=["webcam", "picam", "video"], default="webcam")
    p.add_argument("--camera-index", type=int, default=None, help="webcam index (default from config)")
    p.add_argument("--video", help="video file for --source video (development only)")
    p.add_argument("--fast-video", action="store_true",
                   help="loop --video as fast as possible instead of real time (fps benchmark; use --max-frames)")
    p.add_argument("--model", help="override model path (.pt, .onnx, or *_ncnn_model dir)")
    p.add_argument("--depth", action="store_true", help="use MiDaS small for distance (off by default)")
    p.add_argument("--show", action="store_true", help="draw boxes, callout and fps in a window")
    p.add_argument("--fullscreen", action="store_true", help="make the --show window fullscreen (touchscreens)")
    p.add_argument("--mute", action="store_true", help="log callouts instead of speaking them")
    p.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    p.add_argument("--max-frames", type=int, default=0, help="stop after N processed frames (testing)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_planner(cfg: dict) -> CalloutPlanner:
    c, d, a = cfg["callout"], cfg["direction"], cfg["approach"]
    act = cfg.get("actions", {})
    return CalloutPlanner(
        priority=c["priority"], rate_limit_s=float(c["rate_limit_s"]),
        left_max=float(d["left_max"]), right_min=float(d["right_min"]),
        approach_window=int(a["window"]), approach_min_growth=float(a["min_growth_frac"]),
        near_word=c.get("near_word", "close"), medium_word=c.get("medium_word", "medium"),
        prescriptive_only_ahead=bool(act.get("prescriptive_only_ahead", True)),
        min_stable_frames=int(act.get("min_stable_frames", 3)),
        same_box_iou=float(c.get("same_box_iou", 0.5)),
    )


def build_distance(cfg: dict, use_depth: bool):
    dcfg = cfg["distance"]
    box = BoxHeightEstimator(float(dcfg["near_frac"]), float(dcfg["medium_frac"]))
    if not use_depth:
        return box
    from distance import MidasEstimator

    m = cfg["depth"]
    return MidasEstimator(int(m["every_n_frames"]), float(m["near"]), float(m["medium"]),
                          fallback=box, device=cfg["model"].get("device", "cpu"))


def draw(frame, cands: list[Candidate], callout: Optional[Callout], fps: float):
    import cv2

    for c in cands:
        d = c.det
        color = (0, 0, 255) if c.distance == NEAR else (0, 165, 255) if c.distance == "medium" else (0, 200, 0)
        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 1)
        # L = low, O = overhead; "!" = steppable/duckable, "?" = partly out of view, "~" = not stable yet
        zone = "" if c.obs.zone == NORMAL else f" {c.obs.zone[0].upper()}"
        flags = ("!" if c.obs.action else "") + ("?" if c.obs.truncated else "") + ("" if c.stable else "~")
        tag = f"{c.obs.label} {c.distance[0]}{'+' if c.approaching else ''} {c.direction[0]}{zone}{flags}"
        cv2.putText(frame, tag, (d.x1, max(10, d.y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    if callout:
        cv2.putText(frame, callout.text, (4, frame.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.1f} fps", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    x1, y1, x2, y2 = exit_button(frame.shape[1], frame.shape[0])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 200), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
    cv2.putText(frame, "EXIT", (x1 + 9, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def exit_button(width: int, height: int) -> tuple[int, int, int, int]:
    """Top-right Exit button in frame pixels, big enough to hit with a finger once scaled up."""
    w, h = max(56, width // 6), max(26, height // 9)
    return width - w - 4, 4, width - 4, 4 + h


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)

    cap_cfg, model_cfg = cfg["capture"], cfg["model"]
    width, height = int(cap_cfg["width"]), int(cap_cfg["height"])
    camera_index = args.camera_index if args.camera_index is not None else int(cap_cfg.get("camera_index", 0))

    detector = Detector(
        args.model or model_cfg["path"], imgsz=int(model_cfg["imgsz"]), conf=float(model_cfg["conf"]),
        iou=float(model_cfg.get("iou", 0.5)), device=model_cfg.get("device", "cpu"),
        class_map=cfg.get("class_map", {}), ignore_classes=cfg.get("ignore_classes", []),
    )
    detector.warmup(width, height)
    dist = build_distance(cfg, args.depth)
    zones = cfg.get("zones", {})
    low_dist = BottomEdgeEstimator(float(zones.get("low_near_bottom", 0.90)),
                                   float(zones.get("low_medium_bottom", 0.75)))
    classifier = ObstacleClassifier.from_config(cfg)
    planner = build_planner(cfg)

    from speech import make_speaker
    speaker = make_speaker(cfg["tts"], mute=args.mute)

    source = make_source(args.source, width, height, camera_index, args.video,
                         video_realtime=not args.fast_video)
    grabber = LatestFrameGrabber(source)

    stop = False

    def _sig(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    if args.show:
        import cv2
        cv2.namedWindow("Vision Pin", cv2.WINDOW_NORMAL)

        # A touchscreen tap arrives as a left click, in frame coordinates.
        def _click(event, x, y, *_):
            nonlocal stop
            if event == cv2.EVENT_LBUTTONDOWN:
                x1, y1, x2, y2 = exit_button(width, height)
                if x1 <= x <= x2 and y1 <= y <= y2:
                    stop = True

        cv2.setMouseCallback("Vision Pin", _click)
    fullscreen_pending = args.show and args.fullscreen

    seq = 0
    processed = 0
    fps = 0.0
    win_start = time.monotonic()
    win_frames = 0
    print_every = float(cfg.get("fps_print_interval_s", 1.0))
    last_callout: Optional[Callout] = None
    last_callout_t = 0.0

    log.info("Running. Source=%s model=%s depth=%s", args.source, args.model or model_cfg["path"], args.depth)
    try:
        while not stop:
            frame, seq = grabber.latest(seq, timeout=1.0)
            if frame is None:
                if grabber.ended:
                    log.info("Stream ended")
                    break
                continue  # camera hiccup; try again

            now = time.monotonic()
            dist.tick()
            dets = detector.detect(frame)
            obstacles = classifier.classify(dets, frame.shape[0])
            buckets = [low_dist.estimate(frame, o.det) if o.zone == LOW else dist.estimate(frame, o.det)
                       for o in obstacles]
            callout, cands = planner.plan(obstacles, buckets, now)
            if callout:
                speaker.say(callout)
                last_callout, last_callout_t = callout, now
                log.info("callout: %s", callout.text)

            processed += 1
            win_frames += 1
            elapsed = now - win_start
            if elapsed >= print_every:
                fps = win_frames / elapsed
                print(f"fps: {fps:.1f}  objects: {len(dets)}", flush=True)
                win_start, win_frames = now, 0

            if args.show:
                shown = last_callout if now - last_callout_t < 2.0 else None
                cv2.imshow("Vision Pin", draw(frame.copy(), cands, shown, fps))
                if fullscreen_pending:
                    # Some compositors ignore fullscreen requested before the window is mapped.
                    cv2.setWindowProperty("Vision Pin", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    fullscreen_pending = False
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames and processed >= args.max_frames:
                break
    finally:
        grabber.stop()
        speaker.close()
        if args.show:
            cv2.destroyAllWindows()
    log.info("Processed %d frames", processed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
