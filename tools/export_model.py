#!/usr/bin/env python3
"""Export yolov8n.pt to NCNN (best on Pi 5) or ONNX for faster CPU inference.

    python tools/export_model.py --format ncnn     # -> yolov8n_ncnn_model/
    python tools/export_model.py --format onnx     # -> yolov8n.onnx

Then run with:  python main.py --source picam --model yolov8n_ncnn_model
"""
import argparse

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--format", choices=["ncnn", "onnx"], default="ncnn")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--half", action="store_true", help="fp16 (NCNN only)")
    a = p.parse_args()
    model = YOLO(a.weights)
    out = model.export(format=a.format, imgsz=a.imgsz, half=a.half, simplify=True)
    print(f"Exported to {out}")


if __name__ == "__main__":
    main()
