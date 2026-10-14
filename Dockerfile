# Vision Pin container: CPU-only, multi-arch (linux/amd64 and linux/arm64, so it builds natively on
# Apple Silicon and Intel Macs, x86 Linux and the Raspberry Pi 5).
#
#   docker build -t visionpin .
#   docker run --rm visionpin --source video --video /data/clip.mp4 --mute      # see DOCKER.md
#
# The YOLOv8n weights are downloaded at build time so the container runs offline.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_CONFIG_DIR=/tmp/yolo \
    TORCH_HOME=/root/.cache/torch

# espeak-ng + alsa-utils: speech (pyttsx3 renders with libespeak-ng and plays with aplay).
# libgl1/libglib2.0: opencv-python runtime libs (libgl1 is also what --show needs under X11).
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng alsa-utils libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first (small wheel, exists for both amd64 and arm64), then the rest.
# Export extras (pnnx, ncnn, onnx, onnxruntime) and simpleaudio (needs a C build and an
# ALSA dev package) are left out of the image; the beep is simply disabled without simpleaudio.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && grep -vE '^(pnnx|ncnn|onnx|onnxruntime|simpleaudio)' requirements.txt > /tmp/req.txt \
    && pip install -r /tmp/req.txt

COPY config.yaml main.py capture.py detect.py distance.py callout.py speech.py ./
COPY tools ./tools
COPY tests ./tests

# Bake the detector weights into the image (6 MB) so first start needs no network.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Mount clips or a custom config here.
VOLUME ["/data"]

ENTRYPOINT ["python", "main.py"]
# Default: a Linux host with the webcam passed through (docker run --device /dev/video0 ...).
# On macOS override with --source video --video <stream or file>; see DOCKER.md.
CMD ["--source", "webcam"]
