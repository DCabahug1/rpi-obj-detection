# Vision Pin

Spoken obstacle callouts from a chest-worn camera, for blind and low-vision users.
A Raspberry Pi 5 (or a laptop with a webcam) runs YOLOv8n on the CPU and says short phrases
like **"chair, close, left"** or **"person, ahead"** through the default audio device, so
OS-paired Bluetooth headphones just work. It stays quiet when nothing is near.

```
camera -> latest frame -> YOLOv8n (320px, CPU) -> distance + direction -> priority / rate limit -> TTS thread
```

---

## Laptop quick start (under 5 minutes)

```bash
git clone <this repo> visionpin && cd visionpin
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision   # CPU-only, much smaller
pip install -r requirements.txt
python main.py --source webcam --show
```

The first run downloads `yolov8n.pt` (6 MB). Press `q` in the window or Ctrl-C to quit.

Windows: use `python -m venv .venv` and `.venv\Scripts\activate` (PowerShell:
`.venv\Scripts\Activate.ps1`); everything else is the same. Verified on Windows 11 with a
Logitech C270 and the built-in SAPI voices, see [DECISIONS.md](DECISIONS.md).

What you should see and hear:

- fps printed to the terminal once a second (target >= 5 fps; a modern laptop does 20-60).
- Slide a chair toward the camera from the left: **"chair, close, left"**.
- Walk toward the camera: **"person, medium, ahead"** while approaching, then **"person, ahead"** when close.
- An empty, still room: silence.

Linux TTS needs a speech engine: `sudo apt install espeak-ng` (macOS and Windows use the built-in voices).
If no engine is found the program keeps running and logs the callouts instead of speaking them.
`-v` logs how long each callout took to speak.

### Useful flags

| Flag | Meaning |
|------|---------|
| `--source webcam\|picam\|video` | Camera type. `video` replays a file (`--video clip.mp4`) for development. |
| `--show` | Draw boxes, the current callout, and fps in a window. |
| `--model PATH` | Use `.pt`, `.onnx`, or an exported `yolov8n_ncnn_model/` directory. |
| `--depth` | Use MiDaS small for distance (off by default, slower). The first run downloads ~100 MB of model code and weights from GitHub. Run with `-v` to log the per-box depth values when tuning `depth.near` / `depth.medium`. |
| `--mute` | Log callouts instead of speaking. |
| `--fast-video --max-frames N` | Loop a video file as fast as possible: an fps benchmark. |
| `--config FILE` | Alternative config (default `config.yaml`). |

All thresholds, the class-name map, the ignore list, priorities, the rate limit, and TTS settings
live in [`config.yaml`](config.yaml).

---

## Raspberry Pi 5 setup

Tested instructions target Raspberry Pi OS Bookworm 64-bit. (Written from the docs; see
[DECISIONS.md](DECISIONS.md) for what has and has not been run on real hardware.)

### 1. System packages

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv espeak-ng libportaudio2 git
```

### 2. Project and Python packages

`picamera2` only comes from apt, so the venv must see system packages:

```bash
git clone <this repo> ~/visionpin && cd ~/visionpin
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements-pi.txt
```

### 3. Export the model to NCNN (fastest CPU path on the Pi)

Do this once, on a laptop or on the Pi (on the Pi it needs torch: see the comment in
`requirements-pi.txt`). Copy the resulting `yolov8n_ncnn_model/` folder into the project directory.

```bash
python tools/export_model.py --format ncnn        # -> yolov8n_ncnn_model/
```

### 4. Run

```bash
python main.py --source picam --model yolov8n_ncnn_model
```

Add `--show` if a display is attached. Expect >= 3 fps end to end; NCNN at 320px on a Pi 5 is
usually well above that.

### 5. Pair Bluetooth headphones

Put the headphones in pairing mode, then:

```bash
bluetoothctl
[bluetooth]# power on
[bluetooth]# agent on
[bluetooth]# default-agent
[bluetooth]# scan on
#   ... wait until your headphones appear, note the MAC address XX:XX:XX:XX:XX:XX
[bluetooth]# scan off
[bluetooth]# pair XX:XX:XX:XX:XX:XX
[bluetooth]# trust XX:XX:XX:XX:XX:XX
[bluetooth]# connect XX:XX:XX:XX:XX:XX
[bluetooth]# exit
```

`trust` makes them reconnect automatically on boot. Check they are the default output and test:

```bash
wpctl status                 # the headphones should be marked * under Audio > Sinks
espeak-ng "vision pin ready" # should come out of the headphones
```

If audio goes to the wrong device: `wpctl set-default <sink id>` using the id shown by `wpctl status`.

### 6. Run on boot with systemd

[`visionpin.service`](visionpin.service) is a ready-made unit. It assumes user `pi` and the
project in `/home/pi/visionpin`; edit `User=`, `WorkingDirectory=`, `ExecStart=` and
`XDG_RUNTIME_DIR` (`/run/user/$(id -u)`) if yours differ.

```bash
sudo cp visionpin.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now visionpin
sudo loginctl enable-linger pi        # keep the user's audio session alive without a login
journalctl -u visionpin -f            # watch fps and callouts
```

The unit waits 8 seconds after boot so Bluetooth has time to reconnect before the first callout.

---

## How callouts are decided

1. **Distance** comes from bounding-box height as a fraction of frame height
   (`distance.near_frac`, `distance.medium_frac`). With `--depth`, MiDaS small runs every
   `depth.every_n_frames` and the median depth inside the box is used instead.
2. **Direction** is left / ahead / right from the box centre x (`direction.left_max`, `direction.right_min`).
3. An object is **spoken** only when it is near, or medium and growing across the last
   `approach.window` frames (approaching).
4. When several qualify, **priority** is `callout.priority` order (person first), ties go to the bigger box.
5. **Rate limit**: the same class in the same direction is not repeated within `callout.rate_limit_s`
   (4 s). An object that escalates from medium to near is announced once even inside that window.
6. **Phrasing**: `"{class}, {distance}, {direction}"`; the distance word is dropped for a near
   object straight ahead: `"person, ahead"`. COCO names are mapped to plain words via `class_map`
   and anything in `ignore_classes` (tv, laptop, cell phone, ...) is never spoken.
7. **Speech** runs in its own thread with a priority queue. The capture loop never waits on it.
   A higher-priority callout clears whatever is still queued and is spoken next. A short beep
   precedes near callouts (`tts.beep`).

## Performance notes

- The grabber thread keeps only the newest camera frame, so a slow detection step drops frames
  instead of building up a lag behind a walking person.
- Measured in a 4-core x86 container with synthetic 320x240 clips (`--fast-video`): PyTorch
  `yolov8n.pt` ~55 fps, NCNN ~60 fps, ONNX (onnxruntime) ~17 fps. Details in [DECISIONS.md](DECISIONS.md).
- Measured end to end on a Windows 11 laptop with a Logitech C270 at 320x240 (CPU-only torch):
  ~30 fps with `yolov8n.pt`, ~24 fps with `--depth` (MiDaS every 3rd frame, ~50 ms per run).
- Low fps on the Pi? Use the NCNN export, keep `model.imgsz` at 320, and leave `--depth` off.

## Tests

```bash
pytest tests/
```

Covers distance buckets, direction, approach detection, priority, rate limiting, phrasing,
and the speech queue's non-blocking and preemption behaviour. For an end-to-end check without
a camera: `python main.py --source video --video clip.mp4 --mute`.
