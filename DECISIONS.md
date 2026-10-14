# Decisions

Choices made while building Vision Pin, and why. Newest at the bottom.

## Build environment
- Built and verified in a headless 4-core x86 cloud container with Python 3.13: **no webcam,
  no audio device, no Raspberry Pi**. Verification used synthetic 320x240 clips (a person crop
  and a bus crop sliding in and growing, plus an empty scene) and the `--mute` logging speaker.
- The PyTorch CPU wheel index was blocked by the container's proxy, so the default PyPI torch
  was installed for testing. The README still recommends the CPU index for laptops because it is
  far smaller.
- Pi instructions (picamera2, NCNN, bluetoothctl, systemd) are written from documentation and
  have not been run on hardware yet.

## Capture
- **Latest-frame grabber thread** implements "skip frames rather than queue": the camera is read
  continuously in a background thread and the main loop always takes the newest frame. Anything
  the loop was too slow for is overwritten, never queued.
- Added a **`--source video`** file source (not in the spec) so the pipeline can be developed,
  tested and benchmarked without a camera. `--fast-video` loops the file as fast as possible,
  which turns it into a throughput benchmark; `--max-frames` bounds the run.
- picamera2 is imported lazily inside `PicamSource` so laptops never need it installed.

## Detection
- `ultralytics.YOLO` loads `.pt`, `.onnx` and `*_ncnn_model/` transparently, so one `Detector`
  class serves the laptop (PyTorch) and the Pi (NCNN); only `--model` changes.
- Class mapping and the ignore list are applied inside the detector so downstream code only
  ever sees spoken words.
- **"stairs", "pole", "post" are not COCO classes.** YOLOv8n will never emit them. They stay in
  the priority list so a future custom model with those labels slots in without code changes.

## Distance
- Default is box height as a fraction of frame height with two thresholds (near >= 0.55,
  medium >= 0.30). Simple, zero cost, and adequate for "something is close".
- `--depth` builds a `MidasEstimator` that loads MiDaS_small via `torch.hub` (needs `timm`),
  runs every Nth frame (default 3), caches the depth map in between, min-max normalises it per
  frame, and uses the median inside the box. Falls back to box height for degenerate boxes.
  Never enabled by default; torch.hub is only touched when the flag is given.

## Callout logic
- **Approaching** = history of box heights per (label, direction) over the last 5 frames, with
  total growth >= 0.04 of frame height, at least half the steps growing, and at most one shrink.
  The "half the steps" rule was added after a test showed a single far-to-medium jump followed by
  a plateau being called "approaching".
- History is dropped for objects not seen in the current frame so a reappearing object starts fresh.
- **Rate-limit escalation**: the spec's 4 s limit per (class, direction) suppressed the important
  "person, ahead" when it followed "person, medium, ahead" within 4 s during testing. An escalation
  from medium to near now bypasses the limit once; repeats at the same distance stay limited.
- Priority ties are broken by the larger (closer) box.
- Unlisted classes rank last but are still spoken when near.

## Speech
- `pyttsx3` runs in a dedicated thread fed by a `PriorityQueue`; `say()` is a non-blocking put.
- **Preemption** = a higher-priority callout clears everything still queued and is spoken next.
  pyttsx3 cannot portably interrupt an utterance already in progress, so phrases are kept to
  2-3 words and this is good enough.
- `close()` lets queued callouts finish before stopping (an earlier version dropped them).
- **Fallback speaker**: if `pyttsx3.init()` fails (no espeak, headless CI) or `--mute` is given,
  callouts are logged instead. Audio problems never stop the detection loop.
- Beep: a 120 ms sine generated with numpy and played with `simpleaudio` when available;
  silently disabled otherwise. Kept optional because `simpleaudio` has no wheels for every platform.

## Performance (measured here, 4-core x86, `--fast-video --max-frames 200`, 320x240, run one at a time)
| Model | fps |
|-------|-----|
| `yolov8n.pt` (PyTorch) | ~50-58 |
| `yolov8n.onnx` (onnxruntime) | ~15-19 |
| `yolov8n_ncnn_model` (NCNN) | ~52-63 |

- NCNN matched or beat PyTorch here and is the recommended Pi path. ONNX via onnxruntime was
  the slowest on this machine, so the README only mentions it as an option.
- Both laptop targets (>= 5 fps) are met with a large margin; Pi numbers are still to be measured.

## Export
- NCNN export requires the `pnnx` package in addition to `ncnn`; ultralytics does not pull it in
  automatically. Added to both requirements files.

## Depth path verification
- `--depth` could not be run end to end here: the proxy allows the MiDaS weights download but
  blocks the GitHub repo archive that `torch.hub.load` needs for the model code (HTTP 403/400).
  The estimator's cadence, median-in-box and fallback logic are unit-tested with an injected fake
  depth function instead. The real torch.hub call is unchanged and should work on a laptop.

## Dev / repo
- Tests cover the pure-logic modules only (distance, callout, speech queue). The detector and
  camera classes are exercised by the end-to-end video runs rather than unit tests.
- Model weights and exports are git-ignored; `tools/export_model.py` regenerates them.
