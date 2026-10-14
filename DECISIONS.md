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

## Laptop verification (Windows 11, Logitech C270 webcam, 2026-10-13)
First run on real hardware: Python 3.12 venv, CPU-only torch 2.14, ultralytics 8.4, opencv-python 5.0,
pyttsx3 2.99. The 20 unit tests passed unchanged; the fixes below add three more.

- **Webcam start-up.** OpenCV's default Media Foundation backend took 7 s to open the C270 and 22 s
  to deliver the first frame. DirectShow opened in 0.6 s but was stuck at 15 fps (YUY2, fps and
  fourcc hints ignored). Setting `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` before cv2 is imported
  keeps Media Foundation and gives open 0.5 s, first frame 1 s, 30 fps. `capture.py` sets it on
  Windows only, with `setdefault` so a value from the environment wins.
- **Speech was silent on Windows, for two separate reasons.**
  1. The pyttsx3 engine was created on the main thread and used from the speaker thread. SAPI5 is a
     single-threaded COM object, so the first `runAndWait()` blocked forever with no error. The
     `Speaker` now takes an engine *factory* and creates, uses and closes the engine on its worker
     thread. `Speaker(engine_instance)` still works (tests), and a failing factory falls back to
     `LogEngine` as before.
  2. With the pyttsx3 2.99 sapi5 driver only the first `say()` + `runAndWait()` is heard. After the
     first loop the driver proxy is left not-busy, so the next `say()` dispatches immediately and the
     following `runAndWait()` pops its end-loop command first; that command calls `stop()`, which
     purges the utterance 20 ms in (measured: `finished-utterance` with `completed=False`). On
     Windows the engine now drives the documented external loop instead (`startLoop(False)`, then
     `say()` and `iterate()` until `isBusy()` is false, with a 15 s safety timeout). Five
     consecutive utterances then complete with `completed=True` and realistic durations. Linux keeps
     `runAndWait()`: the espeak driver renders synchronously inside it and has no working external
     loop in 2.99 (its `iterate()` is not a generator), so the two platforms need different calls.
- **Beep on Windows** uses the stdlib `winsound.Beep`; simpleaudio has no Windows wheel.
- **`--depth` crashed without a terminal.** `torch.hub.load("intel-isl/MiDaS", ..., trust_repo=True)`
  is fine, but MiDaS's own hubconf loads its EfficientNet backbone from
  `rwightman/gen-efficientnet-pytorch` *without* `trust_repo`, so torch prompts on stdin and raises
  `EOFError` when there is none (and would stop the app to ask otherwise). `distance.py` now calls
  `torch.hub.help(...)` on that repo with `trust_repo=True` first, which fetches it and adds it to
  torch's trusted list. First run downloads ~100 MB; later runs load from cache in ~3 s. MiDaS small
  takes ~50 ms per 320x240 frame on this CPU.
- **Measured end to end (webcam, 320x240):** ~30 fps with `yolov8n.pt`, ~24 fps with `--depth`
  (every 3rd frame). fps stayed at 27-30 while speech was playing: the loop never waits on TTS.
- **`--show` and `q`:** window up ~4 s after launch (model load + webcam), `q` exits cleanly with
  "Processed N frames".
- **Speech latency, for the acceptance tests.** One 3-word callout takes ~3 s from dequeue to done
  (0.12 s beep, roughly 1 s of SAPI audio start-up, two comma pauses, the words at 190 wpm). Two near
  objects in view therefore produce ~6 s of speech per 4 s rate-limit window and the queue falls
  behind; priority still puts "person" first and `close()` gives up flushing after 3 s. If this is
  too chatty, raise `callout.rate_limit_s` first; dropping the commas from the phrasing (code) is
  the next lever.
- **Depth thresholds look too strict.** A person filling 98 % of the frame height measured a median
  normalised depth of 0.55-0.62, i.e. "medium" with `depth.near: 0.70`; a large plant on the left
  measured 0.04-0.06, "far". For big boxes the median includes background pixels. Left unchanged
  until the chair and walking tests are repeated with `--depth`; `-v` now logs the per-box median so
  the thresholds can be set from data. A high percentile instead of the median is the code-level
  alternative if tuning alone is not enough.
- **Still unverified:** the Raspberry Pi path (picamera2, NCNN, espeak-ng, Bluetooth, systemd) and
  real espeak-ng speech on Linux. The Linux speech path is reasoned from the 2.99 driver source;
  only the Windows path has been run here, and whether the audio is audible and correctly worded is
  for the physical acceptance tests.

## Object abstraction (branch `feature/object-abstraction`)
Goal: tell the user what to *do* about things in the path (step over, duck), and stop announcing
raw COCO names for classes that are not already part of the spoken vocabulary.

- **New module `obstacle.py`.** A `Detection` becomes an `Obstacle` carrying the spoken label,
  whether that label is a real class name (`known`), a vertical `zone` and an `action`. The
  callout planner now works on obstacles; `detect.py` and the camera code are untouched.
- **Vocabulary = priority list + `class_map` words.** Anything else is spoken as
  `callout.ambiguous_word` ("object"). This is the literal reading of "names not already included
  can just be ambiguous". `person` is always known. Identity entries were added to `class_map` for
  couch, bed, toilet and suitcase so big furniture keeps its name.
- **Zones from box geometry, not from a model.** With a chest camera roughly level: `low` = box
  top in the lower half and bottom at or below 0.80 of the frame; `overhead` = box top at the
  frame edge and bottom above mid frame. Thresholds live under `zones:` and will need tuning on the
  real mount and camera tilt. A near person spans the whole frame height and is `normal`; people
  are excluded from zones outright (`zones.exclude`) so a short or seated person is never
  "step over".
- **Low obstacles measure distance by bottom edge.** Box height is useless for a small floor
  object (it stays small however close it is), while the bottom edge slides down the frame as you
  approach. `BottomEdgeEstimator` (`zones.low_near_bottom` 0.90, `low_medium_bottom` 0.75) is used
  for low obstacles only; normal and overhead keep box height (or MiDaS with `--depth`). The
  approach metric follows the same split: bottom edge for low, height for the rest.
- **Ambiguous objects speak only when low or overhead.** A backpack carried at chest height, a
  bottle on a table, a cat on a sofa are not walking hazards and would make "object, close, ahead"
  noise. The same classes on the floor are hazards and are announced with "step over". Known
  classes keep speaking in every zone as before.
- **Ignore list trimmed.** bottle, backpack, handbag, umbrella, frisbee, sports ball, baseball bat,
  skateboard and cat left the ignore list because they can lie in the path. Because they are not in
  the vocabulary they only ever surface as "object ... step over" / "duck".
- **Priority.** `object` sits after `post` and before `chair`: a trip hazard outranks furniture
  but not people or vehicles.
- **Phrasing.** The action is appended: "object, close, left, step over", "object, ahead, step
  over" (the near+ahead drop rule still applies), "sign, medium, right, duck". Rate limit key is
  unchanged (label, direction), so repeated "object" in one direction is throttled.
- **Limits.** YOLOv8n can only name COCO classes. Boxes, curbs, steps, cables and branches have no
  class, so a floor hazard is only announced if the model recognises it as *something*. Getting
  past this needs a different detector (a custom "obstacle" model or depth-based ground-plane
  segmentation); the zone/action plumbing here is detector-agnostic and ready for that.
- **Verification (headless, synthetic 320x240 clips, `--mute`).** 38 unit tests pass (15 new).
  A 35 %-tall bus crop sliding down to the frame bottom: "bus, ahead, step over"; the same clip
  with `bus` removed from the vocabulary: "object, ahead, step over"; a bus at chest height with
  `bus` removed: silent; with `bus` known: "bus, ahead". A bus crop hanging from the top edge:
  "bus, medium, right, duck". Person and empty-room clips behave exactly as before. Throughput
  unchanged at ~50 fps (`--fast-video`). The real tests are a bag on the floor and a low branch or
  sign with the webcam; a tie crop and a 16 %-tall bus crop were simply not detected on a grey
  background, which is a limitation of the synthetic clips, not the pipeline.
- **Observed on the clips:** YOLO flips the same crop between `bus` and `truck` on some frames,
  which produces two callouts because they are different labels. Pre-existing behaviour, not
  introduced here; a label-stability filter is a candidate follow-up.
