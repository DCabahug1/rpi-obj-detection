# Running Vision Pin in Docker

The image is CPU-only and multi-arch (`linux/amd64` and `linux/arm64`), so `docker build` produces a
native image on Intel and Apple Silicon Macs, x86 Linux, and a Raspberry Pi 5. The YOLOv8n weights are
baked in; the container needs no network to start. The image is about 1.5 GB (torch).

```bash
docker build -t visionpin .
docker run --rm visionpin --help
```

Anything after the image name is passed to `main.py`, so every flag in the README works.

## What a container can and cannot reach

| Host | Webcam | Speakers | `--show` window |
|------|--------|----------|-----------------|
| Linux | yes, `--device /dev/video0` | yes, `--device /dev/snd` | yes, with X11 socket mounted |
| macOS (Docker Desktop) | **no** device passthrough; stream it in (below) | **no**; use `--mute` or run natively | only via XQuartz |
| Windows (Docker Desktop) | no | no | no |

Docker Desktop runs containers in a Linux VM, and that VM has no access to the Mac camera or audio
hardware. This is a Docker limitation, not something the image can work around. For live speech on a
Mac, run natively instead: the README laptop quick start works unchanged on macOS and uses the
built-in voices (`python3 -m venv .venv && source .venv/bin/activate`, install, `python main.py
--source webcam --show`).

## Linux host: everything passed through

```bash
docker run --rm -it \
  --device /dev/video0 --device /dev/snd \
  --group-add video --group-add audio \
  visionpin --source webcam
```

or `docker compose --profile linux up`. Add `-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix` and
`--show` for the preview window (run `xhost +local:` on the host first).

## macOS: stream the webcam into the container

1. Install ffmpeg once: `brew install ffmpeg`.
2. Start the camera stream on the Mac (leave it running). `"0"` is the default FaceTime/USB camera;
   list devices with `ffmpeg -f avfoundation -list_devices true -i ""`.

   ```bash
   ffmpeg -f avfoundation -framerate 30 -video_size 640x480 -i "0" \
     -vcodec mpeg1video -b:v 2M -f mpegts "tcp://0.0.0.0:5000?listen"
   ```

   (Bind to `0.0.0.0`, not `127.0.0.1`: the connection from the container arrives on the VM's
   network adapter, not on loopback.)

3. In another terminal, run the container against that stream:

   ```bash
   docker run --rm -it visionpin \
     --source video --video tcp://host.docker.internal:5000 --mute
   ```

   or `docker compose --profile stream up`. Callouts and fps are printed to the terminal
   (`--mute` logs them as `SPEAK: ...`); the detection pipeline, distance, direction, priority and
   rate limiting are all exercised for real. Expect 15-30 fps on an Apple Silicon Mac.

## Windows: same idea, DirectShow instead of AVFoundation

Docker Desktop on Windows has the same limitation as on a Mac: no camera or audio in the container.
Verified on Windows 11 with a Logitech C270 (15 fps end to end, correct callouts).

1. Install ffmpeg once: `winget install Gyan.FFmpeg` (new terminal afterwards).
2. Find the camera name, then start the stream and leave it running (PowerShell):

   ```powershell
   ffmpeg -f dshow -list_devices true -i dummy
   ffmpeg -f dshow -framerate 30 -video_size 640x480 -i video="Logi C270 HD WebCam" -vcodec mpeg1video -b:v 2M -f mpegts "tcp://0.0.0.0:5000?listen"
   ```

   Windows Firewall may ask once to allow ffmpeg; the connection comes from Docker's own virtual
   network, so allowing it on private networks is enough.

3. In a second terminal:

   ```powershell
   docker run --rm -it visionpin --source video --video tcp://host.docker.internal:5000 --mute
   ```

   or `docker compose --profile stream up`. For real speech and the `--show` window on Windows,
   run natively instead (README quick start with `.venv\Scripts\activate`).

## Development without a camera (any host)

Put a clip in `./data` and replay it:

```bash
docker run --rm -v "$PWD/data:/data" visionpin --source video --video /data/clip.mp4 --mute
docker run --rm -v "$PWD/data:/data" visionpin --source video --video /data/clip.mp4 --fast-video --max-frames 200 --mute
```

Run the unit tests inside the image:

```bash
docker run --rm --entrypoint python visionpin -m pytest tests/ -q
```

## Notes

- A custom config: `-v "$PWD/my.yaml:/data/config.yaml" ... --config /data/config.yaml`.
- `--depth` downloads MiDaS (~100 MB) on first use; persist it with `-v visionpin-torch:/root/.cache/torch`
  so it is not fetched again for every container.
- The beep before "near" callouts is disabled in the image (simpleaudio is not installed there);
  speech itself works on a Linux host with `/dev/snd`.
- Multi-arch image for a registry: `docker buildx build --platform linux/amd64,linux/arm64 -t <repo>/visionpin --push .`
