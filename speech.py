"""Offline TTS in its own thread. The capture loop only ever does a non-blocking put()."""
from __future__ import annotations

import itertools
import logging
import queue
import sys
import threading
import time
from typing import Callable, Optional, Protocol, Union

from callout import Callout

log = logging.getLogger(__name__)

_SENTINEL_RANK = 1 << 30
_UTTERANCE_TIMEOUT_S = 15.0   # safety net so one lost "finished" event cannot wedge the speaker
_ENGINE_READY_TIMEOUT_S = 15.0


class Engine(Protocol):
    def say(self, text: str) -> None: ...
    def beep(self) -> None: ...
    def close(self) -> None: ...


class Pyttsx3Engine:
    """pyttsx3 wrapper. Create and use it on ONE thread.

    On Windows the SAPI5 voice is a single-threaded COM object: only the thread that created it
    can drive it, so the Speaker builds this engine inside its worker thread (a voice created on
    the main thread and spoken from the worker hangs forever on the first utterance).
    """

    def __init__(self, rate: int, volume: float, beep: bool, beep_hz: int, beep_ms: int):
        import pyttsx3

        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)
        # Windows: with the pyttsx3 2.9x sapi5 driver, the second and later runAndWait() calls
        # purge the utterance that say() has just started, so only the first phrase is ever heard.
        # The documented external event loop (startLoop(False) + iterate()) never hits that path.
        # Linux: the espeak driver has no working external loop and its runAndWait() is
        # synchronous and correct, so it keeps the simple call.
        self._external_loop = sys.platform == "win32"
        if self._external_loop:
            self.engine.startLoop(False)
        self._beep = _make_beep(beep_hz, beep_ms) if beep else None

    def say(self, text: str) -> None:
        self.engine.say(text)
        if not self._external_loop:
            self.engine.runAndWait()
            return
        deadline = time.monotonic() + _UTTERANCE_TIMEOUT_S
        while self.engine.isBusy():
            if time.monotonic() > deadline:
                log.warning("TTS did not finish %r in %.0fs; dropping it", text, _UTTERANCE_TIMEOUT_S)
                self.engine.stop()
                break
            self.engine.iterate()
            time.sleep(0.02)

    def beep(self) -> None:
        if self._beep:
            self._beep()

    def close(self) -> None:
        try:
            if self._external_loop:
                self.engine.endLoop()
            else:
                self.engine.stop()
        except Exception:  # pragma: no cover
            pass


def _make_beep(hz: int, ms: int):
    """Return a callable that plays a short sine beep, or None if no audio backend is available.

    Windows uses the stdlib winsound module (simpleaudio ships no Windows wheel); elsewhere
    simpleaudio is used when installed.
    """
    if sys.platform == "win32":
        import winsound

        def play_win():
            try:
                winsound.Beep(hz, ms)
            except Exception as e:  # pragma: no cover
                log.warning("beep failed: %s", e)

        return play_win
    try:
        import numpy as np
        import simpleaudio as sa  # type: ignore
    except Exception:
        log.warning("simpleaudio not available; beep disabled")
        return None
    sr = 44100
    t = np.linspace(0, ms / 1000.0, int(sr * ms / 1000.0), False)
    wave = np.sin(2 * np.pi * hz * t)
    # short fade in/out to avoid clicks
    fade = min(len(wave) // 10, 400)
    env = np.ones_like(wave)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    pcm = (wave * env * 32767 * 0.5).astype(np.int16)

    def play():
        try:
            sa.play_buffer(pcm, 1, 2, sr).wait_done()
        except Exception as e:  # pragma: no cover
            log.warning("beep failed: %s", e)

    return play


class LogEngine:
    """Fallback when no TTS backend exists (headless CI, --mute). Logs instead of speaking."""

    def __init__(self, delay_s: float = 0.0):
        self.delay_s = delay_s
        self.spoken: list[str] = []

    def say(self, text: str) -> None:
        log.info("SPEAK: %s", text)
        self.spoken.append(text)
        if self.delay_s:
            time.sleep(self.delay_s)

    def beep(self) -> None:
        log.debug("BEEP")

    def close(self) -> None:
        pass


EngineFactory = Callable[[], Engine]


class Speaker:
    """Priority queue + worker thread.

    say() never blocks. If a callout arrives with a higher priority (lower rank) than
    anything queued or currently being spoken, the queue is cleared so it is spoken next.
    pyttsx3 cannot be interrupted mid-utterance portably, so utterances are kept short.

    engine is either an Engine instance or a zero-argument factory. A factory is called on the
    worker thread, which is what SAPI5 on Windows requires; if it raises, callouts are logged
    instead (LogEngine). The engine is also closed on the worker thread.
    """

    def __init__(self, engine: Union[Engine, EngineFactory]):
        self._factory: EngineFactory = engine if callable(engine) else (lambda: engine)
        self.engine: Optional[Engine] = None  # set by the worker thread
        self._ready = threading.Event()
        self._q: "queue.PriorityQueue[tuple[int, int, Optional[Callout]]]" = queue.PriorityQueue()
        self._seq = itertools.count()
        self._lock = threading.Lock()
        self._current_rank: Optional[int] = None  # rank of utterance being spoken
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="speaker", daemon=True)
        self._thread.start()
        if not self._ready.wait(_ENGINE_READY_TIMEOUT_S):
            log.warning("TTS engine still initialising after %.0fs; continuing", _ENGINE_READY_TIMEOUT_S)

    @property
    def busy(self) -> bool:
        return self._current_rank is not None or not self._q.empty()

    def say(self, callout: Callout) -> None:
        with self._lock:
            ranks = [item[0] for item in list(self._q.queue) if item[2] is not None]
            if self._current_rank is not None:
                ranks.append(self._current_rank)
            if ranks and callout.priority_rank < min(ranks):
                self._drain()  # higher priority than anything pending: it goes next
            self._q.put((callout.priority_rank, next(self._seq), callout))

    def _drain(self) -> None:
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def _run(self) -> None:
        try:
            engine = self._factory()
        except Exception as e:
            log.warning("TTS unavailable (%s); callouts will be logged instead of spoken", e)
            engine = LogEngine()
        self.engine = engine
        self._ready.set()
        try:
            while True:
                try:
                    rank, _, callout = self._q.get(timeout=0.2)
                except queue.Empty:
                    if self._stop.is_set():
                        break
                    continue
                if callout is None:
                    break
                self._current_rank = rank
                t0 = time.monotonic()
                try:
                    if callout.is_near:
                        engine.beep()
                    engine.say(callout.text)
                    log.debug("spoke %r in %.2fs", callout.text, time.monotonic() - t0)
                except Exception as e:  # never let audio kill the loop
                    log.error("TTS error: %s", e)
                finally:
                    self._current_rank = None
        finally:
            try:
                engine.close()
            except Exception as e:  # pragma: no cover
                log.warning("TTS close failed: %s", e)

    def close(self, timeout: float = 3.0) -> None:
        """Speak what is already queued, then stop. The sentinel sorts after every real callout."""
        self._stop.set()
        self._q.put((_SENTINEL_RANK, next(self._seq), None))
        self._thread.join(timeout)
        if self._thread.is_alive():
            log.warning("speaker thread did not finish within %.0fs", timeout)


def make_speaker(tts_cfg: dict, mute: bool = False) -> Speaker:
    if mute:
        return Speaker(LogEngine())

    def factory() -> Engine:
        return Pyttsx3Engine(
            rate=int(tts_cfg.get("rate", 190)), volume=float(tts_cfg.get("volume", 1.0)),
            beep=bool(tts_cfg.get("beep", True)), beep_hz=int(tts_cfg.get("beep_hz", 880)),
            beep_ms=int(tts_cfg.get("beep_ms", 120)),
        )

    return Speaker(factory)
