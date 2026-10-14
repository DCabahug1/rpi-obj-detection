"""Offline TTS in its own thread. The capture loop only ever does a non-blocking put()."""
from __future__ import annotations

import itertools
import logging
import queue
import threading
import time
from typing import Optional, Protocol

from callout import Callout

log = logging.getLogger(__name__)

_SENTINEL_RANK = 1 << 30


class Engine(Protocol):
    def say(self, text: str) -> None: ...
    def beep(self) -> None: ...
    def close(self) -> None: ...


class Pyttsx3Engine:
    def __init__(self, rate: int, volume: float, beep: bool, beep_hz: int, beep_ms: int):
        import pyttsx3

        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)
        self._beep = None
        if beep:
            self._beep = _make_beep(beep_hz, beep_ms)

    def say(self, text: str) -> None:
        self.engine.say(text)
        self.engine.runAndWait()

    def beep(self) -> None:
        if self._beep:
            self._beep()

    def close(self) -> None:
        try:
            self.engine.stop()
        except Exception:  # pragma: no cover
            pass


def _make_beep(hz: int, ms: int):
    """Return a callable that plays a short sine beep, or None if no audio backend is available."""
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


class Speaker:
    """Priority queue + worker thread.

    `say()` never blocks. If a callout arrives with a higher priority (lower rank) than
    anything queued or currently being spoken, the queue is cleared so it is spoken next.
    pyttsx3 cannot be interrupted mid-utterance portably, so utterances are kept short.
    """

    def __init__(self, engine: Engine):
        self.engine = engine
        self._q: "queue.PriorityQueue[tuple[int, int, Optional[Callout]]]" = queue.PriorityQueue()
        self._seq = itertools.count()
        self._lock = threading.Lock()
        self._current_rank: Optional[int] = None  # rank of utterance being spoken
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="speaker", daemon=True)
        self._thread.start()

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
            try:
                if callout.is_near:
                    self.engine.beep()
                self.engine.say(callout.text)
            except Exception as e:  # never let audio kill the loop
                log.error("TTS error: %s", e)
            finally:
                self._current_rank = None

    def close(self, timeout: float = 3.0) -> None:
        """Speak what is already queued, then stop. The sentinel sorts after every real callout."""
        self._stop.set()
        self._q.put((_SENTINEL_RANK, next(self._seq), None))
        self._thread.join(timeout)
        self.engine.close()


def make_speaker(tts_cfg: dict, mute: bool = False) -> Speaker:
    if mute:
        return Speaker(LogEngine())
    try:
        engine = Pyttsx3Engine(
            rate=int(tts_cfg.get("rate", 190)), volume=float(tts_cfg.get("volume", 1.0)),
            beep=bool(tts_cfg.get("beep", True)), beep_hz=int(tts_cfg.get("beep_hz", 880)),
            beep_ms=int(tts_cfg.get("beep_ms", 120)),
        )
        return Speaker(engine)
    except Exception as e:
        log.warning("pyttsx3 unavailable (%s); callouts will be logged instead of spoken", e)
        return Speaker(LogEngine())
