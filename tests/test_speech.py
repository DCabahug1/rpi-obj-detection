import threading
import time

from callout import Callout
from speech import LogEngine, Speaker


def co(text, rank, near=True):
    return Callout(text=text, label=text, distance="near", direction="ahead", priority_rank=rank, is_near=near)


class SlowEngine(LogEngine):
    """Blocks on the first utterance until released, so we can test queue behaviour while 'speaking'."""

    def __init__(self):
        super().__init__()
        self.release = threading.Event()
        self.started = threading.Event()

    def say(self, text):
        self.spoken.append(text)
        if len(self.spoken) == 1:
            self.started.set()
            self.release.wait(2.0)


def test_say_never_blocks():
    eng = SlowEngine()
    s = Speaker(eng)
    t0 = time.monotonic()
    for i in range(50):
        s.say(co(f"x{i}", 5))
    assert time.monotonic() - t0 < 0.1
    eng.release.set()
    s.close()


def test_higher_priority_clears_queue():
    eng = SlowEngine()
    s = Speaker(eng)
    s.say(co("chair", 6))          # starts speaking, blocks
    assert eng.started.wait(1.0)
    s.say(co("table", 7))          # queued
    s.say(co("plant", 8))          # queued
    s.say(co("person", 0))         # higher priority: clears table/plant
    eng.release.set()
    s.close()
    assert eng.spoken == ["chair", "person"]


def test_lower_priority_does_not_clear_queue():
    eng = SlowEngine()
    s = Speaker(eng)
    s.say(co("person", 0))
    assert eng.started.wait(1.0)
    s.say(co("chair", 6))
    s.say(co("plant", 8))
    eng.release.set()
    s.close()
    assert eng.spoken == ["person", "chair", "plant"]


def test_engine_error_does_not_kill_thread():
    class Bad(LogEngine):
        def say(self, text):
            if text == "boom":
                raise RuntimeError("tts died")
            super().say(text)

    eng = Bad()
    s = Speaker(eng)
    s.say(co("boom", 1))
    s.say(co("ok", 2))
    s.close()
    assert eng.spoken == ["ok"]


def test_factory_runs_on_worker_thread():
    created_on = []

    def factory():
        created_on.append(threading.current_thread())
        return LogEngine()

    s = Speaker(factory)
    assert isinstance(s.engine, LogEngine)
    assert created_on and created_on[0] is not threading.main_thread()
    s.say(co("chair", 6))
    s.close()
    assert s.engine.spoken == ["chair"]


def test_factory_failure_falls_back_to_logging():
    def factory():
        raise RuntimeError("no audio device")

    s = Speaker(factory)
    assert isinstance(s.engine, LogEngine)
    s.say(co("person", 0))
    s.close()
    assert s.engine.spoken == ["person"]


def test_engine_closed_on_worker_thread():
    class Closing(LogEngine):
        def __init__(self):
            super().__init__()
            self.closed_on = None

        def close(self):
            self.closed_on = threading.current_thread()

    eng = Closing()
    s = Speaker(eng)
    s.say(co("x", 1))
    s.close()
    assert eng.closed_on is not None and eng.closed_on is not threading.main_thread()
