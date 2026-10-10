"""The voice loop, without any UI: microphone frames in, phases out.

idle -> listening (the child talks) -> thinking (Whisper + the model) -> speaking -> idle.

Talking over Pipo stops him (barge-in), but on laptop speakers the microphone also hears Pipo himself, and
after echo cancellation (audio.py) some of his voice still gets through loud enough to look like speech.
So an interruption is confirmed before he stops: when speech starts while he talks, he softens his voice
(like a person who hears someone start) and Whisper listens to the first moment of it. If it is his own
words (or nothing), he carries on at full volume. If it is someone else, he stops, and the child's sentence
becomes the next turn. The same check covers the echo's tail right after he finishes."""
import logging
import queue
import threading
import time
from collections.abc import Callable
from datetime import datetime
from typing import Protocol

import numpy as np

from .echo import is_own_echo
from .store import Store
from .turn import Models, run_turn
from .vad import MIN_SPEECH_MS, Ended, SpeechDetector, Started
from .values import Kid

log = logging.getLogger(__name__)
CANT_TALK = "O Pipo não conseguiu falar agora. Tenta de novo?"
#: How much of an interruption Whisper hears before deciding whether it is the child or Pipo's own echo.
CONFIRM_MS = 800
#: Speech starting this soon after Pipo stops may still be the room ringing with his voice.
TAIL_S = 0.8


class Output(Protocol):
    def play(self, clip: np.ndarray) -> None: ...
    def stop(self) -> None: ...
    def duck(self, on: bool) -> None: ...
    def wait(self, cancel: threading.Event) -> None: ...
    @property
    def playing(self) -> bool: ...


class Conversation:
    def __init__(self, *, store: Store, models: Models, kid: Kid, output: Output,
                 on_phase: Callable[[str], None], now: Callable[[], datetime],
                 detector: SpeechDetector | None = None) -> None:
        self.store, self.models, self.kid, self.output = store, models, kid, output
        self.now = now
        self._on_phase = on_phase
        self.phase = "starting"
        self.detector = detector or SpeechDetector()
        self.session_id = store.start_session(kid.id, now())
        self._cancel = threading.Event()
        self._work: queue.Queue[tuple[np.ndarray | None, bool, str] | None] = queue.Queue()
        self._closed = False
        self._lock = threading.RLock()
        #: What Pipo has sent to the speaker in this turn, to recognise his own voice in the microphone.
        self._saying: list[str] = []
        #: None: nothing to check; "checking": speech started while he talks; "deciding": Whisper is listening.
        self._barge: str | None = None
        self._ended_while_checking: np.ndarray | None = None
        self._stopped_at = 0.0
        self._tail_suspect = ""
        self.echoes_ignored = 0
        self._worker = threading.Thread(target=self._loop, name="pipo-turns", daemon=True)
        self._worker.start()
        self._work.put((None, True, ""))  # Pipo says hello first

    # ---- called from the microphone thread -------------------------------
    def feed(self, frame: np.ndarray) -> None:
        if self._closed or self.phase in ("starting", "thinking"):
            return
        with self._lock:
            speaking = self.phase == "speaking"
            event = self.detector.feed(frame, pipo_speaking=speaking)
            if speaking:
                self._while_speaking(event)
                return
            if isinstance(event, Started):
                # Right after he stopped, this may be the echo's tail: check it like an interruption.
                self._tail_suspect = " ".join(self._saying) if time.monotonic() - self._stopped_at < TAIL_S else ""
        if isinstance(event, Started):
            self._set("listening")
        elif isinstance(event, Ended):
            self._finish(event)

    def _while_speaking(self, event: Started | Ended | None) -> None:
        """Speech while Pipo talks: soften, and let Whisper decide whether it is the child. Holds the lock."""
        if isinstance(event, Started):
            self._barge = "checking"
            self._ended_while_checking = None
            self.output.duck(True)
            return
        if self._barge != "checking":
            return
        if isinstance(event, Ended):
            self._ended_while_checking = event.audio if event.voiced_ms >= MIN_SPEECH_MS else np.zeros(0, np.float32)
            self._check(event.audio)
        elif self.detector.recorded_ms() >= CONFIRM_MS:
            self._check(self.detector.so_far())

    def _check(self, audio: np.ndarray) -> None:
        self._barge = "deciding"
        said = " ".join(self._saying)
        threading.Thread(target=self._decide, args=(audio, said), name="pipo-barge", daemon=True).start()

    def _decide(self, audio: np.ndarray, said: str) -> None:
        try:
            heard = self.models.transcribe(audio)
        except Exception:  # noqa: BLE001 - when unsure, let Pipo keep talking
            log.exception("could not check an interruption")
            heard = ""
        with self._lock:
            if self._barge != "deciding":
                return  # the turn ended meanwhile
            ended, self._ended_while_checking = self._ended_while_checking, None
            if is_own_echo(heard, said):
                log.info("ignored my own voice: %r", heard)
                self.echoes_ignored += 1
                self._barge = None
                self.detector.drop()
                self.output.duck(False)
                return
            log.info("interrupted by: %r", heard)
            self._barge = None
            self.interrupt()
            if ended is not None and len(ended):
                self._set("thinking")
                self._work.put((ended, False, ""))
            elif ended is not None:
                self._set("idle")
            else:
                self._set("listening")  # still talking; the detector keeps recording the sentence

    def _finish(self, event: Ended) -> None:
        if event.voiced_ms < MIN_SPEECH_MS:
            self._set("idle")
        else:
            self._set("thinking")
            self._work.put((event.audio, False, self._tail_suspect))

    def interrupt(self) -> None:
        """Barge-in: silence now, and stop the turn that is still writing."""
        self._cancel.set()
        self.output.stop()

    def close(self) -> None:
        self._closed = True
        self.interrupt()
        self._work.put(None)
        self._worker.join(timeout=10)
        self.store.end_session(self.session_id, self.now())

    # ---- the turn thread --------------------------------------------------
    def _loop(self) -> None:
        while (job := self._work.get()) is not None:
            audio, greet, tail_of = job
            self._cancel = cancel = threading.Event()
            with self._lock:
                self._saying = []
                self._barge = None
            self._set("thinking")

            def play(clip: np.ndarray, text: str) -> None:
                with self._lock:
                    self._saying.append(text)
                    if self.phase not in ("listening", "speaking"):
                        self._set("speaking")
                self.output.play(clip)

            echo = (lambda heard: is_own_echo(heard, tail_of)) if tail_of else None
            try:
                result = run_turn(store=self.store, models=self.models, kid=self.kid, session_id=self.session_id,
                                  play=play, cancel=cancel, now=self.now, audio=audio, greet=greet, is_echo=echo)
                if result.echo:
                    self.echoes_ignored += 1
                    log.info("ignored my own voice after speaking: %r", result.heard)
                    with self._lock:
                        self._saying = tail_of.split()  # keep recognising the same words
                self.output.wait(cancel)
            except Exception:  # noqa: BLE001 - the child can try again; the reason goes to the log
                log.exception("turn failed for session %s", self.session_id)
                self._set("error")
                cancel.wait(2.5)
            if not cancel.is_set() and not self._closed:
                with self._lock:
                    if self._barge is not None:
                        self.output.duck(False)
                    self._barge = None
                    self._stopped_at = time.monotonic()
                    self._set("listening" if self.detector.recording else "idle")

    def _set(self, phase: str) -> None:
        if self._closed:
            return
        self.phase = phase
        self._on_phase(phase)
