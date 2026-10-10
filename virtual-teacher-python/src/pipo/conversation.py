"""The voice loop, without any UI: microphone frames in, phases out.

idle -> listening (the child talks) -> thinking (Whisper + the model) -> speaking -> idle.

Talking over Pipo stops him (barge-in), but the microphone also hears things that are not the child talking
to him: Pipo himself (after echo cancellation, audio.py, some of his voice still gets through), and the room
(music, a kid singing along across the room). So an interruption is confirmed before he stops:

1. It must be loud compared with the room right now (vad.py: the bar rises with music and singing).
2. Then he softens his voice (like a person who hears someone start) and Whisper listens to its first moment.
   Only someone talking TO him (echo.is_interruption: "Pipo, ...", "para!", or a real phrase that is not his)
   stops him. His own words, a murmur, a sung word or two, or nothing: he carries on at full volume.
3. If he stopped and it turns out nobody said anything after all, he says the cut sentence again and finishes.

When the child really talks, her sentence becomes the next turn. The same echo check covers the echo's tail
right after he finishes. Every decision goes to the log with the numbers behind it.
"""
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np

from .echo import is_interruption, is_own_echo
from .store import Store
from .turn import Models, run_turn
from .vad import MIN_SPEECH_MS, Ended, SpeechDetector, Started
from .values import Kid

log = logging.getLogger(__name__)
CANT_TALK = "O Pipo não conseguiu falar agora. Tenta de novo?"
#: How much of an interruption Whisper hears before deciding whether it is the child or not.
CONFIRM_MS = 800
#: Speech starting this soon after Pipo stops may still be the room ringing with his voice.
TAIL_S = 0.8
#: After a stop that turned out to be nothing, he picks up where he was only if it was this recent.
RESUME_S = 10.0


class Output(Protocol):
    def play(self, clip: np.ndarray, text: str = "") -> None: ...
    def stop(self) -> list[tuple[np.ndarray, str]]: ...
    def duck(self, on: bool) -> None: ...
    def wait(self, cancel: threading.Event) -> None: ...
    def heard_texts(self) -> list[str]: ...
    @property
    def playing(self) -> bool: ...


@dataclass
class _Job:
    audio: np.ndarray | None = None
    greet: bool = False
    #: Pipo's words this may be the echo of (speech that started right after he finished).
    tail_of: str = ""
    #: Say again what an interruption that turned out to be nothing cut off.
    resume: bool = False


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
        self._work: queue.Queue[_Job | None] = queue.Queue()
        self._closed = False
        self._lock = threading.RLock()
        #: What Pipo has sent to the speaker in this turn, to recognise his own voice in the microphone.
        self._saying: list[str] = []
        #: None: nothing to check; "checking": speech started while he talks; "deciding": Whisper is listening.
        self._barge: str | None = None
        self._ended_while_checking: np.ndarray | None = None
        self._stopped_at = 0.0
        self._tail_suspect = ""
        #: What the last interruption cut off (the cut sentence whole, then the rest), and when.
        self._cut: list[tuple[np.ndarray, str]] = []
        self._cut_at = 0.0
        self.echoes_ignored = 0
        self.interruptions_ignored = 0
        self.resumed = 0
        self._worker = threading.Thread(target=self._loop, name="pipo-turns", daemon=True)
        self._worker.start()
        self._work.put(_Job(greet=True))  # Pipo says hello first

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
        """Sound while Pipo talks: soften, and let Whisper decide whether it is the child. Holds the lock."""
        if isinstance(event, Started):
            d = self.detector
            log.info("sound while speaking: level %.3f, bar %.3f, room %.3f - checking",
                     d.level, d.threshold(True), d.room())
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
            if not is_interruption(heard, said):
                own = is_own_echo(heard, said)
                log.info("kept talking: heard %r (%s)", heard, "my own voice" if own else "nobody talking to me")
                if own:
                    self.echoes_ignored += 1
                else:
                    self.interruptions_ignored += 1
                self._barge = None
                self.detector.drop()
                self.output.duck(False)
                return
            log.info("interrupted by: %r", heard)
            self._barge = None
            self.interrupt()
            if ended is not None and len(ended):
                self._set("thinking")
                self._work.put(_Job(audio=ended))
            elif ended is not None:
                self._resume_or_idle()
            else:
                self._set("listening")  # still talking; the detector keeps recording the sentence

    def _finish(self, event: Ended) -> None:
        if event.voiced_ms < MIN_SPEECH_MS:
            self._resume_or_idle()
        else:
            self._set("thinking")
            self._work.put(_Job(audio=event.audio, tail_of=self._tail_suspect))

    def _resume_or_idle(self) -> None:
        """Nothing was said after all: if he was cut off a moment ago, he picks up where he was."""
        with self._lock:
            if self._cut and time.monotonic() - self._cut_at < RESUME_S:
                self._set("thinking")
                self._work.put(_Job(resume=True))
            else:
                self._cut = []
                self._set("idle")

    def interrupt(self) -> None:
        """Barge-in: silence now, and stop the turn that is still writing."""
        self._cancel.set()
        cut = self.output.stop()
        with self._lock:
            self._cut, self._cut_at = list(cut or []), time.monotonic()

    def close(self) -> None:
        self._closed = True
        self.interrupt()
        self._work.put(None)
        self._worker.join(timeout=10)
        self.store.end_session(self.session_id, self.now())

    # ---- the turn thread --------------------------------------------------
    def _loop(self) -> None:
        while (job := self._work.get()) is not None:
            self._cancel = cancel = threading.Event()
            self.output.heard_texts()  # forget what an earlier turn played
            with self._lock:
                self._saying = []
                self._barge = None
            self._set("thinking")

            def play(clip: np.ndarray, text: str) -> None:
                with self._lock:
                    self._saying.append(text)
                    if self.phase not in ("listening", "speaking"):
                        self._set("speaking")
                self.output.play(clip, text)

            def heard_to_end(_sent: list[str], cancel: threading.Event = cancel) -> list[str]:
                self.output.wait(cancel)
                return self.output.heard_texts()

            try:
                if job.resume:
                    self._say_again(play, heard_to_end)
                else:
                    echo = (lambda heard, t=job.tail_of: is_own_echo(heard, t)) if job.tail_of else None
                    result = run_turn(store=self.store, models=self.models, kid=self.kid,
                                      session_id=self.session_id, play=play, cancel=cancel, now=self.now,
                                      audio=job.audio, greet=job.greet, is_echo=echo, heard_to_end=heard_to_end)
                    if result.echo:
                        self.echoes_ignored += 1
                        log.info("ignored my own voice after speaking: %r", result.heard)
                        with self._lock:
                            self._saying = job.tail_of.split()  # keep recognising the same words
                    if result.heard and not result.echo:
                        with self._lock:
                            self._cut = []  # the child really said something; that is what he answers now
                    elif not job.greet and not cancel.is_set():
                        log.info("nothing was said after all (%r)", result.heard)
                        self._say_again(play, heard_to_end)
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

    def _say_again(self, play: Callable[[np.ndarray, str], None],
                   heard_to_end: Callable[[list[str]], list[str]]) -> None:
        """Plays what the last interruption cut off (if recent) and records what was heard of it."""
        with self._lock:
            cut, self._cut = self._cut, []
            if not cut or time.monotonic() - self._cut_at >= RESUME_S:
                return
        self.resumed += 1
        log.info("saying again what was cut: %r", " ".join(t for _, t in cut))
        for clip, text in cut:
            if self._cancel.is_set():
                break
            play(clip, text)
        said = heard_to_end([t for _, t in cut])
        if said:
            self.store.add_line(self.session_id, f"{time.monotonic_ns():x}_again", "assistente", " ".join(said),
                                self.now())

    def _set(self, phase: str) -> None:
        if self._closed:
            return
        self.phase = phase
        self._on_phase(phase)
