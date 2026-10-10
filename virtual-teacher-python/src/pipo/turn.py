"""One turn: hear the child, let the model answer, and speak each sentence as soon as it is written.

Three things overlap: the model keeps writing (GPU) while Kokoro speaks the previous sentence (CPU) while the
speaker plays the one before. They are plain function calls and a queue inside this process.
"""
import logging
import queue
import threading
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import numpy as np

from . import config
from .store import Store
from .text import SentenceSplitter, for_speech
from .values import GREETING, Kid, kid_instructions, load_topics, load_values, speed_for

log = logging.getLogger(__name__)

#: Said when the model wrote nothing speakable.
FALLBACK = "Hmm, me conta de novo?"


class Models(Protocol):
    def transcribe(self, audio: np.ndarray) -> str:
        """16 kHz mono float32 -> text."""
        ...

    def reply(self, messages: list[dict], cancel: threading.Event) -> Iterator[str]:
        """The answer in pieces while it is written. Stops writing when `cancel` is set."""
        ...

    def speak(self, text: str, speed: float) -> np.ndarray:
        """Text -> 24 kHz mono float32."""
        ...


@dataclass
class TurnResult:
    heard: str
    said: list[str] = field(default_factory=list)
    cancelled: bool = False
    #: What was heard was Pipo's own voice coming back through the microphone; nothing was answered.
    echo: bool = False


def build_messages(store: Store, session_id: str, kid: Kid, user_text: str) -> list[dict]:
    history = store.lines(session_id, last=config.HISTORY_LINES)
    return [
        {"role": "system", "content": kid_instructions(kid, load_values(), load_topics())},
        *({"role": "assistant" if role == "assistente" else "user", "content": text} for role, text in history),
        {"role": "user", "content": user_text},
    ]


def run_turn(
    *,
    store: Store,
    models: Models,
    kid: Kid,
    session_id: str,
    play: Callable[[np.ndarray, str], None],
    cancel: threading.Event,
    now: Callable[[], datetime],
    audio: np.ndarray | None = None,
    greet: bool = False,
    is_echo: Callable[[str], bool] | None = None,
) -> TurnResult:
    """Runs one turn to the end, or until `cancel` is set (the child talked over Pipo).

    `play` receives each sentence's audio and text, in order, as soon as it is spoken. The transcript gets the child's
    line and what Pipo actually said (sentences already sent to the speaker), never what was cut off."""
    heard = models.transcribe(audio).strip() if audio is not None else ""
    if not heard and not greet:
        return TurnResult(heard="")
    if heard and is_echo and is_echo(heard):
        return TurnResult(heard=heard, echo=True)

    messages = build_messages(store, session_id, kid, GREETING if greet else heard)
    turn_id = uuid.uuid4().hex[:8]
    if heard:
        store.add_line(session_id, f"{turn_id}_in", "crianca", heard, now())

    speed = speed_for(kid)
    said: list[str] = []
    errors: list[BaseException] = []
    sentences: queue.Queue[str | None] = queue.Queue()

    def voice() -> None:
        while (s := sentences.get()) is not None:
            if cancel.is_set() or errors:
                continue
            try:
                clip = models.speak(s, speed)
            except Exception as err:  # noqa: BLE001 - reported by the turn below
                errors.append(err)
                continue
            if not cancel.is_set():
                play(clip, s)
                said.append(s)

    speaker = threading.Thread(target=voice, name="pipo-voice", daemon=True)
    speaker.start()
    queued = 0
    splitter = SentenceSplitter()
    try:
        for piece in models.reply(messages, cancel):
            if cancel.is_set() or errors:
                break
            for s in map(for_speech, splitter.push(piece)):
                if s:
                    sentences.put(s)
                    queued += 1
        if not cancel.is_set() and not errors:
            for s in map(for_speech, splitter.flush()):
                if s:
                    sentences.put(s)
                    queued += 1
            if not queued:
                sentences.put(FALLBACK)
    finally:
        sentences.put(None)
        speaker.join()

    if said:
        store.add_line(session_id, f"{turn_id}_out", "assistente", " ".join(said), now())
    if errors:
        raise errors[0]
    return TurnResult(heard=heard, said=said, cancelled=cancel.is_set())
