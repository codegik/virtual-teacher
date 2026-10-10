"""Shared fakes. The models are the only boundary faked here: SQLite, the splitter, the turn and the conversation
loop are real. Fakes serve recorded real output (tests/fixtures), not invented shapes."""
import json
import threading
from collections.abc import Callable
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from pipo.store import Store
from pipo.values import Kid

FIXTURES = Path(__file__).parent / "fixtures"


def read_wav(name: str) -> np.ndarray:
    with wave.open(str(FIXTURES / name)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


#: A real reply from qwen2.5:7b with the real kid instructions, in the pieces the model streamed it.
QWEN_PIECES: list[str] = json.loads((FIXTURES / "qwen-stream.json").read_text("utf-8"))["pieces"]
#: What Whisper heard in child-7x8.wav.
HEARD = json.loads((FIXTURES / "voice-transcribe.json").read_text("utf-8"))["body"]["text"]
#: Kokoro saying "Oi".
OI = read_wav("voice-speak-oi.wav")


class FakeModels:
    """Captures everything the turn asks for. `replies` is a queue of answers, each a list of streamed pieces."""

    def __init__(self) -> None:
        self.heard = HEARD
        self.hear_next: list[str] = []  # what Whisper hears next, one per call, before falling back to `heard`
        self.replies: list[list[str]] = []
        self.transcribed: list[np.ndarray] = []
        self.chats: list[list[dict]] = []
        self.spoken: list[tuple[str, float]] = []
        self.pieces_sent: list[int] = []  # how many pieces the model had sent when each sentence was spoken
        self.stopped_early = False
        self.fail_reply: Exception | None = None
        self.piece_delay = 0.0
        self._sent = 0

    def transcribe(self, audio: np.ndarray) -> str:
        self.transcribed.append(audio)
        return self.hear_next.pop(0) if self.hear_next else self.heard

    def reply(self, messages, cancel):
        self.chats.append(messages)
        if self.fail_reply:
            raise self.fail_reply
        pieces = self.replies.pop(0) if self.replies else ["Que legal! Me conta mais?"]
        self._sent = 0
        for i, p in enumerate(pieces):
            if cancel.is_set():
                self.stopped_early = i < len(pieces)
                return
            self._sent = i + 1
            yield p
            if self.piece_delay:
                time.sleep(self.piece_delay)

    def speak(self, text: str, speed: float) -> np.ndarray:
        self.spoken.append((text, speed))
        self.pieces_sent.append(self._sent)
        return OI


class FakeOutput:
    """Plays instantly: every clip is heard the moment it is played, so stop() has nothing left to cut."""

    def __init__(self) -> None:
        self.clips: list[np.ndarray] = []
        self.texts: list[str] = []
        self.stops = 0
        self.ducks: list[bool] = []
        self.on_play: Callable[[int], None] | None = None
        self._heard: list[str] = []

    def play(self, clip, text=""):
        self.clips.append(clip)
        self.texts.append(text)
        self._heard.append(text)
        if self.on_play:
            self.on_play(len(self.clips))

    def stop(self):
        self.stops += 1
        return []

    def duck(self, on: bool):
        self.ducks.append(on)

    def wait(self, cancel: threading.Event):
        return

    def heard_texts(self):
        done, self._heard = [t for t in self._heard if t], []
        return done

    @property
    def playing(self) -> bool:
        return False


@pytest.fixture
def store(tmp_path) -> Store:
    # A real SQLite file per test; nothing is ever wiped.
    return Store(tmp_path / "pipo.db")


@pytest.fixture
def models() -> FakeModels:
    return FakeModels()


KID_4 = Kid(id="crianca-4", name="Filha de 4 anos", age=4, emoji="🌼")
KID_9 = Kid(id="crianca-9", name="Theo", age=9, emoji="🚀")
