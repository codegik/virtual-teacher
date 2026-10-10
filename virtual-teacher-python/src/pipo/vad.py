"""Is someone talking? An energy detector on the microphone, ported from the browser's (web/src/conversation.ts).

Pure: frames in, events out. The microphone gives 16 kHz mono float32 frames, which is also what Whisper takes,
so the utterance goes to Whisper as it is, with no WAV in between."""
import collections
from dataclasses import dataclass

import numpy as np

RATE = 16000
FRAME = 640  # 40 ms
START_FRAMES = 4  # ~160 ms of loud audio to count as speech
END_SILENCE_MS = 1100  # kids pause mid-sentence; wait a bit longer than for adults
MAX_UTTERANCE_MS = 15_000
MIN_SPEECH_MS = 350
PREROLL_FRAMES = 8  # keep the start of the first word
CALIBRATION_FRAMES = 12  # ~0.5 s to learn how loud the room is
#: While Pipo speaks, his own voice leaks into the microphone; the child must be this much louder to interrupt.
SPEAKING_FACTOR = 2.5
#: How much of the recent room sound is remembered (~3 s): music, a kid singing across the room, the TV.
ROOM_FRAMES = 75
#: The room level only counts once this much of it was heard (~1 s).
ROOM_MIN_FRAMES = 25
#: While Pipo speaks, an interruption must be this much louder than the room's recent level. A child at the
#: laptop is much louder at the microphone than one singing along to music across the room.
ROOM_FACTOR = 2.5


@dataclass
class Started:
    """Speech began."""


@dataclass
class Ended:
    audio: np.ndarray
    voiced_ms: float


class SpeechDetector:
    def __init__(self, end_silence_ms: float = END_SILENCE_MS, speaking_factor: float = SPEAKING_FACTOR,
                 room_factor: float = ROOM_FACTOR) -> None:
        self.end_silence_ms = end_silence_ms
        self.speaking_factor = speaking_factor
        self.room_factor = room_factor
        self._room: collections.deque[float] = collections.deque(maxlen=ROOM_FRAMES)
        self._recording_threshold = 0.0
        #: Loudness of the last frame, for the log.
        self.level = 0.0
        self.frame_ms = FRAME / RATE * 1000
        self.floor = 0.0
        self._calibration: list[float] = []
        self._preroll: list[np.ndarray] = []
        self._chunks: list[np.ndarray] = []
        self.recording = False
        self._loud = 0
        self._voiced_ms = 0.0
        self._quiet_ms = 0.0

    def so_far(self) -> np.ndarray:
        """The utterance being recorded, up to now (empty when not recording)."""
        return np.concatenate(self._chunks).astype(np.float32) if self.recording and self._chunks else np.zeros(0, np.float32)

    def recorded_ms(self) -> float:
        return len(self._chunks) * self.frame_ms if self.recording else 0.0

    def drop(self) -> None:
        """Forget the utterance being recorded (it was Pipo's own voice)."""
        self.recording = False
        self._loud = 0
        self._chunks = []
        self._preroll = []

    def room(self) -> float:
        """The room's typical level over the last ~3 s (0 until ~1 s was heard)."""
        return float(np.median(self._room)) if len(self._room) >= ROOM_MIN_FRAMES else 0.0

    def threshold(self, pipo_speaking: bool) -> float:
        base = max(self.floor * 3.5, 0.012)
        if not pipo_speaking:
            return base
        return max(base * self.speaking_factor, self.room() * self.room_factor)

    def feed(self, frame: np.ndarray, pipo_speaking: bool = False) -> Started | Ended | None:
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))) if len(frame) else 0.0
        self.level = rms

        if len(self._calibration) < CALIBRATION_FRAMES:
            self._calibration.append(rms)
            self.floor = sum(self._calibration) / len(self._calibration)
            return None
        # The room is everything heard while Pipo talks (music that never stops must raise the bar even though
        # it looked like speech at first) plus the quiet between turns, but never the child's own turn: right
        # after she spoke, her voice must not become the bar she has to beat. The end of an utterance uses the
        # bar from when it started.
        threshold = self.threshold(pipo_speaking)
        if pipo_speaking or not self.recording:
            self._room.append(rms)

        if not self.recording:
            self._preroll.append(frame)
            if len(self._preroll) > PREROLL_FRAMES:
                self._preroll.pop(0)
            # Slowly follow the room's noise while nobody talks.
            if rms < threshold and not pipo_speaking:
                self.floor = self.floor * 0.98 + rms * 0.02
            self._loud = self._loud + 1 if rms > threshold else 0
            if self._loud >= START_FRAMES:
                self.recording = True
                self._chunks = list(self._preroll)
                self._voiced_ms = self._quiet_ms = 0.0
                self._recording_threshold = threshold
                return Started()
            return None

        self._chunks.append(frame)
        if rms > self._recording_threshold * 0.6:
            self._voiced_ms += self.frame_ms
            self._quiet_ms = 0.0
        else:
            self._quiet_ms += self.frame_ms
        total = len(self._chunks) * self.frame_ms
        if self._quiet_ms >= self.end_silence_ms or total >= MAX_UTTERANCE_MS:
            self.recording = False
            self._loud = 0
            self._preroll = []
            audio = np.concatenate(self._chunks).astype(np.float32)
            self._chunks = []
            return Ended(audio, self._voiced_ms)
        return None
