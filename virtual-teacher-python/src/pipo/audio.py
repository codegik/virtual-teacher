"""Microphone and speaker, straight from the sound server (PipeWire/PulseAudio) through PortAudio."""
import collections
import threading

import numpy as np
import sounddevice as sd

from . import config
from .echo import EchoCanceller, condition
from .models import VOICE_RATE
from .vad import FRAME, RATE

BLOCK = 480  # 20 ms at 24 kHz
#: Pipo's volume while an interruption is being confirmed: he softens, like a person who hears someone start.
DUCKED = 0.35
#: How much of what the speaker played is kept for the echo canceller (it consumes it in real time).
PLAYED_MAX = RATE  # 1 s


class Speaker:
    """Plays clips one after another. `play` returns at once (Kokoro keeps making the next sentence while this
    one plays); `stop` drops everything for a barge-in and returns what was not finished (the cut sentence
    first, whole) so it can be said again; `duck` softens him while an interruption is checked.
    `heard_texts` is what actually reached the child, so the transcript never claims more than she heard.
    `level` is how loud the voice is right now, for Pipo's mouth. Everything played (silence included) is kept
    at 16 kHz for the echo canceller."""

    def __init__(self) -> None:
        #: [whole clip, samples already played, its text]
        self._clips: collections.deque[list] = collections.deque()
        self._heard: list[str] = []
        self._lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self.level = 0.0
        self._volume = 1.0
        self._target_volume = 1.0
        self._played = np.zeros(0, np.float32)
        self._played_lock = threading.Lock()
        self._stream = sd.OutputStream(samplerate=VOICE_RATE, channels=1, dtype="float32", blocksize=BLOCK,
                                       callback=self._fill)
        self._stream.start()

    def play(self, clip: np.ndarray, text: str = "") -> None:
        with self._lock:
            self._clips.append([clip, 0, text])
            self._idle.clear()

    def stop(self) -> list[tuple[np.ndarray, str]]:
        with self._lock:
            unfinished = [(clip, text) for clip, _, text in self._clips]
            self._clips.clear()
            self._idle.set()
        self._target_volume = 1.0
        self.level = 0.0
        return unfinished

    def heard_texts(self) -> list[str]:
        """Texts of the clips that started playing since the last call (the one cut mid-way included; the
        ones still waiting in the queue are not: the child never heard them)."""
        with self._lock:
            done, self._heard = self._heard, []
        return [t for t in done if t]

    def duck(self, on: bool) -> None:
        self._target_volume = DUCKED if on else 1.0

    def wait(self, cancel: threading.Event) -> None:
        """Until everything queued has been heard, or the turn is cancelled."""
        while not self._idle.wait(0.05):
            if cancel.is_set():
                return

    @property
    def playing(self) -> bool:
        return not self._idle.is_set()

    def take_played(self, n: int) -> np.ndarray:
        """The next `n` samples (16 kHz) the speaker played, padded with silence if it is behind."""
        with self._played_lock:
            got, self._played = self._played[:n], self._played[n:]
        return got if len(got) == n else np.concatenate([got, np.zeros(n - len(got), np.float32)])

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()

    def _fill(self, out: np.ndarray, frames: int, _time, _status) -> None:
        buf = out[:, 0]
        buf.fill(0)
        written = 0
        with self._lock:
            while written < frames and self._clips:
                item = self._clips[0]
                clip, pos = item[0], item[1]
                if pos == 0:
                    self._heard.append(item[2])
                n = min(frames - written, len(clip) - pos)
                buf[written:written + n] = clip[pos:pos + n]
                written += n
                item[1] = pos + n
                if item[1] >= len(clip):
                    self._clips.popleft()
            if not self._clips:
                self._idle.set()
        # Glide to the target volume over the block, so ducking doesn't click.
        ramp = np.linspace(self._volume, self._target_volume, frames, dtype=np.float32)
        self._volume = self._target_volume
        buf *= ramp
        self.level = float(np.sqrt(np.mean(np.square(buf[:written])))) if written else 0.0
        # 24 kHz -> 16 kHz for the echo canceller: exactly 2 out of every 3 samples' worth.
        played = np.interp(np.arange(0, frames, 1.5), np.arange(frames), buf).astype(np.float32)
        with self._played_lock:
            self._played = np.concatenate([self._played, played])[-PLAYED_MAX:]


class Microphone:
    """Calls `on_frame` with 40 ms of 16 kHz mono float32, from PortAudio's thread: levelled, limited and,
    with a speaker, without Pipo's own voice."""

    def __init__(self, on_frame, speaker: Speaker | None = None) -> None:
        self._on_frame = on_frame
        self._speaker = speaker
        self._aec = EchoCanceller() if speaker else None
        self._stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=FRAME,
                                      callback=self._read)
        self._stream.start()

    def _read(self, data: np.ndarray, *_) -> None:
        frame = condition(data[:, 0], config.MIC_GAIN)
        if self._aec and self._speaker:
            frame = self._aec.process(frame, self._speaker.take_played(len(frame)))
        self._on_frame(frame)

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
        # Freed now, while its native library is alive; at interpreter exit it raised an AssertionError.
        self._aec = None
