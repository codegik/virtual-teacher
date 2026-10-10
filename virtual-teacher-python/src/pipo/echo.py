"""Keeping Pipo from hearing himself.

On laptop speakers the microphone hears Pipo's voice louder than a child sitting in front of the screen
(measured on this laptop: his echo peaked at 1.46 RMS, past full scale, while the "child talks over Pipo"
threshold was 0.23), so he kept interrupting himself after a sentence or two.

Two steps, before the speech detector sees anything:
1. `condition`: a fixed gain plus a soft limiter, so the microphone never clips (clipped echo can't be cancelled).
2. `EchoCanceller`: WebRTC's echo canceller (the one browsers use), fed with exactly what the speaker played.
What is left of his voice after this is still sometimes loud enough to look like speech; the conversation
confirms an interruption with Whisper before stopping him (see conversation.py).
"""
import numpy as np

RATE = 16000
CHUNK = 160  # WebRTC processes 10 ms at a time


def condition(frame: np.ndarray, gain: float) -> np.ndarray:
    """Gain, then a soft limiter: quiet speech passes almost unchanged, nothing goes past full scale."""
    return np.tanh(frame * gain).astype(np.float32)


def _i16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


class EchoCanceller:
    def __init__(self) -> None:
        from livekit import rtc

        self._rtc = rtc
        self._apm = rtc.AudioProcessingModule(echo_cancellation=True, noise_suppression=True, high_pass_filter=True)

    def process(self, mic: np.ndarray, played: np.ndarray) -> np.ndarray:
        """`mic` and `played` are 16 kHz float32 of the same length (a multiple of 10 ms): what the microphone
        heard and what the speaker played over the same stretch of time. Returns the microphone without Pipo."""
        rtc = self._rtc
        out = np.empty(len(mic), np.float32)
        for i in range(0, len(mic) - CHUNK + 1, CHUNK):
            self._apm.process_reverse_stream(rtc.AudioFrame(_i16(played[i:i + CHUNK]).tobytes(), RATE, 1, CHUNK))
            frame = rtc.AudioFrame(_i16(mic[i:i + CHUNK]).tobytes(), RATE, 1, CHUNK)
            self._apm.process_stream(frame)
            out[i:i + CHUNK] = np.frombuffer(bytes(frame.data), np.int16).astype(np.float32) / 32768
        return out


_WORD = __import__("re").compile(r"\w+")


def _words(text: str) -> list[str]:
    import unicodedata

    plain = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    return _WORD.findall(plain)


def is_own_echo(heard: str, pipo_said: str) -> bool:
    """True when what the microphone caught is Pipo's own words (or nothing at all), not the child.
    Whisper on an echo hears fragments of the sentence he is saying; a child interrupting says something else."""
    heard_words = _words(heard)
    if not heard_words:
        return True
    said = set(_words(pipo_said))
    if not said:
        return False
    return sum(w in said for w in heard_words) / len(heard_words) >= 0.6
