"""Times real turns with the real models in this process: time to first audio, and the whole turn.
Uses the recorded child's voice (tests/fixtures/child-7x8.wav). Run with `make bench`."""
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .models import LocalModels
from .store import Store
from .turn import run_turn
from .values import load_kids


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = Path(__file__).resolve().parents[2]
    import wave

    with wave.open(str(root / "tests/fixtures/child-7x8.wav")) as w:
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        rate = w.getframerate()
    audio = np.interp(np.linspace(0, len(raw) - 1, int(len(raw) * 16000 / rate)), np.arange(len(raw)), raw).astype(np.float32)

    t = time.time()
    models = LocalModels()
    print(f"models loaded in {time.time() - t:.1f}s")
    store = Store(":memory:")
    kid = next(k for k in load_kids() if k.age >= 9)
    sid = store.start_session(kid.id, datetime.now(timezone.utc))

    for n in range(3):
        start = time.time()
        first: list[float] = []
        clips: list[float] = []

        def play(clip: np.ndarray, _text: str) -> None:
            if not first:
                first.append(time.time() - start)
            clips.append(len(clip) / 24000)

        t0 = time.time()
        heard = models.transcribe(audio)
        whisper_s = time.time() - t0
        start = time.time()
        r = run_turn(store=store, models=models, kid=kid, session_id=sid, play=play, cancel=threading.Event(),
                     now=lambda: datetime.now(timezone.utc), audio=audio)
        total = time.time() - start
        print(f"turn {n + 1}: whisper {whisper_s:.2f}s | first audio {first[0]:.2f}s after the child stopped "
              f"| whole answer made in {total:.2f}s | {len(clips)} clips, {sum(clips):.1f}s of speech")
        print(f"   heard: {heard!r}")
        print(f"   said:  {' '.join(r.said)!r}")


if __name__ == "__main__":
    main()
