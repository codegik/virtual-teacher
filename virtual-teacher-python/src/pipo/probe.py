"""Measures the real room for the barge-in bar (`make probe`). Pipo tells a story on the speakers through the
app's own microphone chain (gain + echo canceller), in two parts:

1. ~12 s with the room as it is (music on, the kid singing where she usually is): how loud the room is at the
   microphone, and how often it would still have crossed the bar.
2. ~8 s where someone talks to him FROM WHERE THE CHILD SITS at the laptop: how loud a real interruption is.

Prints both levels next to the bar the app uses, and saves the frame levels to data/probe-<time>.npz.
"""
import threading
import time
from datetime import datetime

import numpy as np

from . import config
from .audio import Microphone, Speaker
from .models import _load_kokoro
from .vad import CALIBRATION_FRAMES, ROOM_FACTOR, SPEAKING_FACTOR, START_FRAMES

STORY = ("Era uma vez um gatinho muito curioso que morava numa casa amarela, perto de um rio bem grande. "
         "Todo dia de manhã, ele subia no telhado para olhar os passarinhos. Um dia, ele viu um passarinho azul "
         "que cantava uma música diferente, e resolveu seguir o passarinho pela floresta inteira. ") * 2
ROOM_S, TALK_S = 12, 8


def main() -> None:
    print("Preparando a voz do Pipo...")
    clip, rate = _load_kokoro().create(STORY, voice=config.KOKORO_VOICE, speed=1.0, lang="pt-br")
    clip = np.asarray(clip, np.float32)
    if rate != 24000:
        clip = np.interp(np.linspace(0, len(clip) - 1, int(len(clip) * 24000 / rate)), np.arange(len(clip)), clip)
    levels: list[float] = []
    lock = threading.Lock()

    def on_frame(frame: np.ndarray) -> None:
        with lock:
            levels.append(float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))))

    spk = Speaker()
    mic = Microphone(on_frame, speaker=spk)
    time.sleep(0.6)  # the room before he talks, like the app's calibration
    with lock:
        calib = list(levels[:CALIBRATION_FRAMES])
        start_room = len(levels)
    spk.play(clip.astype(np.float32))
    print(f"\n1/2  Deixe a sala como sempre (música, criança cantando onde ela costuma ficar). {ROOM_S} s...")
    time.sleep(ROOM_S)
    with lock:
        start_talk = len(levels)
    print(f"\n2/2  AGORA: do lugar da criança no laptop, fale com o Pipo (ex.: 'Pipo, eu já sei!'). {TALK_S} s...")
    time.sleep(TALK_S)
    with lock:
        room = np.array(levels[start_room + 25:start_talk])  # after ~1 s, like the app's room level
        talk = np.array(levels[start_talk:])
    spk.stop()
    mic.close()
    spk.close()

    floor = float(np.mean(calib)) if calib else 0.0
    base = max(floor * 3.5, 0.012) * SPEAKING_FACTOR
    bar = max(base, float(np.median(room)) * ROOM_FACTOR)
    runs = np.convolve((room > bar).astype(int), np.ones(START_FRAMES, int), "valid")
    false_starts = int(np.sum((runs[1:] == START_FRAMES) & (runs[:-1] < START_FRAMES)))
    loud_talk = np.percentile(talk, 90)
    print("\n--- resultado (nível RMS no microfone, depois do cancelamento de eco) ---")
    print(f"sala parada antes dele falar: {floor:.4f}")
    print(f"sala enquanto ele fala:       mediana {np.median(room):.4f}  p90 {np.percentile(room, 90):.4f}  "
          f"máx {room.max():.4f}")
    print(f"alguém falando com ele:       mediana {np.median(talk):.4f}  p90 {loud_talk:.4f}")
    print(f"barra do app (x{ROOM_FACTOR} a sala): {bar:.4f}")
    print(f"vezes que a sala passaria a barra: {false_starts} (cada uma só vira checagem com o Whisper)")
    print(f"a fala passa a barra? {'sim' if loud_talk > bar else 'NÃO'} (p90 {loud_talk / bar:.1f}x a barra)")
    out = config.DATA_DIR / f"probe-{datetime.now():%Y%m%d-%H%M%S}.npz"
    np.savez(out, calib=np.array(calib), room=room, talk=talk)
    print(f"salvo em {out}")


if __name__ == "__main__":
    main()
