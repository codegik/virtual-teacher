"""Paths and knobs. Every value can be overridden by an environment variable of the same name."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# The parent's files are shared with the Node app: one place to write the family's values.
VALUES_DIR = Path(_env("VALUES_DIR", str(ROOT.parent / "values")))
DATA_DIR = Path(_env("DATA_DIR", str(ROOT / "data")))
MODELS_DIR = Path(_env("MODELS_DIR", str(ROOT / "models")))

#: Must match QWEN_FILE in scripts/setup.sh (pinned by tests/test_guards.py).
QWEN_FILE = _env("QWEN_FILE", "qwen2.5-7b-instruct-q4_k_m.gguf")
#: The card has 6 GB. Measured with Whisper turbo loaded in the same process: 22 layers write 12 tokens/s,
#: 27 layers write 23 tokens/s (5.6 GB used), 28 run out of memory. Flash attention and the smaller context
#: below are what make 27 fit.
GPU_LAYERS = int(_env("LOCAL_GPU_LAYERS", "27"))
#: Tokens the model can see: the instructions are ~1.5k, 30 lines of chat ~1.2k, the answer 200.
CONTEXT = int(_env("LOCAL_CONTEXT", "4096"))
#: Threads for the model's CPU layers, and for Kokoro. When both used every core, one sentence took 3.4-3.9 s to speak.
CPU_THREADS = int(_env("LOCAL_CPU_THREADS", "6"))
KOKORO_THREADS = int(_env("KOKORO_THREADS", "4"))
KOKORO_VOICE = _env("KOKORO_VOICE", "pf_dora")
WHISPER_MODEL = _env("WHISPER_MODEL", "large-v3-turbo")
WHISPER_DEVICE = _env("WHISPER_DEVICE", "cuda")

#: Microphone gain before the soft limiter. This laptop's mic clipped at full scale (peaks of 2.3) with the
#: system input at 81%; clipped echo can't be cancelled. Scaled here so the system setting stays untouched.
MIC_GAIN = float(_env("MIC_GAIN", "0.4"))

#: Lines of the current chat the model sees.
HISTORY_LINES = 30
#: Most tokens in one answer. Spoken answers are short; this stops a runaway one.
MAX_TOKENS = 200
