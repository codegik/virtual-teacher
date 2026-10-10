"""The real models, loaded once into this process: Whisper (hearing), qwen2.5 via llama.cpp (thinking),
Kokoro (speaking)."""
import logging
import threading
import time
from collections.abc import Iterator
from typing import Any

import numpy as np

from . import config

log = logging.getLogger(__name__)
VOICE_RATE = 24000


class LocalModels:
    def __init__(self) -> None:
        t = time.time()
        self._whisper = _load_whisper()
        self._llm = _load_llm()
        self._kokoro = _load_kokoro()
        # Warm up, so the first answer isn't the slow one.
        self.transcribe(np.zeros(16000, dtype=np.float32))
        self.speak("Oi.", 1.0)
        log.info("models ready in %.1fs", time.time() - t)

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self._whisper.transcribe(
            audio,
            language="pt",
            beam_size=1,
            vad_filter=True,
            # Kids' speech: don't let Whisper invent text on silence or mumbling.
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    def reply(self, messages: list[dict], cancel: threading.Event) -> Iterator[str]:
        stream: Any = self._llm.create_chat_completion(
            messages=messages,  # type: ignore[arg-type]
            stream=True, max_tokens=config.MAX_TOKENS, temperature=0.6
        )
        try:
            for chunk in stream:
                if cancel.is_set():
                    return  # closing the generator stops llama.cpp; the GPU is free for the next turn
                piece = chunk["choices"][0]["delta"].get("content")
                if piece:
                    yield piece
        finally:
            stream.close()

    def speak(self, text: str, speed: float) -> np.ndarray:
        samples, rate = self._kokoro.create(
            text[:600], voice=config.KOKORO_VOICE, speed=max(0.6, min(speed, 1.4)), lang="pt-br"
        )
        samples = np.asarray(samples, dtype=np.float32)
        if rate != VOICE_RATE:
            n = int(len(samples) * VOICE_RATE / rate)
            samples = np.interp(np.linspace(0, len(samples) - 1, n), np.arange(len(samples)), samples).astype(np.float32)
        return samples


def _load_whisper():
    from faster_whisper import WhisperModel

    try:
        return WhisperModel(config.WHISPER_MODEL, device=config.WHISPER_DEVICE,
                            compute_type="int8_float16" if config.WHISPER_DEVICE == "cuda" else "int8")
    except Exception as err:  # no GPU or CUDA libs: the CPU is slower but the child can still talk
        log.warning("whisper on %s failed (%s); using cpu", config.WHISPER_DEVICE, err)
        return WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")


def _load_llm():
    from llama_cpp import Llama

    return Llama(
        model_path=str(config.MODELS_DIR / config.QWEN_FILE),
        n_gpu_layers=config.GPU_LAYERS,
        n_ctx=config.CONTEXT,
        flash_attn=True,
        n_threads=config.CPU_THREADS,
        verbose=False,
    )


def _load_kokoro():
    import onnxruntime as rt
    from kokoro_onnx import Kokoro

    opts = rt.SessionOptions()
    opts.intra_op_num_threads = config.KOKORO_THREADS
    session = rt.InferenceSession(str(config.MODELS_DIR / "kokoro-v1.0.onnx"), opts, providers=["CPUExecutionProvider"])
    return Kokoro.from_session(session, str(config.MODELS_DIR / "voices-v1.0.bin"))
