"""Pipo must not hear himself. Measured on this laptop, his echo was louder than the child, and he cut himself
off after a sentence. These run the real WebRTC echo canceller on the real recording of his voice coming
back through the laptop microphone (tests/fixtures/echo-*.npy, 16 kHz)."""
import numpy as np

from pipo.echo import EchoCanceller, condition, is_interruption, is_own_echo

from .conftest import FIXTURES


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def test_the_limiter_never_lets_the_mic_clip_and_keeps_quiet_speech_almost_unchanged():
    loud = np.array([2.3, -2.3, 1.0], np.float32)
    assert np.abs(condition(loud, 0.4)).max() < 1.0
    quiet = np.full(10, 0.05, np.float32)
    assert abs(condition(quiet, 1.0)[0] - 0.05) < 0.001


def test_the_echo_canceller_removes_most_of_his_voice_from_the_microphone():
    mic = condition(np.load(FIXTURES / "echo-mic.npy"), 0.4)
    played = np.load(FIXTURES / "echo-played.npy")
    n = len(mic) // 160 * 160
    clean = EchoCanceller().process(mic[:n], played[:n])
    talking = slice(16000, n)  # after the first second, when he is talking
    assert rms(clean[talking]) < rms(mic[talking]) / 4


def test_his_own_words_heard_back_are_an_echo_and_other_words_are_the_child():
    said = "Oi, Theo! Você já notou que o céu muda de cores ao longo do dia?"
    assert is_own_echo("o céu muda de cores", said)
    assert is_own_echo("ceu muda de cor", said)  # Whisper drops accents and endings on echoes
    assert is_own_echo("", said)  # nothing at all: noise
    assert not is_own_echo("Pipo, para, eu quero brincar de outra coisa", said)
    assert not is_own_echo("eu já sei", said)


def test_only_someone_talking_to_pipo_interrupts_him():
    said = "Oi, Theo! Hoje vamos falar sobre planetas. O Pipo adora Júpiter."
    assert is_interruption("Pipo, eu já sei", said)
    assert is_interruption("Para!", said)
    assert is_interruption("Espera, eu quero contar uma coisa", said)
    assert is_interruption("eu quero brincar de outra coisa", said)
    # Not him: nothing, Whisper's inventions on noise, his own words, one sung word, a stop word mid-lyric.
    assert not is_interruption("", said)
    assert not is_interruption("Obrigado.", said)
    assert not is_interruption("Legendas pela comunidade Amara.org", said)
    assert not is_interruption("falar sobre planetas", said)
    assert not is_interruption("cantando", said)
    assert not is_interruption("o Pipo adora", said)  # his own "Pipo" coming back is not a call
