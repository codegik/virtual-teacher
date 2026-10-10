"""The voice loop without a window: real detector, real turn thread, real SQLite; fake models and speaker.
Barge-in is the case that silently breaks: if the detector ignored the child while Pipo spoke, he would
talk over her forever."""
import threading
import time
from datetime import datetime, timezone

import numpy as np

from pipo.conversation import Conversation
from pipo.vad import CALIBRATION_FRAMES, FRAME, SpeechDetector

from .conftest import HEARD, KID_9, FakeOutput, read_wav

T0 = datetime(2026, 10, 3, 19, 0, tzinfo=timezone.utc)
QUIET = np.full(FRAME, 0.001, np.float32)


def loud(level=0.2):
    return (np.sin(np.arange(FRAME) / 3) * level).astype(np.float32)


class SlowOutput(FakeOutput):
    """Plays like a speaker: 'playing' until stop() or until `seconds` pass."""

    def __init__(self, seconds=5.0):
        super().__init__()
        self.seconds = seconds
        self._until = 0.0

    def play(self, clip):
        super().play(clip)
        self._until = time.time() + self.seconds

    def stop(self):
        super().stop()
        self._until = 0.0

    def wait(self, cancel):
        while time.time() < self._until and not cancel.is_set():
            time.sleep(0.01)

    @property
    def playing(self):
        return time.time() < self._until


def wait_for(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def conversation(store, models, output, phases):
    return Conversation(store=store, models=models, kid=KID_9, output=output, on_phase=phases.append,
                        now=lambda: T0, detector=SpeechDetector(end_silence_ms=200))


def calibrate(conv):
    for _ in range(CALIBRATION_FRAMES):
        conv.feed(QUIET)


def test_pipo_says_hello_first_then_waits_for_the_child(store, models):
    phases: list[str] = []
    models.replies.append(["Oi, Theo! Do que vamos brincar hoje?"])
    conv = conversation(store, models, FakeOutput(), phases)
    assert wait_for(lambda: conv.phase == "idle")
    conv.close()
    assert phases == ["thinking", "speaking", "idle"]
    assert store.lines(conv.session_id) == [("assistente", "Oi, Theo! Do que vamos brincar hoje?")]


def test_the_child_talks_and_pipo_answers_with_the_real_recording(store, models):
    phases: list[str] = []
    out = FakeOutput()
    conv = conversation(store, models, out, phases)
    assert wait_for(lambda: conv.phase == "idle")
    calibrate(conv)
    child = read_wav("child-7x8.wav")[::3]  # 24 kHz -> roughly 16 kHz, like the microphone gives
    for i in range(0, len(child) - FRAME, FRAME):
        conv.feed(child[i:i + FRAME])
    for _ in range(10):
        conv.feed(QUIET)
    assert wait_for(lambda: len(models.transcribed) == 1 and conv.phase == "idle")
    conv.close()
    assert "listening" in phases and phases[-1] == "idle"
    assert store.lines(conv.session_id)[-2:] == [("crianca", "Quanto é 7 vezes 8? Pensa comigo."), ("assistente", "Que legal! Me conta mais?")]


def test_a_cough_shorter_than_350_ms_is_not_a_turn(store, models):
    phases: list[str] = []
    conv = conversation(store, models, FakeOutput(), phases)
    assert wait_for(lambda: conv.phase == "idle")
    calibrate(conv)
    for _ in range(5):  # 200 ms
        conv.feed(loud())
    for _ in range(10):
        conv.feed(QUIET)
    time.sleep(0.1)
    conv.close()
    assert models.transcribed == []
    assert phases[-2:] == ["listening", "idle"]


def start_speaking(store, models, out, phases):
    models.replies.append(["Oi, Theo! Hoje vamos falar sobre planetas. ", "Sabia que Júpiter é enorme?"])
    models.piece_delay = 0.05
    conv = conversation(store, models, out, phases)
    assert wait_for(lambda: conv.phase == "speaking")
    calibrate(conv)
    return conv


def test_his_own_voice_leaking_under_the_threshold_is_not_even_checked(store, models):
    out = SlowOutput()
    conv = start_speaking(store, models, out, [])
    leak = conv.detector.threshold(pipo_speaking=False) * 1.5  # over the idle threshold, under the speaking one
    for _ in range(10):
        conv.feed(np.full(FRAME, leak, np.float32))
    assert conv.phase == "speaking" and out.stops == 0 and out.ducks == []
    conv.close()


def test_a_loud_echo_of_his_own_words_does_not_stop_him(store, models):
    # The bug: on laptop speakers his echo is louder than the threshold, and he cut himself off mid-sentence.
    out = SlowOutput()
    conv = start_speaking(store, models, out, [])
    models.hear_next.append("vamos falar sobre planetas")  # what Whisper makes of the echo
    for _ in range(24):  # loud sound until the check on its first moment says it was him
        conv.feed(loud(0.3))
        if wait_for(lambda: conv.echoes_ignored == 1, timeout=0.05):
            break
    for _ in range(10):
        conv.feed(QUIET)
    assert conv.echoes_ignored == 1
    assert out.stops == 0
    assert out.ducks == [True, False]  # softened while checking, then back to full voice
    assert conv.phase == "speaking"
    conv.close()


def test_the_child_talking_over_pipo_stops_him_and_her_sentence_is_the_next_turn(store, models):
    phases: list[str] = []
    out = SlowOutput()
    conv = start_speaking(store, models, out, phases)
    models.hear_next.append("Pipo, eu já sei")  # the confirmation
    models.hear_next.append("Pipo, eu já sei tudo sobre Júpiter!")  # her whole sentence, as the turn
    models.replies.append(["Que legal! Então me conta uma coisa sobre ele?"])
    models.piece_delay = 0
    for _ in range(25):
        conv.feed(loud(0.3))
    assert wait_for(lambda: out.stops >= 1)
    assert conv.phase == "listening"
    for _ in range(10):
        conv.feed(QUIET)
    assert wait_for(lambda: len(store.lines(conv.session_id)) == 3)
    conv.close()
    assert store.lines(conv.session_id) == [
        ("assistente", "Oi, Theo! Hoje vamos falar sobre planetas."),
        ("crianca", "Pipo, eu já sei tudo sobre Júpiter!"),
        ("assistente", "Que legal! Então me conta uma coisa sobre ele?"),
    ]


def test_a_short_word_from_the_child_while_pipo_talks_is_answered(store, models):
    out = SlowOutput()
    conv = start_speaking(store, models, out, [])
    models.hear_next.append("Para!")  # Whisper on the whole short utterance
    models.piece_delay = 0
    for _ in range(15):  # 600 ms of speech, then silence: over before the 800 ms check
        conv.feed(loud(0.3))
    for _ in range(10):
        conv.feed(QUIET)
    assert wait_for(lambda: out.stops >= 1)
    assert wait_for(lambda: any(r == "crianca" for r, _ in store.lines(conv.session_id)))
    conv.close()
    assert ("crianca", HEARD) in store.lines(conv.session_id)


def test_the_echo_tail_right_after_he_finishes_is_not_answered(store, models):
    models.replies.append(["Oi, Theo! Do que vamos brincar hoje?"])
    conv = conversation(store, models, FakeOutput(), [])
    assert wait_for(lambda: conv.phase == "idle")
    calibrate(conv)
    models.hear_next.append("brincar hoje")
    for _ in range(15):
        conv.feed(loud(0.3))
    for _ in range(10):
        conv.feed(QUIET)
    assert wait_for(lambda: conv.echoes_ignored == 1)
    conv.close()
    assert len(models.chats) == 1  # only the greeting; the echo was not answered
    assert store.lines(conv.session_id) == [("assistente", "Oi, Theo! Do que vamos brincar hoje?")]


def test_closing_ends_the_session_once(store, models):
    conv = conversation(store, models, FakeOutput(), [])
    assert wait_for(lambda: conv.phase == "idle")
    conv.close()
    first = store.session(conv.session_id)["ended_at"]
    store.end_session(conv.session_id, datetime(2027, 1, 1, tzinfo=timezone.utc))
    assert first == T0.isoformat()
    assert store.session(conv.session_id)["ended_at"] == first


def test_a_failed_turn_shows_the_error_then_lets_the_child_try_again(store, models):
    phases: list[str] = []
    models.fail_reply = RuntimeError("model crashed")
    conv = conversation(store, models, FakeOutput(), phases)
    assert wait_for(lambda: "error" in phases)
    assert wait_for(lambda: conv.phase == "idle", timeout=5)
    conv.close()
    assert phases == ["thinking", "error", "idle"]
