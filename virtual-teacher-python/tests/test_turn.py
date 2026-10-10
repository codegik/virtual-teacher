"""A turn, end to end against a real SQLite file. The point of this app is that the child hears the first
sentence while the model is still writing; if that quietly broke, Pipo would just feel slow again."""
import threading
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from pipo.turn import FALLBACK, run_turn
from pipo.values import GREETING

from .conftest import HEARD, KID_4, KID_9, QWEN_PIECES, FakeOutput

T0 = datetime(2026, 10, 3, 19, 0, tzinfo=timezone.utc)


def clock():
    t = [T0]

    def now():
        t[0] += timedelta(seconds=1)
        return t[0]

    return now


def turn(store, models, kid=KID_9, audio=None, greet=False, cancel=None, output=None):
    sid = store.start_session(kid.id, T0)
    output = output or FakeOutput()
    result = run_turn(store=store, models=models, kid=kid, session_id=sid, play=lambda c, t: output.play(c),
                      cancel=cancel or threading.Event(), now=clock(),
                      audio=np.zeros(16000, np.float32) if audio is None and not greet else audio, greet=greet)
    return sid, result, output


def test_a_turn_speaks_one_clip_per_sentence_in_order_and_the_texts_are_the_answer(store, models):
    models.replies.append(QWEN_PIECES)
    _, result, out = turn(store, models)
    spoken = [t for t, _ in models.spoken]
    assert spoken == [
        "Oi Theo! Você já notou que o céu muda de cores ao longo do dia?",
        "Às vezes ele pode ficar mais claro ou até mesmo rosa no por do sol.",
        "A gente ainda não descobriu tudo sobre por que ele é azul, mas tem uma teoria legal: os pequenos pedaços de ar nos fazem ver azul ao olhar para cima.",
        "E você acha que isso é legal ou estranho?",
    ]
    assert result.said == spoken
    assert len(out.clips) == 4


def test_the_first_sentence_is_spoken_while_the_model_is_still_writing(store, models):
    models.replies.append(QWEN_PIECES)
    models.piece_delay = 0.01  # the real model writes a piece every ~40 ms
    turn(store, models)
    assert models.pieces_sent[0] < len(QWEN_PIECES) // 3


def test_a_short_sentence_joins_the_next_and_a_sentence_split_across_pieces_is_spoken_whole(store, models):
    models.replies.append(["Oi", "! Vamos con", "tar até dez juntos", "? Um, dois."])
    turn(store, models)
    assert [t for t, _ in models.spoken] == ["Oi! Vamos contar até dez juntos?", "Um, dois."]


def test_a_long_opening_sentence_is_spoken_from_its_first_comma_after_25_characters(store, models):
    models.replies.append(["Sabe, ", "quando a gente olha para o céu bem cedinho, ", "ele fica laranja, ", "depois azul. Legal né?"])
    turn(store, models)
    assert [t for t, _ in models.spoken] == [
        "Sabe, quando a gente olha para o céu bem cedinho,",
        "ele fica laranja, depois azul.",
        "Legal né?",
    ]


def test_emojis_and_markdown_never_reach_the_voice_nor_the_transcript(store, models):
    models.replies.append(["**Muito bem!** 🎉🌟 Você ", "acertou a conta. 👏"])
    sid, _, _ = turn(store, models)
    assert [t for t, _ in models.spoken] == ["Muito bem!", "Você acertou a conta."]
    assert store.lines(sid)[-1] == ("assistente", "Muito bem! Você acertou a conta.")


def test_silence_answers_nothing_and_the_model_is_not_called(store, models):
    models.heard = "  "
    sid, result, out = turn(store, models)
    assert result.heard == "" and models.chats == [] and out.clips == []
    assert store.lines(sid) == []


def test_the_4_year_old_hears_speed_0_9_and_the_9_year_old_1_0(store, models):
    turn(store, models, kid=KID_4)
    turn(store, models, kid=KID_9)
    assert [s for _, s in models.spoken] == [0.9, 1.0]


def test_both_sides_of_the_turn_are_stored_in_order(store, models):
    models.replies.append(["Boa pergunta! ", "Quanto é 7 vezes 7?"])
    sid, _, _ = turn(store, models)
    assert store.lines(sid) == [("crianca", HEARD), ("assistente", "Boa pergunta! Quanto é 7 vezes 7?")]


def test_the_model_sees_the_instructions_then_the_chat_so_far_then_the_new_line(store, models):
    models.replies.append(["Oi, Theo! Tudo bem?"])
    sid = store.start_session(KID_9.id, T0)
    store.add_line(sid, "a_in", "crianca", "Oi Pipo", T0)
    store.add_line(sid, "a_out", "assistente", "Oi! Que bom te ouvir.", T0)
    run_turn(store=store, models=models, kid=KID_9, session_id=sid, play=lambda c, t: None,
             cancel=threading.Event(), now=clock(), audio=np.zeros(160, np.float32))
    msgs = models.chats[0]
    assert msgs[0]["role"] == "system" and "Você é o Pipo, amigo e professor virtual de Theo" in msgs[0]["content"]
    assert msgs[0]["content"].rstrip().endswith("conversando com você.")  # safety rules come last
    assert msgs[1:] == [
        {"role": "user", "content": "Oi Pipo"},
        {"role": "assistant", "content": "Oi! Que bom te ouvir."},
        {"role": "user", "content": HEARD},
    ]


def test_the_greeting_asks_the_model_to_say_hello_and_stores_only_pipos_line(store, models):
    models.replies.append(["Oi, Theo! Vamos conversar?"])
    sid, result, _ = turn(store, models, greet=True)
    assert models.transcribed == []
    assert models.chats[0][-1] == {"role": "user", "content": GREETING}
    assert store.lines(sid) == [("assistente", "Oi, Theo! Vamos conversar?")]


def test_an_answer_with_nothing_speakable_says_the_fallback(store, models):
    models.replies.append(["🎉 **"])
    _, result, _ = turn(store, models)
    assert result.said == [FALLBACK]


def test_talking_over_pipo_stops_the_model_and_stores_only_what_he_actually_said(store, models):
    models.replies.append(QWEN_PIECES)
    models.piece_delay = 0.005
    cancel = threading.Event()
    out = FakeOutput()
    out.on_play = lambda n: cancel.set()  # the child interrupts during the first sentence
    sid, result, _ = turn(store, models, cancel=cancel, output=out)
    assert result.cancelled
    assert models.stopped_early
    assert len(out.clips) == 1
    assert store.lines(sid)[-1] == ("assistente", "Oi Theo! Você já notou que o céu muda de cores ao longo do dia?")


def test_a_model_failure_is_raised_and_the_childs_line_is_still_stored(store, models):
    models.fail_reply = RuntimeError("CUDA out of memory")
    sid = store.start_session(KID_9.id, T0)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        run_turn(store=store, models=models, kid=KID_9, session_id=sid, play=lambda c, t: None,
                 cancel=threading.Event(), now=clock(), audio=np.zeros(160, np.float32))
    assert store.lines(sid) == [("crianca", HEARD)]
