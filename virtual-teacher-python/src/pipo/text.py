"""Turning the model's text into what Kokoro speaks, sentence by sentence while it is written."""
import re

# Emoji and pictographs, plus the joiners that glue them. Kokoro reads some of them aloud.
_EMOJI = re.compile(
    "[\U0001f000-\U0001faff\U00002600-\U000027bf\U00002b00-\U00002bff\U0001f1e6-\U0001f1ff\ufe0f\u200d\u2300-\u23ff]"
)
_MARKDOWN = re.compile(r"[*_#`>]+")
_PARENS = re.compile(r"\(([^)]*)\)")
_SPACES = re.compile(r"\s+")


def for_speech(text: str) -> str:
    """Text for speech: the model sometimes adds emojis or markdown even when told not to, and
    Kokoro would read "asterisco" aloud."""
    text = _EMOJI.sub("", text)
    text = _MARKDOWN.sub("", text)
    text = _PARENS.sub(r", \1,", text)
    return _SPACES.sub(" ", text).strip()


#: The first spoken piece may end at a comma once it has this many characters.
FIRST_PIECE_MIN = 25
#: Pieces shorter than this ("Oi!") are joined to the next, so the voice doesn't stop after every word.
SHORT_PIECE = 12
_END = re.compile(r"[.!?…]+[\"')\]*_]*\s|,\s")


class SentenceSplitter:
    """Cuts the model's text into sentences while it is being written, so each one can be spoken as soon
    as it is complete. A sentence ends at . ! ? or … (plus closing quotes, brackets or markdown) followed
    by a space; "7.5" is not an end.

    The first piece may also end at a comma once it is long enough: the child hears nothing until it is
    written and spoken, and the real model's first sentences ran to 150 characters. Later pieces are
    made while the earlier ones play, so they keep whole sentences."""

    def __init__(self) -> None:
        self._text = ""
        self._short = ""
        self._first = True

    def push(self, piece: str) -> list[str]:
        """Adds a piece of text; returns the sentences it completed."""
        self._text += piece
        out: list[str] = []
        start = 0
        for m in _END.finditer(self._text):
            if m.group(0).startswith(","):
                candidate = f"{self._short} {self._text[start:m.end()]}".strip()
                if not (self._first and len(candidate) >= FIRST_PIECE_MIN):
                    continue
            s = self._join(self._text[start:m.end()])
            start = m.end()
            if s:
                out.append(s)
                self._first = False
        self._text = self._text[start:]
        return out

    def flush(self) -> list[str]:
        """The text left when the answer ends."""
        rest = f"{self._short} {self._text}".strip()
        self._short = self._text = ""
        return [rest] if rest else []

    def _join(self, sentence: str) -> str | None:
        s = f"{self._short} {sentence}".strip()
        if len(s) < SHORT_PIECE:
            self._short = s
            return None
        self._short = ""
        return s
