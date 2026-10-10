"""The window. Python owns the state (screen, phase, mouth); QML (ui/*.qml) draws it and animates it on the GPU."""
import logging
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import Property, QObject, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from . import config
from .conversation import CANT_TALK, Conversation
from .store import Store
from .values import load_kids

log = logging.getLogger(__name__)
UI = Path(__file__).parent / "ui"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Bridge(QObject):
    """What QML sees. Signals carry changes from the turn and audio threads to the UI thread."""

    phaseChanged = Signal()
    mouthChanged = Signal()
    readyChanged = Signal()
    errorChanged = Signal()
    kidNameChanged = Signal()
    _phaseFromWorker = Signal(str)
    _loaded = Signal(object, str)

    def __init__(self, store: Store) -> None:
        super().__init__()
        self._store = store
        self._models = None
        self._speaker = None
        self._mic = None
        self._conv: Conversation | None = None
        self._phase = "starting"
        self._mouth = 0.0
        self._ready = False
        self._error = ""
        self._kid_name = ""
        self._phaseFromWorker.connect(self._apply_phase)
        self._loaded.connect(self._on_loaded)
        # A timer, so the mouth follows the voice ~30 times a second.
        self._mouth_timer = QTimer(self)
        self._mouth_timer.setInterval(33)
        self._mouth_timer.timeout.connect(self._tick_mouth)
        threading.Thread(target=self._load, name="pipo-load", daemon=True).start()

    # ---- properties --------------------------------------------------------
    @Property(str, notify=phaseChanged)
    def phase(self) -> str:
        return self._phase

    @Property(float, notify=mouthChanged)
    def mouth(self) -> float:
        return self._mouth

    @Property(bool, notify=readyChanged)
    def ready(self) -> bool:
        return self._ready

    @Property(str, notify=errorChanged)
    def error(self) -> str:
        return self._error

    @Property(str, notify=kidNameChanged)
    def kidName(self) -> str:
        return self._kid_name

    @Property("QVariantList", constant=True)  # type: ignore[arg-type]
    def kids(self) -> list:
        return [{"id": k.id, "name": k.name, "emoji": k.emoji} for k in load_kids()]

    # ---- actions from QML ----------------------------------------------------
    @Slot(str)
    def start(self, kid_id: str) -> None:
        assert self._models is not None and self._speaker is not None, "start is only offered once ready"
        kid = next(k for k in load_kids() if k.id == kid_id)
        self._kid_name = kid.name
        self.kidNameChanged.emit()
        self._apply_phase("starting")
        self._conv = Conversation(store=self._store, models=self._models, kid=kid, output=self._speaker,
                                  on_phase=self._phaseFromWorker.emit, now=utc_now)
        from .audio import Microphone

        conv = self._conv
        self._mic = Microphone(conv.feed, speaker=self._speaker)
        self._mouth_timer.start()

    @Slot()
    def stop(self) -> None:
        if self._mic:
            self._mic.close()
            self._mic = None
        if self._conv:
            self._conv.close()
            self._conv = None
        self._mouth_timer.stop()
        self._set_mouth(0.0)

    # ---- internals -----------------------------------------------------------
    def _load(self) -> None:
        try:
            from .audio import Speaker
            from .models import LocalModels

            self._loaded.emit((LocalModels(), Speaker()), "")
        except Exception as err:  # noqa: BLE001
            log.exception("could not load the models")
            self._loaded.emit(None, f"Não consegui acordar o Pipo: {err}")

    def _on_loaded(self, loaded, error: str) -> None:
        if loaded:
            self._models, self._speaker = loaded
            self._ready = True
            self.readyChanged.emit()
        else:
            self._error = error
            self.errorChanged.emit()

    def _apply_phase(self, phase: str) -> None:
        self._phase = phase
        if phase == "error":
            self._error = CANT_TALK
            self.errorChanged.emit()
        self.phaseChanged.emit()

    def _tick_mouth(self) -> None:
        level = min(1.0, self._speaker.level * 6) if self._speaker and self._speaker.playing else 0.0
        self._set_mouth(self._mouth * 0.55 + level * 0.45)

    def _set_mouth(self, value: float) -> None:
        if abs(value - self._mouth) > 0.005 or value == 0.0:
            self._mouth = value
            self.mouthChanged.emit()

    def shutdown(self) -> None:
        self.stop()
        if self._speaker:
            self._speaker.close()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = QGuiApplication(sys.argv)
    app.setApplicationName("Pipo")
    store = Store(config.DATA_DIR / "pipo.db")
    bridge = Bridge(store)
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("pipo", bridge)
    engine.load(QUrl.fromLocalFile(str(UI / "Main.qml")))
    if not engine.rootObjects():
        return 1
    app.aboutToQuit.connect(bridge.shutdown)
    code = app.exec()
    # The QML goes first: if Python freed the bridge while the bindings were alive, every one of them
    # would re-read `pipo` as null and print a TypeError on the way out.
    del engine
    return code
