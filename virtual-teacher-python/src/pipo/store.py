"""The conversation record, in one SQLite file inside this process. No audio is ever stored."""
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path

# Applied in order; PRAGMA user_version remembers how many ran.
MIGRATIONS = [
    """
    create table sessions (
      id text primary key,
      kid_id text not null,
      started_at text not null,
      ended_at text
    );
    create index sessions_kid_started on sessions (kid_id, started_at);
    create table transcript_lines (
      id integer primary key autoincrement,
      session_id text not null references sessions(id),
      item_id text not null,
      role text not null check (role in ('crianca', 'assistente')),
      text text not null,
      said_at text not null,
      unique (session_id, item_id)
    );
    """,
]


class Store:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # One connection shared by the UI and the turn threads; the lock keeps them from interleaving.
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("pragma journal_mode = wal")
            self._db.execute("pragma foreign_keys = on")
            done = self._db.execute("pragma user_version").fetchone()[0]
            for i, sql in enumerate(MIGRATIONS[done:], start=done + 1):
                self._db.executescript(f"begin; {sql}; pragma user_version = {i}; commit;")

    def schema_version(self) -> int:
        with self._lock:
            return self._db.execute("pragma user_version").fetchone()[0]

    def start_session(self, kid_id: str, now: datetime) -> str:
        sid = str(uuid.uuid4())
        with self._lock:
            self._db.execute("insert into sessions (id, kid_id, started_at) values (?, ?, ?)", (sid, kid_id, now.isoformat()))
        return sid

    def end_session(self, session_id: str, now: datetime) -> None:
        """Idempotent: the first end time stays."""
        with self._lock:
            self._db.execute("update sessions set ended_at = ? where id = ? and ended_at is null", (now.isoformat(), session_id))

    def session(self, session_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("select id, kid_id, started_at, ended_at from sessions where id = ?", (session_id,)).fetchone()
        return dict(zip(("id", "kid_id", "started_at", "ended_at"), row)) if row else None

    def add_line(self, session_id: str, item_id: str, role: str, text: str, now: datetime) -> bool:
        """Stores one transcript line once; a repeated item_id changes nothing. True when it was new."""
        with self._lock:
            cur = self._db.execute(
                "insert into transcript_lines (session_id, item_id, role, text, said_at) values (?, ?, ?, ?, ?)"
                " on conflict (session_id, item_id) do nothing",
                (session_id, item_id, role, text.strip(), now.isoformat()),
            )
            return cur.rowcount == 1

    def lines(self, session_id: str, last: int | None = None) -> list[tuple[str, str]]:
        """(role, text) in the order they were said; only the last `last` when given."""
        with self._lock:
            rows = self._db.execute(
                "select role, text from (select id, role, text from transcript_lines where session_id = ?"
                " order by id desc limit ?) order by id",
                (session_id, last if last is not None else -1),
            ).fetchall()
        return [(r, t) for r, t in rows]
