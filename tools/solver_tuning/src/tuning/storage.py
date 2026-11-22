"""SQLite is authoritative. Human-readable exports can always be regenerated."""

from __future__ import annotations
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import sqlite3
import threading
from .model import Evaluation
from .util import canonical


class Store:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.on_complete = (
            None  # optional callback(evaluation), e.g. the live CSV table
        )
        self.db = sqlite3.connect(
            self.directory / "trials.sqlite3", check_same_thread=False
        )
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS trials (id INTEGER PRIMARY KEY, phase TEXT NOT NULL, request TEXT NOT NULL, result TEXT, state TEXT NOT NULL)"
        )
        self.db.commit()

    def reserve(self, phase, request):
        with self.lock, self.db:
            row = self.db.execute(
                "INSERT INTO trials (phase,request,state) VALUES (?, ?, ?)",
                (phase, canonical(request), "running"),
            )
            return row.lastrowid

    def complete(self, evaluation):
        with self.lock, self.db:
            self.db.execute(
                "UPDATE trials SET state=?, result=? WHERE id=?",
                ("complete", canonical(evaluation.to_dict()), evaluation.trial_id),
            )
        if self.on_complete is not None:
            self.on_complete(evaluation)

    def records(self, phase=None):
        with self.lock:
            rows = self.db.execute(
                "SELECT id,phase,request,result,state FROM trials ORDER BY id"
            ).fetchall()
        return [
            {
                "id": i,
                "phase": p,
                "request": json.loads(q),
                "result": json.loads(r) if r else None,
                "state": s,
            }
            for i, p, q, r, s in rows
            if phase is None or p == phase
        ]

    def results(self, phase=None):
        return [
            Evaluation.from_dict(r["result"])
            for r in self.records(phase)
            if r["state"] == "complete"
        ]

    def recover(self):
        with self.lock, self.db:
            self.db.execute("UPDATE trials SET state='abandoned' WHERE state='running'")

    def cached(self, key):
        # Fresh deserialization prevents later metadata changes from mutating history.
        for record in reversed(self.records()):
            if (
                record["state"] == "complete"
                and record["request"].get("cache_key") == key
            ):
                ev = Evaluation.from_dict(record["result"])
                if ev.runs and all(not r.get("error") for r in ev.runs):
                    return ev
        return None

    def close(self):
        self.db.close()


@contextmanager
def exclusive_study(directory):
    """OS-owned local file lock; automatically released after a process crash.

    The lock file is intentionally retained: unlinking a locked path can create
    two independent locks. Use a local filesystem for SQLite and this lock.
    """
    path = Path(directory) / ".controller.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = open(path, "a+")
    acquired = False
    try:
        if os.name == "posix":
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("another controller owns this study") from exc
        elif os.name == "nt":
            import msvcrt

            stream.seek(0, 2)
            if stream.tell() == 0:
                stream.write(" ")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("another controller owns this study") from exc
        else:
            raise RuntimeError("study locking is supported on POSIX and Windows")
        acquired = True
        stream.seek(0)
        stream.truncate()
        json.dump({"pid": os.getpid(), "host": socket.gethostname()}, stream)
        stream.flush()
        yield
    finally:
        if acquired:
            if os.name == "posix":
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            else:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        stream.close()
