"""Repository abstraction + SQLite implementation.

All SQL lives in this file. Everything else depends on the abstract `Repository`, so adding a
PostgreSQL implementation (e.g. psycopg + the same schema) will not touch recognition/indexing/API code.
"""
from __future__ import annotations

import abc
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from backend.database.connection import sqlite_connection
from backend.database.models import FaceInsert, FaceRow, IndexStats, PhotoRecord, PhotoRow


class Repository(abc.ABC):
    @abc.abstractmethod
    def init_schema(self) -> None: ...
    @abc.abstractmethod
    def get_meta(self, key: str) -> str | None: ...
    @abc.abstractmethod
    def set_meta(self, key: str, value: str) -> None: ...
    @abc.abstractmethod
    def get_photo_by_key(self, source: str, rel_path: str) -> PhotoRow | None: ...
    @abc.abstractmethod
    def get_photo(self, photo_id: int) -> PhotoRow | None: ...
    @abc.abstractmethod
    def save_photo_result(self, rec: PhotoRecord, status: str, error: str | None, width: int, height: int,
                          faces: Sequence[FaceInsert]) -> tuple[int, list[int]]:
        """Insert/replace one photo and its faces atomically. Returns (photo_id, new_face_ids)."""
    @abc.abstractmethod
    def delete_photos(self, photo_ids: Sequence[int]) -> list[int]:
        """Delete photos (cascade to faces). Returns the removed face ids."""
    @abc.abstractmethod
    def list_photo_keys(self, source: str) -> dict[str, int]:
        """rel_path -> photo_id for all photos of a source."""
    @abc.abstractmethod
    def all_face_ids(self) -> set[int]: ...
    @abc.abstractmethod
    def face_ids_for_photo(self, photo_id: int) -> list[int]: ...
    @abc.abstractmethod
    def iter_embeddings(self, ids: Sequence[int] | None = None, chunk: int = 5000) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (face_ids int64[n], embeddings float32[n,d]) chunks."""
    @abc.abstractmethod
    def get_faces_with_photos(self, face_ids: Sequence[int]) -> dict[int, tuple[FaceRow, PhotoRow]]: ...
    @abc.abstractmethod
    def face_ids_matching(self, year: int | None, category: str | None) -> list[int]: ...
    @abc.abstractmethod
    def stats(self) -> IndexStats: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    rel_path TEXT NOT NULL,
    abs_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    identity TEXT,
    year INTEGER,
    category TEXT,
    photo_number TEXT,
    file_size INTEGER NOT NULL DEFAULT 0,
    mtime_ns INTEGER NOT NULL DEFAULT 0,
    width INTEGER NOT NULL DEFAULT 0,
    height INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    error TEXT,
    n_faces INTEGER NOT NULL DEFAULT 0,
    indexed_at TEXT,
    UNIQUE (source, rel_path)
);
CREATE INDEX IF NOT EXISTS idx_photos_identity ON photos(identity);
CREATE INDEX IF NOT EXISTS idx_photos_year_cat ON photos(year, category);
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    face_index INTEGER NOT NULL,
    x INTEGER, y INTEGER, w INTEGER, h INTEGER,
    det_score REAL, quality REAL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_faces_photo ON faces(photo_id);
"""

_PHOTO_COLS = ("id, source, rel_path, abs_path, filename, identity, year, category, photo_number, file_size, "
               "mtime_ns, width, height, status, error, n_faces, indexed_at")


def _photo(r: sqlite3.Row) -> PhotoRow:
    return PhotoRow(source=r["source"], rel_path=r["rel_path"], abs_path=r["abs_path"], filename=r["filename"],
                    identity=r["identity"], year=r["year"], category=r["category"], photo_number=r["photo_number"],
                    file_size=r["file_size"], mtime_ns=r["mtime_ns"], id=r["id"], status=r["status"], error=r["error"],
                    n_faces=r["n_faces"], width=r["width"], height=r["height"], indexed_at=r["indexed_at"])


def _chunks(seq: Sequence, n: int) -> Iterable[Sequence]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


class SqliteRepository(Repository):
    def __init__(self, path: Path):
        self.path = Path(path)

    def _conn(self):
        return sqlite_connection(self.path)

    def init_schema(self) -> None:
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def get_meta(self, key: str) -> str | None:
        with self._conn() as c:
            r = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r["value"] if r else None

    def set_meta(self, key: str, value: str) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                      (key, value))

    def get_photo_by_key(self, source: str, rel_path: str) -> PhotoRow | None:
        with self._conn() as c:
            r = c.execute(f"SELECT {_PHOTO_COLS} FROM photos WHERE source=? AND rel_path=?", (source, rel_path)).fetchone()
        return _photo(r) if r else None

    def get_photo(self, photo_id: int) -> PhotoRow | None:
        with self._conn() as c:
            r = c.execute(f"SELECT {_PHOTO_COLS} FROM photos WHERE id=?", (photo_id,)).fetchone()
        return _photo(r) if r else None

    def save_photo_result(self, rec, status, error, width, height, faces):
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn() as c:
            old = c.execute("SELECT id FROM photos WHERE source=? AND rel_path=?", (rec.source, rec.rel_path)).fetchone()
            if old:   # re-index of a changed file: replace faces, keep photo id stable
                c.execute("DELETE FROM faces WHERE photo_id=?", (old["id"],))
                c.execute(
                    "UPDATE photos SET abs_path=?, filename=?, identity=?, year=?, category=?, photo_number=?, file_size=?, "
                    "mtime_ns=?, width=?, height=?, status=?, error=?, n_faces=?, indexed_at=? WHERE id=?",
                    (rec.abs_path, rec.filename, rec.identity, rec.year, rec.category, rec.photo_number, rec.file_size,
                     rec.mtime_ns, width, height, status, error, len(faces), now, old["id"]))
                pid = old["id"]
            else:
                cur = c.execute(
                    "INSERT INTO photos(source, rel_path, abs_path, filename, identity, year, category, photo_number, "
                    "file_size, mtime_ns, width, height, status, error, n_faces, indexed_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (rec.source, rec.rel_path, rec.abs_path, rec.filename, rec.identity, rec.year, rec.category,
                     rec.photo_number, rec.file_size, rec.mtime_ns, width, height, status, error, len(faces), now))
                pid = cur.lastrowid
            ids: list[int] = []
            for f in faces:
                cur = c.execute(
                    "INSERT INTO faces(photo_id, face_index, x, y, w, h, det_score, quality, embedding) VALUES (?,?,?,?,?,?,?,?,?)",
                    (pid, f.face_index, *f.bbox, f.det_score, f.quality, f.embedding))
                ids.append(cur.lastrowid)
        return pid, ids

    def delete_photos(self, photo_ids):
        removed: list[int] = []
        with self._conn() as c:
            for chunk in _chunks(list(photo_ids), 500):
                q = ",".join("?" * len(chunk))
                removed += [r["id"] for r in c.execute(f"SELECT id FROM faces WHERE photo_id IN ({q})", chunk)]
                c.execute(f"DELETE FROM photos WHERE id IN ({q})", chunk)   # faces cascade
        return removed

    def list_photo_keys(self, source):
        with self._conn() as c:
            return {r["rel_path"]: r["id"] for r in c.execute("SELECT id, rel_path FROM photos WHERE source=?", (source,))}

    def all_face_ids(self):
        with self._conn() as c:
            return {r["id"] for r in c.execute("SELECT id FROM faces")}

    def face_ids_for_photo(self, photo_id):
        with self._conn() as c:
            return [r["id"] for r in c.execute("SELECT id FROM faces WHERE photo_id=?", (photo_id,))]

    def iter_embeddings(self, ids=None, chunk=5000):
        with self._conn() as c:
            if ids is None:
                cur = c.execute("SELECT id, embedding FROM faces ORDER BY id")
                while True:
                    rows = cur.fetchmany(chunk)
                    if not rows:
                        break
                    yield (np.array([r["id"] for r in rows], dtype=np.int64),
                           np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]))
            else:
                for part in _chunks(list(ids), 500):
                    q = ",".join("?" * len(part))
                    rows = c.execute(f"SELECT id, embedding FROM faces WHERE id IN ({q})", list(part)).fetchall()
                    if rows:
                        yield (np.array([r["id"] for r in rows], dtype=np.int64),
                               np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]))

    def get_faces_with_photos(self, face_ids):
        out: dict[int, tuple[FaceRow, PhotoRow]] = {}
        with self._conn() as c:
            for part in _chunks(list(face_ids), 500):
                q = ",".join("?" * len(part))
                rows = c.execute(
                    f"SELECT f.id AS fid, f.photo_id, f.face_index, f.x, f.y, f.w, f.h, f.det_score, f.quality, "
                    f"p.id, p.source, p.rel_path, p.abs_path, p.filename, p.identity, p.year, p.category, p.photo_number, "
                    f"p.file_size, p.mtime_ns, p.width, p.height, p.status, p.error, p.n_faces, p.indexed_at "
                    f"FROM faces f JOIN photos p ON p.id=f.photo_id WHERE f.id IN ({q})", list(part)).fetchall()
                for r in rows:
                    out[r["fid"]] = (FaceRow(r["fid"], r["photo_id"], r["face_index"], (r["x"], r["y"], r["w"], r["h"]),
                                             r["det_score"], r["quality"]), _photo(r))
        return out

    def face_ids_matching(self, year, category):
        sql, args = "SELECT f.id FROM faces f JOIN photos p ON p.id=f.photo_id WHERE 1=1", []
        if year is not None:
            sql += " AND p.year=?"
            args.append(year)
        if category:
            sql += " AND lower(p.category)=lower(?)"
            args.append(category)
        with self._conn() as c:
            return [r["id"] for r in c.execute(sql, args)]

    def stats(self) -> IndexStats:
        with self._conn() as c:
            g = lambda sql: c.execute(sql).fetchone()[0]  # noqa: E731
            return IndexStats(
                photos_total=g("SELECT COUNT(*) FROM photos"),
                photos_indexed=g("SELECT COUNT(*) FROM photos WHERE status='indexed'"),
                photos_no_face=g("SELECT COUNT(*) FROM photos WHERE status='no_face'"),
                photos_failed=g("SELECT COUNT(*) FROM photos WHERE status='failed'"),
                photos_multi_face=g("SELECT COUNT(*) FROM photos WHERE n_faces>1"),
                faces=g("SELECT COUNT(*) FROM faces"),
                identities=g("SELECT COUNT(DISTINCT identity) FROM photos WHERE status='indexed' AND identity IS NOT NULL"),
            )
