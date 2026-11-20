"""Connection handling. SQLite now; a PostgreSQL implementation only needs to supply a Repository
(see repository.py) and a factory branch in `create_repository`."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from backend.config.settings import Settings, get_settings
from backend.errors import FinderError


@contextmanager
def sqlite_connection(path: Path) -> Iterator[sqlite3.Connection]:
    """Short-lived connection (safe across threads and across the API and indexer processes)."""
    try:
        conn = sqlite3.connect(str(path), timeout=30)
    except sqlite3.Error as e:
        raise FinderError(f"Database unavailable ({path}): {e}") from e
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")   # safe with WAL; avoids an fsync per commit
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_repository(settings: Settings | None = None):
    from backend.database.repository import SqliteRepository
    s = settings or get_settings()
    if s.database_url.startswith("sqlite:///"):
        s.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        repo = SqliteRepository(s.sqlite_path)
        repo.init_schema()
        return repo
    raise FinderError(f"Unsupported DATABASE_URL scheme: {s.database_url.split(':')[0]} (only sqlite is implemented)")
