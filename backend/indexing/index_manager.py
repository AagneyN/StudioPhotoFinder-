"""FAISS vector index: wrapper, atomic persistence, model-compatibility and DB-consistency checks.

The SQLite `faces.embedding` column is the source of truth; the FAISS file is a derived, rebuildable cache.
That makes crash recovery trivial: `reconcile()` adds/removes vectors until FAISS matches the database.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np

from backend.config.settings import Settings
from backend.database.repository import Repository
from backend.errors import IndexUnavailableError

log = logging.getLogger(__name__)


class FaceIndex:
    """Exact cosine search (inner product on L2-normalised vectors) keyed by face id."""

    def __init__(self, dim: int, fingerprint: str):
        self.dim = dim
        self.fingerprint = fingerprint
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    def ids(self) -> set[int]:
        return set(faiss.vector_to_array(self._index.id_map).tolist())

    def add(self, ids: Sequence[int], vectors: np.ndarray) -> None:
        v = np.ascontiguousarray(vectors, dtype=np.float32)
        if v.ndim != 2 or v.shape[1] != self.dim:
            raise ValueError(f"Embedding dimension {v.shape[-1]} does not match index dimension {self.dim}")
        if len(ids) != len(v):
            raise ValueError("ids and vectors length mismatch")
        if len(ids):
            self._index.add_with_ids(v, np.asarray(ids, dtype=np.int64))

    def remove(self, ids: Sequence[int]) -> int:
        if not len(ids):
            return 0
        return int(self._index.remove_ids(np.asarray(ids, dtype=np.int64)))

    def search(self, query: np.ndarray, k: int, allowed_ids: Sequence[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (ids, scores) sorted by descending cosine similarity for ONE query vector."""
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        q = q / max(float(np.linalg.norm(q)), 1e-12)     # guarantee cosine scores whatever the caller passes
        if q.shape[1] != self.dim:
            raise ValueError(f"Query dimension {q.shape[1]} does not match index dimension {self.dim}")
        if self.ntotal == 0 or k <= 0:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        if allowed_ids is not None:
            if not len(allowed_ids):
                return np.zeros(0, np.int64), np.zeros(0, np.float32)
            params = faiss.SearchParameters()
            params.sel = faiss.IDSelectorBatch(np.asarray(allowed_ids, dtype=np.int64))
            scores, ids = self._index.search(q, min(k, self.ntotal), params=params)
        else:
            scores, ids = self._index.search(q, min(k, self.ntotal))
        keep = ids[0] >= 0
        return ids[0][keep], scores[0][keep]

    # ---- persistence (atomic; serialise to bytes so unicode paths are never passed to FAISS) ----
    def save(self, path: Path, meta_path: Path, model_name: str = "") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(faiss.serialize_index(self._index).tobytes())
        os.replace(tmp, path)
        meta = {"fingerprint": self.fingerprint, "model_name": model_name, "dim": self.dim, "ntotal": self.ntotal,
                "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        mtmp = meta_path.with_suffix(".tmp")
        mtmp.write_text(json.dumps(meta, indent=2))
        os.replace(mtmp, meta_path)

    @classmethod
    def load(cls, path: Path, meta_path: Path, expected_fingerprint: str, expected_dim: int) -> "FaceIndex":
        if not path.exists():
            raise IndexUnavailableError(f"FAISS index not found at {path}. Build it with `python scripts/index_dataset.py`.")
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError) as e:
            raise IndexUnavailableError(f"Index metadata missing/corrupt ({meta_path}); rebuild the index.") from e
        if meta.get("fingerprint") != expected_fingerprint or meta.get("dim") != expected_dim:
            raise IndexUnavailableError(
                f"Index was built with model '{meta.get('fingerprint')}' (dim {meta.get('dim')}) but the application is "
                f"configured for '{expected_fingerprint}' (dim {expected_dim}). Embeddings are incompatible: rebuild the index.")
        try:
            raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            inner = faiss.deserialize_index(raw)
        except Exception as e:
            raise IndexUnavailableError(f"FAISS index file is corrupt ({path}): {e}. Rebuild the index.") from e
        if inner.d != expected_dim:
            raise IndexUnavailableError(f"Index dimension {inner.d} != expected {expected_dim}.")
        idx = cls(expected_dim, expected_fingerprint)
        idx._index = inner
        return idx


class IndexManager:
    """Ties the FAISS index to the database for one application/model configuration."""

    def __init__(self, settings: Settings, repo: Repository, fingerprint: str, model_name: str):
        self.s, self.repo, self.fingerprint, self.model_name = settings, repo, fingerprint, model_name

    def new_index(self) -> FaceIndex:
        return FaceIndex(self.s.embedding_dim, self.fingerprint)

    def save(self, index: FaceIndex) -> None:
        index.save(self.s.index_path, self.s.index_meta_path, self.model_name)

    def _check_db_model(self) -> None:
        stored = self.repo.get_meta("model_fingerprint")
        if stored is None:
            self.repo.set_meta("model_fingerprint", self.fingerprint)
        elif stored != self.fingerprint:
            raise IndexUnavailableError(
                f"Database holds embeddings from model '{stored}' but the application uses '{self.fingerprint}'. "
                "Use a fresh database/index (Rebuild) after changing the recognition model.")

    def load_strict(self) -> FaceIndex:
        """Load for SEARCH. Raises IndexUnavailableError on missing/incompatible/inconsistent state."""
        self._check_db_model()
        idx = FaceIndex.load(self.s.index_path, self.s.index_meta_path, self.fingerprint, self.s.embedding_dim)
        db_ids = self.repo.all_face_ids()
        if idx.ntotal != len(db_ids) or idx.ids() != db_ids:
            raise IndexUnavailableError(
                f"Index/database mismatch: FAISS has {idx.ntotal} vectors, database has {len(db_ids)} faces. "
                "Use 'Reload / repair index' (or `python scripts/index_dataset.py --repair`) to resynchronise.")
        return idx

    def load_or_create(self) -> FaceIndex:
        """Load for INDEXING: tolerant. Missing/corrupt/incompatible file -> rebuilt from DB embeddings."""
        self._check_db_model()
        try:
            idx = FaceIndex.load(self.s.index_path, self.s.index_meta_path, self.fingerprint, self.s.embedding_dim)
        except IndexUnavailableError as e:
            log.warning("Starting from a fresh FAISS index (%s)", e)
            idx = self.new_index()
        self.reconcile(idx)
        return idx

    def reconcile(self, idx: FaceIndex) -> dict[str, int]:
        db_ids = self.repo.all_face_ids()
        have = idx.ids()
        extra, missing = sorted(have - db_ids), sorted(db_ids - have)
        if extra:
            idx.remove(extra)
        for ids, vecs in self.repo.iter_embeddings(missing):
            idx.add(ids.tolist(), vecs)
        if extra or missing:
            log.warning("Index reconciled with database: removed %d stale, added %d missing vectors", len(extra), len(missing))
        return {"removed": len(extra), "added": len(missing)}
