"""Application service: owns the (single, reused) detector/embedder/index and exposes detect + search.

Thread-safe: searches take a read lock-free snapshot of the index reference; reload swaps it under a lock.
The API, tests and any future front end go through this class, so the engine stays UI-agnostic.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from backend.config.settings import Settings, get_settings
from backend.database.connection import create_repository
from backend.database.repository import Repository
from backend.errors import IndexUnavailableError, NoFaceError
from backend.indexing.image_processor import decode_image
from backend.indexing.index_manager import FaceIndex, IndexManager
from backend.recognition.detector import DetectedFace, FaceDetector
from backend.recognition.embedder import FaceEmbedder

log = logging.getLogger(__name__)

DISCLAIMER = ("Results are ranked by facial similarity. A high score is a lead for a human to check, "
              "not a verified identity match.")


@dataclass
class QueryFaces:
    image_bgr: np.ndarray
    faces: list[DetectedFace]


class FinderService:
    def __init__(self, settings: Settings | None = None, repo: Repository | None = None,
                 detector: FaceDetector | None = None, embedder: FaceEmbedder | None = None):
        self.s = settings or get_settings()
        self.s.ensure_dirs()
        self.repo = repo or create_repository(self.s)
        self.detector = detector or FaceDetector(self.s)
        self.embedder = embedder or FaceEmbedder(self.s)
        self.manager = IndexManager(self.s, self.repo, self.embedder.info.fingerprint, self.embedder.info.name)
        self._lock = threading.Lock()
        self._index: FaceIndex | None = None
        self.index_error: str | None = None
        self.reload_index()

    # ---- index lifecycle -------------------------------------------------
    def reload_index(self, repair: bool = False) -> dict:
        """(Re)load the persisted index. With repair=True, resynchronise FAISS from the database first."""
        with self._lock:
            try:
                if repair:
                    idx = self.manager.load_or_create()
                    self.manager.save(idx)
                else:
                    idx = self.manager.load_strict()
                self._index, self.index_error = idx, None
                log.info("Index loaded: %d vectors", idx.ntotal)
            except IndexUnavailableError as e:
                self._index, self.index_error = None, str(e)
                log.warning("Index unavailable: %s", e)
        return self.status()

    def set_index(self, index: FaceIndex) -> None:
        with self._lock:
            self._index, self.index_error = index, None

    def status(self) -> dict:
        st = self.repo.stats()
        return {
            "index_loaded": self._index is not None,
            "index_vectors": self._index.ntotal if self._index else 0,
            "index_error": self.index_error,
            "model": self.embedder.info.to_dict(),
            "detector": self.s.detector_model,
            "stats": st.to_dict(),
            "threshold": self.s.effective_threshold(),
            "threshold_source": ("configured" if self.s.similarity_threshold is not None
                                 else "calibrated" if self.s.effective_threshold() is not None else "none"),
        }

    # ---- query ------------------------------------------------------------
    def detect_faces(self, data: bytes) -> QueryFaces:
        """Decode+validate an upload and detect faces. Raises InvalidImageError; returns [] faces if none."""
        img = decode_image(data, self.s)
        return QueryFaces(img, self.detector.detect(img))

    def search(self, data: bytes, *, face_index: int = 0, top_k: int | None = None, min_score: float | None = None,
               year: int | None = None, category: str | None = None) -> dict:
        index = self._index
        if index is None:
            raise IndexUnavailableError(self.index_error or "No index loaded. Build one with `python scripts/index_dataset.py`.")
        top_k = max(1, min(top_k or self.s.default_top_k, self.s.max_top_k))
        t0 = time.perf_counter()
        q = self.detect_faces(data)
        if not q.faces:
            raise NoFaceError("No usable face was detected in the uploaded photograph. "
                              "Try a clearer, front-facing photo (face at least ~40 px wide).")
        if not 0 <= face_index < len(q.faces):
            raise ValueError(f"face_index {face_index} out of range (the image has {len(q.faces)} faces)")
        face = q.faces[face_index]
        vec = self.embedder.embed_faces(q.image_bgr, [face])[0]
        t_embed = time.perf_counter()

        allowed = None
        if year is not None or category:
            allowed = self.repo.face_ids_matching(year, category)
        ids, scores = index.search(vec, top_k, allowed)
        t_search = time.perf_counter()
        meta = self.repo.get_faces_with_photos(ids.tolist())
        threshold = min_score if min_score is not None else self.s.effective_threshold()
        results = []
        for fid, sc in zip(ids.tolist(), scores.tolist()):
            if fid not in meta:      # index/db drift: skip but never crash
                log.warning("Face id %s present in index but not in database", fid)
                continue
            if min_score is not None and sc < min_score:
                continue
            f, p = meta[fid]
            results.append({
                "rank": len(results) + 1, "score": round(float(sc), 4), "face_id": fid, "photo_id": p.id,
                "identity": p.identity, "filename": p.filename, "source": p.source, "year": p.year,
                "category": p.category, "photo_number": p.photo_number, "location": p.abs_path,
                "face_bbox": list(f.bbox), "face_det_score": round(f.det_score, 3), "faces_in_photo": p.n_faces,
                "meets_threshold": (sc >= threshold) if threshold is not None else None,
            })
        return {
            "query": {"faces_detected": len(q.faces), "face_used": face_index,
                      "faces": [f.to_dict() for f in q.faces]},
            "results": results,
            "threshold": threshold,
            "threshold_note": ("Threshold is user-set." if min_score is not None else
                               "Calibrated on LFW validation pairs." if threshold is not None else
                               "No calibrated threshold: scores are rankings only. Run scripts/evaluate_model.py to calibrate."),
            "disclaimer": DISCLAIMER,
            "timing_ms": {"detect_embed": round((t_embed - t0) * 1000, 1),
                          "search": round((t_search - t_embed) * 1000, 2),
                          "total": round((time.perf_counter() - t0) * 1000, 1)},
        }
