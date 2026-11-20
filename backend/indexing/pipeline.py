"""Resumable, batched indexing pipeline. Independent of the API and the UI (runs from scripts/index_dataset.py)."""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Callable, Iterable

import numpy as np

from backend.config.settings import Settings
from backend.database.models import FaceInsert, PhotoRecord
from backend.database.repository import Repository
from backend.errors import FinderError, InvalidImageError
from backend.indexing.image_processor import load_image
from backend.indexing.index_manager import IndexManager
from backend.recognition.detector import FaceDetector
from backend.recognition.embedder import FaceEmbedder

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class IndexingLock:
    """Single-writer lock across processes (pid file; a dead owner's lock is taken over)."""

    def __init__(self, path):
        self.path = path

    @staticmethod
    def _alive(pid: int) -> bool:
        if pid == os.getpid():
            return True
        try:
            if os.name == "nt":
                import ctypes
                h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
                if not h:
                    return False
                ctypes.windll.kernel32.CloseHandle(h)
                return True
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                pid = int(self.path.read_text().strip())
            except ValueError:
                pid = -1
            if pid > 0 and self._alive(pid):
                raise FinderError(f"Another indexing job is already running (pid {pid}).")
            log.warning("Removing stale indexing lock (pid %s)", pid)
        self.path.write_text(str(os.getpid()))
        return self

    def __exit__(self, *exc):
        try:
            self.path.unlink()
        except OSError:
            pass


class IndexingPipeline:
    def __init__(self, settings: Settings, repo: Repository, detector: FaceDetector, embedder: FaceEmbedder):
        self.s, self.repo, self.det, self.emb = settings, repo, detector, embedder
        self.mgr = IndexManager(settings, repo, embedder.info.fingerprint, embedder.info.name)
        self.cancel_path = settings.metadata_dir / "indexing.cancel"

    # -- status file read by the UI --
    def _write_status(self, st: dict) -> None:
        st["updated_at"] = _now()
        tmp = self.s.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=2))
        os.replace(tmp, self.s.status_path)

    def run(self, records: Iterable[PhotoRecord], *, prune: bool = False, retry_failed: bool = False,
            limit: int | None = None, progress: Callable[[dict], None] | None = None) -> dict:
        self.s.ensure_dirs()
        with IndexingLock(self.s.lock_path):
            self.cancel_path.unlink(missing_ok=True)
            return self._run(list(records), prune, retry_failed, limit, progress)

    def _run(self, records, prune, retry_failed, limit, progress):
        t0 = time.perf_counter()
        st = {"state": "running", "started_at": _now(), "pid": os.getpid(), "total": len(records), "processed": 0,
              "skipped_unchanged": 0, "photos_indexed": 0, "faces_added": 0, "no_face": 0, "failed": 0,
              "multi_face": 0, "pruned": 0, "images_per_sec": 0.0, "eta_sec": None, "recent_errors": [],
              "provider": self.emb.info.execution_provider, "model": self.emb.info.name}
        self._write_status(st)
        index = self.mgr.load_or_create()
        wanted = {r.rel_path for r in records}
        source = records[0].source if records else None

        if prune and source:  # removed from disk / no longer part of the selection
            gone = [pid for rel, pid in self.repo.list_photo_keys(source).items() if rel not in wanted]
            if gone:
                index.remove(self.repo.delete_photos(gone))
                st["pruned"] = len(gone)
                log.info("Pruned %d photos that are no longer present/selected", len(gone))

        # decide what needs work (incremental + resumable: unchanged files already in the DB are skipped)
        todo: list[PhotoRecord] = []
        for r in records:
            old = self.repo.get_photo_by_key(r.source, r.rel_path)
            if old and old.file_size == r.file_size and old.mtime_ns == r.mtime_ns and (old.status != "failed" or not retry_failed):
                st["skipped_unchanged"] += 1
                continue
            todo.append(r)
        if limit is not None:
            todo = todo[:limit]
        st["to_process"] = len(todo)
        log.info("%d images to process (%d already indexed and unchanged)", len(todo), st["skipped_unchanged"])

        since_ckpt = 0
        bs = self.s.batch_size
        try:
            for start in range(0, len(todo), bs):
                if self.cancel_path.exists():
                    st["state"] = "interrupted"
                    st["message"] = "Cancelled by user; progress saved. Run again to resume."
                    break
                batch = todo[start:start + bs]
                self._process_batch(batch, index, st)
                st["processed"] += len(batch)
                since_ckpt += len(batch)
                el = max(time.perf_counter() - t0, 1e-6)
                st["images_per_sec"] = round(st["processed"] / el, 2)
                st["eta_sec"] = round((len(todo) - st["processed"]) / max(st["images_per_sec"], 1e-6))
                if since_ckpt >= self.s.checkpoint_every:
                    self.mgr.save(index)
                    since_ckpt = 0
                self._write_status(st)
                if progress:
                    progress(st)
            else:
                st["state"] = "completed"
        except KeyboardInterrupt:
            st["state"], st["message"] = "interrupted", "Interrupted; progress saved. Run again to resume."
        except MemoryError:
            st["state"], st["message"] = "failed", "Ran out of memory; lower BATCH_SIZE and run again to resume."
            log.error(st["message"])
        except Exception as e:
            st["state"], st["message"] = "failed", f"{type(e).__name__}: {e}"
            log.exception("Indexing failed")
        finally:
            self.mgr.save(index)       # always leave a consistent checkpoint on disk
            st["elapsed_sec"] = round(time.perf_counter() - t0, 1)
            st["index_vectors"] = index.ntotal
            self._write_status(st)
        if st["state"] == "completed":
            self.repo.set_meta("last_completed_index", _now())
        return st

    def _process_batch(self, batch: list[PhotoRecord], index, st: dict) -> None:
        loaded: list[tuple[PhotoRecord, np.ndarray, list]] = []
        results: list[tuple[PhotoRecord, str, str | None, int, int, list]] = []
        for rec in batch:
            try:
                img = load_image(rec.abs_path, self.s)
            except InvalidImageError as e:
                log.warning("Corrupted/invalid image %s: %s", rec.rel_path, e)
                results.append((rec, "failed", str(e), 0, 0, []))
                continue
            try:
                faces = self.det.detect(img)
            except Exception as e:   # detector failure on one image must not abort the run
                log.warning("Detection failed on %s: %s", rec.rel_path, e)
                results.append((rec, "failed", f"detection error: {e}", img.shape[1], img.shape[0], []))
                continue
            loaded.append((rec, img, faces))

        # embed every face of the batch in a single model call
        crops, owners = [], []
        for i, (rec, img, faces) in enumerate(loaded):
            for f in faces:
                crops.append(self.emb.align(img, f))
                owners.append((i, f))
        vecs = self.emb.embed_aligned(crops)
        per_img: dict[int, list[FaceInsert]] = {}
        for (i, f), v in zip(owners, vecs):
            per_img.setdefault(i, []).append(FaceInsert(f.index, f.bbox, f.det_score, f.quality, v.astype(np.float32).tobytes()))
        for i, (rec, img, faces) in enumerate(loaded):
            ins = per_img.get(i, [])
            results.append((rec, "indexed" if ins else "no_face", None if ins else "no face detected",
                            img.shape[1], img.shape[0], ins))

        for rec, status, err, w, h, ins in results:
            # replacing a changed file: drop its old vectors first
            old = self.repo.get_photo_by_key(rec.source, rec.rel_path)
            if old:
                index.remove(self.repo.face_ids_for_photo(old.id))
            _, face_ids = self.repo.save_photo_result(rec, status, err, w, h, ins)
            if face_ids:
                index.add(face_ids, np.vstack([np.frombuffer(f.embedding, dtype=np.float32) for f in ins]))
            if status == "indexed":
                st["photos_indexed"] += 1
                st["faces_added"] += len(face_ids)
                if len(face_ids) > 1:
                    st["multi_face"] += 1
            elif status == "no_face":
                st["no_face"] += 1
            else:
                st["failed"] += 1
                st["recent_errors"] = (st["recent_errors"] + [f"{rec.rel_path}: {err}"])[-20:]
