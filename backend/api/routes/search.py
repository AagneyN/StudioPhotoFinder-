from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.api.main import get_service
from backend.config.settings import get_settings
from backend.errors import InvalidImageError
from backend.indexing.image_processor import make_thumbnail

log = logging.getLogger(__name__)
router = APIRouter(tags=["search"])


async def _read_upload(file: UploadFile) -> bytes:
    s = get_settings()
    name = (file.filename or "").lower()
    if Path(name).suffix not in s.allowed_extensions:
        raise InvalidImageError("Only .jpg, .jpeg and .png files are accepted.")
    data = await file.read(s.max_upload_mb * 1024 * 1024 + 1)   # never buffer more than the limit
    if len(data) > s.max_upload_mb * 1024 * 1024:
        raise InvalidImageError(f"File is larger than the {s.max_upload_mb} MB limit.")
    _maybe_store(data)
    return data


def _maybe_store(data: bytes) -> None:
    """Query photos are processed in memory and NOT written to disk unless KEEP_UPLOADS=true.
    When kept, they get random names (no user-controlled path) and expire after UPLOAD_RETENTION_HOURS."""
    s = get_settings()
    if not s.keep_uploads:
        return
    s.uploads_dir.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - s.upload_retention_hours * 3600
    for old in s.uploads_dir.glob("*.bin"):
        if old.stat().st_mtime < cutoff:
            old.unlink(missing_ok=True)
    (s.uploads_dir / f"{uuid.uuid4().hex}.bin").write_bytes(data)


@router.post("/search/faces")
async def detect_faces(file: UploadFile = File(...)):
    """Step 1: detect faces in the query photo so the user can choose one when there are several."""
    data = await _read_upload(file)
    q = get_service().detect_faces(data)
    h, w = q.image_bgr.shape[:2]
    return {"width": w, "height": h, "faces": [f.to_dict() for f in q.faces]}


@router.post("/search")
async def search(file: UploadFile = File(...), face_index: int = Form(0), top_k: int = Form(10),
                 min_score: float | None = Form(None), year: int | None = Form(None),
                 category: str | None = Form(None)):
    """Step 2: similarity search with the chosen face. Results are rankings, not verified matches."""
    data = await _read_upload(file)
    return get_service().search(data, face_index=face_index, top_k=top_k, min_score=min_score, year=year,
                                category=category or None)


def _photo_path(photo_id: int) -> Path:
    """Resolve a photo by database id (never by a client-supplied path) and make sure it lives under the dataset root."""
    s = get_settings()
    row = get_service().repo.get_photo(photo_id)
    if row is None:
        raise HTTPException(404, "Photo not found")
    p = Path(row.abs_path).resolve()
    if not p.is_relative_to(s.dataset_root.resolve()):
        log.error("Refusing to serve %s: outside the dataset root", p)
        raise HTTPException(403, "File is outside the configured dataset location")
    if not p.exists():
        raise HTTPException(404, "Original file is no longer available (missing or the shared folder is offline)")
    return p


@router.get("/photos/{photo_id}/image")
def photo_image(photo_id: int):
    return FileResponse(_photo_path(photo_id))


@router.get("/photos/{photo_id}/thumbnail")
def photo_thumbnail(photo_id: int):
    s = get_settings()
    thumb = s.thumbnails_dir / f"{int(photo_id)}.jpg"
    if not thumb.exists():
        make_thumbnail(_photo_path(photo_id), thumb)
    return FileResponse(thumb, media_type="image/jpeg")


@router.get("/samples")
def sample_queries(n: int = 12):
    """Held-out LFW query photos (never in the gallery) that the UI offers as demo inputs."""
    import json
    s = get_settings()
    try:
        split = json.loads(s.eval_split_path.read_text())
    except (OSError, ValueError):
        return {"samples": [], "note": "No evaluation split yet. Run scripts/index_dataset.py or scripts/evaluate_model.py."}
    out = []
    for rel in split["queries"][:: max(1, len(split["queries"]) // max(n, 1))][:n]:
        out.append({"rel_path": rel, "identity": rel.split("/")[0]})
    return {"samples": out}


@router.get("/samples/image")
def sample_image(rel_path: str):
    """Serve a held-out query image. Only paths listed in the split file are allowed (prevents path traversal)."""
    import json
    s = get_settings()
    try:
        split = json.loads(s.eval_split_path.read_text())
    except (OSError, ValueError):
        raise HTTPException(404, "No evaluation split")
    allowed = set(split["queries"]) | set(split["unknown_queries"])
    if rel_path not in allowed:
        raise HTTPException(404, "Not a held-out sample")
    p = (s.dataset_root / rel_path).resolve()
    if not p.is_relative_to(s.dataset_root.resolve()) or not p.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(p)
