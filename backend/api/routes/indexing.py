from __future__ import annotations

import json
import subprocess
import sys

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.api.main import get_service
from backend.config.settings import PROJECT_ROOT, get_settings
from backend.indexing.pipeline import IndexingLock

router = APIRouter(prefix="/index", tags=["indexing"])


class StartRequest(BaseModel):
    rebuild: bool = False
    split: str = "gallery"       # gallery | all
    limit: int | None = None


def _running_pid() -> int | None:
    s = get_settings()
    if not s.lock_path.exists():
        return None
    try:
        pid = int(s.lock_path.read_text().strip())
    except ValueError:
        return None
    return pid if IndexingLock._alive(pid) else None


@router.get("/status")
def index_status():
    s = get_settings()
    try:
        st = json.loads(s.status_path.read_text())
    except (OSError, ValueError):
        st = {"state": "never_run"}
    pid = _running_pid()
    if st.get("state") == "running" and pid is None:
        st["state"] = "interrupted"
        st["message"] = "The indexing process is no longer running (crashed or stopped). Start it again to resume."
    return {"job": st, "running": pid is not None, "dataset_root": str(s.dataset_root),
            "dataset_exists": s.dataset_root.exists()}


@router.post("/start")
def start_indexing(req: StartRequest):
    """Runs scripts/index_dataset.py as a separate process so the UI and searches stay responsive."""
    s = get_settings()
    if req.split not in ("gallery", "all"):
        raise HTTPException(400, "split must be 'gallery' or 'all'")
    if not s.dataset_root.exists():
        raise HTTPException(400, f"Dataset folder not found: {s.dataset_root}. Run scripts/download_dataset.py first.")
    if _running_pid() is not None:
        raise HTTPException(409, "An indexing job is already running.")
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "index_dataset.py"), "--split", req.split]
    if req.rebuild:
        cmd.append("--rebuild")
    if req.limit:
        cmd += ["--limit", str(int(req.limit))]
    s.logs_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(s.logs_dir / "indexing_job.log", "ab")
    subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=log_f, stderr=subprocess.STDOUT,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return {"started": True, "command": " ".join(cmd[1:])}


@router.post("/cancel")
def cancel_indexing():
    s = get_settings()
    if _running_pid() is None:
        raise HTTPException(409, "No indexing job is running.")
    (s.metadata_dir / "indexing.cancel").write_text("cancel")
    return {"cancelling": True}


@router.post("/reload")
def reload_index(repair: bool = False):
    """Reload the saved index from disk (after indexing), optionally repairing it from the database."""
    if _running_pid() is not None and repair:
        raise HTTPException(409, "Cannot repair while an indexing job is running.")
    return get_service().reload_index(repair=repair)
