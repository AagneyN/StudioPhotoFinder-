from __future__ import annotations

from fastapi import APIRouter

from backend.api.main import get_service, state
from backend.config.settings import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    """Always answers, even when the models failed to load, so the UI can explain what is wrong."""
    s = get_settings()
    if state.service is None:
        return {"status": "degraded", "error": state.startup_error}
    st = state.service.status()
    return {"status": "ok" if st["index_loaded"] else "degraded", "index_error": st["index_error"],
            "execution_provider": st["model"]["execution_provider"], "provider_note": st["model"]["fallback_reason"],
            "dataset_root_exists": s.dataset_root.exists()}


@router.get("/status")
def status():
    """Dashboard data: counts, model and execution provider. Contains no biometric data."""
    return get_service().status()
