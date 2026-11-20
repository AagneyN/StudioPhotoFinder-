from __future__ import annotations

import json

from fastapi import APIRouter

from backend.config.settings import get_settings

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.get("/report")
def report():
    """Last saved evaluation report, or an explicit 'not run' marker. Never fabricates numbers."""
    s = get_settings()
    try:
        return {"available": True, "report": json.loads(s.eval_report_path.read_text())}
    except (OSError, ValueError):
        return {"available": False, "message": "Evaluation has not been run. Run: python scripts/evaluate_model.py"}
