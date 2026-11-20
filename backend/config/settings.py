"""Central configuration. Every tunable lives here and can be overridden via environment / .env."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _path(name: str, default: str) -> Path:
    p = Path(os.getenv(name, default))
    return p if p.is_absolute() else PROJECT_ROOT / p


def _opt_float(name: str) -> float | None:
    v = os.getenv(name, "").strip()
    return float(v) if v else None


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    # --- storage -------------------------------------------------------
    dataset_root: Path = field(default_factory=lambda: _path("DATASET_ROOT", "data/datasets/lfw"))
    dataset_name: str = field(default_factory=lambda: os.getenv("DATASET_NAME", "lfw"))
    database_url: str = field(default_factory=lambda: os.getenv(
        "DATABASE_URL", f"sqlite:///{_path('SQLITE_PATH', 'indexes/metadata/photos.db').as_posix()}"))
    faiss_dir: Path = field(default_factory=lambda: _path("FAISS_DIR", "indexes/faiss"))
    metadata_dir: Path = field(default_factory=lambda: _path("METADATA_DIR", "indexes/metadata"))
    processed_dir: Path = field(default_factory=lambda: _path("PROCESSED_DIR", "data/processed"))
    thumbnails_dir: Path = field(default_factory=lambda: _path("THUMBNAILS_DIR", "data/thumbnails"))
    uploads_dir: Path = field(default_factory=lambda: _path("UPLOADS_DIR", "data/uploads"))
    models_dir: Path = field(default_factory=lambda: _path("MODELS_DIR", "models"))
    logs_dir: Path = field(default_factory=lambda: _path("LOGS_DIR", "logs"))

    # --- recognition model ----------------------------------------------
    detector_model: str = "face_detection_yunet_2023mar.onnx"
    embedder_model: str = "face_recognition_sface_2021dec.onnx"
    model_name: str = "SFace-2021dec"          # recorded in index metadata
    embedding_dim: int = 128
    # "auto" tries CUDA then CPU. Use "cpu" to force CPU.
    execution_provider: str = field(default_factory=lambda: os.getenv("EXECUTION_PROVIDER", "auto").lower())

    # --- detection ------------------------------------------------------
    detection_score_threshold: float = field(default_factory=lambda: _float("DETECTION_SCORE_THRESHOLD", 0.6))
    min_face_size_px: int = field(default_factory=lambda: _int("MIN_FACE_SIZE_PX", 40))
    min_quality: float = field(default_factory=lambda: _float("MIN_FACE_QUALITY", 0.3))
    max_detect_side: int = field(default_factory=lambda: _int("MAX_DETECT_SIDE", 1280))

    # --- search ---------------------------------------------------------
    default_top_k: int = 10
    max_top_k: int = 100
    # None = no calibrated threshold. Scores are model dependent; calibrate with scripts/evaluate_model.py.
    similarity_threshold: float | None = field(default_factory=lambda: _opt_float("SIMILARITY_THRESHOLD"))

    # --- indexing -------------------------------------------------------
    batch_size: int = field(default_factory=lambda: _int("BATCH_SIZE", 64))
    checkpoint_every: int = field(default_factory=lambda: _int("CHECKPOINT_EVERY", 500))

    # --- uploads / security --------------------------------------------
    max_upload_mb: int = field(default_factory=lambda: _int("MAX_UPLOAD_MB", 10))
    max_image_pixels: int = 50_000_000
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")
    keep_uploads: bool = field(default_factory=lambda: os.getenv("KEEP_UPLOADS", "false").lower() == "true")
    upload_retention_hours: int = field(default_factory=lambda: _int("UPLOAD_RETENTION_HOURS", 24))

    # --- server ---------------------------------------------------------
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: _int("API_PORT", 8000))
    streamlit_port: int = field(default_factory=lambda: _int("STREAMLIT_PORT", 8501))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @property
    def sqlite_path(self) -> Path:
        if not self.database_url.startswith("sqlite:///"):
            raise ValueError("Only sqlite:/// URLs are implemented; see backend/database/repository.py for the PostgreSQL seam.")
        return Path(self.database_url[len("sqlite:///"):])

    @property
    def index_path(self) -> Path:
        return self.faiss_dir / "faces.index"

    @property
    def index_meta_path(self) -> Path:
        return self.faiss_dir / "faces.index.json"

    @property
    def status_path(self) -> Path:
        return self.metadata_dir / "indexing_status.json"

    @property
    def lock_path(self) -> Path:
        return self.metadata_dir / "indexing.lock"

    @property
    def eval_report_path(self) -> Path:
        return self.processed_dir / "evaluation_report.json"

    @property
    def calibration_path(self) -> Path:
        return self.processed_dir / "calibration.json"

    @property
    def eval_split_path(self) -> Path:
        return self.processed_dir / "eval_split.json"

    def ensure_dirs(self) -> None:
        for d in (self.faiss_dir, self.metadata_dir, self.processed_dir, self.thumbnails_dir,
                  self.uploads_dir, self.models_dir, self.logs_dir, self.sqlite_path.parent):
            d.mkdir(parents=True, exist_ok=True)

    def effective_threshold(self) -> float | None:
        """Explicit setting wins; otherwise the calibrated value from the last evaluation run, if any."""
        if self.similarity_threshold is not None:
            return self.similarity_threshold
        try:
            return float(json.loads(self.calibration_path.read_text())["threshold"])
        except (OSError, KeyError, ValueError):
            return None


def get_settings() -> Settings:
    return Settings()


def setup_logging(settings: Settings | None = None, name: str = "app") -> None:
    s = settings or get_settings()
    s.logs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, s.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(s.logs_dir / f"{name}.log", encoding="utf-8")],
        force=True,
    )
