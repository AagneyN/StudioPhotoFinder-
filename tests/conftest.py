"""Shared fixtures. Face tests use a handful of images from the local LFW copy (not redistributed in the repo)
and skip, with a clear reason, when the dataset or models are missing. Everything else runs on synthetic data."""
from __future__ import annotations

import dataclasses
import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config.settings import get_settings  # noqa: E402


@pytest.fixture()
def settings(tmp_path):
    """Real model/dataset locations, but all writable state redirected to a temp dir."""
    s = get_settings()
    return dataclasses.replace(
        s, database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}", faiss_dir=tmp_path / "faiss",
        metadata_dir=tmp_path / "meta", processed_dir=tmp_path / "proc", thumbnails_dir=tmp_path / "thumbs",
        uploads_dir=tmp_path / "uploads", logs_dir=tmp_path / "logs", batch_size=4, checkpoint_every=4)


@pytest.fixture(scope="session")
def models_ok():
    s = get_settings()
    if not (s.models_dir / s.detector_model).exists() or not (s.models_dir / s.embedder_model).exists():
        pytest.skip("Models missing: run `python scripts/download_dataset.py --models`")


@pytest.fixture(scope="session")
def detector(models_ok):
    from backend.recognition.detector import FaceDetector
    return FaceDetector()


@pytest.fixture(scope="session")
def embedder(models_ok):
    from backend.recognition.embedder import FaceEmbedder
    return FaceEmbedder()


@pytest.fixture(scope="session")
def lfw_people():
    """6 identities x 4 images, chosen deterministically (alphabetical) from the local LFW copy."""
    root = get_settings().dataset_root
    if not root.exists():
        pytest.skip("LFW not downloaded: run `python scripts/download_dataset.py --dataset`")
    people = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = sorted(d.glob("*.jpg"))
        if len(imgs) >= 4:
            people[d.name] = imgs[:4]
        if len(people) == 6:
            break
    if len(people) < 6:
        pytest.skip("Not enough LFW identities available")
    return people


def jpeg_bytes(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr)
    assert ok
    return buf.tobytes()


@pytest.fixture()
def blank_jpeg():
    return jpeg_bytes(np.full((300, 300, 3), 127, np.uint8))
