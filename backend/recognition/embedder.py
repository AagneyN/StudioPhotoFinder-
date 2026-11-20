"""Face embeddings using SFace (Apache-2.0) executed through ONNX Runtime.

Alignment uses OpenCV's reference `FaceRecognizerSF.alignCrop`, so preprocessing is identical for indexing
and searching. ONNX Runtime runs inference so a CUDA provider can be used when it is actually functional,
with a CPU fallback that is always reported (never silent).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

from backend.config.settings import Settings, get_settings
from backend.errors import ModelUnavailableError
from backend.recognition.detector import DetectedFace, FaceDetector
from backend.recognition.similarity import l2_normalize

log = logging.getLogger(__name__)
ort.set_default_logger_severity(3)   # silence harmless initializer warnings from this model


@dataclass
class ModelInfo:
    name: str
    embedding_dim: int
    execution_provider: str
    requested_provider: str
    fallback_reason: str | None
    fingerprint: str     # identifies the embedding space; an index is only valid for the same fingerprint

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _select_providers(requested: str) -> tuple[list[str], str | None]:
    available = ort.get_available_providers()
    if requested == "cpu":
        return ["CPUExecutionProvider"], None
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], None
    return ["CPUExecutionProvider"], (
        f"CUDAExecutionProvider not available in this onnxruntime build (available: {available}); using CPU.")


class FaceEmbedder:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        path = self.s.models_dir / self.s.embedder_model
        if not path.exists():
            raise ModelUnavailableError(
                f"Embedding model not found: {path}. Run `python scripts/download_dataset.py --models`.")
        providers, reason = _select_providers(self.s.execution_provider)
        probe = {"input": np.zeros((1, 3, 112, 112), np.float32)}
        try:
            self._sess = ort.InferenceSession(str(path), providers=providers)
            self._sess.run(None, {self._sess.get_inputs()[0].name: probe["input"]})  # surface provider errors now
        except Exception as e:
            if providers[0] == "CPUExecutionProvider":
                raise ModelUnavailableError(f"Could not load embedding model {path}: {e}") from e
            reason = f"CUDA provider failed to initialise ({str(e).splitlines()[0][:200]}); fell back to CPU."
            log.warning(reason)
            self._sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name
        used = self._sess.get_providers()[0]
        self._aligner = cv2.FaceRecognizerSF.create(str(path), "")   # used only for alignCrop
        dim = self._sess.get_outputs()[0].shape[-1]
        if isinstance(dim, int) and dim != self.s.embedding_dim:
            raise ModelUnavailableError(f"Model outputs {dim}-d embeddings but config says {self.s.embedding_dim}.")
        self.info = ModelInfo(self.s.model_name, self.s.embedding_dim, used, self.s.execution_provider, reason,
                              f"{self.s.model_name}:{self.s.embedding_dim}:l2norm:cosine")
        log.info("Embedder ready: %s on %s%s", self.info.name, used, f" ({reason})" if reason else "")

    def align(self, bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        return self._aligner.alignCrop(bgr, FaceDetector.raw_row(face))

    def embed_aligned(self, crops: list[np.ndarray]) -> np.ndarray:
        """Aligned 112x112 BGR crops -> (n, dim) L2-normalised float32."""
        if not crops:
            return np.zeros((0, self.s.embedding_dim), np.float32)
        # The published SFace ONNX graph has a fixed batch dimension of 1, so crops are run one by one.
        outs = []
        for crop in crops:
            blob = cv2.dnn.blobFromImage(crop, 1.0, (112, 112), (0, 0, 0), swapRB=True, crop=False)
            outs.append(self._sess.run(None, {self._input: blob.astype(np.float32)})[0].reshape(-1))
        return l2_normalize(np.vstack(outs))

    def embed_faces(self, bgr: np.ndarray, faces: list[DetectedFace]) -> np.ndarray:
        return self.embed_aligned([self.align(bgr, f) for f in faces])

    def embed_reference_cv2(self, crop: np.ndarray) -> np.ndarray:
        """OpenCV's own implementation; used by tests to verify ONNX Runtime parity."""
        return l2_normalize(self._aligner.feature(crop))
