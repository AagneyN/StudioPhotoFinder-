"""Face detection using OpenCV's YuNet (MIT licence)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from backend.config.settings import Settings, get_settings
from backend.errors import ModelUnavailableError

log = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    index: int                         # position in the sorted face list for this image
    bbox: tuple[int, int, int, int]    # x, y, w, h in ORIGINAL image pixels
    landmarks: np.ndarray              # (5, 2) original pixels: r-eye, l-eye, nose, r-mouth, l-mouth
    det_score: float
    quality: float                     # 0..1 heuristic: detection confidence x size x frontal-ness
    area: int = 0

    def to_dict(self) -> dict:
        return {"index": self.index, "bbox": list(self.bbox), "det_score": round(self.det_score, 4),
                "quality": round(self.quality, 4)}


class FaceDetector:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        path = self.s.models_dir / self.s.detector_model
        if not path.exists():
            raise ModelUnavailableError(
                f"Detector model not found: {path}. Run `python scripts/download_dataset.py --models`.")
        try:
            self._det = cv2.FaceDetectorYN.create(str(path), "", (320, 320), self.s.detection_score_threshold, 0.3, 5000)
        except cv2.error as e:  # pragma: no cover
            raise ModelUnavailableError(f"Could not load detector model {path}: {e}") from e

    def detect(self, bgr: np.ndarray, apply_quality_filter: bool = True) -> list[DetectedFace]:
        """Faces sorted by quality*area descending. Empty list when no acceptable face is found."""
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("detect() expects a BGR uint8 image of shape (H, W, 3)")
        h0, w0 = bgr.shape[:2]
        scale, img, side = 1.0, bgr, max(h0, w0)
        if side > self.s.max_detect_side:
            scale = self.s.max_detect_side / side
            img = cv2.resize(bgr, (int(round(w0 * scale)), int(round(h0 * scale))), interpolation=cv2.INTER_AREA)
        elif side < 160:   # tiny images: upsample so the detector has something to work with
            scale = 160 / side
            img = cv2.resize(bgr, (int(round(w0 * scale)), int(round(h0 * scale))), interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]
        self._det.setInputSize((w, h))
        _, raw = self._det.detect(img)
        if raw is None:
            return []
        faces: list[DetectedFace] = []
        for r in raw:
            x, y, fw, fh = (r[0:4] / scale)
            lm = (r[4:14].reshape(5, 2) / scale).astype(np.float32)
            score = float(r[14])
            quality = self._quality(score, fw, fh, lm)
            if apply_quality_filter and (min(fw, fh) < self.s.min_face_size_px or quality < self.s.min_quality):
                log.debug("rejected low-quality face (size=%.0fx%.0f q=%.2f)", fw, fh, quality)
                continue
            faces.append(DetectedFace(0, (max(int(round(x)), 0), max(int(round(y)), 0), int(round(fw)), int(round(fh))),
                                      lm, score, quality, int(fw * fh)))
        faces.sort(key=lambda f: f.quality * f.area, reverse=True)
        for i, f in enumerate(faces):
            f.index = i
        return faces

    @staticmethod
    def _quality(score: float, fw: float, fh: float, lm: np.ndarray) -> float:
        size = min(1.0, min(fw, fh) / 112.0)       # saturates at the recognition model's input size
        eye_r, eye_l, nose = lm[0], lm[1], lm[2]    # frontal-ness: nose roughly midway between the eyes
        span = abs(eye_l[0] - eye_r[0]) + 1e-6
        frontal = 1.0 - min(1.0, abs((nose[0] - (eye_r[0] + eye_l[0]) / 2) / span) * 1.5)
        return float(np.clip(score * (0.4 + 0.6 * size) * (0.5 + 0.5 * frontal), 0.0, 1.0))

    @staticmethod
    def raw_row(face: DetectedFace) -> np.ndarray:
        """15-float row in the layout expected by cv2.FaceRecognizerSF.alignCrop."""
        x, y, w, h = face.bbox
        return np.array([x, y, w, h, *face.landmarks.reshape(-1), face.det_score], dtype=np.float32)
