import io

import cv2
import numpy as np
import pytest
from PIL import Image

from backend.errors import InvalidImageError
from backend.indexing.image_processor import decode_image, load_image
from tests.conftest import jpeg_bytes


# ---- image loading / validation (no models needed) --------------------------------
def test_load_valid_image(lfw_people):
    img = load_image(next(iter(lfw_people.values()))[0])
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8


def test_invalid_images_rejected(settings, tmp_path):
    with pytest.raises(InvalidImageError, match="empty"):
        decode_image(b"", settings)
    with pytest.raises(InvalidImageError):
        decode_image(b"this is not an image", settings)
    buf = io.BytesIO()
    Image.new("RGB", (20, 20)).save(buf, "GIF")
    with pytest.raises(InvalidImageError, match="Unsupported"):
        decode_image(buf.getvalue(), settings)
    with pytest.raises(InvalidImageError):          # JPEG header followed by garbage (corrupted file)
        decode_image(b"\xff\xd8\xff" + b"junk" * 20, settings)
    good = jpeg_bytes(np.full((50, 50, 3), 90, np.uint8))
    with pytest.raises(InvalidImageError):          # truncated real JPEG
        decode_image(good[: len(good) // 2], settings)


def test_extension_must_be_allowed_and_file_must_exist(tmp_path):
    p = tmp_path / "x.exe"
    p.write_bytes(b"MZ")
    with pytest.raises(InvalidImageError, match="extension"):
        load_image(p)
    with pytest.raises(InvalidImageError, match="Cannot read"):
        load_image(tmp_path / "missing.jpg")


def test_disguised_file_rejected_by_content_not_extension(tmp_path):
    p = tmp_path / "evil.jpg"
    p.write_bytes(b"#!/bin/sh\nrm -rf /\n")
    with pytest.raises(InvalidImageError):
        load_image(p)


def test_oversized_upload_rejected(settings):
    import dataclasses
    tiny = dataclasses.replace(settings, max_upload_mb=0)
    with pytest.raises(InvalidImageError, match="larger"):
        decode_image(jpeg_bytes(np.zeros((10, 10, 3), np.uint8)), tiny)


# ---- detection ----------------------------------------------------------------------
def test_detects_single_face_with_sane_box(detector, lfw_people):
    img = load_image(next(iter(lfw_people.values()))[0])
    faces = detector.detect(img)
    assert len(faces) >= 1
    x, y, w, h = faces[0].bbox
    assert 0 <= x < img.shape[1] and 0 <= y < img.shape[0] and w > 20 and h > 20
    assert 0 < faces[0].quality <= 1 and faces[0].index == 0


def test_no_face_returns_empty_list(detector):
    rng = np.random.default_rng(0)
    assert detector.detect(np.full((300, 300, 3), 120, np.uint8)) == []
    assert detector.detect(rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)) == []


def test_multiple_faces_detected_and_sorted(detector, lfw_people):
    a, b = [load_image(v[0]) for v in list(lfw_people.values())[:2]]
    both = np.hstack([a, b])
    faces = detector.detect(both)
    assert len(faces) >= 2
    assert [f.index for f in faces] == list(range(len(faces)))
    assert faces[0].quality * faces[0].area >= faces[-1].quality * faces[-1].area
    xs = sorted(f.bbox[0] for f in faces[:2])
    assert xs[1] - xs[0] > 100                      # two distinct places, not duplicates


def test_large_image_boxes_in_original_coordinates(detector, lfw_people):
    img = load_image(next(iter(lfw_people.values()))[0])
    big = cv2.resize(img, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)     # 2000x2000 > max_detect_side
    small_face, big_face = detector.detect(img)[0], detector.detect(big)[0]
    assert big_face.bbox[2] == pytest.approx(small_face.bbox[2] * 8, rel=0.2)


def test_tiny_face_is_rejected_by_quality_filter(detector, lfw_people):
    img = load_image(next(iter(lfw_people.values()))[0])
    canvas = np.full((900, 900, 3), 127, np.uint8)
    small = cv2.resize(img, (50, 50))
    canvas[400:450, 400:450] = small
    assert detector.detect(canvas) == []            # < MIN_FACE_SIZE_PX or low quality
