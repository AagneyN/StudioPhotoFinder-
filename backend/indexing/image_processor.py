"""Safe image loading and validation. Used for dataset images AND user uploads."""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from backend.config.settings import Settings, get_settings
from backend.errors import InvalidImageError

_ALLOWED_FORMATS = {"JPEG", "PNG"}


def decode_image(data: bytes, settings: Settings | None = None) -> np.ndarray:
    """Decode bytes -> BGR uint8. Validates the real format (not the extension), size and pixel count."""
    s = settings or get_settings()
    if not data:
        raise InvalidImageError("The file is empty.")
    if len(data) > s.max_upload_mb * 1024 * 1024:
        raise InvalidImageError(f"File is larger than the {s.max_upload_mb} MB limit.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in _ALLOWED_FORMATS:
                raise InvalidImageError(f"Unsupported image format '{im.format}'. Use JPG or PNG.")
            if im.width * im.height > s.max_image_pixels:
                raise InvalidImageError("Image dimensions are too large.")
            im = ImageOps.exif_transpose(im).convert("RGB")
            arr = np.asarray(im)
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as e:
        raise InvalidImageError(f"Not a valid or readable image: {e}") from e
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def load_image(path: str | Path, settings: Settings | None = None) -> np.ndarray:
    """Load an image file (BGR). Raises InvalidImageError for corrupted/unreadable/unsupported files."""
    p, s = Path(path), settings or get_settings()
    if p.suffix.lower() not in s.allowed_extensions:
        raise InvalidImageError(f"Unsupported extension: {p.suffix}")
    try:
        data = p.read_bytes()
    except OSError as e:
        raise InvalidImageError(f"Cannot read file {p}: {e}") from e
    return decode_image(data, s)


def make_thumbnail(path: str | Path, out_path: Path, size: int = 220) -> Path:
    img = load_image(path)
    h, w = img.shape[:2]
    scale = size / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise InvalidImageError("Could not encode thumbnail")
    out_path.write_bytes(buf.tobytes())
    return out_path
