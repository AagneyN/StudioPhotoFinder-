"""Plain data classes shared between the loaders, repository and API (no ORM, so the DB backend can change)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class PhotoRecord:
    """A photograph as discovered by a dataset loader (before indexing).

    The same structure serves LFW now and the studio archive later:
    year/category/photo_number are None for LFW and derived from the folder layout for studio photos.
    """
    source: str                 # logical dataset name, e.g. "lfw" or "studio"
    rel_path: str               # posix path relative to the dataset root (unique with source)
    abs_path: str               # original file location (never copied)
    filename: str
    identity: str | None = None
    year: int | None = None
    category: str | None = None
    photo_number: str | None = None
    file_size: int = 0
    mtime_ns: int = 0


@dataclass
class PhotoRow(PhotoRecord):
    id: int = 0
    status: str = "indexed"     # indexed | no_face | failed
    error: str | None = None
    n_faces: int = 0
    width: int = 0
    height: int = 0
    indexed_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FaceInsert:
    face_index: int
    bbox: tuple[int, int, int, int]
    det_score: float
    quality: float
    embedding: bytes            # float32 little-endian, L2-normalised


@dataclass
class FaceRow:
    id: int
    photo_id: int
    face_index: int
    bbox: tuple[int, int, int, int]
    det_score: float
    quality: float


@dataclass
class IndexStats:
    photos_total: int = 0
    photos_indexed: int = 0
    photos_no_face: int = 0
    photos_failed: int = 0
    photos_multi_face: int = 0
    faces: int = 0
    identities: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
