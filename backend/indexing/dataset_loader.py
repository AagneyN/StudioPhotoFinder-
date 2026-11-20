"""Dataset loaders: turn a folder tree into `PhotoRecord`s. The recognition engine never imports this module's
dataset-specific parts; it only sees PhotoRecord. A StudioFolderLoader (year/category/photo-number from the
folder layout, UNC paths, incremental scans) can be added next to LfwLoader without touching anything else."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Protocol

from backend.database.models import PhotoRecord
from backend.errors import DatasetNotFoundError

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


class DatasetLoader(Protocol):
    source: str

    def scan(self) -> Iterator[PhotoRecord]: ...


class LfwLoader:
    """LFW layout: <root>/<Person_Name>/<Person_Name_0001>.jpg ; the folder name is the identity label."""

    def __init__(self, root: Path, source: str = "lfw"):
        self.root = Path(root)
        self.source = source

    def validate(self) -> None:
        if not self.root.exists():
            raise DatasetNotFoundError(
                f"Dataset folder not found: {self.root}. Run `python scripts/download_dataset.py --dataset`.")
        if not any(self.root.iterdir()):
            raise DatasetNotFoundError(f"Dataset folder is empty: {self.root}.")

    def scan(self) -> Iterator[PhotoRecord]:
        self.validate()
        for person_dir in sorted((p for p in os.scandir(self.root) if p.is_dir()), key=lambda e: e.name):
            for f in sorted(os.scandir(person_dir.path), key=lambda e: e.name):
                if not f.is_file() or Path(f.name).suffix.lower() not in IMAGE_EXTS:
                    continue
                st = f.stat()
                yield PhotoRecord(
                    source=self.source,
                    rel_path=f"{person_dir.name}/{f.name}",
                    abs_path=str(Path(f.path)),
                    filename=f.name,
                    identity=person_dir.name,
                    file_size=st.st_size,
                    mtime_ns=st.st_mtime_ns,
                )


def get_loader(name: str, root: Path) -> DatasetLoader:
    if name == "lfw":
        return LfwLoader(root, "lfw")
    raise ValueError(f"Unknown dataset '{name}'. Implemented: lfw")
