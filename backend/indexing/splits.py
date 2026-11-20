"""Deterministic gallery / query splits (no leakage by construction).

Rules (identity-aware, closed-set + open-set):
  * identities with >=2 images are "enrolled": up to `max_queries` images (~`query_fraction` of them, min 1)
    become QUERIES; ALL the others go in the GALLERY (so every query identity has >=1 gallery image,
    and no query image is ever in the gallery).
  * identities with exactly 1 image can't be queried against themselves. Half go into the gallery as
    distractors; the other half are UNKNOWN queries (person not in the gallery) used to measure false matches.
Ordering uses SHA-256 of (seed, relative path) so the split is identical on every machine and Python version.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

from backend.database.models import PhotoRecord


def _rank(seed: int, rel_path: str) -> str:
    return hashlib.sha256(f"{seed}|{rel_path}".encode()).hexdigest()


@dataclass
class Split:
    gallery: list[str]            # rel_paths
    queries: list[str]            # known-identity queries (identity present in gallery)
    unknown_queries: list[str]    # identity absent from gallery
    seed: int

    def to_dict(self) -> dict:
        return {"seed": self.seed, "gallery": self.gallery, "queries": self.queries,
                "unknown_queries": self.unknown_queries}


def make_split(records: list[PhotoRecord], seed: int = 42, query_fraction: float = 0.3, max_queries: int = 5) -> Split:
    by_id: dict[str, list[PhotoRecord]] = defaultdict(list)
    for r in records:
        if r.identity is None:
            raise ValueError(f"{r.rel_path} has no identity label; cannot build an evaluation split")
        by_id[r.identity].append(r)
    gallery: list[str] = []
    queries: list[str] = []
    unknown: list[str] = []
    for identity in sorted(by_id):
        imgs = sorted(by_id[identity], key=lambda r: _rank(seed, r.rel_path))
        n = len(imgs)
        if n == 1:
            (unknown if int(_rank(seed, "id|" + identity), 16) % 2 else gallery).append(imgs[0].rel_path)
            continue
        nq = max(1, min(max_queries, round(n * query_fraction), n - 1))
        queries += [r.rel_path for r in imgs[:nq]]
        gallery += [r.rel_path for r in imgs[nq:]]
    gallery.sort(); queries.sort(); unknown.sort()
    assert not (set(gallery) & set(queries)) and not (set(gallery) & set(unknown)), "split leakage"
    return Split(gallery, queries, unknown, seed)
