"""Index a dataset: detect faces -> embeddings -> FAISS + SQLite. Resumable and incremental.

Examples:
    python scripts/index_dataset.py                    # index the evaluation GALLERY split (default, no leakage)
    python scripts/index_dataset.py --split all        # index every image
    python scripts/index_dataset.py --limit 500        # quick trial
    python scripts/index_dataset.py --rebuild          # wipe index + database first
    python scripts/index_dataset.py --repair           # resync FAISS from database embeddings and exit
Ctrl+C stops safely; running the same command again resumes where it left off.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config.settings import get_settings, setup_logging  # noqa: E402
from backend.database.connection import create_repository  # noqa: E402
from backend.errors import FinderError  # noqa: E402
from backend.indexing.dataset_loader import get_loader  # noqa: E402
from backend.indexing.index_manager import IndexManager  # noqa: E402
from backend.indexing.pipeline import IndexingPipeline  # noqa: E402
from backend.indexing.splits import make_split  # noqa: E402
from backend.recognition.detector import FaceDetector  # noqa: E402
from backend.recognition.embedder import FaceEmbedder  # noqa: E402

log = logging.getLogger("index_dataset")


def wipe(s) -> None:
    for p in (s.sqlite_path, Path(str(s.sqlite_path) + "-wal"), Path(str(s.sqlite_path) + "-shm"),
              s.index_path, s.index_meta_path, s.status_path):
        p.unlink(missing_ok=True)
    log.warning("Existing index and database removed (--rebuild)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["gallery", "all"], default="gallery",
                    help="gallery = only the evaluation gallery images (queries stay held out). Default: gallery")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, help="process at most N new images")
    ap.add_argument("--rebuild", action="store_true", help="delete the existing index and database first")
    ap.add_argument("--repair", action="store_true", help="resync FAISS with the database and exit")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--batch-size", type=int)
    args = ap.parse_args()

    s = get_settings()
    setup_logging(s, "indexing")
    s.ensure_dirs()
    try:
        if args.rebuild:
            wipe(s)
        repo = create_repository(s)
        embedder = FaceEmbedder(s)
        print(f"Model: {embedder.info.name} | execution provider: {embedder.info.execution_provider}"
              + (f"\n  note: {embedder.info.fallback_reason}" if embedder.info.fallback_reason else ""))
        if args.repair:
            mgr = IndexManager(s, repo, embedder.info.fingerprint, embedder.info.name)
            idx = mgr.load_or_create()
            mgr.save(idx)
            print(f"Index repaired: {idx.ntotal} vectors")
            return 0
        loader = get_loader(s.dataset_name, s.dataset_root)
        records = list(loader.scan())
        if not records:
            raise FinderError(f"No images found under {s.dataset_root}")
        if args.split == "gallery":
            split = make_split(records, seed=args.seed)
            s.processed_dir.mkdir(parents=True, exist_ok=True)
            s.eval_split_path.write_text(json.dumps(split.to_dict()))
            keep = set(split.gallery)
            records = [r for r in records if r.rel_path in keep]
            print(f"Gallery split (seed {args.seed}): {len(records)} images; "
                  f"{len(split.queries)} known-identity + {len(split.unknown_queries)} unknown queries held out.")
        if args.batch_size:
            object.__setattr__(s, "batch_size", args.batch_size)
        pipe = IndexingPipeline(s, repo, FaceDetector(s), embedder)

        def show(st: dict) -> None:
            n, tot = st["processed"], st.get("to_process", 0)
            print(f"\r[{n}/{tot}] indexed={st['photos_indexed']} faces={st['faces_added']} no_face={st['no_face']} "
                  f"failed={st['failed']} multi={st['multi_face']} {st['images_per_sec']} img/s ETA {st['eta_sec']}s   ",
                  end="", flush=True)

        st = pipe.run(records, prune=args.limit is None, retry_failed=args.retry_failed, limit=args.limit, progress=show)
        print()
        print(json.dumps({k: v for k, v in st.items() if k != "recent_errors"}, indent=2))
        for e in st["recent_errors"]:
            print("  error:", e)
        return 0 if st["state"] in ("completed", "interrupted") else 1
    except FinderError as e:
        log.error("%s", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
