"""Reproducible evaluation of the recognition + retrieval pipeline on LFW.

    python scripts/evaluate_model.py                 # full run (embeds all 13,233 images once, then caches)
    python scripts/evaluate_model.py --limit-identities 200   # quick smoke run on a subset
    python scripts/evaluate_model.py --seed 7        # a different (but still reproducible) split

Outputs: data/processed/evaluation_report.json (+ .csv summary) and data/processed/calibration.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config.settings import get_settings, setup_logging  # noqa: E402
from backend.evaluation import (GalleryFaces, identification_metrics, open_set_false_matches,  # noqa: E402
                                pair_scores, rates_at_threshold, roc_from_scores, threshold_for_fmr)
from backend.indexing.dataset_loader import get_loader  # noqa: E402
from backend.indexing.image_processor import load_image  # noqa: E402
from backend.indexing.splits import make_split  # noqa: E402
from backend.recognition.detector import FaceDetector  # noqa: E402
from backend.recognition.embedder import FaceEmbedder  # noqa: E402

log = logging.getLogger("evaluate")

METHODOLOGY = [
    "Dataset: original LFW (13,233 images, 5,749 identities) read from the local copy; labels = folder names.",
    "Split: deterministic (SHA-256 of seed+path). Identities with >=2 images: up to 5 images (~30%) become queries, "
    "the rest go to the gallery, so every query identity has >=1 gallery image and no query image is in the gallery. "
    "Identities with 1 image: half are gallery distractors, half are 'unknown' queries (person absent from the gallery).",
    "Gallery: every acceptable face (YuNet, quality filter) of every gallery photo is embedded and indexed (exact cosine search).",
    "Query: the best-quality face (largest, most frontal, highest confidence) of the query photo, as the UI does by default.",
    "Identification: ranked list of distinct gallery photos; a result is correct only if its identity equals the query identity. "
    "Queries where no face is detected count as misses in topK_accuracy (end-to-end) and are excluded in *_detected_only.",
    "Open-set false matches: unknown-person queries; FMR(t) = share whose best gallery score is >= t.",
    "Verification: threshold calibrated on gallery-vs-gallery pairs only, then tested on disjoint query-vs-gallery pairs.",
]
LIMITATIONS = [
    "LFW is mostly frontal, celebrity/news photos; studio passport photos differ (lighting, aging over years, glasses).",
    "Many LFW identities have 2 images, so per-identity results rest on few samples; identity 'George W Bush' etc. are over-represented in micro averages.",
    "Thresholds calibrated on LFW must be re-calibrated on studio data before any operational use.",
    "Verification FMR floors: with N impostor pairs the smallest measurable FMR is ~1/N.",
    "LFW may contain a small number of mislabeled/duplicate images; they are not corrected here.",
    "Similarity ranking is a decision-support tool, not proof of identity.",
]


class EmbeddingCache:
    """Per-image embeddings cached on disk so a rerun (or an interrupted run) does not redo work."""

    def __init__(self, path: Path, fingerprint: str):
        self.path, self.fp = path, fingerprint
        self.data: dict[str, dict] = {}
        z = None
        if path.exists():
            try:
                z = np.load(path, allow_pickle=False)
                _ = z["fingerprint"]
            except Exception as e:  # truncated/corrupt cache (e.g. interrupted write): rebuild instead of crashing
                log.warning("Embedding cache %s is unreadable (%s); discarding and re-embedding.", path.name, e)
                z = None
        if z is not None:
            if str(z["fingerprint"]) == fingerprint:
                counts, vecs, paths = z["counts"], z["vecs"], z["paths"]   # read each array once (npz re-reads per access)
                det, embm = z["det_ms"], z["emb_ms"]
                offs = np.concatenate([[0], np.cumsum(counts)])
                for i, p in enumerate(paths):
                    self.data[str(p)] = {"vecs": vecs[offs[i]:offs[i + 1]], "det_ms": float(det[i]), "emb_ms": float(embm[i])}
            else:
                log.warning("Embedding cache is for another model; ignoring it.")

    def save(self) -> None:
        paths = sorted(self.data)
        counts = np.array([len(self.data[p]["vecs"]) for p in paths], dtype=np.int64)
        vecs = np.vstack([self.data[p]["vecs"] for p in paths]) if paths and counts.sum() else np.zeros((0, 128), np.float32)
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez(tmp, fingerprint=self.fp, paths=np.array(paths), counts=counts, vecs=vecs,
                 det_ms=np.array([self.data[p]["det_ms"] for p in paths]), emb_ms=np.array([self.data[p]["emb_ms"] for p in paths]))
        tmp.replace(self.path)


def embed_all(records, cache: EmbeddingCache, det, emb) -> dict:
    todo = [r for r in records if r.rel_path not in cache.data]
    log.info("Embedding %d images (%d cached)", len(todo), len(records) - len(todo))
    t0 = time.perf_counter()
    for i, r in enumerate(todo, 1):
        try:
            img = load_image(r.abs_path)
            t1 = time.perf_counter()
            faces = det.detect(img)
            t2 = time.perf_counter()
            vecs = emb.embed_faces(img, faces) if faces else np.zeros((0, emb.s.embedding_dim), np.float32)
            t3 = time.perf_counter()
            cache.data[r.rel_path] = {"vecs": vecs, "det_ms": (t2 - t1) * 1000, "emb_ms": (t3 - t2) * 1000}
        except Exception as e:
            log.warning("Skipping unreadable image %s: %s", r.rel_path, e)
            cache.data[r.rel_path] = {"vecs": np.zeros((0, emb.s.embedding_dim), np.float32), "det_ms": 0.0, "emb_ms": 0.0}
        if i % 500 == 0:
            cache.save()
            el = time.perf_counter() - t0
            print(f"\r  {i}/{len(todo)}  {i / el:.1f} img/s", end="", flush=True)
    if todo:
        cache.save()
        print()
    return {"images_embedded_now": len(todo), "wall_sec": time.perf_counter() - t0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-identities", type=int, help="subset of identities for a quick run")
    ap.add_argument("--target-fmr", type=float, default=1e-3, help="FMR used to calibrate the saved threshold")
    ap.add_argument("--no-calibration-write", action="store_true")
    a = ap.parse_args()
    s = get_settings()
    setup_logging(s, "evaluation")
    s.ensure_dirs()

    loader = get_loader(s.dataset_name, s.dataset_root)
    records = list(loader.scan())
    if a.limit_identities:
        keep = sorted({r.identity for r in records})[:a.limit_identities]
        records = [r for r in records if r.identity in set(keep)]
    by_path = {r.rel_path: r for r in records}
    split = make_split(records, seed=a.seed)
    det, emb = FaceDetector(s), FaceEmbedder(s)
    cache_name = "eval_embeddings.npz" if not a.limit_identities else f"eval_embeddings_subset{a.limit_identities}.npz"
    cache = EmbeddingCache(s.processed_dir / cache_name, emb.info.fingerprint)
    timing = embed_all(records, cache, det, emb)

    # ---- gallery faces (all faces of gallery photos) ----
    g_vecs, g_pidx, g_ident = [], [], []
    for p_i, rel in enumerate(split.gallery):
        v = cache.data[rel]["vecs"]
        g_ident.append(by_path[rel].identity)
        if len(v):
            g_vecs.append(v); g_pidx += [p_i] * len(v)
    gallery = GalleryFaces(np.vstack(g_vecs), np.array(g_pidx), g_ident)
    gal_photo_vec = {}          # best face per gallery photo, for the pair protocols
    for rel in split.gallery:
        v = cache.data[rel]["vecs"]
        if len(v):
            gal_photo_vec[rel] = v[0]
    n_no_face_gallery = sum(1 for rel in split.gallery if not len(cache.data[rel]["vecs"]))

    def qvec(rel):
        v = cache.data[rel]["vecs"]
        return v[0] if len(v) else None     # faces are stored best-first (detector sort order)

    # ---- identification ----
    q_list = [(by_path[r].identity, qvec(r)) for r in split.queries]
    ident = identification_metrics(gallery, q_list)
    top1_scores, top1_correct = ident.pop("_top1_scores"), ident.pop("_top1_correct")
    unknown_vecs = [v for v in (qvec(r) for r in split.unknown_queries) if v is not None]

    # ---- verification: calibrate on gallery pairs ----
    rng = np.random.default_rng(a.seed)
    gal_by_id: dict[str, list[str]] = {}
    for rel in gal_photo_vec:
        gal_by_id.setdefault(by_path[rel].identity, []).append(rel)
    gen_a, gen_b = [], []
    for idn in sorted(gal_by_id):
        rels = sorted(gal_by_id[idn])
        pairs = [(x, y) for i, x in enumerate(rels) for y in rels[i + 1:]]
        if len(pairs) > 10:
            pairs = [pairs[j] for j in rng.choice(len(pairs), 10, replace=False)]
        for x, y in pairs:
            gen_a.append(gal_photo_vec[x]); gen_b.append(gal_photo_vec[y])
    rels_all = sorted(gal_photo_vec)
    ids_all = np.array([by_path[r].identity for r in rels_all])
    mat = np.vstack([gal_photo_vec[r] for r in rels_all])
    n_imp = 200_000
    ia, ib = rng.integers(0, len(rels_all), n_imp), rng.integers(0, len(rels_all), n_imp)
    ok = ids_all[ia] != ids_all[ib]
    cal_gen = pair_scores(np.array(gen_a), np.array(gen_b)) if gen_a else np.array([])
    cal_imp = pair_scores(mat[ia[ok]], mat[ib[ok]])

    verification = {"calibration_pairs": {"genuine": int(len(cal_gen)), "impostor": int(len(cal_imp)), "source": "gallery vs gallery"}}
    thresholds = {}
    for tf in (1e-2, 1e-3):
        thresholds[tf] = threshold_for_fmr(cal_imp, tf)

    # ---- verification: test on query-vs-gallery pairs ----
    t_gen, t_imp = [], []
    for rel in split.queries:
        qv = qvec(rel)
        if qv is None:
            continue
        same = gal_by_id.get(by_path[rel].identity, [])
        for gr in same[:5]:
            t_gen.append(float(qv @ gal_photo_vec[gr]))
    for rel in split.queries + split.unknown_queries:
        qv = qvec(rel)
        if qv is None:
            continue
        idn = by_path[rel].identity
        for j in rng.integers(0, len(rels_all), 20):
            if ids_all[j] != idn:
                t_imp.append(float(qv @ mat[j]))
    t_gen, t_imp = np.array(t_gen), np.array(t_imp)
    verification["test_pairs"] = {"genuine": int(len(t_gen)), "impostor": int(len(t_imp)), "source": "query vs gallery"}
    verification["test_roc"] = roc_from_scores(t_gen, t_imp)
    verification["operating_points"] = [
        {"target_fmr_on_calibration": tf, **rates_at_threshold(t_gen, t_imp, th)} for tf, th in thresholds.items()]

    open_set = open_set_false_matches(gallery, unknown_vecs, [thresholds[1e-2], thresholds[1e-3]])
    chosen = thresholds[a.target_fmr] if a.target_fmr in thresholds else threshold_for_fmr(cal_imp, a.target_fmr)

    # ---- end-to-end latency on real files (detect + embed + search), 200 queries ----
    from backend.evaluation import build_index, ranked_photos
    index = build_index(gallery)
    lat = []
    for rel in split.queries[:200]:
        t = time.perf_counter()
        img = load_image(by_path[rel].abs_path)
        faces = det.detect(img)
        if faces:
            v = emb.embed_faces(img, faces[:1])[0]
            ranked_photos(index, gallery, v, 10)
        lat.append((time.perf_counter() - t) * 1000)

    det_ms = np.array([cache.data[r.rel_path]["det_ms"] for r in records if r.rel_path in cache.data])
    emb_ms = np.array([cache.data[r.rel_path]["emb_ms"] for r in records if r.rel_path in cache.data and len(cache.data[r.rel_path]["vecs"])])
    n_faces_total = int(sum(len(cache.data[r.rel_path]["vecs"]) for r in records))
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": emb.info.to_dict(), "detector": s.detector_model, "seed": a.seed,
        "dataset": {"name": s.dataset_name, "images": len(records), "identities": len({r.identity for r in records}),
                    "subset_identities": a.limit_identities},
        "split": {"gallery_photos": len(split.gallery), "gallery_faces": int(len(gallery.vectors)),
                  "gallery_photos_without_face": n_no_face_gallery, "known_queries": len(split.queries),
                  "unknown_queries": len(split.unknown_queries)},
        "identification": ident,
        "open_set": open_set,
        "verification": verification,
        "calibrated_threshold": {"threshold": round(chosen, 4), "target_fmr": a.target_fmr,
                                 "calibrated_on": "gallery-vs-gallery impostor pairs"},
        "timing": {
            "detect_ms_per_image_mean": float(det_ms.mean()) if len(det_ms) else None,
            "embed_ms_per_image_mean": float(emb_ms.mean()) if len(emb_ms) else None,
            "end_to_end_query_ms_mean": float(np.mean(lat)) if lat else None,
            "end_to_end_query_ms_p95": float(np.percentile(lat, 95)) if lat else None,
            "indexing_images_per_sec": (len(records) / float((det_ms + 0).sum() / 1000 + emb_ms.sum() / 1000)) if len(det_ms) and (det_ms.sum() + emb_ms.sum()) else None,
            "faces_total": n_faces_total,
            "this_run": timing, "execution_provider": emb.info.execution_provider,
            "note": "per-image times come from the embedding cache (recorded when each image was first processed)",
        },
        "methodology": METHODOLOGY, "limitations": LIMITATIONS,
    }
    out = s.eval_report_path if not a.limit_identities else s.processed_dir / "evaluation_report_subset.json"
    out.write_text(json.dumps(report, indent=2))
    if not a.limit_identities:
        s.eval_split_path.write_text(json.dumps(split.to_dict()))
    if not a.limit_identities and not a.no_calibration_write:
        s.calibration_path.write_text(json.dumps(report["calibrated_threshold"] | {"model": emb.info.fingerprint}, indent=2))
    with out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in ident.items():
            w.writerow([f"identification.{k}", v])
        for k, v in report["timing"].items():
            if isinstance(v, (int, float)):
                w.writerow([f"timing.{k}", v])
        for row in report["verification"]["operating_points"]:
            for k, v in row.items():
                w.writerow([f"verification.fmr{row['target_fmr_on_calibration']}.{k}", v])
        w.writerow(["verification.auc", verification["test_roc"]["auc"]])
        w.writerow(["verification.eer", verification["test_roc"]["eer"]])

    print(json.dumps({k: report[k] for k in ("split", "identification", "open_set", "calibrated_threshold", "timing")}, indent=2))
    print("verification:", json.dumps({"test_pairs": verification["test_pairs"], "roc": verification["test_roc"],
                                       "operating_points": verification["operating_points"]}, indent=2))
    print(f"\nReport written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
