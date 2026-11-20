"""Evaluation metrics (pure functions on embeddings, so they can be unit-tested on tiny fixtures).

Protocols (see METHODOLOGY in scripts/evaluate_model.py for the human-readable version):
  * Closed-set identification: query -> ranked list of distinct gallery PHOTOS; correct iff the photo's identity
    equals the query identity. Top-k accuracy = fraction of queries with a correct photo within the first k.
  * Open-set false matches: queries of identities absent from the gallery; FMR(t) = fraction whose best score >= t.
  * Verification: genuine/impostor pair score distributions -> FMR/FNMR/EER/AUC; the operating threshold is
    calibrated on gallery-only pairs and then *tested* on query-vs-gallery pairs (disjoint images).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from backend.indexing.index_manager import FaceIndex
from backend.recognition.similarity import l2_normalize


@dataclass
class GalleryFaces:
    """All indexed faces of the gallery, with the parent photo + identity of every face."""
    vectors: np.ndarray        # (n_faces, d) L2-normalised
    photo_idx: np.ndarray      # (n_faces,) index of parent photo
    photo_identity: list[str]  # identity per photo


def build_index(g: GalleryFaces) -> FaceIndex:
    idx = FaceIndex(g.vectors.shape[1], "eval")
    idx.add(np.arange(len(g.vectors)), g.vectors)
    return idx


def ranked_photos(index: FaceIndex, g: GalleryFaces, q: np.ndarray, max_photos: int = 10,
                  search_k: int = 200) -> list[tuple[int, float]]:
    """Distinct parent photos by best face score, descending. Widens the face search until enough photos are found."""
    k = min(search_k, index.ntotal)
    while True:
        ids, scores = index.search(q, k)
        seen: dict[int, float] = {}
        for i, s in zip(ids.tolist(), scores.tolist()):
            p = int(g.photo_idx[i])
            if p not in seen:
                seen[p] = s
        if len(seen) >= max_photos or k >= index.ntotal:
            return list(seen.items())[:max_photos]
        k = min(index.ntotal, k * 4)


def identification_metrics(g: GalleryFaces, queries: list[tuple[str, np.ndarray | None]],
                           ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    """queries: (true identity, query embedding or None if no face was detected). Returns metrics + per-query top-1 info."""
    index = build_index(g)
    hits = {k: 0 for k in ks}
    per_id: dict[str, list[int]] = {}
    top1_scores, top1_correct, latencies = [], [], []
    detected = 0
    for ident, vec in queries:
        per_id.setdefault(ident, [])
        if vec is None:       # undetected query counts as a miss at every k (end-to-end view)
            per_id[ident].append(0)
            top1_scores.append(None); top1_correct.append(False)
            continue
        detected += 1
        t = time.perf_counter()
        ranked = ranked_photos(index, g, vec, max(ks))
        latencies.append((time.perf_counter() - t) * 1000)
        ids_ranked = [g.photo_identity[p] for p, _ in ranked]
        for k in ks:
            if ident in ids_ranked[:k]:
                hits[k] += 1
        per_id[ident].append(int(ident in ids_ranked[:1]))
        top1_scores.append(ranked[0][1] if ranked else None); top1_correct.append(bool(ids_ranked[:1] == [ident]))
    n = len(queries)
    out: dict = {"n_queries": n, "n_detected": detected, "query_detection_rate": detected / n if n else 0.0}
    for k in ks:
        out[f"top{k}_accuracy"] = hits[k] / n if n else 0.0                       # undetected = miss
        out[f"top{k}_accuracy_detected_only"] = hits[k] / detected if detected else 0.0
    out["top1_accuracy_macro_per_identity"] = float(np.mean([np.mean(v) for v in per_id.values()])) if per_id else 0.0
    out["n_query_identities"] = len(per_id)
    if latencies:
        out["search_latency_ms_mean"] = float(np.mean(latencies))
        out["search_latency_ms_p95"] = float(np.percentile(latencies, 95))
    out["_top1_scores"] = top1_scores
    out["_top1_correct"] = top1_correct
    return out


def open_set_false_matches(g: GalleryFaces, unknown_vecs: list[np.ndarray], thresholds: list[float]) -> dict:
    """FMR(t) for queries of people who are NOT in the gallery (any returned match is a false match)."""
    index = build_index(g)
    best = []
    for v in unknown_vecs:
        ids, sc = index.search(v, 1)
        best.append(float(sc[0]) if len(sc) else -1.0)
    best_arr = np.array(best)
    return {"n_unknown_queries": len(best),
            "by_threshold": [{"threshold": round(float(t), 4),
                              "false_match_rate": float((best_arr >= t).mean()) if len(best_arr) else None,
                              "false_matches": int((best_arr >= t).sum())} for t in thresholds]}


def roc_from_scores(genuine: np.ndarray, impostor: np.ndarray) -> dict:
    """AUC and EER from score samples (higher score = more similar)."""
    genuine, impostor = np.sort(genuine), np.sort(impostor)
    thr = np.unique(np.concatenate([genuine, impostor]))
    fnmr = np.searchsorted(genuine, thr, side="left") / len(genuine)             # genuine < t -> false non-match
    fmr = 1.0 - np.searchsorted(impostor, thr, side="left") / len(impostor)      # impostor >= t -> false match
    # AUC via rank statistic (probability genuine > impostor, ties 0.5)
    ranks = np.searchsorted(impostor, genuine, side="left") + np.searchsorted(impostor, genuine, side="right")
    auc = float(ranks.sum() / (2.0 * len(genuine) * len(impostor)))
    i = int(np.argmin(np.abs(fmr - fnmr)))
    return {"auc": auc, "eer": float((fmr[i] + fnmr[i]) / 2), "eer_threshold": float(thr[i])}


def threshold_for_fmr(impostor: np.ndarray, target_fmr: float) -> float:
    """Smallest threshold whose empirical FMR on `impostor` is <= target_fmr."""
    s = np.sort(impostor)
    n_allowed = int(np.floor(target_fmr * len(s)))
    # threshold just above the (n_allowed+1)-th highest score
    cut = s[len(s) - n_allowed - 1] if n_allowed < len(s) else s[0]
    return float(np.nextafter(np.float32(cut), np.float32(np.inf)))


def rates_at_threshold(genuine: np.ndarray, impostor: np.ndarray, t: float) -> dict:
    return {"threshold": round(float(t), 4),
            "false_match_rate": float((impostor >= t).mean()),
            "false_non_match_rate": float((genuine < t).mean())}


def pair_scores(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity of two (n,d) arrays."""
    return np.sum(l2_normalize(a) * l2_normalize(b), axis=1)
