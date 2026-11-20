import dataclasses

import numpy as np
import pytest

from backend.database.connection import create_repository
from backend.database.models import FaceInsert, PhotoRecord
from backend.errors import IndexUnavailableError
from backend.evaluation import (GalleryFaces, identification_metrics, open_set_false_matches, ranked_photos,
                                build_index, roc_from_scores, threshold_for_fmr)
from backend.indexing.index_manager import FaceIndex, IndexManager
from backend.indexing.splits import make_split
from backend.recognition.similarity import cosine_similarity, l2_normalize

FP = "test-model:128"


def unit(n, seed=0, d=128):
    return l2_normalize(np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32))


# ---- similarity -----------------------------------------------------------
def test_l2_normalize_unit_norm_and_zero_vector_safe():
    x = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(np.linalg.norm(x[0]), 1.0) and np.all(np.isfinite(x))


def test_cosine_similarity_known_values():
    a = np.array([[1.0, 0.0]])
    b = np.array([[2.0, 0.0], [0.0, 5.0], [-1.0, 0.0]])
    assert np.allclose(cosine_similarity(a, b), [[1.0, 0.0, -1.0]])    # scale invariant


# ---- FAISS ranking ---------------------------------------------------------
def test_search_ranking_descending_and_exact():
    v = unit(50)
    idx = FaceIndex(128, FP)
    idx.add(list(range(100, 150)), v)
    q = v[7] * 0.9 + v[8] * 0.1
    ids, scores = idx.search(q, 5)
    assert ids[0] == 107 and list(scores) == sorted(scores, reverse=True)
    expected = (l2_normalize(q) @ v.T)
    assert np.isclose(scores[0], expected.max(), atol=1e-5)


def test_search_with_allowed_ids_and_remove():
    v = unit(10)
    idx = FaceIndex(128, FP)
    idx.add(list(range(10)), v)
    ids, _ = idx.search(v[3], 3, allowed_ids=[1, 2])
    assert set(ids.tolist()) <= {1, 2}
    assert idx.remove([3]) == 1
    assert 3 not in idx.search(v[3], 10)[0].tolist()
    assert len(idx.search(v[3], 3, allowed_ids=[])[0]) == 0


def test_dimension_mismatch_rejected():
    idx = FaceIndex(128, FP)
    with pytest.raises(ValueError):
        idx.add([1], np.zeros((1, 64), np.float32))
    with pytest.raises(ValueError):
        idx.search(np.zeros(64, np.float32), 1)


# ---- persistence / compatibility --------------------------------------------
def test_index_save_and_reload(settings):
    v = unit(20)
    idx = FaceIndex(128, FP)
    idx.add(list(range(20)), v)
    idx.save(settings.index_path, settings.index_meta_path, "m")
    again = FaceIndex.load(settings.index_path, settings.index_meta_path, FP, 128)
    assert again.ntotal == 20 and again.search(v[5], 1)[0][0] == 5


def test_incompatible_model_index_cannot_be_loaded(settings):
    idx = FaceIndex(128, FP)
    idx.add([1], unit(1))
    idx.save(settings.index_path, settings.index_meta_path)
    with pytest.raises(IndexUnavailableError, match="incompatible"):
        FaceIndex.load(settings.index_path, settings.index_meta_path, "other-model:128", 128)
    with pytest.raises(IndexUnavailableError):
        FaceIndex.load(settings.index_path, settings.index_meta_path, FP, 512)


def test_missing_and_corrupt_index(settings):
    with pytest.raises(IndexUnavailableError, match="not found"):
        FaceIndex.load(settings.index_path, settings.index_meta_path, FP, 128)
    settings.faiss_dir.mkdir(parents=True)
    settings.index_path.write_bytes(b"garbage")
    settings.index_meta_path.write_text('{"fingerprint": "%s", "dim": 128}' % FP)
    with pytest.raises(IndexUnavailableError, match="corrupt"):
        FaceIndex.load(settings.index_path, settings.index_meta_path, FP, 128)


def _mgr(settings, repo):
    return IndexManager(settings, repo, FP, "m")


def test_index_database_mismatch_detected_then_repaired(settings):
    repo = create_repository(settings)
    vecs = unit(3)
    for i in range(3):
        repo.save_photo_result(PhotoRecord("lfw", f"A/{i}.jpg", f"x/{i}", f"{i}.jpg", "A"), "indexed", None, 1, 1,
                               [FaceInsert(0, (0, 0, 1, 1), .9, .9, vecs[i].tobytes())])
    mgr = _mgr(settings, repo)
    idx = mgr.new_index()
    idx.add(sorted(repo.all_face_ids())[:2], vecs[:2])          # simulate a crash: FAISS is missing one face
    mgr.save(idx)
    with pytest.raises(IndexUnavailableError, match="mismatch"):
        mgr.load_strict()
    fixed = mgr.load_or_create()                                # repair from DB embeddings
    mgr.save(fixed)
    assert mgr.load_strict().ntotal == 3


def test_database_model_change_is_refused(settings):
    repo = create_repository(settings)
    _mgr(settings, repo)._check_db_model()
    other = IndexManager(settings, repo, "different:128", "x")
    with pytest.raises(IndexUnavailableError):
        other.load_or_create()


# ---- evaluation protocol -----------------------------------------------------
def _records():
    out = []
    for n, count in {"A": 6, "B": 3, "C": 2, "D": 1, "E": 1, "F": 1, "G": 1}.items():
        out += [PhotoRecord("lfw", f"{n}/{n}_{i}.jpg", "", f"{n}_{i}.jpg", n) for i in range(count)]
    return out


def test_split_is_deterministic_and_leak_free():
    r = _records()
    a, b = make_split(r, seed=1), make_split(list(reversed(r)), seed=1)
    assert a.to_dict() == b.to_dict()                                         # order independent + reproducible
    assert make_split(r, seed=2).to_dict() != a.to_dict()
    assert not set(a.gallery) & set(a.queries) and not set(a.gallery) & set(a.unknown_queries)
    ident = {x.rel_path: x.identity for x in r}
    gal_ids = {ident[p] for p in a.gallery}
    assert all(ident[q] in gal_ids for q in a.queries)                        # closed-set: query identity in gallery
    assert all(ident[u] not in gal_ids for u in a.unknown_queries)            # open-set: truly unseen people
    assert len(a.gallery) + len(a.queries) + len(a.unknown_queries) == len(r)


def _gallery():
    # 3 identities x 2 photos; photo 0 of identity 'A' has TWO faces (second is a background person)
    centers = unit(3, seed=5)
    vecs, pidx, pid = [], [], []
    rng = np.random.default_rng(1)
    photo = 0
    for k, n in enumerate("ABC"):
        for _ in range(2):
            vecs.append(l2_normalize(centers[k] + 0.1 * rng.normal(size=128))); pidx.append(photo); pid.append(n); photo += 1
    vecs.append(unit(1, seed=99)[0]); pidx.append(0)
    return GalleryFaces(np.vstack(vecs), np.array(pidx), pid), centers


def test_identification_metrics_and_ranked_photos_dedupe():
    g, c = _gallery()
    queries = [("A", l2_normalize(c[0])), ("B", l2_normalize(c[1])), ("C", None), ("Z", l2_normalize(c[2]))]
    m = identification_metrics(g, queries)
    assert m["top1_accuracy"] == 0.5 and m["n_detected"] == 3                  # A, B correct; C undetected=miss; Z wrong
    assert m["top1_accuracy_detected_only"] == pytest.approx(2 / 3)
    assert m["top10_accuracy"] >= m["top5_accuracy"] >= m["top1_accuracy"]
    ranked = ranked_photos(build_index(g), g, c[0], 10)
    assert len({p for p, _ in ranked}) == len(ranked)                          # distinct parent photos only
    assert [s for _, s in ranked] == sorted([s for _, s in ranked], reverse=True)


def test_open_set_false_match_rate():
    g, c = _gallery()
    out = open_set_false_matches(g, [g.vectors[0], unit(1, 77)[0]], [0.95, -2.0])
    assert out["by_threshold"][0]["false_matches"] == 1 and out["by_threshold"][1]["false_match_rate"] == 1.0


def test_roc_and_threshold_for_fmr():
    rng = np.random.default_rng(0)
    gen, imp = rng.normal(0.7, 0.1, 5000), rng.normal(0.1, 0.1, 100000)
    roc = roc_from_scores(gen, imp)
    assert 0.99 < roc["auc"] <= 1.0 and roc["eer"] < 0.01
    t = threshold_for_fmr(imp, 1e-3)
    assert (imp >= t).mean() <= 1e-3 + 1e-9 and (imp >= t).mean() > 0
