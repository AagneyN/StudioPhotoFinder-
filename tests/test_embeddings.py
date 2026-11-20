"""Embedding properties + the full index -> search pipeline on a tiny LFW fixture (6 identities x 4 photos)."""
import numpy as np
import pytest

from backend.database.connection import create_repository
from backend.database.models import PhotoRecord
from backend.indexing.image_processor import load_image
from backend.indexing.index_manager import IndexManager
from backend.indexing.pipeline import IndexingLock, IndexingPipeline
from backend.errors import FinderError, IndexUnavailableError, NoFaceError
from backend.service import FinderService
from tests.conftest import jpeg_bytes


def embed_first(det, emb, path):
    img = load_image(path)
    return emb.embed_faces(img, det.detect(img)[:1])[0]


def test_embedding_dimension_normalisation_determinism(detector, embedder, lfw_people):
    p = next(iter(lfw_people.values()))[0]
    a, b = embed_first(detector, embedder, p), embed_first(detector, embedder, p)
    assert a.shape == (embedder.s.embedding_dim,) == (128,) and a.dtype == np.float32
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)
    assert np.allclose(a, b)
    assert embedder.info.fingerprint.startswith(embedder.info.name) and str(embedder.info.embedding_dim) in embedder.info.fingerprint


def test_onnxruntime_matches_opencv_reference(detector, embedder, lfw_people):
    img = load_image(next(iter(lfw_people.values()))[0])
    face = detector.detect(img)[0]
    crop = embedder.align(img, face)
    assert crop.shape[:2] == (112, 112)
    assert float(embedder.embed_aligned([crop])[0] @ embedder.embed_reference_cv2(crop).ravel()) > 0.9999


def test_same_person_scores_higher_than_different_people(detector, embedder, lfw_people):
    E = {n: np.vstack([embed_first(detector, embedder, p) for p in paths]) for n, paths in lfw_people.items()}
    names = list(E)
    same = [float(E[n][0] @ E[n][i]) for n in names for i in range(1, 4)]
    diff = [float(E[n][0] @ E[m][i]) for n in names for m in names if m != n for i in range(4)]
    assert np.mean(same) > np.mean(diff) + 0.2


def test_provider_reported_and_cpu_fallback_works(settings, models_ok):
    import dataclasses
    from backend.recognition.embedder import FaceEmbedder
    cpu = FaceEmbedder(dataclasses.replace(settings, execution_provider="cpu"))
    assert cpu.info.execution_provider == "CPUExecutionProvider"
    auto = FaceEmbedder(dataclasses.replace(settings, execution_provider="auto"))
    assert auto.info.execution_provider in ("CPUExecutionProvider", "CUDAExecutionProvider")
    if auto.info.execution_provider == "CPUExecutionProvider":
        assert auto.info.fallback_reason                 # the fallback must be explained, never silent


def build_records(lfw_people, per_identity_gallery=3):
    gal, qry = [], []
    for n, paths in lfw_people.items():
        for i, p in enumerate(paths):
            r = PhotoRecord("lfw", f"{n}/{p.name}", str(p), p.name, n, file_size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
            (gal if i < per_identity_gallery else qry).append(r)
    return gal, qry


@pytest.fixture()
def indexed(settings, detector, embedder, lfw_people, tmp_path):
    gal, qry = build_records(lfw_people)
    bad = tmp_path / "corrupt.jpg"
    bad.write_bytes(b"\xff\xd8\xff" + b"junk" * 50)
    blank = tmp_path / "blank.jpg"
    blank.write_bytes(jpeg_bytes(np.full((250, 250, 3), 127, np.uint8)))
    extra = [PhotoRecord("lfw", "zz/corrupt.jpg", str(bad), "corrupt.jpg", "zz", file_size=bad.stat().st_size, mtime_ns=1),
             PhotoRecord("lfw", "zz/blank.jpg", str(blank), "blank.jpg", "yy", file_size=blank.stat().st_size, mtime_ns=1)]
    repo = create_repository(settings)
    pipe = IndexingPipeline(settings, repo, detector, embedder)
    st = pipe.run(gal + extra)
    return settings, repo, pipe, gal, qry, st


def test_pipeline_counts_corrupt_and_noface_and_persists(indexed):
    settings, repo, pipe, gal, qry, st = indexed
    assert st["state"] == "completed" and st["failed"] == 1 and st["no_face"] == 1
    assert st["photos_indexed"] == len(gal)
    s = repo.stats()
    assert s.photos_failed == 1 and s.photos_no_face == 1 and s.faces >= len(gal)
    assert settings.index_path.exists() and settings.status_path.exists()


def test_incremental_run_skips_unchanged_and_detects_changed(indexed):
    settings, repo, pipe, gal, qry, st = indexed
    st2 = pipe.run(gal)
    assert st2["to_process"] == 0 and st2["skipped_unchanged"] == len(gal)
    changed = [PhotoRecord(**{**gal[0].__dict__, "mtime_ns": gal[0].mtime_ns + 1})]
    st3 = pipe.run(changed + gal[1:])
    assert st3["to_process"] == 1 and st3["photos_indexed"] == 1
    assert repo.stats().photos_indexed == len(gal)             # re-indexed in place, no duplicate


def test_prune_removes_deleted_photos_from_db_and_index(indexed):
    settings, repo, pipe, gal, qry, st = indexed
    before = repo.stats().faces
    pipe.run(gal[1:], prune=True)
    assert repo.stats().photos_indexed == len(gal) - 1 and repo.stats().faces < before
    mgr = IndexManager(settings, repo, pipe.emb.info.fingerprint, pipe.emb.info.name)
    assert mgr.load_strict().ntotal == repo.stats().faces        # index and DB still consistent


def test_interrupted_run_resumes(settings, detector, embedder, lfw_people):
    gal, _ = build_records(lfw_people)
    repo = create_repository(settings)
    pipe = IndexingPipeline(settings, repo, detector, embedder)
    pipe.run(gal, limit=5)                                       # "interrupted" after 5 images
    assert repo.stats().photos_indexed == 5
    st = pipe.run(gal)                                           # resume
    assert st["skipped_unchanged"] == 5 and repo.stats().photos_indexed == len(gal)


def test_crash_between_db_commit_and_index_save_is_recovered(indexed):
    settings, repo, pipe, gal, qry, st = indexed
    settings.index_path.unlink()                                 # lose the FAISS file entirely
    pipe2 = IndexingPipeline(settings, repo, pipe.det, pipe.emb)
    pipe2.run(gal)                                               # rebuilds from DB embeddings, no re-embedding needed
    assert IndexManager(settings, repo, pipe.emb.info.fingerprint, pipe.emb.info.name).load_strict().ntotal == repo.stats().faces


def test_lock_prevents_second_writer(settings):
    with IndexingLock(settings.lock_path):
        with pytest.raises(FinderError, match="already running"):
            with IndexingLock(settings.lock_path):
                pass
    assert not settings.lock_path.exists()


def test_end_to_end_search_retrieves_same_identity(indexed, detector, embedder, lfw_people):
    settings, repo, pipe, gal, qry, st = indexed
    svc = FinderService(settings, repo, detector, embedder)
    assert svc.status()["index_loaded"]
    correct = 0
    for r in qry:
        res = svc.search(open(r.abs_path, "rb").read(), top_k=10)
        scores = [x["score"] for x in res["results"]]
        assert scores == sorted(scores, reverse=True) and len(res["results"]) <= 10
        assert all(k in res["results"][0] for k in ("filename", "identity", "location", "score", "photo_id"))
        assert "embedding" not in str(res["results"][0]).lower()          # biometric vectors never leave the service
        correct += res["results"][0]["identity"] == r.identity
    assert correct >= len(qry) - 1                                         # 6 identities: expect (almost) perfect top-1
    assert "not a verified identity" in res["disclaimer"]


def test_search_min_score_filters_and_reload(indexed, detector, embedder):
    settings, repo, pipe, gal, qry, st = indexed
    svc = FinderService(settings, repo, detector, embedder)
    data = open(qry[0].abs_path, "rb").read()
    assert svc.search(data, min_score=0.99)["results"] == []
    n = svc.status()["index_vectors"]
    assert svc.reload_index()["index_vectors"] == n


def test_search_errors_are_informative(indexed, detector, embedder, blank_jpeg, settings, lfw_people):
    settings, repo, pipe, gal, qry, st = indexed
    svc = FinderService(settings, repo, detector, embedder)
    with pytest.raises(NoFaceError):
        svc.search(blank_jpeg)
    with pytest.raises(ValueError, match="face_index"):
        svc.search(open(qry[0].abs_path, "rb").read(), face_index=9)
    settings.index_path.unlink()
    svc.reload_index()
    assert "not found" in svc.index_error
    with pytest.raises(IndexUnavailableError):
        svc.search(open(qry[0].abs_path, "rb").read())


def test_multiple_faces_query_lets_user_choose(indexed, detector, embedder, lfw_people):
    import cv2
    settings, repo, pipe, gal, qry, st = indexed
    svc = FinderService(settings, repo, detector, embedder)
    names = list(lfw_people)
    a, b = load_image(qry[0].abs_path), None
    other = next(q for q in qry if q.identity != qry[0].identity)
    b = load_image(other.abs_path)
    both = jpeg_bytes(np.hstack([a, b]))
    q = svc.detect_faces(both)
    assert len(q.faces) >= 2
    got = set()
    for i in range(2):
        res = svc.search(both, face_index=i, top_k=3)
        assert res["query"]["faces_detected"] >= 2 and res["query"]["face_used"] == i
        got.add(res["results"][0]["identity"])
    assert got == {qry[0].identity, other.identity}                       # each selected face finds its own person
