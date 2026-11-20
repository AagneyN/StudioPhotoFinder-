import numpy as np

from backend.database.connection import create_repository
from backend.database.models import FaceInsert, PhotoRecord


def rec(name="A/A_0001.jpg", identity="A", year=None, category=None, mtime=1):
    return PhotoRecord("lfw", name, "C:/x/" + name, name.split("/")[-1], identity, year, category, None, 10, mtime)


def face(i=0, seed=0):
    v = np.random.default_rng(seed).normal(size=128).astype(np.float32)
    v /= np.linalg.norm(v)
    return FaceInsert(i, (1, 2, 3, 4), 0.9, 0.8, v.tobytes()), v


def test_persistence_across_instances(settings):
    repo = create_repository(settings)
    f, v = face()
    pid, fids = repo.save_photo_result(rec(), "indexed", None, 250, 250, [f])
    repo2 = create_repository(settings)             # new object, same file: data must persist
    p = repo2.get_photo_by_key("lfw", "A/A_0001.jpg")
    assert p and p.id == pid and p.identity == "A" and p.n_faces == 1
    (ids, vecs), = list(repo2.iter_embeddings())
    assert ids.tolist() == fids and np.allclose(vecs[0], v)


def test_metadata_retrieval_joins_face_and_photo(settings):
    repo = create_repository(settings)
    _, fids = repo.save_photo_result(rec(year=2021, category="Passport"), "indexed", None, 10, 10, [face()[0]])
    fr, pr = repo.get_faces_with_photos(fids)[fids[0]]
    assert fr.bbox == (1, 2, 3, 4) and pr.year == 2021 and pr.category == "Passport"
    assert repo.face_ids_matching(2021, "passport") == fids       # case-insensitive category
    assert repo.face_ids_matching(2020, None) == []


def test_reindex_replaces_faces_and_keeps_photo_id(settings):
    repo = create_repository(settings)
    pid, f1 = repo.save_photo_result(rec(mtime=1), "indexed", None, 1, 1, [face(0, 1)[0], face(1, 2)[0]])
    pid2, f2 = repo.save_photo_result(rec(mtime=2), "indexed", None, 1, 1, [face(0, 3)[0]])
    assert pid == pid2 and len(f2) == 1
    assert repo.all_face_ids() == set(f2)


def test_delete_cascades_and_stats(settings):
    repo = create_repository(settings)
    p1, _ = repo.save_photo_result(rec("A/1.jpg", "A"), "indexed", None, 1, 1, [face(0, 1)[0], face(1, 2)[0]])
    repo.save_photo_result(rec("B/1.jpg", "B"), "no_face", "no face detected", 1, 1, [])
    repo.save_photo_result(rec("C/1.jpg", "C"), "failed", "corrupt", 0, 0, [])
    st = repo.stats()
    assert (st.photos_total, st.photos_indexed, st.photos_no_face, st.photos_failed, st.faces, st.photos_multi_face) == (3, 1, 1, 1, 2, 1)
    removed = repo.delete_photos([p1])
    assert len(removed) == 2 and repo.all_face_ids() == set()


def test_model_fingerprint_meta_roundtrip(settings):
    repo = create_repository(settings)
    assert repo.get_meta("model_fingerprint") is None
    repo.set_meta("model_fingerprint", "m1")
    repo.set_meta("model_fingerprint", "m2")
    assert repo.get_meta("model_fingerprint") == "m2"
