# Development status (2026-11-18)

## Completed and verified
- Full LFW (13,233 images) present; YuNet + SFace models with licenses documented.
- Detection (no face / multi face / tiny faces), 128-d L2-normalised embeddings, cosine search via FAISS.
- SQLite metadata behind a repository layer; index save/reload; model-fingerprint guard; index/DB mismatch repair.
- Resumable, incremental indexing with checkpoints, lock file, corrupt-image handling.
- FastAPI (health, search, search/faces, indexing, evaluation, thumbnails): live search returned the correct identity at rank 1.
- Streamlit UI (Dashboard, Search, Dataset, Evaluation): starts and serves HTTP 200. Not walked through manually.
- Evaluation: reproducible leak-free split; Top-1/5/10, ROC/EER, FMR/FNMR, open-set, latency; real report generated.
- Tests: 43 passed.

## Fixed this session
- `scripts/evaluate_model.py` crashed on a corrupt embedding cache, was very slow/out-of-memory loading it, and had two
  `cache.data[r]` indexing bugs. All fixed; the full evaluation now completes.
- Added the missing README.md, requirements.txt, .env.example, .gitignore.

## Remaining / not done
- GPU inference not enabled or benchmarked (CPU-only onnxruntime installed); no CPU-vs-GPU benchmark.
- Studio archive loader (year/category/photo number, UNC paths): architecture-ready, deliberately not built.
- Authentication, roles, PostgreSQL, backups, LAN deployment.
- No automated tests for the evaluation script itself or the Streamlit UI.
- Open-set rejection is weak (see README caveat).

## Test results
pytest: 43 passed. Evaluation (seed 42): Top-1 97.2%, Top-5 99.0%, Top-10 99.0%, AUC 0.9941, EER 1.32%.

## Completion
Prototype definition of done: 17 of 19 items met. Not met: GPU inference (not attempted) and a manual UI walkthrough
(smoke test only). About 89% by item count.
