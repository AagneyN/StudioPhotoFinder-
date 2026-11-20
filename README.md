# STUDIO PHOTO FINDER

Local face-recognition photo retrieval for a photography studio. An employee uploads a reference photo; the system detects
the face, computes an embedding, searches a FAISS index and shows the most similar photos (year, category, filename,
path, similarity score). **Results are a similarity ranking for a human to verify, never a confirmed identity.**

This prototype is demonstrated on the public LFW dataset. The real studio archive is **not** processed yet.

## Architecture
```
Streamlit UI (frontend/)  --HTTP-->  FastAPI (backend/api)  -->  backend/service.py
                                                                  |- recognition/  YuNet detector, SFace embedder, cosine similarity
                                                                  |- indexing/     dataset loader, resumable pipeline, FAISS index manager, eval splits
                                                                  '- database/     SQLite via repository layer (swap for PostgreSQL later)
scripts/  download_dataset.py | index_dataset.py | evaluate_model.py   (independent of the UI)
```
The dataset loader is separate from the recognition engine, so a studio loader (year/category folders) can replace LFW.
Embeddings never leave the server: API responses contain no vectors. Index files carry a model fingerprint
(`SFace-2021dec:128:l2norm:cosine`); an index built with another model is refused.

## Stack
Python 3.12, OpenCV (YuNet + SFace via ONNX), onnxruntime, FAISS (`IndexFlatIP` on L2-normalised vectors = cosine),
SQLite, FastAPI, Streamlit, pytest.

## Licensing (checked)
| Item | License |
|---|---|
| YuNet face detector (OpenCV Zoo) | MIT - `docs/licenses/YuNet_MIT.txt` |
| SFace recognizer weights (OpenCV Zoo) | Apache-2.0 - `docs/licenses/SFace_Apache-2.0.txt` |
| LFW dataset | Research use; see https://vis-www.cs.umass.edu/lfw/ |

InsightFace's popular pretrained packs (buffalo_l etc.) are non-commercial, so they were **not** used. SFace is
permissively licensed but less accurate than those models. Re-check licenses before any commercial deployment.
Real deployment also needs customer consent and compliance with local biometric/data-protection law.

## Install (Windows, PowerShell)
```powershell
cd "path\to\project"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
copy .env.example .env                 # optional
```
## Download dataset and models
```powershell
python scripts/download_dataset.py
```
Downloads the full original LFW (13,233 images, 5,749 identities; official site, figshare mirror as fallback) and
the two ONNX models (sha256 logged) into `data/datasets/lfw` and `models/`. It is not the filtered scikit-learn subset.

## Index the dataset
```powershell
python scripts/index_dataset.py            # gallery split only (default, needed for honest evaluation)
python scripts/index_dataset.py --split all
python scripts/index_dataset.py --rebuild  # start over
python scripts/index_dataset.py --repair   # resync FAISS and database after a crash
```
Shows progress, counts of failed / no-face / multi-face images, checkpoints every 500 images, resumes after
interruption, skips unchanged files and removes deleted ones. The index reloads from disk on start.

## Run
```powershell
python app.py        # API + UI, opens http://127.0.0.1:8501  (Ctrl+C stops both)
python main.py       # API only; docs at http://127.0.0.1:8000/docs
```
Pages: Dashboard, Search (drag and drop, face picker when several faces, top-k, threshold), Dataset Management, Evaluation.

## Evaluate
```powershell
python scripts/evaluate_model.py                          # full, deterministic (seed 42)
python scripts/evaluate_model.py --limit-identities 200   # quick run
```
Writes `data/processed/evaluation_report.json/.csv` and `calibration.json`. Methodology: per identity, photos are split
deterministically into gallery and query; query photos are never in the gallery (leak test included); half of the
single-photo identities are held out of the gallery and used as "unknown" queries for the open-set test. A hit counts only if the returned photo
has the query's identity. Verification uses query-vs-gallery genuine/impostor pairs, thresholds calibrated on
gallery-only pairs.

### Results (seed 42, CPU, this machine)
Gallery 8,710 photos / 10,341 faces; 2,499 known queries, 2,024 unknown queries.

| Metric | Value |
|---|---|
| Top-1 / Top-5 / Top-10 accuracy | 97.2% / 99.0% / 99.0% |
| Verification AUC / EER | 0.9941 / 1.32% |
| FMR / FNMR at threshold 0.2975 | 0.93% / 1.35% |
| FMR / FNMR at threshold 0.3644 | 0.11% / 2.10% |
| Search latency (mean / p95) | 2.3 ms / 2.8 ms |
| End-to-end query incl. detect+embed | 23 ms mean |
| Indexing throughput | ~72 images/s |

Caveats:
- The open-set test (unknown person searched against the whole gallery) gave a **93.5% false-match rate even at the
  0.1%-FMR threshold**: a query is compared against ~10k faces and a single exceedance counts. A score threshold alone
  cannot say "this person is not in the archive"; employees must verify visually.
- LFW is mostly celebrity photos in unconstrained settings; these numbers do not predict accuracy on studio
  passport photos. Re-calibrate on real data.

## GPU configuration
Only CPU is verified: the installed onnxruntime exposes `CPUExecutionProvider` only. The app reports the provider and
any fallback reason in `/health` and on the Dashboard. CPU speed already suffices at this scale. To try the GPU, install
`onnxruntime-gpu` instead of `onnxruntime` with matching CUDA 12 / cuDNN, then set `EXECUTION_PROVIDER=cuda`.
This is untested on the RTX 3050; if initialisation fails the app falls back to CPU.

## Tests
```powershell
python -m pytest -q          # 43 tests, ~7 s, small fixtures
```

## Troubleshooting
- "Index missing / mismatch": `python scripts/index_dataset.py --repair` or `--rebuild`.
- "Dataset not found": run the download script or fix `DATASET_ROOT`.
- Corrupt `data/processed/eval_embeddings.npz`: discarded automatically and rebuilt (~6 min on CPU).
- Port in use: change `API_PORT` / `STREAMLIT_PORT` in `.env`.
- "Another indexer is running": a lock protects the index; wait, or remove a stale lock.

## Limitations
No authentication or roles; single SQLite writer; LFW only; year/category/photo-number extraction not implemented yet
(fields exist and return null); GPU untested; SFace is less accurate than larger models.

## Future studio integration
Add a loader that scans `\\STUDIO-PC\StudioPhotos\<year>\<Passport|Studio>` recursively, extracts year, category and
photo number, and stores original network paths (no copies). Incremental indexing, modified/deleted-file detection and
multi-face photos already exist in the pipeline. Then add authentication and roles, year/category filters, PostgreSQL,
backups, handling of an unavailable share, LAN-only binding (never the internet), and optionally a React front end on
the same API.
