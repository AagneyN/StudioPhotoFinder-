"""STUDIO PHOTO FINDER: Streamlit front end. A pure HTTP client of the FastAPI backend (no recognition code here),
so it can be replaced by a React app without touching the engine."""
from __future__ import annotations

import io
import os

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

API = os.getenv("API_URL", f"http://127.0.0.1:{os.getenv('API_PORT', '8000')}").rstrip("/")

st.set_page_config(page_title="Studio Photo Finder", page_icon="📷", layout="wide")

st.markdown("""
<style>
  .stApp { background: #f6f5f2; }
  .stApp, .stApp p, .stApp label, .stApp span, .stApp li, .stApp div[data-testid="stMetricValue"], .stApp div[data-testid="stMetricLabel"] { color: #1d1d1b; }
  .stApp [data-testid="stCaptionContainer"], .stApp small { color: #55524b; }
  html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }
  h1, h2, h3 { letter-spacing: -0.01em; color: #1d1d1b; }
  section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e6e3dd; }
  div[data-testid="stMetric"] { background: #fff; border: 1px solid #e6e3dd; border-radius: 10px; padding: 14px 16px; }
  .brand { font-weight: 700; letter-spacing: .14em; font-size: 1.05rem; color: #1d1d1b; }
  .brand small { display:block; font-weight: 400; letter-spacing: .02em; color: #77736b; font-size: .78rem; margin-top: 2px; }
  .notice { background:#fff8e6; border:1px solid #f0dca6; border-radius:8px; padding:10px 14px; color:#6b5413; font-size:.9rem; }
  .card-meta { font-size:.82rem; color:#55524b; line-height:1.35; }
  .score { font-weight:700; font-size:1.05rem; color:#1d1d1b; }
  .pill { display:inline-block; padding:1px 8px; border-radius:99px; font-size:.72rem; border:1px solid #d8d4cb; color:#55524b; background:#fff; }
  .pill.ok { border-color:#bcd9c2; color:#2f6b3c; background:#eef7f0; }
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- API helpers
def api(method: str, path: str, **kw):
    """Returns (json | None, error message | None). Never raises into the UI."""
    try:
        r = requests.request(method, f"{API}{path}", timeout=kw.pop("timeout", 120), **kw)
    except requests.RequestException:
        return None, f"Cannot reach the API at {API}. Start it with `python main.py` (or launch everything with `python app.py`)."
    try:
        body = r.json()
    except ValueError:
        body = {}
    if r.status_code >= 400:
        return None, body.get("detail", f"Request failed ({r.status_code})")
    return body, None


@st.cache_data(ttl=600, show_spinner=False)
def fetch_bytes(path: str) -> bytes | None:
    try:
        r = requests.get(f"{API}{path}", timeout=30)
        return r.content if r.status_code == 200 else None
    except requests.RequestException:
        return None


def draw_faces(img: Image.Image, faces: list[dict], selected: int) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    lw = max(2, out.width // 200)
    for f in faces:
        x, y, w, h = f["bbox"]
        sel = f["index"] == selected
        d.rectangle([x, y, x + w, y + h], outline="#d9480f" if sel else "#9a968c", width=lw)
        d.text((x + 4, max(y - 14, 0)), str(f["index"] + 1), fill="#d9480f" if sel else "#9a968c")
    return out


# --------------------------------------------------------------------------- pages
def page_dashboard():
    st.title("Dashboard")
    st_, err = api("GET", "/status")
    if err:
        st.error(err)
        h, _ = api("GET", "/health")
        if h and h.get("error"):
            st.warning(h["error"])
        return
    s = st_["stats"]
    c = st.columns(4)
    c[0].metric("Indexed photographs", f"{s['photos_indexed']:,}")
    c[1].metric("Detected faces", f"{s['faces']:,}")
    c[2].metric("Identities", f"{s['identities']:,}")
    c[3].metric("Photos with several faces", f"{s['photos_multi_face']:,}")
    c = st.columns(4)
    c[0].metric("No face detected", f"{s['photos_no_face']:,}")
    c[1].metric("Failed / corrupted", f"{s['photos_failed']:,}")
    c[2].metric("Vectors in search index", f"{st_['index_vectors']:,}")
    c[3].metric("Execution provider", st_["model"]["execution_provider"].replace("ExecutionProvider", ""))
    if st_["index_error"]:
        st.error(st_["index_error"])
    elif not st_["index_loaded"]:
        st.warning("No search index is loaded yet. Go to Dataset Management and start indexing.")
    job, _ = api("GET", "/index/status")
    if job:
        j = job["job"]
        st.subheader("Indexing status")
        st.write(f"**{j.get('state', 'never_run').replace('_', ' ').title()}**"
                 + (f": {j['processed']:,} of {j.get('to_process', 0):,} processed" if "processed" in j else ""))
    st.subheader("Recognition model")
    m = st_["model"]
    st.write(f"**Model:** {m['name']} ({m['embedding_dim']}-dimensional embeddings, cosine similarity)  \n"
             f"**Detector:** {st_['detector']}  \n**Execution provider:** {m['execution_provider']}")
    if m.get("fallback_reason"):
        st.info(m["fallback_reason"])
    thr = st_["threshold"]
    st.write("**Similarity threshold:** " + (f"{thr:.3f} ({st_['threshold_source']})" if thr is not None
                                              else "none (not calibrated; results are rankings only)"))


def page_search():
    st.title("Search")
    st.markdown('<div class="notice">Results are ranked by <b>facial similarity</b>. They are leads for a person to check, '
                'not verified identities.</div>', unsafe_allow_html=True)
    stat, _ = api("GET", "/status")
    left, right = st.columns([1, 1.4], gap="large")
    data, fname = None, None
    with left:
        up = st.file_uploader("Reference photograph (drag and drop, or browse)", type=["jpg", "jpeg", "png"])
        samples, _ = api("GET", "/samples")
        pick = None
        if samples and samples.get("samples"):
            with st.expander("Or try a held-out sample (never in the index)"):
                labels = {f"{x['identity'].replace('_', ' ')}  -  {x['rel_path'].split('/')[-1]}": x["rel_path"] for x in samples["samples"]}
                choice = st.selectbox("Sample query", ["-"] + list(labels))
                if choice != "-":
                    pick = labels[choice]
        if up is not None:
            data, fname = up.getvalue(), up.name
        elif pick:
            data, fname = fetch_bytes(f"/samples/image?rel_path={requests.utils.quote(pick)}"), pick.split("/")[-1]
        top_k = st.slider("Maximum number of results", 1, 50, 10)
        thr_default = (stat or {}).get("threshold")
        use_thr = st.checkbox("Apply a similarity threshold", value=False,
                              help="Hide results scoring below the value. Scores are model-specific; see the Evaluation page.")
        min_score = None
        if use_thr:
            min_score = st.number_input("Minimum similarity (cosine)", -1.0, 1.0,
                                        float(round(thr_default, 3)) if thr_default is not None else 0.4, 0.01)

    faces, face_idx = [], 0
    with right:
        if data is None:
            st.info("Upload a photograph to begin.")
        else:
            info, err = api("POST", "/search/faces", files={"file": (fname, data)})
            if err:
                st.error(err)
            elif not info["faces"]:
                st.warning("No usable face was found in this photograph. Try a clearer, front-facing photo.")
            else:
                faces = info["faces"]
                if len(faces) > 1:
                    st.warning(f"{len(faces)} faces detected. Choose the face to search for.")
                    face_idx = st.radio("Face to search", [f["index"] for f in faces],
                                        format_func=lambda i: f"Face {i + 1}  (quality {faces[i]['quality']:.2f})", horizontal=True)
                img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
                st.image(draw_faces(img, faces, face_idx) if len(faces) > 1 else img, caption=fname, use_container_width=True)

    if data is not None and faces:
        if st.button("Search", type="primary"):
            with st.spinner("Searching..."):
                res, err = api("POST", "/search", files={"file": (fname, data)},
                               data={"face_index": face_idx, "top_k": top_k, **({"min_score": min_score} if min_score is not None else {})})
            st.session_state["result"] = (res, err)
    out = st.session_state.get("result")
    if out and data is not None and faces:
        res, err = out
        if err:
            st.error(err)
            return
        show_results(res)


def show_results(res: dict):
    t = res["timing_ms"]
    st.caption(f"{len(res['results'])} results in {t['total']:.0f} ms (search {t['search']:.1f} ms). {res['threshold_note']}")
    if not res["results"]:
        st.warning("No photographs matched. Try lowering the threshold, choosing another face, or using a clearer photo.")
        return
    cols_n = 5
    for row_start in range(0, len(res["results"]), cols_n):
        cols = st.columns(cols_n)
        for col, r in zip(cols, res["results"][row_start:row_start + cols_n]):
            with col:
                tb = fetch_bytes(f"/photos/{r['photo_id']}/thumbnail")
                if tb:
                    st.image(tb, use_container_width=True)
                else:
                    st.warning("Original file unavailable")
                badge = ("" if r["meets_threshold"] is None else
                         ' <span class="pill ok">above threshold</span>' if r["meets_threshold"] else ' <span class="pill">below threshold</span>')
                st.markdown(
                    f'<div class="score">#{r["rank"]} &nbsp; {r["score"]:.3f}{badge}</div>'
                    f'<div class="card-meta"><b>{(r["identity"] or "-").replace("_", " ")}</b><br>{r["filename"]}<br>'
                    f'<span class="pill">candidate, not verified</span>'
                    f'{" &nbsp;<span class=pill>" + str(r["faces_in_photo"]) + " faces in photo</span>" if r["faces_in_photo"] > 1 else ""}'
                    f'{"<br>" + str(r["year"]) + " / " + str(r["category"]) if r["year"] else ""}'
                    f'{"<br>No. " + r["photo_number"] if r["photo_number"] else ""}</div>', unsafe_allow_html=True)
                with st.expander("Larger view"):
                    full = fetch_bytes(f"/photos/{r['photo_id']}/image")
                    if full:
                        st.image(full, use_container_width=True)
                    st.caption(f"Location: {r['location']}  \nFace detection score: {r['face_det_score']}")


def page_dataset():
    st.title("Dataset Management")
    job, err = api("GET", "/index/status")
    if err:
        st.error(err)
        return
    st.write(f"**Dataset location:** `{job['dataset_root']}`" + ("" if job["dataset_exists"] else "  \n:red[Folder not found. Run `python scripts/download_dataset.py`.]"))

    @st.fragment(run_every=2)
    def live():
        j, e = api("GET", "/index/status")
        if e:
            st.error(e)
            return
        jb = j["job"]
        state = jb.get("state", "never_run")
        total = max(jb.get("to_process", 0), 1)
        prog = min(jb.get("processed", 0) / total, 1.0) if state != "never_run" else 0.0
        st.progress(prog, text=f"{state.replace('_', ' ').title()}: {jb.get('processed', 0):,} / {jb.get('to_process', 0):,} images"
                    + (f"  |  {jb.get('images_per_sec', 0)} img/s  |  ETA {jb['eta_sec']}s" if state == "running" and jb.get("eta_sec") is not None else ""))
        c = st.columns(6)
        for col, (label, key) in zip(c, [("Indexed", "photos_indexed"), ("Faces added", "faces_added"), ("No face", "no_face"),
                                         ("Failed", "failed"), ("Multi-face", "multi_face"), ("Skipped (unchanged)", "skipped_unchanged")]):
            col.metric(label, f"{jb.get(key, 0):,}")
        if jb.get("message"):
            (st.error if state == "failed" else st.info)(jb["message"])
        if jb.get("recent_errors"):
            with st.expander(f"Errors ({len(jb['recent_errors'])} most recent)"):
                st.code("\n".join(jb["recent_errors"]))
        if state == "completed" and st.session_state.get("last_done") != jb.get("started_at"):
            st.session_state["last_done"] = jb.get("started_at")
            api("POST", "/index/reload")
            st.rerun()

    live()
    st_, _ = api("GET", "/status")
    if st_:
        s = st_["stats"]
        st.write(f"**Index:** {st_['index_vectors']:,} vectors, {s['photos_indexed']:,} photos, {s['identities']:,} identities, "
                 f"{s['photos_failed']:,} failed")
        if st_["index_error"]:
            st.error(st_["index_error"])
    st.subheader("Actions")
    running = job["running"]
    c = st.columns(4)
    if c[0].button("Start / resume indexing", disabled=running, type="primary",
                   help="Indexes the gallery split. Already indexed, unchanged images are skipped."):
        _, e = api("POST", "/index/start", json={"split": "gallery"})
        st.error(e) if e else st.rerun()
    if c[1].button("Cancel job", disabled=not running):
        _, e = api("POST", "/index/cancel")
        st.error(e) if e else st.rerun()
    if c[2].button("Reload index"):
        r, e = api("POST", "/index/reload")
        st.error(e) if e else st.success(f"Index reloaded: {r['index_vectors']:,} vectors") if r["index_loaded"] else st.warning(r["index_error"])
    if c[3].button("Repair index", disabled=running, help="Resynchronise the FAISS index from the database."):
        r, e = api("POST", "/index/reload", params={"repair": "true"})
        st.error(e) if e else st.success(f"Repaired: {r['index_vectors']:,} vectors")
    with st.expander("Rebuild from scratch"):
        st.write("Deletes the existing index and database, then indexes again.")
        ok = st.checkbox("I understand the current index will be deleted")
        if st.button("Rebuild index", disabled=running or not ok):
            _, e = api("POST", "/index/start", json={"split": "gallery", "rebuild": True})
            st.error(e) if e else st.rerun()


def page_evaluation():
    st.title("Evaluation")
    r, err = api("GET", "/evaluation/report")
    if err:
        st.error(err)
        return
    if not r["available"]:
        st.warning(r["message"])
        return
    rep = r["report"]
    ident, ver, tim, sp = rep["identification"], rep["verification"], rep["timing"], rep["split"]
    st.caption(f"Generated {rep['generated_at']}  |  model {rep['model']['name']}  |  split seed {rep['seed']}  |  "
               f"{sp['gallery_photos']:,} gallery photos / {sp['known_queries']:,} known queries / {sp['unknown_queries']:,} unknown queries")
    st.subheader("Closed-set identification (query identity is in the gallery)")
    c = st.columns(3)
    for col, k in zip(c, (1, 5, 10)):
        col.metric(f"Top-{k} accuracy", f"{ident[f'top{k}_accuracy'] * 100:.1f}%")
    st.caption(f"Queries with no detected face count as misses. Among detected queries ({ident['n_detected']:,} of {ident['n_queries']:,}): "
               f"top-1 {ident['top1_accuracy_detected_only'] * 100:.1f}%, top-5 {ident['top5_accuracy_detected_only'] * 100:.1f}%, "
               f"top-10 {ident['top10_accuracy_detected_only'] * 100:.1f}%.  Per-identity (macro) top-1: {ident['top1_accuracy_macro_per_identity'] * 100:.1f}%.")
    st.subheader("False matches and verification")
    roc = ver["test_roc"]
    c = st.columns(2)
    c[0].metric("Verification AUC", f"{roc['auc']:.4f}")
    c[1].metric("Equal error rate", f"{roc['eer'] * 100:.2f}%")
    rows = [{"Calibration target FMR": f"{o['target_fmr_on_calibration']:g}", "Threshold": o["threshold"],
             "Test false match rate": f"{o['false_match_rate']:.4%}", "Test false non-match rate": f"{o['false_non_match_rate']:.2%}"}
            for o in ver["operating_points"]]
    st.table(rows)
    osr = rep["open_set"]
    st.write(f"**Open-set** ({osr['n_unknown_queries']:,} people not in the gallery; any returned match is a false match):")
    st.table([{"Threshold": o["threshold"], "False matches": o["false_matches"], "Rate": f"{o['false_match_rate']:.3%}"} for o in osr["by_threshold"]])
    st.subheader("Speed")
    c = st.columns(4)
    fmt = lambda v, u: "n/a" if v is None else f"{v:.1f} {u}"  # noqa: E731
    c[0].metric("Detection / image", fmt(tim["detect_ms_per_image_mean"], "ms"))
    c[1].metric("Embedding / image", fmt(tim["embed_ms_per_image_mean"], "ms"))
    c[2].metric("End-to-end query", fmt(tim["end_to_end_query_ms_mean"], "ms"))
    c[3].metric("Indexing throughput", fmt(tim["indexing_images_per_sec"], "img/s"))
    st.caption(f"Execution provider: {tim['execution_provider']}. Search over the gallery alone: "
               f"{ident.get('search_latency_ms_mean', float('nan')):.2f} ms mean, {ident.get('search_latency_ms_p95', float('nan')):.2f} ms p95.")
    with st.expander("Methodology"):
        st.write("\n".join(f"- {m}" for m in rep["methodology"]))
    with st.expander("Limitations", expanded=True):
        st.write("\n".join(f"- {m}" for m in rep["limitations"]))


with st.sidebar:
    st.markdown('<div class="brand">STUDIO PHOTO FINDER<small>Facial similarity search for studio archives</small></div>', unsafe_allow_html=True)
    st.write("")
    page = st.radio("Navigate", ["Dashboard", "Search", "Dataset Management", "Evaluation"], label_visibility="collapsed")
    st.caption("All processing is local. No photographs leave this machine.")

{"Dashboard": page_dashboard, "Search": page_search, "Dataset Management": page_dataset, "Evaluation": page_evaluation}[page]()
