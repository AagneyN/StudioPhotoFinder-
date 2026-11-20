"""Download and verify the LFW dataset and the face models.

Usage:
    python scripts/download_dataset.py            # dataset + models
    python scripts/download_dataset.py --models   # models only
    python scripts/download_dataset.py --dataset  # dataset only

Sources (only these, never anything else):
  * LFW original images: official site first, then the figshare mirror that scikit-learn itself uses.
  * Models: OpenCV Zoo (github.com/opencv/opencv_zoo): YuNet (MIT) and SFace (Apache-2.0).
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
import tarfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config.settings import get_settings, setup_logging  # noqa: E402

log = logging.getLogger("download")

LFW_URLS = [
    "https://vis-www.cs.umass.edu/lfw/lfw.tgz",          # official
    "https://ndownloader.figshare.com/files/5976018",    # figshare mirror used by scikit-learn
]
LFW_EXPECTED_IMAGES = 13233

_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    "face_detection_yunet_2023mar.onnx": f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(urls: list[str] | str, dest: Path) -> bool:
    urls = [urls] if isinstance(urls, str) else urls
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            log.info("Downloading %s", url)
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                tmp = dest.with_suffix(dest.suffix + ".part")
                done = 0
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
                print()
                tmp.replace(dest)
            return True
        except requests.RequestException as e:
            log.warning("Failed: %s", e)
    return False


def get_models(models_dir: Path) -> int:
    models_dir.mkdir(parents=True, exist_ok=True)
    rc = 0
    for name, url in MODELS.items():
        dest = models_dir / name
        if dest.exists() and dest.stat().st_size > 1000:
            log.info("Model present: %s (sha256 %s)", name, sha256(dest))
            continue
        if not download(url, dest):
            log.error("Could not download %s. Download it manually from %s and place it in %s", name, url, models_dir)
            rc = 1
        else:
            log.info("Downloaded %s (sha256 %s)", name, sha256(dest))
    return rc


def count_images(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.jpg"))


def get_dataset(dataset_root: Path) -> int:
    if dataset_root.exists() and count_images(dataset_root) >= LFW_EXPECTED_IMAGES:
        log.info("LFW already present at %s (%d images)", dataset_root, count_images(dataset_root))
        return 0
    work = dataset_root.parent / "_download"
    work.mkdir(parents=True, exist_ok=True)
    archive = work / "lfw.tgz"
    if not archive.exists():
        if not download(LFW_URLS, archive):
            log.error(
                "Could not download LFW from any known source. Manual option: download lfw.tgz from "
                "https://vis-www.cs.umass.edu/lfw/ and extract so that %s/<Person_Name>/<Person_Name_0001>.jpg exists.",
                dataset_root)
            return 1
    log.info("Archive sha256: %s", sha256(archive))
    log.info("Extracting (only regular files, no path escapes) ...")
    extract_to = dataset_root.parent
    with tarfile.open(archive) as tf:
        tf.extractall(extract_to, filter="data")   # 'data' filter blocks absolute paths / traversal
    n = count_images(dataset_root)
    log.info("Extracted %d images to %s", n, dataset_root)
    if n != LFW_EXPECTED_IMAGES:
        log.warning("Expected %d images but found %d. This may not be the complete original LFW.", LFW_EXPECTED_IMAGES, n)
    shutil.rmtree(work, ignore_errors=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", action="store_true")
    ap.add_argument("--dataset", action="store_true")
    a = ap.parse_args()
    s = get_settings()
    setup_logging(s, "download")
    both = not (a.models or a.dataset)
    rc = 0
    if a.models or both:
        rc |= get_models(s.models_dir)
    if a.dataset or both:
        rc |= get_dataset(s.dataset_root)
    return rc


if __name__ == "__main__":
    sys.exit(main())
