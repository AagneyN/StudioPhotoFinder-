"""Embedding normalisation and similarity.

Metric: cosine similarity. Embeddings are L2-normalised once (here), so cosine == inner product,
which lets FAISS IndexFlatIP return exact cosine scores. Scores are in [-1, 1]; higher = more similar.
Scores are model-specific: do not compare them across models or treat them as probabilities.
"""
from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x / max(float(np.linalg.norm(x)), eps)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of a (n,d) and rows of b (m,d) -> (n,m). Inputs need not be normalised."""
    return l2_normalize(np.atleast_2d(a)) @ l2_normalize(np.atleast_2d(b)).T
