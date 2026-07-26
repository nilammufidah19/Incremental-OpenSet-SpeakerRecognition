"""Prototype update rule (F6-02/03) [3.8, Pers. 4.4-4.6].

    z̄_new  = (1/m) * sum_i f_phi(x_new,i)                          (4.4)
    n_k(t) = n_k(t-1) + m                                          (4.5)
    c_k(t) = [n_k(t-1)*c_k(t-1) + m*z̄_new] / n_k(t)                (4.6)

This is a running-average (weighted, unbiased) update -- NOT an exponential
moving average -- so it needs no decay hyperparameter and is mathematically
identical to recomputing the mean from the full sample history each time
(see tests/test_continual.py::test_update_prototype_matches_recompute_from_scratch,
F6-07). Only `(c_k, n_k)` is kept between updates; raw embeddings are
never retained (src/continual/speaker_database.py).
"""
from __future__ import annotations

import numpy as np


def new_sample_mean(new_embeddings: np.ndarray) -> np.ndarray:
    """Pers. 4.4: mean embedding of m newly observed samples."""
    if len(new_embeddings) == 0:
        raise ValueError("new_embeddings must contain at least one sample")
    return np.mean(new_embeddings, axis=0)


def update_prototype(
    old_prototype: np.ndarray, old_n: int, new_embeddings: np.ndarray
) -> tuple[np.ndarray, int]:
    """Pers. 4.5-4.6: update an existing prototype with `new_embeddings`
    (m new samples for the same speaker). Returns (new_prototype, new_n)."""
    if old_n < 0:
        raise ValueError(f"old_n must be >= 0, got {old_n}")
    m = len(new_embeddings)
    z_new = new_sample_mean(new_embeddings)
    new_n = old_n + m
    new_prototype = (old_n * old_prototype + m * z_new) / new_n
    return new_prototype.astype(np.float32), new_n


def initialize_prototype(embeddings: np.ndarray) -> tuple[np.ndarray, int]:
    """Pers. 4.4 applied directly to initialize a brand-new speaker's
    prototype: c_k(0) = mean of their first support-set embeddings,
    n_k(0) = number of those embeddings."""
    prototype = new_sample_mean(embeddings)
    return prototype.astype(np.float32), len(embeddings)
