"""Two-backbone complementarity analysis helpers (Experiment 4, exp4a/4b/4c).

Pure numpy utilities shared by scripts/exp4_complementarity_asnorm.py (the
Experiment 4 decision-gate analysis) and, later, Experiment 5's candidate-
backbone screening. Nothing here touches the production decision path --
these functions only *analyze* per-space distance matrices that the caller
computed with the existing machinery (src/prototypical/score_norm.py etc.),
so every prior experiment tag keeps behaving exactly as before.

Conventions (same as the rest of the codebase):
  * scores are DISTANCES -- lower = more genuine, prediction = argmin;
  * a "space" is one backbone's embedding space (native or AS-normalized);
  * matrices are (n_queries, n_prototypes), labels are prototype row indices.
"""
from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization (match ScoreFusionEmbed's per-backbone
    unit-norm convention, so Euclidean distance == monotone cosine distance)."""
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def oracle_report(
    dist_a: np.ndarray, dist_b: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Complementarity-ceiling report for two per-space distance matrices.

    Returns accuracies of space A alone, space B alone, and the ORACLE that
    counts a query correct when EITHER space's argmin is correct (the upper
    bound of any per-query selection between the two spaces), plus
    `delta_ceiling` = oracle - acc_a (the headroom any fusion of B into A
    could ever add, Experiment 4 gate G4.1).
    """
    dist_a = np.asarray(dist_a)
    dist_b = np.asarray(dist_b)
    labels = np.asarray(labels)
    if dist_a.shape != dist_b.shape or len(labels) != len(dist_a):
        raise ValueError(
            f"shape mismatch: {dist_a.shape} vs {dist_b.shape} vs {labels.shape}"
        )
    pred_a = dist_a.argmin(axis=1)
    pred_b = dist_b.argmin(axis=1)
    ok_a = pred_a == labels
    ok_b = pred_b == labels
    return {
        "acc_a": float(ok_a.mean()),
        "acc_b": float(ok_b.mean()),
        "acc_oracle": float((ok_a | ok_b).mean()),
        "delta_ceiling": float((ok_a | ok_b).mean() - ok_a.mean()),
        "b_rescues_a": int((ok_b & ~ok_a).sum()),   # queries only B gets right
        "a_rescues_b": int((ok_a & ~ok_b).sum()),
        "n_queries": int(len(labels)),
    }


def fusion_accuracy(
    dist_a: np.ndarray, dist_b: np.ndarray, labels: np.ndarray, weight_a: float
) -> float:
    """Accuracy of late score fusion  w*d_a + (1-w)*d_b  (argmin decision).

    Both matrices must already live on comparable scales -- for Experiment 4b
    that means each is AS-normalized in its own space first (z-score units),
    which is exactly what made naive raw-distance fusion ill-posed in exp2.
    """
    fused = weight_a * np.asarray(dist_a) + (1.0 - weight_a) * np.asarray(dist_b)
    return float((fused.argmin(axis=1) == np.asarray(labels)).mean())


def detection_scores(
    dist_a: np.ndarray, dist_b: np.ndarray, disagree_penalty: float = 0.5
) -> dict[str, np.ndarray]:
    """Per-query open-set detection scores (lower = accept) from two spaces.

    Rules (all training-free, Experiment 4c):
      a_only / b_only : best distance in one space (baselines);
      mean / max      : symmetric combinations of the two best distances
                        ("max" = both spaces must look genuine);
      a_disagree      : space-A score plus a fixed penalty when the two
                        spaces disagree on WHO the nearest speaker is --
                        the "second witness" heuristic.
    """
    dist_a = np.atleast_2d(np.asarray(dist_a, dtype=np.float64))
    dist_b = np.atleast_2d(np.asarray(dist_b, dtype=np.float64))
    best_a = dist_a.min(axis=1)
    best_b = dist_b.min(axis=1)
    disagree = (dist_a.argmin(axis=1) != dist_b.argmin(axis=1)).astype(np.float64)
    return {
        "a_only": best_a,
        "b_only": best_b,
        "mean": 0.5 * (best_a + best_b),
        "max": np.maximum(best_a, best_b),
        "a_disagree": best_a + disagree_penalty * disagree,
    }
