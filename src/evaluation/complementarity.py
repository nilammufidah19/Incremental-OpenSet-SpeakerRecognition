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
    """SELECTION-ceiling report for two per-space distance matrices.

    Returns accuracies of space A alone, space B alone, and the ORACLE that
    counts a query correct when EITHER space's argmin is correct, plus
    `delta_ceiling` = oracle - acc_a.

    IMPORTANT (corrected 2026-08-15): `delta_ceiling` is the upper bound of
    per-query SELECTION between the two spaces -- it is NOT "the headroom any
    fusion mechanism could ever add", which is how Experiment 4 used it to
    decide gate G4.1. A weighted SUM of the two scores can be right on queries
    where BOTH argmins are wrong, so it can exceed this number: measured
    +0.0333 vs this function's +0.0142 on the exp4 validation task. Use
    `linear_fusion_oracle` for the score-fusion ceiling and report both.
    See docs/experiment-4-reaudit.md §3.
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


def _validate_pair(dist_a: np.ndarray, dist_b: np.ndarray, labels: np.ndarray):
    dist_a = np.asarray(dist_a, dtype=np.float64)
    dist_b = np.asarray(dist_b, dtype=np.float64)
    labels = np.asarray(labels)
    if dist_a.shape != dist_b.shape or len(labels) != len(dist_a):
        raise ValueError(
            f"shape mismatch: {dist_a.shape} vs {dist_b.shape} vs {labels.shape}"
        )
    return dist_a, dist_b, labels


def linear_fusion_feasible(
    dist_a: np.ndarray, dist_b: np.ndarray, labels: np.ndarray, eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-query: is there ANY w in [0, 1] making the true class the argmin of
    `w*d_a + (1-w)*d_b`?  Returns (ok, w_lo, w_hi).

    For a competitor j, the true class t wins iff

        w*(d_a[t] - d_a[j]) + (1-w)*(d_b[t] - d_b[j]) < 0
        <=>  w*C_j + B_j < 0     with  C_j = A_j - B_j,  A_j = d_a[t]-d_a[j],
                                       B_j = d_b[t]-d_b[j]

    which is LINEAR in w, so each competitor contributes a half-line and the
    feasible set is the intersection of all of them with [0, 1] -- an interval
    [w_lo, w_hi), open because the inequality is strict (at the endpoint the
    argmin is a tie). C_j == 0 makes w drop out: that competitor is either
    always beaten (B_j < 0) or never (B_j >= 0, query infeasible).
    """
    dist_a, dist_b, labels = _validate_pair(dist_a, dist_b, labels)
    n, m = dist_a.shape
    rows = np.arange(n)

    A = dist_a[rows, labels][:, None] - dist_a
    B = dist_b[rows, labels][:, None] - dist_b
    C = A - B

    is_true = np.zeros((n, m), dtype=bool)
    is_true[rows, labels] = True          # the true column is trivially 0, skip it

    # C == 0: w cancels out -- competitor must already be beaten in both spaces
    tie = (np.abs(C) <= eps) & ~is_true
    infeasible = (tie & (B >= 0.0)).any(axis=1)

    safe_C = np.where(np.abs(C) > eps, C, 1.0)
    ratio = -B / safe_C
    upper = (C > eps) & ~is_true          # w < ratio
    lower = (C < -eps) & ~is_true         # w > ratio

    w_hi = np.minimum(np.where(upper, ratio, np.inf).min(axis=1), 1.0)
    w_lo = np.maximum(np.where(lower, ratio, -np.inf).max(axis=1), 0.0)
    ok = (w_lo < w_hi) & ~infeasible
    return ok, w_lo, w_hi


def linear_fusion_oracle(
    dist_a: np.ndarray, dist_b: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Exact ceiling of the LINEAR SCORE-FUSION family `w*d_a + (1-w)*d_b`.

    This is NOT the same thing as `oracle_report`'s `delta_ceiling`. That one is
    the oracle of per-query SELECTION between the two spaces (take whichever
    argmin is right). A weighted SUM can additionally be right on queries where
    BOTH argmins are wrong -- e.g. the true class ranks 2nd in both spaces but
    wins once the two scores are added -- so the selection oracle is *not* an
    upper bound for score fusion, and using it as one understates the headroom.

    Re-audit 2026-08-15 measured selection +0.0142 vs linear-fusion +0.0333 on
    the exp4 validation task: a factor of 2.3 (docs/experiment-4-reaudit.md §3).
    Report BOTH; gate the "is fusion worth pursuing" decision on this one, and
    the "does a deployable single weight actually win" decision on
    `best_global_weight` below.
    """
    dist_a, dist_b, labels = _validate_pair(dist_a, dist_b, labels)
    ok_lin, w_lo, w_hi = linear_fusion_feasible(dist_a, dist_b, labels)
    ok_a = dist_a.argmin(axis=1) == labels
    ok_b = dist_b.argmin(axis=1) == labels
    return {
        "acc_a": float(ok_a.mean()),
        "acc_b": float(ok_b.mean()),
        "acc_linear_oracle": float(ok_lin.mean()),
        "delta_linear_ceiling": float(ok_lin.mean() - ok_a.mean()),
        # queries no per-query SELECTION could ever get -- the gap between the
        # two ceilings lives entirely here
        "recovered_both_argmin_wrong": float((ok_lin & ~ok_a & ~ok_b).mean()),
        "n_queries": int(len(labels)),
    }


def best_global_weight(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    labels: np.ndarray,
    grid: np.ndarray | None = None,
) -> dict:
    """Best SINGLE weight over a grid -- the deployable counterpart of
    `linear_fusion_oracle` (one w must serve every query).

    Default grid is [0, 1] at 0.01 resolution. Experiment 4 swept only
    {0.5 ... 0.95} at 6 points and missed the true optimum at w = 0.883
    (docs/experiment-4-reaudit.md §3), so the default deliberately spans the
    full interval including w < 0.5 and the w = 0 / w = 1 single-space ends.
    """
    dist_a, dist_b, labels = _validate_pair(dist_a, dist_b, labels)
    if grid is None:
        grid = np.round(np.arange(0.0, 1.0 + 1e-9, 0.01), 2)
    grid = np.asarray(grid, dtype=np.float64)
    accs = {
        float(w): float(((w * dist_a + (1.0 - w) * dist_b).argmin(axis=1) == labels).mean())
        for w in grid
    }
    best_w = max(accs, key=lambda w: (accs[w], w))  # tie-break toward space A
    return {
        "best_w": float(best_w),
        "best_acc": float(accs[best_w]),
        "acc_a": float(accs[1.0]) if 1.0 in accs else float((dist_a.argmin(axis=1) == labels).mean()),
        "grid_accuracies": accs,
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
