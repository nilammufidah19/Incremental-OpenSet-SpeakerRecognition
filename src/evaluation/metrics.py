"""Evaluation metrics (F8-01..04) [3.9, Pers. 3.11, 3.14, 3.15].

EER (Pers. 3.12-3.13) is intentionally NOT reimplemented here -- it's
already implemented once in src/prototypical/calibration.py
(`find_eer_threshold`/`compute_far_frr`), and F8-02 just re-exports it under
the evaluation module so callers doing session-by-session reporting don't
need to import from two different subsystems for what is the same formula.
"""
from __future__ import annotations

import numpy as np

from src.prototypical.calibration import compute_far_frr, find_eer_threshold  # noqa: F401 (F8-02 re-export)

__all__ = [
    "accuracy",
    "average_accuracy",
    "forgetting_measure",
    "compute_far_frr",
    "find_eer_threshold",
    "auroc",
    "tar_at_far",
]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pers. 3.11: proportion of correct predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        raise ValueError("y_true/y_pred must be non-empty")
    return float(np.mean(y_true == y_pred))


def average_accuracy(session_task_accuracy: dict[int, dict[int, float]], final_session: int) -> float:
    """Pers. 3.14: A_T = (1/T) * sum_{i=1}^{T} a_{T,i}

    `session_task_accuracy[t][i]` = accuracy measured on task i's query set,
    using the model/prototypes as they stood right after session t. This
    function averages the accuracies recorded at `final_session` (T) across
    every task i in [0, final_session].
    """
    row = session_task_accuracy[final_session]
    values = [row[i] for i in range(final_session + 1) if i in row]
    if not values:
        raise ValueError(f"No task accuracies recorded for session {final_session}")
    return float(np.mean(values))


def forgetting_measure(session_task_accuracy: dict[int, dict[int, float]], final_session: int) -> float:
    """Pers. 3.15: F_T = (1/(T-1)) * sum_{i=1}^{T-1} (a_{l,i} - a_{T,i})

    For every task i seen *before* `final_session` (T), `a_{l,i}` is the
    best accuracy task i ever achieved at any evaluation point STRICTLY
    BEFORE T (l in [i, T-1]), and `a_{T,i}` is its accuracy at the final
    session T. Excludes the just-introduced final task (no prior session to
    have forgotten anything from). The "best" must come from before T --
    including T itself would make it structurally impossible to ever
    observe negative forgetting (positive backward transfer), since T's own
    accuracy would then always be one of the candidates being maxed against.
    """
    if final_session == 0:
        raise ValueError("forgetting_measure needs at least 2 sessions (final_session >= 1)")

    per_task_forgetting = []
    for i in range(final_session):  # tasks 0..final_session-1
        accs_over_time = [
            session_task_accuracy[t][i]
            for t in range(i, final_session)  # strictly before T
            if i in session_task_accuracy.get(t, {})
        ]
        if not accs_over_time:
            continue
        best_acc = max(accs_over_time)
        final_acc = session_task_accuracy[final_session][i]
        per_task_forgetting.append(best_acc - final_acc)

    if not per_task_forgetting:
        raise ValueError("No prior-task accuracies available to compute forgetting")
    return float(np.mean(per_task_forgetting))


# --------------------------------------------------------------------------- #
# Experiment 3c (exp3c_dualmetric): threshold-free open-set DETECTION metrics.
# Both take DISTANCES (lower = more genuine), matching the rest of the
# codebase (calibration.py's FAR/FRR convention, Pers. 3.12-3.13).
# --------------------------------------------------------------------------- #

def auroc(genuine_distances: np.ndarray, impostor_distances: np.ndarray) -> float:
    """Area under the ROC of the genuine-vs-unknown detection problem:
    P(genuine distance < impostor distance), with ties counted 1/2
    (Mann-Whitney U formulation -- exact, no curve interpolation)."""
    g = np.asarray(genuine_distances, dtype=np.float64)
    i = np.asarray(impostor_distances, dtype=np.float64)
    if len(g) == 0 or len(i) == 0:
        raise ValueError("need at least one genuine and one impostor distance")
    diff = g[:, None] - i[None, :]
    return float(((diff < 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def tar_at_far(
    genuine_distances: np.ndarray, impostor_distances: np.ndarray, far_target: float = 0.01
) -> float:
    """True-Accept Rate at the threshold whose False-Accept Rate is
    `far_target`: threshold = far_target-quantile of impostor distances
    (accept iff distance < threshold), TAR = P(genuine < threshold)."""
    g = np.asarray(genuine_distances, dtype=np.float64)
    i = np.asarray(impostor_distances, dtype=np.float64)
    if len(g) == 0 or len(i) == 0:
        raise ValueError("need at least one genuine and one impostor distance")
    threshold = np.quantile(i, far_target)
    return float((g < threshold).mean())
