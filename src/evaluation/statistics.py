"""Statistical significance testing (F11) [4.11].

Implements the exact procedure Bab 4.11 specifies:
  1. Normality check (Shapiro-Wilk) on paired differences.
  2. Paired t-test if normal, Wilcoxon signed-rank otherwise, for Accuracy.
  3. Bootstrap resampling for a 95% CI on EER differences (Bengio &
     Mariethoz, 2004) -- EER isn't a simple sample mean, so a parametric
     test doesn't apply the way it does for Accuracy.
  4. Bonferroni correction across however many pairwise comparisons are
     actually run (F9's A3-vs-A1/A3-vs-A2/B2-vs-B1 plus F10's
     proposed-vs-each-baseline).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

from src.prototypical.calibration import find_eer_threshold


@dataclass
class PairedTestResult:
    test_used: str  # "paired_t_test" | "wilcoxon_signed_rank"
    statistic: float
    p_value: float
    normality_p_value: float


def paired_significance_test(values_a: np.ndarray, values_b: np.ndarray, alpha: float = 0.05) -> PairedTestResult:
    """F11-03/04: pick paired t-test or Wilcoxon based on the Shapiro-Wilk
    normality of the paired differences, matched samples (e.g. per-seed
    Accuracy across two configurations)."""
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    if len(values_a) != len(values_b):
        raise ValueError("values_a and values_b must be paired (same length)")
    if len(values_a) < 3:
        raise ValueError("Need >= 3 paired samples for a meaningful normality test")

    diffs = values_a - values_b
    if np.allclose(diffs, diffs[0]):
        # zero-variance differences (e.g. identical repeated runs) make
        # Shapiro-Wilk and ttest_rel both degenerate; report directly.
        return PairedTestResult(
            test_used="degenerate_identical_diffs",
            statistic=0.0,
            p_value=1.0 if diffs[0] == 0 else 0.0,
            normality_p_value=float("nan"),
        )

    _, normality_p = scipy_stats.shapiro(diffs)

    if normality_p > alpha:
        statistic, p_value = scipy_stats.ttest_rel(values_a, values_b)
        test_used = "paired_t_test"
    else:
        statistic, p_value = scipy_stats.wilcoxon(values_a, values_b)
        test_used = "wilcoxon_signed_rank"

    return PairedTestResult(
        test_used=test_used, statistic=float(statistic), p_value=float(p_value), normality_p_value=float(normality_p)
    )


@dataclass
class BootstrapEERResult:
    mean_diff: float
    ci_lower: float
    ci_upper: float
    significant: bool  # True if 0 lies OUTSIDE the CI


def bootstrap_eer_difference(
    genuine_a: np.ndarray,
    impostor_a: np.ndarray,
    genuine_b: np.ndarray,
    impostor_b: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapEERResult:
    """F11-05: bootstrap resampling for a CI on EER_a - EER_b (Bengio &
    Marietoz, 2004 methodology for authentication-error metrics).

    genuine_a/genuine_b (and impostor_a/impostor_b) must be PAIRED: the same
    trials (same held-out queries) scored by system A and system B, e.g.
    scripts/exp5_bootstrap_eer.py's genuine_a/genuine_b are the same
    final-session query set scored under two different configs. Bengio &
    Mariethoz's paired bootstrap therefore resamples ONE shared set of trial
    indices per iteration and applies it to both systems, preserving the
    per-trial correlation; resampling A and B independently would inflate
    the variance of the diff and bias the CI toward "not significant."
    """
    genuine_a = np.asarray(genuine_a)
    genuine_b = np.asarray(genuine_b)
    impostor_a = np.asarray(impostor_a)
    impostor_b = np.asarray(impostor_b)
    if len(genuine_a) != len(genuine_b) or len(impostor_a) != len(impostor_b):
        raise ValueError(
            "genuine_a/genuine_b and impostor_a/impostor_b must be paired "
            "(same length): the same trials scored by two systems."
        )

    rng = np.random.default_rng(seed)
    n_genuine, n_impostor = len(genuine_a), len(impostor_a)
    diffs = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        genuine_idx = rng.integers(0, n_genuine, size=n_genuine)
        impostor_idx = rng.integers(0, n_impostor, size=n_impostor)
        g_a, g_b = genuine_a[genuine_idx], genuine_b[genuine_idx]
        imp_a, imp_b = impostor_a[impostor_idx], impostor_b[impostor_idx]

        eer_a = find_eer_threshold(g_a, imp_a).eer
        eer_b = find_eer_threshold(g_b, imp_b).eer
        diffs[i] = eer_a - eer_b

    alpha = 1 - confidence
    lower = float(np.percentile(diffs, alpha / 2 * 100))
    upper = float(np.percentile(diffs, (1 - alpha / 2) * 100))
    significant = not (lower <= 0.0 <= upper)

    return BootstrapEERResult(mean_diff=float(diffs.mean()), ci_lower=lower, ci_upper=upper, significant=significant)


def bonferroni_correction(alpha: float, n_comparisons: int) -> float:
    """F11-06: corrected significance threshold for `n_comparisons`
    simultaneous pairwise tests."""
    if n_comparisons < 1:
        raise ValueError("n_comparisons must be >= 1")
    return alpha / n_comparisons
