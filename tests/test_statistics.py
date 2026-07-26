"""Tests for src/evaluation/statistics.py (F11)."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.statistics import (
    bonferroni_correction,
    bootstrap_eer_difference,
    paired_significance_test,
)


def test_bonferroni_correction_basic():
    assert bonferroni_correction(0.05, 5) == pytest.approx(0.01)


def test_bonferroni_correction_raises_on_zero_comparisons():
    with pytest.raises(ValueError, match="n_comparisons"):
        bonferroni_correction(0.05, 0)


def test_paired_significance_test_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="paired"):
        paired_significance_test(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_paired_significance_test_raises_on_too_few_samples():
    with pytest.raises(ValueError, match="Need >= 3"):
        paired_significance_test(np.array([1.0, 2.0]), np.array([1.5, 2.5]))


def test_paired_significance_test_detects_clear_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.95, scale=0.01, size=10)
    b = rng.normal(loc=0.80, scale=0.01, size=10)
    result = paired_significance_test(a, b)
    assert result.p_value < 0.05
    assert result.test_used in ("paired_t_test", "wilcoxon_signed_rank")


def test_paired_significance_test_no_difference_high_p_value():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.90, scale=0.02, size=10)
    b = a + rng.normal(loc=0.0, scale=0.001, size=10)  # near-identical
    result = paired_significance_test(a, b)
    assert result.p_value > 0.05


def test_paired_significance_test_handles_identical_values():
    a = np.array([0.9, 0.9, 0.9, 0.9])
    b = np.array([0.9, 0.9, 0.9, 0.9])
    result = paired_significance_test(a, b)
    assert result.test_used == "degenerate_identical_diffs"
    assert result.p_value == 1.0


def test_bootstrap_eer_difference_detects_clear_gap():
    rng = np.random.default_rng(0)
    genuine_a = rng.normal(loc=1.0, scale=0.1, size=200)
    impostor_a = rng.normal(loc=5.0, scale=0.1, size=200)  # well separated -> low EER
    genuine_b = rng.normal(loc=1.0, scale=1.5, size=200)
    impostor_b = rng.normal(loc=1.5, scale=1.5, size=200)  # heavily overlapping -> high EER

    result = bootstrap_eer_difference(genuine_a, impostor_a, genuine_b, impostor_b, n_bootstrap=200, seed=0)
    assert result.mean_diff < 0  # config A has much lower EER than config B
    assert result.significant is True


def test_bootstrap_eer_difference_no_gap_not_significant():
    rng = np.random.default_rng(0)
    genuine = rng.normal(loc=1.0, scale=0.1, size=200)
    impostor = rng.normal(loc=5.0, scale=0.1, size=200)
    # same distributions for both "configs" -> true difference is 0
    result = bootstrap_eer_difference(genuine, impostor, genuine.copy(), impostor.copy(), n_bootstrap=200, seed=0)
    assert result.significant is False
