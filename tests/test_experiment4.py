"""Unit tests for Experiment 4's complementarity-analysis helpers
(src/evaluation/complementarity.py). Analysis-only code: none of these touch
the production decision path, mirroring the experiment's feature-flag rule."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.complementarity import (
    best_global_weight,
    detection_scores,
    fusion_accuracy,
    l2_normalize,
    linear_fusion_feasible,
    linear_fusion_oracle,
    oracle_report,
)
from src.prototypical.score_norm import ASNorm


def test_l2_normalize_rows_unit_norm():
    x = np.array([[3.0, 4.0], [0.0, 2.0], [1e-15, 0.0]])
    out = l2_normalize(x)
    np.testing.assert_allclose(np.linalg.norm(out[:2], axis=1), 1.0)
    assert np.all(np.isfinite(out))  # near-zero row must not divide by zero


def test_oracle_report_counts_either_correct():
    # 4 queries, 3 prototypes; labels: 0,1,2,0
    labels = np.array([0, 1, 2, 0])
    # space A correct on queries 0,1; space B correct on queries 1,2
    d_a = np.array([[0.1, 0.5, 0.5],
                    [0.5, 0.1, 0.5],
                    [0.1, 0.5, 0.5],    # A predicts 0, true 2 -> wrong
                    [0.5, 0.1, 0.5]])   # A predicts 1, true 0 -> wrong
    d_b = np.array([[0.5, 0.1, 0.5],    # B wrong
                    [0.5, 0.1, 0.5],    # B correct
                    [0.5, 0.5, 0.1],    # B correct (rescues A)
                    [0.5, 0.1, 0.5]])   # B wrong
    rep = oracle_report(d_a, d_b, labels)
    assert rep["acc_a"] == pytest.approx(0.5)
    assert rep["acc_b"] == pytest.approx(0.5)
    assert rep["acc_oracle"] == pytest.approx(0.75)   # queries 0,1,2 covered
    assert rep["delta_ceiling"] == pytest.approx(0.25)
    assert rep["b_rescues_a"] == 1
    assert rep["a_rescues_b"] == 1


def test_oracle_report_shape_mismatch_raises():
    with pytest.raises(ValueError):
        oracle_report(np.zeros((2, 3)), np.zeros((2, 4)), np.zeros(2))


def test_fusion_accuracy_recovers_better_space_at_extremes():
    labels = np.array([0, 1])
    d_good = np.array([[0.1, 0.9], [0.9, 0.1]])   # always correct
    d_bad = np.array([[0.9, 0.1], [0.1, 0.9]])    # always wrong
    assert fusion_accuracy(d_good, d_bad, labels, weight_a=1.0) == 1.0
    assert fusion_accuracy(d_good, d_bad, labels, weight_a=0.0) == 0.0


def test_fusion_accuracy_can_beat_both_single_spaces():
    # complementary errors: each space is confidently right on its own query
    # and mildly wrong on the other; equal-weight fusion fixes both.
    labels = np.array([0, 0])
    d_a = np.array([[0.0, 1.0],     # A very confident, correct
                    [0.6, 0.4]])    # A mildly wrong
    d_b = np.array([[0.6, 0.4],     # B mildly wrong
                    [0.0, 1.0]])    # B very confident, correct
    assert fusion_accuracy(d_a, d_b, labels, 1.0) == 0.5
    assert fusion_accuracy(d_a, d_b, labels, 0.0) == 0.5
    assert fusion_accuracy(d_a, d_b, labels, 0.5) == 1.0


def test_detection_scores_rules():
    d_a = np.array([[0.2, 0.8], [0.5, 0.6]])
    d_b = np.array([[0.7, 0.3], [0.4, 0.9]])
    s = detection_scores(d_a, d_b, disagree_penalty=0.5)
    np.testing.assert_allclose(s["a_only"], [0.2, 0.5])
    np.testing.assert_allclose(s["b_only"], [0.3, 0.4])
    np.testing.assert_allclose(s["mean"], [0.25, 0.45])
    np.testing.assert_allclose(s["max"], [0.3, 0.5])
    # query 0: argmins differ (0 vs 1) -> penalty; query 1: both argmin 0 -> none
    np.testing.assert_allclose(s["a_disagree"], [0.7, 0.5])


# --------------------------------------------------------------------------
# linear_fusion_oracle (C1 fix, 2026-08-15): the ceiling of `w*d_a+(1-w)*d_b`,
# which is NOT the same as oracle_report's selection ceiling.
# --------------------------------------------------------------------------


def test_linear_fusion_oracle_exceeds_selection_oracle_when_both_argmins_wrong():
    """The whole point of the correction: a query where NEITHER space's argmin
    is right, but some w makes the sum right. oracle_report must miss it and
    linear_fusion_oracle must catch it."""
    labels = np.array([0])
    #                     t=0   j=1   j=2
    d_a = np.array([[0.50, 0.40, 0.90]])   # A picks 1 -> wrong
    d_b = np.array([[0.50, 0.90, 0.40]])   # B picks 2 -> wrong
    # w=0.5: [0.50, 0.65, 0.65] -> true class 0 wins
    assert oracle_report(d_a, d_b, labels)["acc_oracle"] == pytest.approx(0.0)
    rep = linear_fusion_oracle(d_a, d_b, labels)
    assert rep["acc_linear_oracle"] == pytest.approx(1.0)
    assert rep["recovered_both_argmin_wrong"] == pytest.approx(1.0)
    assert rep["delta_linear_ceiling"] == pytest.approx(1.0)


def test_linear_fusion_feasible_interval_matches_brute_force():
    """The closed-form interval must agree with an exhaustive w sweep."""
    rng = np.random.default_rng(7)
    d_a = rng.random((60, 5))
    d_b = rng.random((60, 5))
    labels = rng.integers(0, 5, size=60)
    ok, w_lo, w_hi = linear_fusion_feasible(d_a, d_b, labels)

    grid = np.linspace(0.0, 1.0, 2001)
    brute = np.zeros(len(labels), dtype=bool)
    for w in grid:
        brute |= (w * d_a + (1 - w) * d_b).argmin(axis=1) == labels
    # a dense sweep can only miss feasible queries whose interval is narrower
    # than the grid step, never invent one
    assert np.all(brute <= ok)
    narrow = (w_hi - w_lo) < (grid[1] - grid[0])
    assert np.all(ok[~narrow] == brute[~narrow])


def test_linear_fusion_oracle_is_upper_bound_of_any_weight():
    rng = np.random.default_rng(11)
    d_a = rng.random((80, 4))
    d_b = rng.random((80, 4))
    labels = rng.integers(0, 4, size=80)
    ceiling = linear_fusion_oracle(d_a, d_b, labels)["acc_linear_oracle"]
    for w in np.linspace(0.0, 1.0, 51):
        assert fusion_accuracy(d_a, d_b, labels, w) <= ceiling + 1e-12


def test_linear_fusion_oracle_single_prototype_is_trivially_feasible():
    d = np.array([[0.3], [0.9]])
    rep = linear_fusion_oracle(d, d, np.array([0, 0]))
    assert rep["acc_linear_oracle"] == pytest.approx(1.0)


def test_linear_fusion_oracle_identical_spaces_matches_single_space():
    """d_a == d_b makes w irrelevant, so the ceiling collapses to plain accuracy."""
    rng = np.random.default_rng(3)
    d = rng.random((40, 6))
    labels = rng.integers(0, 6, size=40)
    rep = linear_fusion_oracle(d, d, labels)
    assert rep["acc_linear_oracle"] == pytest.approx(rep["acc_a"])
    assert rep["delta_linear_ceiling"] == pytest.approx(0.0)


def test_linear_fusion_oracle_shape_mismatch_raises():
    with pytest.raises(ValueError):
        linear_fusion_oracle(np.zeros((2, 3)), np.zeros((2, 4)), np.zeros(2))


def test_best_global_weight_finds_optimum_missed_by_coarse_grid():
    """C5: exp4's 6-point grid {0.5..0.95} can miss the real optimum."""
    labels = np.array([0, 0])
    # true class wins the sum only for w in a narrow band around ~0.88
    d_a = np.array([[0.10, 0.00], [0.30, 0.40]])
    d_b = np.array([[0.90, 1.00], [0.60, 0.00]])
    res = best_global_weight(d_a, d_b, labels)
    assert set(res) == {"best_w", "best_acc", "acc_a", "grid_accuracies"}
    assert 0.0 <= res["best_w"] <= 1.0
    # the fine grid can never do worse than any single point of the coarse one
    coarse = max(fusion_accuracy(d_a, d_b, labels, w)
                 for w in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95))
    assert res["best_acc"] >= coarse
    assert res["best_acc"] <= linear_fusion_oracle(d_a, d_b, labels)["acc_linear_oracle"]


def test_best_global_weight_spans_full_interval():
    """Must include w<0.5 and both single-space endpoints -- exp4's grid did not."""
    labels = np.array([0, 1])
    d_bad = np.array([[0.9, 0.1], [0.1, 0.9]])   # always wrong
    d_good = np.array([[0.1, 0.9], [0.9, 0.1]])  # always right
    res = best_global_weight(d_bad, d_good, labels)   # space A useless
    assert res["best_acc"] == pytest.approx(1.0)
    assert res["best_w"] < 0.5
    assert 0.0 in res["grid_accuracies"] and 1.0 in res["grid_accuracies"]


def test_asnorm_per_space_scale_invariance():
    """AS-Norm z-scoring makes each space's distances scale-free, which is the
    property that makes cross-space score fusion well-posed in exp4b (raw
    cosine distances in exp2 had no such guarantee)."""
    rng = np.random.default_rng(0)
    cohort = rng.normal(size=(50, 8))
    queries = rng.normal(size=(4, 8))
    protos = rng.normal(size=(3, 8))
    z1 = ASNorm(cohort, top_k=20).normalize(queries, protos)
    z2 = ASNorm(cohort * 7.0, top_k=20).normalize(queries * 7.0, protos * 7.0)
    np.testing.assert_allclose(z1, z2, atol=1e-10)
