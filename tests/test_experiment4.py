"""Unit tests for Experiment 4's complementarity-analysis helpers
(src/evaluation/complementarity.py). Analysis-only code: none of these touch
the production decision path, mirroring the experiment's feature-flag rule."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.complementarity import (
    detection_scores,
    fusion_accuracy,
    l2_normalize,
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
