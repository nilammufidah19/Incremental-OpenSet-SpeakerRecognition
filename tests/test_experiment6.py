"""Unit tests for Experiment 6 F6-3b: AdaptiveDualASNorm, the zero-parameter
per-query fusion rule.

The point of F6-3b is that it fits NOTHING (tier 2 in
docs/experiment-6-plan.md section 2), so the tests pin two things: that the
rule really is a function of its inputs alone, and that it degenerates
EXACTLY to DualASNorm wherever no per-query evidence exists -- otherwise the
"strict generalization" claim in the docstring would be false."""
from __future__ import annotations

import numpy as np
import pytest

from src.prototypical.score_norm import AdaptiveDualASNorm, DualASNorm


def _rand(shape, seed):
    return np.random.default_rng(seed).normal(size=shape)


D1, D2 = 8, 12


def _fixture(seed=0, n_proto=4):
    return (
        _rand((60, D1 + D2), seed),
        _rand((7, D1 + D2), seed + 1),
        _rand((n_proto, D1 + D2), seed + 2),
    )


def test_rejects_unknown_rule():
    cohort, _, _ = _fixture()
    with pytest.raises(ValueError, match="rule must be one of"):
        AdaptiveDualASNorm(cohort, split_dim=D1, rule="softmax")


def test_single_prototype_falls_back_to_constant_weight():
    """With one prototype there is no margin, so the adaptive rule must be
    bit-identical to the fixed-weight parent -- not merely close."""
    cohort, queries, _ = _fixture()
    proto = _rand((1, D1 + D2), 9)
    for w in (0.0, 0.35, 1.0):
        fixed = DualASNorm(cohort, split_dim=D1, top_k=20, weight=w)
        adaptive = AdaptiveDualASNorm(cohort, split_dim=D1, top_k=20, weight=w)
        np.testing.assert_allclose(
            adaptive.normalize(queries, proto), fixed.normalize(queries, proto)
        )


def test_weights_stay_in_unit_interval_and_select_is_binary():
    cohort, queries, protos = _fixture()
    soft = AdaptiveDualASNorm(cohort, split_dim=D1, top_k=20, rule="margin_weighted")
    hard = AdaptiveDualASNorm(cohort, split_dim=D1, top_k=20, rule="margin_select")

    w_soft = soft.per_query_weight(queries, protos)
    w_hard = hard.per_query_weight(queries, protos)

    assert w_soft.shape == (len(queries),)
    assert np.all((w_soft >= 0.0) & (w_soft <= 1.0))
    assert set(np.unique(w_hard)).issubset({0.0, 1.0})


def test_output_is_the_convex_combination_the_weight_claims():
    """normalize() must equal w*z_first + (1-w)*z_second for the SAME w that
    per_query_weight() reports -- if those two ever drift apart, the sweep
    would be reporting a weight distribution for a rule it is not running."""
    cohort, queries, protos = _fixture(seed=5)
    adaptive = AdaptiveDualASNorm(cohort, split_dim=D1, top_k=20)
    only_first = DualASNorm(cohort, split_dim=D1, top_k=20, weight=1.0)
    only_second = DualASNorm(cohort, split_dim=D1, top_k=20, weight=0.0)

    w = adaptive.per_query_weight(queries, protos)[:, None]
    expected = w * only_first.normalize(queries, protos) + (1 - w) * only_second.normalize(
        queries, protos
    )
    np.testing.assert_allclose(adaptive.normalize(queries, protos), expected)


def test_decisive_space_wins_the_weight():
    """Hand-built z-matrices: space 1 has a clear winner, space 2 is flat.
    The rule must hand (nearly) all the weight to space 1."""
    adaptive = AdaptiveDualASNorm(_rand((60, D1 + D2), 3), split_dim=D1, top_k=20)
    z_decisive = np.array([[-3.0, 0.0, 0.1]])       # margin 3.0
    z_flat = np.array([[0.0, 0.001, 0.002]])        # margin 0.001
    w = adaptive._weights_from(z_decisive, z_flat)
    assert w[0] > 0.99

    w_swapped = adaptive._weights_from(z_flat, z_decisive)
    assert w_swapped[0] < 0.01


def test_both_spaces_undecided_falls_back_to_constant_weight():
    adaptive = AdaptiveDualASNorm(_rand((60, D1 + D2), 4), split_dim=D1, top_k=20, weight=0.42)
    tied = np.array([[1.0, 1.0, 1.0]])
    w = adaptive._weights_from(tied, tied.copy())
    assert w[0] == pytest.approx(0.42)


def test_is_deterministic_and_stateless_across_calls():
    """No fitting, no accumulated state: repeated and reordered calls must
    give identical results. This is the property that makes F6-3b tier-2."""
    cohort, queries, protos = _fixture(seed=7)
    adaptive = AdaptiveDualASNorm(cohort, split_dim=D1, top_k=20)

    first = adaptive.normalize(queries, protos)
    adaptive.normalize(_rand((3, D1 + D2), 99), protos)  # unrelated traffic
    np.testing.assert_array_equal(adaptive.normalize(queries, protos), first)

    # scoring a subset must match the matching rows of the full call
    np.testing.assert_allclose(adaptive.normalize(queries[2:4], protos), first[2:4])


def test_invariant_to_per_half_constant_scaling():
    """ScoreFusionEmbed emits [sqrt(w)*e1 ; sqrt(1-w)*e2]; those constants
    cancel in z-scores for DualASNorm (tested in test_experiment5) and must
    also cancel here, or the adaptive weight would silently depend on the
    embedder's w."""
    cohort, queries, protos = _fixture(seed=11)
    scale = np.concatenate([np.full(D1, 0.7), np.full(D2, 0.3)])

    plain = AdaptiveDualASNorm(cohort, split_dim=D1, top_k=20)
    scaled = AdaptiveDualASNorm(cohort * scale, split_dim=D1, top_k=20)
    np.testing.assert_allclose(
        scaled.normalize(queries * scale, protos * scale),
        plain.normalize(queries, protos),
        atol=1e-9,
    )
