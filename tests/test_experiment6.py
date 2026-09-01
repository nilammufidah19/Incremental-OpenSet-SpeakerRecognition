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


# --------------------------------------------------------------------------- #
# F6-1: training-free Whisper-PMFA readout                                     #
# --------------------------------------------------------------------------- #
from pathlib import Path  # noqa: E402

import pytest as _pytest  # noqa: E402

from src.models import whisper_encoder  # noqa: E402
from src.preprocessing.pipeline import preprocess_audio  # noqa: E402
from tests.conftest import requires_sample_audio  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pmfa_rejects_wrong_sample_rate():
    with _pytest.raises(ValueError, match="sr=16000"):
        whisper_encoder.extract_embedding_pmfa(np.zeros(16000, dtype=np.float32), sr=8000)


@requires_sample_audio
def test_pmfa_rejects_out_of_range_layers(sample_audio_manifest):
    path = REPO_ROOT / sample_audio_manifest.iloc[0]["path"]
    window = preprocess_audio(path, mode="whisper_inference")[0]
    with _pytest.raises(ValueError, match="out of range"):
        whisper_encoder.extract_embedding_pmfa(window, layers=(3, 99))


@requires_sample_audio
def test_pmfa_shape_layout_and_finiteness(sample_audio_manifest):
    path = REPO_ROOT / sample_audio_manifest.iloc[0]["path"]
    window = preprocess_audio(path, mode="whisper_inference")[0]
    emb = whisper_encoder.extract_embedding_pmfa(window)

    d = whisper_encoder.embedding_dim()
    assert emb.shape == (len(whisper_encoder.PMFA_LAYERS) * 2 * d,)
    assert emb.dtype == np.float32
    assert np.isfinite(emb).all()
    # deliberately NOT unit-norm: the cache stores raw statistics (see the
    # extractor docstring). If this ever starts passing, someone baked a
    # normalization choice into the cache.
    assert abs(np.linalg.norm(emb) - 1.0) > 1e-3


@requires_sample_audio
def test_pmfa_layer3_mean_slice_matches_the_validated_single_layer_readout(
    sample_audio_manifest,
):
    """The default "whisper" backbone is layer_fraction 0.5 -> hidden state 3,
    masked mean, then L2. PMFA's first mean block is the same quantity before
    normalization, so their directions must agree. This pins the masking and
    pooling of the new readout against the path that produced every existing
    Whisper result -- if they ever diverge, the new cache is not comparable
    to the old one."""
    path = REPO_ROOT / sample_audio_manifest.iloc[0]["path"]
    window = preprocess_audio(path, mode="whisper_inference")[0]

    d = whisper_encoder.embedding_dim()
    pmfa_layer3_mean = whisper_encoder.extract_embedding_pmfa(window)[:d]
    single = whisper_encoder.extract_embedding(window, layer_fraction=0.5)

    cosine = float(single @ (pmfa_layer3_mean / np.linalg.norm(pmfa_layer3_mean)))
    assert cosine > 1 - 1e-5


@requires_sample_audio
def test_pmfa_windows_averages_without_normalizing(sample_audio_manifest):
    path = REPO_ROOT / sample_audio_manifest.iloc[0]["path"]
    window = preprocess_audio(path, mode="whisper_inference")[0]
    single = whisper_encoder.extract_embedding_pmfa(window)

    averaged = whisper_encoder.extract_embedding_pmfa_windows([window, window])
    np.testing.assert_allclose(averaged, single, rtol=1e-5)


# --------------------------------------------------------------------------- #
# Margin-shift rule (follow-up to F6-3b)                                       #
# --------------------------------------------------------------------------- #
from src.prototypical.score_norm import MarginShiftDualASNorm  # noqa: E402


def test_margin_shift_zero_scale_is_exactly_the_fixed_weight_rule():
    """shift_scale=0 must be bit-identical to constant-weight DualASNorm, so
    the sweep's zero point doubles as the A3-fixed arm."""
    cohort, queries, protos = _fixture(seed=21)
    for w in (0.3, 0.5):
        fixed = DualASNorm(cohort, split_dim=D1, top_k=20, weight=w)
        shifted = MarginShiftDualASNorm(cohort, split_dim=D1, top_k=20,
                                        weight=w, shift_scale=0.0)
        np.testing.assert_allclose(
            shifted.normalize(queries, protos), fixed.normalize(queries, protos)
        )


def test_margin_shift_moves_weight_toward_the_decided_space():
    rule = MarginShiftDualASNorm(_rand((60, D1 + D2), 22), split_dim=D1,
                                 top_k=20, weight=0.3, shift_scale=1.0)
    z_decided = np.array([[-3.0, 0.0, 0.1]])   # margin 3.0
    z_flat = np.array([[0.0, 0.05, 0.10]])     # margin 0.05
    # first space decided -> weight rises above the 0.3 anchor
    assert rule._weights_from(z_decided, z_flat)[0] > 0.9
    # second space decided -> weight pinned at the floor
    assert rule._weights_from(z_flat, z_decided)[0] == 0.0


def test_margin_shift_weights_are_clipped_to_unit_interval():
    cohort, queries, protos = _fixture(seed=23)
    rule = MarginShiftDualASNorm(cohort, split_dim=D1, top_k=20,
                                 weight=0.3, shift_scale=100.0)
    w = rule.per_query_weight(queries, protos)
    assert np.all((w >= 0.0) & (w <= 1.0))


def test_margin_shift_single_prototype_falls_back_to_base_weight():
    cohort, queries, _ = _fixture(seed=24)
    proto = _rand((1, D1 + D2), 25)
    rule = MarginShiftDualASNorm(cohort, split_dim=D1, top_k=20,
                                 weight=0.37, shift_scale=2.0)
    fixed = DualASNorm(cohort, split_dim=D1, top_k=20, weight=0.37)
    np.testing.assert_allclose(rule.normalize(queries, proto),
                               fixed.normalize(queries, proto))
