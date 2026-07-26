"""Tests for the Experiment 3 additions (docs/experiment-3.md):

- exp3a: find_operating_point(strategy="target_frr") in calibration.py
- exp3b: ASNorm score normalization (score_norm.py) + its manager hook
- exp3c: auroc / tar_at_far metrics, closed-set tracking and the
  state-change-free score_sample() detection path
"""
from __future__ import annotations

import numpy as np
import pytest

from src.continual.manager import ContinualLearningManager
from src.continual.speaker_database import SpeakerDatabase
from src.evaluation.metrics import auroc, tar_at_far
from src.prototypical.calibration import find_operating_point
from src.prototypical.score_norm import ASNorm, build_cohort


# --------------------------------------------------------------------------- #
# exp3a -- low-FRR operating point
# --------------------------------------------------------------------------- #

def _clusters(seed=0, n=500):
    rng = np.random.default_rng(seed)
    genuine = rng.normal(0.4, 0.1, n)
    impostor = rng.normal(1.0, 0.15, n)
    return genuine, impostor


def test_target_frr_threshold_hits_requested_frr():
    genuine, impostor = _clusters()
    result = find_operating_point(genuine, impostor, strategy="target_frr", target_frr=0.05)
    assert result.strategy == "target_frr"
    # by construction the threshold is the 95th genuine percentile
    assert result.frr_at_threshold == pytest.approx(0.05, abs=0.01)
    assert result.threshold == pytest.approx(np.quantile(genuine, 0.95))


def test_target_frr_threshold_is_more_permissive_than_eer():
    # overlapping clusters (EER ~0.2, like the real 1-shot score distributions):
    # only when EER > target_frr is the low-FRR point the more permissive one
    rng = np.random.default_rng(0)
    genuine = rng.normal(0.5, 0.15, 500)
    impostor = rng.normal(0.8, 0.15, 500)
    eer_point = find_operating_point(genuine, impostor, strategy="eer")
    lowfrr_point = find_operating_point(genuine, impostor, strategy="target_frr", target_frr=0.05)
    # low-FRR accepts more genuine (higher distance threshold) whenever
    # the EER of the score distribution exceeds the requested FRR
    assert lowfrr_point.threshold > eer_point.threshold
    # the EER value itself is a distribution property: identical either way
    assert lowfrr_point.eer == pytest.approx(eer_point.eer)


def test_eer_strategy_reproduces_original_behaviour():
    genuine, impostor = _clusters()
    from src.prototypical.calibration import find_eer_threshold
    old = find_eer_threshold(genuine, impostor)
    new = find_operating_point(genuine, impostor, strategy="eer")
    assert new.threshold == old.threshold
    assert new.eer == old.eer


def test_invalid_strategy_and_frr_raise():
    genuine, impostor = _clusters()
    with pytest.raises(ValueError):
        find_operating_point(genuine, impostor, strategy="nope")
    with pytest.raises(ValueError):
        find_operating_point(genuine, impostor, strategy="target_frr", target_frr=1.5)


# --------------------------------------------------------------------------- #
# exp3b -- AS-Norm
# --------------------------------------------------------------------------- #

def test_asnorm_shape_and_orientation():
    rng = np.random.default_rng(0)
    cohort = rng.normal(0, 1, (50, 8))
    queries = rng.normal(0, 1, (4, 8))
    prototypes = rng.normal(0, 1, (3, 8))
    norm = ASNorm(cohort, top_k=10)
    out = norm.normalize(queries, prototypes)
    assert out.shape == (4, 3)
    # a query placed exactly on a prototype must be far more genuine
    # (smaller normalized score) than an arbitrary query
    on_proto = norm.normalize(prototypes[0:1], prototypes)[0, 0]
    assert on_proto < out.min()


def test_asnorm_improves_separation_under_shifted_score_scales():
    """Two 'speakers' whose raw distances live on different scales: a single
    raw threshold cannot separate both, but per-trial cohort normalization
    can -- the mechanism AS-Norm is included for."""
    rng = np.random.default_rng(1)
    dim = 16
    # cohort spread around the origin
    cohort = rng.normal(0, 1, (200, dim))
    # speaker A sits in a dense region (small distances everywhere),
    # speaker B in a sparse region (large distances everywhere)
    proto_a = np.zeros(dim)
    proto_b = np.full(dim, 4.0)
    genuine_a = proto_a + rng.normal(0, 0.3, (30, dim))
    genuine_b = proto_b + rng.normal(0, 1.2, (30, dim))
    impostors = rng.normal(0, 1, (30, dim))  # near A's region -> hard for A

    prototypes = np.stack([proto_a, proto_b])
    norm = ASNorm(cohort, top_k=50)

    def eer_of(genuine_scores, impostor_scores):
        return find_operating_point(np.asarray(genuine_scores), np.asarray(impostor_scores), "eer").eer

    raw_gen = [np.linalg.norm(g - prototypes, axis=1).min() for g in np.vstack([genuine_a, genuine_b])]
    raw_imp = [np.linalg.norm(i - prototypes, axis=1).min() for i in impostors]
    norm_gen = norm.normalize(np.vstack([genuine_a, genuine_b]), prototypes).min(axis=1)
    norm_imp = norm.normalize(impostors, prototypes).min(axis=1)

    assert eer_of(norm_gen, norm_imp) <= eer_of(raw_gen, raw_imp)


def test_build_cohort_round_robin_and_size():
    index = {f"spk{i}": [np.full(4, float(i)), np.full(4, float(i)) + 0.1] for i in range(5)}
    fuse = lambda raws: np.stack(raws)  # noqa: E731
    cohort = build_cohort(index, fuse, cohort_size=5, seed=0)
    assert cohort.shape == (5, 4)
    # round-robin: 5 picks across 5 speakers = one utterance per speaker
    assert len({tuple(np.round(r).astype(int)) for r in cohort}) == 5
    # asking for more than exists caps at the total utterance count
    assert build_cohort(index, fuse, cohort_size=999, seed=0).shape[0] == 10


def test_manager_with_asnorm_decides_in_normalized_space():
    rng = np.random.default_rng(2)
    cohort = rng.normal(0, 1, (100, 8)).astype(np.float32)
    norm = ASNorm(cohort, top_k=20)
    db = SpeakerDatabase()
    proto = rng.normal(0, 1, 8).astype(np.float32)
    db.set("spk_a", proto / np.linalg.norm(proto), 1)
    # normalized genuine scores are strongly negative -> a threshold of 0
    # accepts self-queries and rejects far-away queries
    manager = ContinualLearningManager(db, threshold=-1.0, mode="static", score_normalizer=norm)
    self_result = manager.process_sample(db.get("spk_a").prototype)
    assert self_result.is_known and self_result.predicted_speaker_id == "spk_a"
    far = rng.normal(6, 0.1, 8).astype(np.float32)
    far_result = manager.process_sample(far)
    assert not far_result.is_known
    assert far_result.nearest_speaker_id == "spk_a"  # exp3c closed-set field


# --------------------------------------------------------------------------- #
# exp3c -- detection metrics + state-change-free scoring
# --------------------------------------------------------------------------- #

def test_auroc_perfect_and_chance():
    g = np.array([0.1, 0.2, 0.3])
    i = np.array([0.9, 1.0, 1.1])
    assert auroc(g, i) == 1.0
    assert auroc(i, g) == 0.0
    assert auroc(g, g) == pytest.approx(0.5)


def test_tar_at_far_bounds():
    genuine, impostor = _clusters()
    tar1 = tar_at_far(genuine, impostor, 0.01)
    tar10 = tar_at_far(genuine, impostor, 0.10)
    assert 0.0 <= tar1 <= tar10 <= 1.0


def test_score_sample_changes_no_state():
    rng = np.random.default_rng(3)
    db = SpeakerDatabase()
    proto = rng.normal(0, 1, 8).astype(np.float32)
    db.set("spk_a", proto, 1)
    manager = ContinualLearningManager(db, threshold=0.5, mode="running_average")

    before_proto = db.get("spk_a").prototype.copy()
    before_n = db.get("spk_a").n_samples
    # genuine-looking query: process_sample WOULD have updated the prototype
    near = proto + rng.normal(0, 0.01, 8).astype(np.float32)
    res = manager.score_sample(near)
    assert res.is_known and res.nearest_speaker_id == "spk_a"
    # unknown-looking query: process_sample WOULD have buffered/registered
    far = proto + 10.0
    res_far = manager.score_sample(far)
    assert not res_far.is_known

    assert np.array_equal(db.get("spk_a").prototype, before_proto)
    assert db.get("spk_a").n_samples == before_n
    assert len(db) == 1
    assert manager._unknown_buffer.size() == 0
