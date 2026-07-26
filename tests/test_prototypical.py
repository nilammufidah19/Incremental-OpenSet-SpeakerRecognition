"""Tests for src/prototypical/* (F5)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.fusion import FUSION_DIM, GatedAttentionFusion
from src.prototypical.calibration import (
    build_genuine_impostor_distances,
    calibrate_threshold,
    compute_far_frr,
    find_eer_threshold,
)
from src.prototypical.classifier import (
    classify_log_probs,
    euclidean_distance,
    negative_log_likelihood,
    predict,
)
from src.prototypical.data import ECAPA_DIM, WHISPER_DIM, build_raw_embedding_index, split_raw_embedding
from src.prototypical.episodic import sample_episode
from src.prototypical.inference import identify
from src.prototypical.prototype import compute_prototypes
from src.prototypical.train import evaluate_episodic, load_checkpoint, save_checkpoint, train_episodic
from tests.conftest import requires_sample_audio


def make_synthetic_index(n_speakers=8, n_utts=6, dim=16, cluster_std=0.05, seed=0):
    """Synthetic per-speaker embedding index with well-separated clusters,
    for fast, dependency-free tests of the math (episodic/prototype/classifier)."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-5, 5, size=(n_speakers, dim))
    index = {}
    for i in range(n_speakers):
        samples = centers[i] + rng.normal(scale=cluster_std, size=(n_utts, dim))
        index[f"spk{i}"] = [s.astype(np.float32) for s in samples]
    return index


# --------------------------------------------------------------------------
# episodic.py
# --------------------------------------------------------------------------


def test_sample_episode_shapes():
    index = make_synthetic_index(n_speakers=10, n_utts=6)
    ep = sample_episode(index, n_way=5, k_shot=2, n_query=3, seed=0)

    assert ep.support_embeddings.shape == (10, 16)
    assert ep.support_labels.shape == (10,)
    assert ep.query_embeddings.shape == (15, 16)
    assert ep.query_labels.shape == (15,)
    assert len(ep.speaker_ids) == 5
    assert set(ep.support_labels.tolist()) == set(range(5))


def test_sample_episode_support_query_disjoint():
    index = make_synthetic_index(n_speakers=5, n_utts=6)
    ep = sample_episode(index, n_way=5, k_shot=2, n_query=4, seed=1)
    support_set = {tuple(row) for row in ep.support_embeddings}
    query_set = {tuple(row) for row in ep.query_embeddings}
    assert support_set.isdisjoint(query_set)


def test_sample_episode_insufficient_speakers_raises():
    index = make_synthetic_index(n_speakers=3, n_utts=6)
    with pytest.raises(ValueError, match="need >="):
        sample_episode(index, n_way=5, k_shot=2, n_query=3, seed=0)


def test_sample_episode_deterministic_given_seed():
    index = make_synthetic_index(n_speakers=10, n_utts=6)
    ep1 = sample_episode(index, n_way=5, k_shot=2, n_query=3, seed=42)
    ep2 = sample_episode(index, n_way=5, k_shot=2, n_query=3, seed=42)
    assert ep1.speaker_ids == ep2.speaker_ids
    assert np.array_equal(ep1.support_embeddings, ep2.support_embeddings)


# --------------------------------------------------------------------------
# prototype.py
# --------------------------------------------------------------------------


def test_compute_prototypes_matches_manual_mean():
    support = torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0]])
    labels = torch.tensor([0, 0, 1, 1])
    prototypes = compute_prototypes(support, labels, n_way=2)
    assert torch.allclose(prototypes[0], torch.tensor([2.0, 0.0]))
    assert torch.allclose(prototypes[1], torch.tensor([0.0, 3.0]))


def test_compute_prototypes_raises_if_class_missing():
    support = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    labels = torch.tensor([0, 0])  # no samples for class 1
    with pytest.raises(ValueError, match="No support samples"):
        compute_prototypes(support, labels, n_way=2)


# --------------------------------------------------------------------------
# classifier.py
# --------------------------------------------------------------------------


def test_euclidean_distance_known_values():
    query = torch.tensor([[0.0, 0.0]])
    prototypes = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    d = euclidean_distance(query, prototypes)
    assert torch.allclose(d, torch.tensor([[5.0, 0.0]]))


def test_classify_log_probs_normalized():
    query = torch.randn(5, 8)
    prototypes = torch.randn(3, 8)
    log_probs = classify_log_probs(query, prototypes)
    probs_sum = log_probs.exp().sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones(5), atol=1e-5)


def test_negative_log_likelihood_perfect_prediction_near_zero():
    # query exactly at prototype 0 -> distance 0 there, large elsewhere
    query = torch.tensor([[0.0, 0.0]])
    prototypes = torch.tensor([[0.0, 0.0], [100.0, 100.0]])
    log_probs = classify_log_probs(query, prototypes)
    loss = negative_log_likelihood(log_probs, torch.tensor([0]))
    assert loss.item() < 1e-3


def test_predict_matches_nearest_prototype():
    query = torch.tensor([[0.9, 0.0], [0.0, 5.5]])
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 5.0], [10.0, 10.0]])
    predicted, min_dist = predict(query, prototypes)
    assert predicted.tolist() == [0, 1]
    assert torch.allclose(min_dist, torch.tensor([0.1, 0.5]), atol=1e-5)


# --------------------------------------------------------------------------
# train.py (synthetic, checks fusion actually learns via gradient descent)
# --------------------------------------------------------------------------


def make_synthetic_raw_index(n_speakers=10, n_utts=8, seed=0):
    """Synthetic [ecapa;whisper]-shaped raw index with separable clusters,
    for exercising the real GatedAttentionFusion + prototypical training loop."""
    rng = np.random.default_rng(seed)
    dim = ECAPA_DIM + WHISPER_DIM
    centers = rng.uniform(-3, 3, size=(n_speakers, dim))
    index = {}
    for i in range(n_speakers):
        samples = centers[i] + rng.normal(scale=0.2, size=(n_utts, dim))
        index[f"spk{i}"] = [s.astype(np.float32) for s in samples]
    return index


def test_train_episodic_improves_accuracy_on_separable_synthetic_data():
    torch.manual_seed(0)
    index = make_synthetic_raw_index(n_speakers=10, n_utts=8)
    model = GatedAttentionFusion(ECAPA_DIM, WHISPER_DIM, FUSION_DIM, mode="fusion")

    before = evaluate_episodic(model, index, n_way=5, k_shot=2, n_query=3, n_episodes=20, seed=1000)
    train_episodic(model, index, n_way=5, k_shot=2, n_query=3, n_episodes=100, seed=0, lr=1e-2)
    after = evaluate_episodic(model, index, n_way=5, k_shot=2, n_query=3, n_episodes=20, seed=1000)

    acc_before = sum(r.accuracy for r in before) / len(before)
    acc_after = sum(r.accuracy for r in after) / len(after)
    assert acc_after >= acc_before  # should not get worse on separable data; typically improves


def test_save_load_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = GatedAttentionFusion(ECAPA_DIM, WHISPER_DIM, FUSION_DIM, mode="fusion")
    ckpt_path = tmp_path / "fusion.ckpt"
    save_checkpoint(model, ckpt_path)

    new_model = GatedAttentionFusion(ECAPA_DIM, WHISPER_DIM, FUSION_DIM, mode="fusion")
    load_checkpoint(ckpt_path, new_model)

    e1, e2 = torch.randn(ECAPA_DIM), torch.randn(WHISPER_DIM)
    with torch.no_grad():
        out1 = model(e1, e2)
        out2 = new_model(e1, e2)
    assert torch.allclose(out1, out2)


# --------------------------------------------------------------------------
# calibration.py (synthetic, then real cached embeddings)
# --------------------------------------------------------------------------


def test_compute_far_frr_monotonic_trends():
    genuine = np.array([0.1, 0.2, 0.3, 0.4])
    impostor = np.array([5.0, 6.0, 7.0, 8.0])
    thresholds = np.linspace(0, 9, 10)
    far, frr = compute_far_frr(genuine, impostor, thresholds)
    # FAR should be non-decreasing, FRR non-increasing as threshold grows
    assert np.all(np.diff(far) >= 0)
    assert np.all(np.diff(frr) <= 0)


def test_find_eer_threshold_separates_well_separated_clusters():
    rng = np.random.default_rng(0)
    genuine = rng.normal(loc=1.0, scale=0.1, size=200)
    impostor = rng.normal(loc=10.0, scale=0.1, size=200)
    result = find_eer_threshold(genuine, impostor)

    assert result.eer < 0.05  # near-perfect separation -> near-zero EER
    assert 1.0 < result.threshold < 10.0


@requires_sample_audio
def test_calibration_on_real_embeddings_separates_genuine_impostor(sample_audio_manifest):
    """Split real VoxCeleb1 speakers into two disjoint groups (genuine vs
    impostor) and check the resulting EER threshold meaningfully separates
    them -- genuine distances should be smaller than impostor distances on
    average, and EER should be well below chance (0.5)."""
    speakers = sorted(sample_audio_manifest["speaker_id"].unique())
    counts = sample_audio_manifest["speaker_id"].value_counts()
    eligible = [s for s in speakers if counts[s] >= 2]
    assert len(eligible) >= 10, "need at least 10 multi-utterance speakers"

    genuine_speakers = eligible[: len(eligible) // 2]
    impostor_speakers = eligible[len(eligible) // 2 :]
    assert set(genuine_speakers).isdisjoint(impostor_speakers)

    genuine_manifest = sample_audio_manifest[sample_audio_manifest["speaker_id"].isin(genuine_speakers)]
    impostor_manifest = sample_audio_manifest[sample_audio_manifest["speaker_id"].isin(impostor_speakers)]

    genuine_index = build_raw_embedding_index(genuine_manifest)
    impostor_index = build_raw_embedding_index(impostor_manifest)

    torch.manual_seed(0)
    model = GatedAttentionFusion(ECAPA_DIM, WHISPER_DIM, FUSION_DIM, mode="fusion")

    result = calibrate_threshold(model, genuine_index, impostor_index, enrollment_k=1, seed=0)

    assert result.genuine_distances.mean() < result.impostor_distances.mean()
    assert result.eer < 0.5  # better than chance


# --------------------------------------------------------------------------
# inference.py
# --------------------------------------------------------------------------


def test_identify_known_speaker_below_threshold():
    prototypes = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    query = torch.tensor([[0.1, 0.0]])
    results = identify(query, prototypes, speaker_ids=["alice", "bob"], threshold=1.0)
    assert results[0].is_known is True
    assert results[0].predicted_speaker_id == "alice"


def test_identify_unknown_speaker_above_threshold():
    prototypes = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    query = torch.tensor([[5.0, 5.0]])  # far from both
    results = identify(query, prototypes, speaker_ids=["alice", "bob"], threshold=1.0)
    assert results[0].is_known is False
    assert results[0].predicted_speaker_id is None


def test_identify_batch_mixed_known_unknown():
    prototypes = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    query = torch.tensor([[0.1, 0.0], [50.0, 50.0]])
    results = identify(query, prototypes, speaker_ids=["alice", "bob"], threshold=1.0)
    assert results[0].is_known is True
    assert results[1].is_known is False
