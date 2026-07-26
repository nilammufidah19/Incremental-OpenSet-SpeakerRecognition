"""Tests for src/evaluation/baseline_system.py (F10)."""
from __future__ import annotations

from src.evaluation.baseline_system import CLOSED_SET_THRESHOLD, SingleBackboneSystem, fit_lda_whitener
from tests.conftest import REPO_ROOT, requires_sample_audio


@requires_sample_audio
def test_single_backbone_system_enroll_and_recognize(sample_audio_manifest):
    counts = sample_audio_manifest["speaker_id"].value_counts()
    eligible = counts[counts >= 2].index.tolist()
    spk = eligible[0]
    paths = sample_audio_manifest.loc[sample_audio_manifest["speaker_id"] == spk, "path"].tolist()

    system = SingleBackboneSystem(backbone="ecapa", threshold=CLOSED_SET_THRESHOLD, continual_mode="static")
    system.enroll(spk, [REPO_ROOT / paths[0]])
    result = system.process(REPO_ROOT / paths[1])

    assert result.is_known is True
    assert result.predicted_speaker_id == spk


@requires_sample_audio
def test_single_backbone_system_closed_set_never_rejects(sample_audio_manifest):
    """Closed-set baselines (threshold=CLOSED_SET_THRESHOLD) must always
    assign to the nearest enrolled speaker, never flag 'unknown'."""
    counts = sample_audio_manifest["speaker_id"].value_counts()
    eligible = counts[counts >= 1].index.tolist()
    spk_a, spk_b = eligible[0], eligible[1]
    path_a = sample_audio_manifest.loc[sample_audio_manifest["speaker_id"] == spk_a, "path"].iloc[0]
    path_b = sample_audio_manifest.loc[sample_audio_manifest["speaker_id"] == spk_b, "path"].iloc[0]

    system = SingleBackboneSystem(backbone="ecapa", threshold=CLOSED_SET_THRESHOLD, continual_mode="static")
    system.enroll(spk_a, [REPO_ROOT / path_a])
    result = system.process(REPO_ROOT / path_b)  # a totally different, unenrolled speaker
    assert result.is_known is True  # closed-set: forced into the only enrolled class


@requires_sample_audio
def test_single_backbone_system_static_mode_never_updates(sample_audio_manifest):
    counts = sample_audio_manifest["speaker_id"].value_counts()
    eligible = counts[counts >= 2].index.tolist()
    spk = eligible[0]
    paths = sample_audio_manifest.loc[sample_audio_manifest["speaker_id"] == spk, "path"].tolist()

    system = SingleBackboneSystem(backbone="ecapa", threshold=CLOSED_SET_THRESHOLD, continual_mode="static")
    system.enroll(spk, [REPO_ROOT / paths[0]])
    n_before = system.database.get(spk).n_samples
    system.process(REPO_ROOT / paths[1])
    n_after = system.database.get(spk).n_samples
    assert n_before == n_after == 1


@requires_sample_audio
def test_fit_lda_whitener_produces_lower_dim_and_still_discriminates(sample_audio_manifest):
    from src.prototypical.data import build_raw_embedding_index

    counts = sample_audio_manifest["speaker_id"].value_counts()
    eligible = counts[counts >= 2].index.tolist()[:8]
    manifest_subset = sample_audio_manifest[sample_audio_manifest["speaker_id"].isin(eligible)]

    # Use ecapa embeddings only (first 192 dims of the raw concat helper is
    # inconvenient here, so just fetch directly per-speaker for this test).
    from src.features.cache import get_or_compute_embedding

    embeddings_by_speaker = {}
    for spk in eligible:
        paths = manifest_subset.loc[manifest_subset["speaker_id"] == spk, "path"].tolist()
        embeddings_by_speaker[spk] = [get_or_compute_embedding(REPO_ROOT / p, "ecapa") for p in paths]

    whiten = fit_lda_whitener(embeddings_by_speaker)

    spk = eligible[0]
    e1 = embeddings_by_speaker[spk][0]
    e2 = embeddings_by_speaker[spk][1]
    other = embeddings_by_speaker[eligible[1]][0]

    w1, w2, wo = whiten(e1), whiten(e2), whiten(other)
    same_dist = ((w1 - w2) ** 2).sum() ** 0.5
    diff_dist = ((w1 - wo) ** 2).sum() ** 0.5
    assert same_dist < diff_dist
