"""Tests for src/system.py (F7-01, F7-04 regression test)."""
from __future__ import annotations

import torch

from src.models.fusion import FUSION_DIM, GatedAttentionFusion
from src.prototypical.data import ECAPA_DIM, WHISPER_DIM
from src.system import SpeakerIdentificationSystem
from tests.conftest import REPO_ROOT, requires_sample_audio


def make_system(mode="running_average") -> SpeakerIdentificationSystem:
    torch.manual_seed(0)
    fusion = GatedAttentionFusion(ECAPA_DIM, WHISPER_DIM, FUSION_DIM, mode="fusion")
    return SpeakerIdentificationSystem(fusion, threshold=0.8, continual_mode=mode)


@requires_sample_audio
def test_embed_real_audio_produces_normalized_vector(sample_audio_manifest):
    system = make_system()
    path = REPO_ROOT / sample_audio_manifest.iloc[0]["path"]
    emb = system.embed(path)
    assert emb.shape == (FUSION_DIM,)
    assert abs(float((emb**2).sum() ** 0.5) - 1.0) < 1e-4


@requires_sample_audio
def test_enroll_then_process_recognizes_same_speaker(sample_audio_manifest):
    counts = sample_audio_manifest["speaker_id"].value_counts()
    eligible = counts[counts >= 2].index.tolist()
    spk = eligible[0]
    paths = sample_audio_manifest.loc[sample_audio_manifest["speaker_id"] == spk, "path"].tolist()

    system = make_system()
    system.enroll(spk, [REPO_ROOT / paths[0]])
    result = system.process(REPO_ROOT / paths[1])

    assert result.is_known is True
    assert result.predicted_speaker_id == spk


@requires_sample_audio
def test_process_unknown_speaker_not_recognized_as_enrolled(sample_audio_manifest):
    counts = sample_audio_manifest["speaker_id"].value_counts()
    eligible = counts[counts >= 1].index.tolist()
    spk_a, spk_b = eligible[0], eligible[1]
    path_a = sample_audio_manifest.loc[sample_audio_manifest["speaker_id"] == spk_a, "path"].iloc[0]
    path_b = sample_audio_manifest.loc[sample_audio_manifest["speaker_id"] == spk_b, "path"].iloc[0]

    system = make_system()
    system.enroll(spk_a, [REPO_ROOT / path_a])
    # low threshold: only near-identical embeddings count as known
    system.threshold = 0.05
    system.manager.threshold = 0.05
    result = system.process(REPO_ROOT / path_b)

    assert result.predicted_speaker_id != spk_a or result.is_known is False


@requires_sample_audio
def test_save_load_roundtrip_preserves_behavior(tmp_path, sample_audio_manifest):
    counts = sample_audio_manifest["speaker_id"].value_counts()
    eligible = counts[counts >= 2].index.tolist()
    spk = eligible[0]
    paths = sample_audio_manifest.loc[sample_audio_manifest["speaker_id"] == spk, "path"].tolist()

    system = make_system()
    system.enroll(spk, [REPO_ROOT / paths[0]])
    system.save(tmp_path / "ckpt")

    loaded = SpeakerIdentificationSystem.load(tmp_path / "ckpt")
    result = loaded.process(REPO_ROOT / paths[1])

    assert result.is_known is True
    assert result.predicted_speaker_id == spk


@requires_sample_audio
def test_full_pipeline_regression_no_crash_small_subset(sample_audio_manifest):
    """F7-04: run the entire pipeline (preprocessing -> backbone -> fusion
    -> prototype decision -> continual update) end-to-end on a small real
    subset and make sure nothing crashes / produces NaNs."""
    system = make_system()
    subset = sample_audio_manifest.iloc[:10]
    for _, row in subset.iterrows():
        result = system.process(REPO_ROOT / row["path"])
        assert result.min_distance == result.min_distance  # not NaN
    assert len(system.database) >= 1  # at least some novel speakers accumulated/registered
