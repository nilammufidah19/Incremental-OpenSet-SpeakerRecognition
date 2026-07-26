"""Tests for src/models/{ecapa,whisper_encoder}.py (F3-03/04/05)."""
from __future__ import annotations

import numpy as np
import pytest

from src.models import ecapa, whisper_encoder
from src.preprocessing.pipeline import preprocess_audio
from tests.conftest import REPO_ROOT, requires_sample_audio


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# --------------------------------------------------------------------------
# ECAPA-TDNN (F3-03)
# --------------------------------------------------------------------------


@requires_sample_audio
def test_ecapa_embedding_shape_and_normalization(sample_audio_manifest):
    path = REPO_ROOT / sample_audio_manifest.iloc[0]["path"]
    windows = preprocess_audio(path, mode="ecapa_inference")
    emb = ecapa.extract_embedding(windows[0])

    assert emb.shape == (ecapa.EMBEDDING_DIM,)
    assert emb.dtype == np.float32
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-4  # L2-normalized


@requires_sample_audio
def test_ecapa_rejects_wrong_sample_rate():
    with pytest.raises(ValueError, match="sr=16000"):
        ecapa.extract_embedding(np.zeros(16000, dtype=np.float32), sr=8000)


# --------------------------------------------------------------------------
# Whisper encoder (F3-04)
# --------------------------------------------------------------------------


@requires_sample_audio
def test_whisper_embedding_shape_and_normalization(sample_audio_manifest):
    path = REPO_ROOT / sample_audio_manifest.iloc[0]["path"]
    windows = preprocess_audio(path, mode="whisper_inference")
    emb = whisper_encoder.extract_embedding(windows[0])

    assert emb.shape == (whisper_encoder.embedding_dim(),)
    assert emb.dtype == np.float32
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-4


def test_whisper_rejects_wrong_sample_rate():
    with pytest.raises(ValueError, match="sr=16000"):
        whisper_encoder.extract_embedding(np.zeros(16000, dtype=np.float32), sr=8000)


# --------------------------------------------------------------------------
# F3-05: quick speaker-verification sanity check on REAL multi-utterance
# speakers -- same-speaker pairs must be closer than different-speaker pairs
# for embeddings to be usable at all downstream.
# --------------------------------------------------------------------------


def _pick_two_speakers_with_multiple_utterances(manifest, n_each: int = 2):
    counts = manifest["speaker_id"].value_counts()
    eligible = counts[counts >= n_each].index.tolist()
    assert len(eligible) >= 2, "need at least 2 speakers with multiple utterances"
    spk_a, spk_b = eligible[0], eligible[1]
    utts_a = manifest.loc[manifest["speaker_id"] == spk_a, "path"].tolist()[:n_each]
    utts_b = manifest.loc[manifest["speaker_id"] == spk_b, "path"].tolist()[:n_each]
    return utts_a, utts_b


@requires_sample_audio
@pytest.mark.parametrize("backbone", ["ecapa", "whisper"])
def test_same_speaker_closer_than_different_speaker(sample_audio_manifest, backbone):
    utts_a, utts_b = _pick_two_speakers_with_multiple_utterances(sample_audio_manifest, n_each=2)

    mode = "ecapa_inference" if backbone == "ecapa" else "whisper_inference"
    extract = ecapa.extract_embedding if backbone == "ecapa" else whisper_encoder.extract_embedding

    emb_a1 = extract(preprocess_audio(REPO_ROOT / utts_a[0], mode=mode)[0])
    emb_a2 = extract(preprocess_audio(REPO_ROOT / utts_a[1], mode=mode)[0])
    emb_b1 = extract(preprocess_audio(REPO_ROOT / utts_b[0], mode=mode)[0])

    sim_same_speaker = cosine_sim(emb_a1, emb_a2)
    sim_diff_speaker = cosine_sim(emb_a1, emb_b1)

    assert sim_same_speaker > sim_diff_speaker, (
        f"[{backbone}] expected same-speaker similarity ({sim_same_speaker:.3f}) "
        f"> different-speaker similarity ({sim_diff_speaker:.3f})"
    )


@requires_sample_audio
def test_extract_embedding_windows_averages_multiple_windows(sample_audio_manifest):
    path = REPO_ROOT / sample_audio_manifest.iloc[0]["path"]
    windows = preprocess_audio(path, mode="ecapa_inference")
    single = ecapa.extract_embedding(windows[0])
    averaged = ecapa.extract_embedding_windows(windows)
    # with exactly one window, averaging must be a no-op (post-renormalization)
    assert np.allclose(single, averaged, atol=1e-5)
