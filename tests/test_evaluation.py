"""Tests for src/evaluation/{metrics,fscil}.py (F8)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation.fscil import run_fscil
from src.evaluation.metrics import accuracy, average_accuracy, forgetting_measure
from src.models.fusion import FUSION_DIM, GatedAttentionFusion
from src.prototypical.data import ECAPA_DIM, WHISPER_DIM
from src.system import SpeakerIdentificationSystem
from tests.conftest import REPO_ROOT, requires_sample_audio


# --------------------------------------------------------------------------
# metrics.py
# --------------------------------------------------------------------------


def test_accuracy_basic():
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 2])
    assert accuracy(y_true, y_pred) == pytest.approx(3 / 5)


def test_accuracy_raises_on_empty():
    with pytest.raises(ValueError, match="non-empty"):
        accuracy(np.array([]), np.array([]))


def test_average_accuracy_simple():
    session_task_accuracy = {0: {0: 0.9}, 1: {0: 0.8, 1: 0.7}}
    assert average_accuracy(session_task_accuracy, final_session=1) == pytest.approx((0.8 + 0.7) / 2)


def test_average_accuracy_raises_if_no_entries():
    with pytest.raises(ValueError, match="No task accuracies"):
        average_accuracy({0: {}}, final_session=0)


def test_forgetting_measure_no_forgetting_is_zero():
    # task 0 always at 0.9, never drops -> zero forgetting
    session_task_accuracy = {0: {0: 0.9}, 1: {0: 0.9, 1: 0.8}}
    fm = forgetting_measure(session_task_accuracy, final_session=1)
    assert fm == pytest.approx(0.0)


def test_forgetting_measure_detects_drop():
    # task 0 peaked at 0.9 right after session 0, dropped to 0.5 by session 1
    session_task_accuracy = {0: {0: 0.9}, 1: {0: 0.5, 1: 0.8}}
    fm = forgetting_measure(session_task_accuracy, final_session=1)
    assert fm == pytest.approx(0.4)


def test_forgetting_measure_requires_at_least_two_sessions():
    with pytest.raises(ValueError, match="at least 2 sessions"):
        forgetting_measure({0: {0: 0.9}}, final_session=0)


# --------------------------------------------------------------------------
# fscil.py -- small real-data run (substitute sessions from vox1_sample,
# since real task_speakers audio is being fetched separately in the
# background -- see scripts/fetch_capped_eval_audio.py)
# --------------------------------------------------------------------------


@requires_sample_audio
def test_run_fscil_small_real_scenario(sample_audio_manifest):
    counts = sample_audio_manifest["speaker_id"].value_counts()
    eligible = counts[counts >= 3].index.tolist()
    assert len(eligible) >= 6, "need at least 6 multi-utterance speakers for a 2-session x 3-way test"

    speaker_audio_paths = {
        spk: [REPO_ROOT / p for p in sample_audio_manifest.loc[sample_audio_manifest["speaker_id"] == spk, "path"]]
        for spk in eligible
    }
    sessions = [eligible[0:3], eligible[3:6]]  # 2 sessions x 3-way

    torch.manual_seed(0)
    fusion = GatedAttentionFusion(ECAPA_DIM, WHISPER_DIM, FUSION_DIM, mode="fusion")
    system = SpeakerIdentificationSystem(fusion, threshold=0.9, continual_mode="running_average")

    result = run_fscil(system, sessions, speaker_audio_paths, k_shot=1, n_query=2, seed=0)

    assert set(result.keys()) == {0, 1}
    assert set(result[0].keys()) == {0}       # after session 0, only task 0 exists
    assert set(result[1].keys()) == {0, 1}    # after session 1, tasks 0 and 1 both tested
    for session_accs in result.values():
        for acc in session_accs.values():
            assert 0.0 <= acc <= 1.0

    avg_acc = average_accuracy(result, final_session=1)
    fm = forgetting_measure(result, final_session=1)
    assert 0.0 <= avg_acc <= 1.0
    assert isinstance(fm, float)
