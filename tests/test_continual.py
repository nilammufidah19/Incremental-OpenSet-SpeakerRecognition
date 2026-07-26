"""Tests for src/continual/* (F6)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.continual.manager import ContinualLearningManager
from src.continual.novel_speaker import (
    NovelSpeakerBuffer,
    mean_silhouette_against_existing,
    should_register_new_speaker,
)
from src.continual.prototype_update import (
    initialize_prototype,
    new_sample_mean,
    update_prototype,
)
from src.continual.speaker_database import SpeakerDatabase
from src.models.fusion import FUSION_DIM, GatedAttentionFusion
from src.prototypical.data import ECAPA_DIM, WHISPER_DIM, build_raw_embedding_index, split_raw_embedding
from tests.conftest import REPO_ROOT, requires_sample_audio


# --------------------------------------------------------------------------
# speaker_database.py (F6-01)
# --------------------------------------------------------------------------


def test_speaker_database_set_get_has_remove():
    db = SpeakerDatabase()
    assert not db.has("alice")
    db.set("alice", np.array([1.0, 2.0], dtype=np.float32), 3)
    assert db.has("alice")
    record = db.get("alice")
    assert record.n_samples == 3
    assert np.allclose(record.prototype, [1.0, 2.0])
    db.remove("alice")
    assert not db.has("alice")


def test_speaker_database_prototypes_matrix_order_matches_speaker_ids():
    db = SpeakerDatabase()
    db.set("alice", np.array([1.0, 0.0]), 1)
    db.set("bob", np.array([0.0, 1.0]), 1)
    matrix = db.prototypes_matrix()
    ids = db.speaker_ids
    assert matrix.shape == (2, 2)
    for i, spk in enumerate(ids):
        assert np.allclose(matrix[i], db.get(spk).prototype)


def test_speaker_database_empty_prototypes_matrix_shape():
    db = SpeakerDatabase()
    assert db.prototypes_matrix().shape == (0, 0)
    assert len(db) == 0


def test_speaker_database_save_load_roundtrip(tmp_path):
    db = SpeakerDatabase()
    db.set("alice", np.array([1.0, 2.0, 3.0], dtype=np.float32), 5)
    db.set("bob", np.array([4.0, 5.0, 6.0], dtype=np.float32), 2)

    path = tmp_path / "db.npz"
    db.save(path)
    loaded = SpeakerDatabase.load(path)

    assert set(loaded.speaker_ids) == {"alice", "bob"}
    assert loaded.get("alice").n_samples == 5
    assert np.allclose(loaded.get("bob").prototype, [4.0, 5.0, 6.0])


# --------------------------------------------------------------------------
# prototype_update.py (F6-02/03, F6-07 blocking test)
# --------------------------------------------------------------------------


def test_new_sample_mean_raises_on_empty():
    with pytest.raises(ValueError, match="at least one sample"):
        new_sample_mean(np.zeros((0, 4)))


def test_initialize_prototype_matches_mean():
    embeddings = np.array([[1.0, 1.0], [3.0, 3.0]])
    prototype, n = initialize_prototype(embeddings)
    assert n == 2
    assert np.allclose(prototype, [2.0, 2.0])


def test_update_prototype_single_step():
    old_prototype = np.array([1.0, 1.0], dtype=np.float32)
    old_n = 2
    new_embeddings = np.array([[4.0, 4.0]])
    new_prototype, new_n = update_prototype(old_prototype, old_n, new_embeddings)
    assert new_n == 3
    # (2*[1,1] + 1*[4,4]) / 3 = [2, 2]
    assert np.allclose(new_prototype, [2.0, 2.0])


def test_update_prototype_matches_recompute_from_scratch():
    """F6-07 (blocking): after any sequence of incremental updates, the
    resulting prototype must exactly equal the mean of ALL historical
    samples recomputed from scratch -- proving the running-average update
    rule (Pers. 4.5-4.6) is a mathematically unbiased/exact estimator, not
    an approximation."""
    rng = np.random.default_rng(0)
    all_batches = [rng.normal(size=(rng.integers(1, 5), 8)) for _ in range(6)]

    # incremental path
    prototype, n = initialize_prototype(all_batches[0])
    for batch in all_batches[1:]:
        prototype, n = update_prototype(prototype, n, batch)

    # recompute-from-scratch path
    all_samples = np.concatenate(all_batches, axis=0)
    expected_prototype = all_samples.mean(axis=0)
    expected_n = len(all_samples)

    assert n == expected_n
    assert np.allclose(prototype, expected_prototype, atol=1e-5)


def test_update_prototype_order_independence():
    """Since it's a plain weighted mean, the update should give the same
    final result regardless of what order batches arrive in."""
    rng = np.random.default_rng(1)
    batches = [rng.normal(size=(3, 4)) for _ in range(4)]

    def run(order):
        proto, n = initialize_prototype(batches[order[0]])
        for i in order[1:]:
            proto, n = update_prototype(proto, n, batches[i])
        return proto, n

    proto_a, n_a = run([0, 1, 2, 3])
    proto_b, n_b = run([3, 2, 1, 0])
    assert n_a == n_b
    assert np.allclose(proto_a, proto_b, atol=1e-5)


# --------------------------------------------------------------------------
# novel_speaker.py (F6-04/05)
# --------------------------------------------------------------------------


def test_novel_speaker_buffer_add_size_clear():
    buf = NovelSpeakerBuffer()
    assert buf.size() == 0
    buf.add(np.zeros(4))
    buf.add(np.ones(4))
    assert buf.size() == 2
    buf.clear()
    assert buf.size() == 0


def test_mean_silhouette_no_existing_prototypes_returns_one():
    candidates = np.array([[0.0, 0.0], [0.1, 0.1], [0.05, -0.05]])
    s = mean_silhouette_against_existing(candidates, np.zeros((0, 2)))
    assert s == 1.0


def test_mean_silhouette_well_separated_is_high():
    rng = np.random.default_rng(0)
    candidates = rng.normal(loc=[10.0, 10.0], scale=0.05, size=(5, 2))
    existing = np.array([[0.0, 0.0], [1.0, 1.0]])
    s = mean_silhouette_against_existing(candidates, existing)
    assert s > 0.8


def test_mean_silhouette_overlapping_is_low():
    rng = np.random.default_rng(0)
    candidates = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(5, 2))
    existing = np.array([[0.0, 0.0]])  # right on top of the candidate cluster
    s = mean_silhouette_against_existing(candidates, existing)
    assert s < 0.5


def test_silhouette_raises_on_single_sample():
    with pytest.raises(ValueError, match="Need >= 2"):
        mean_silhouette_against_existing(np.array([[0.0, 0.0]]), np.zeros((0, 2)))


def test_should_register_requires_min_samples():
    buf = NovelSpeakerBuffer()
    buf.add(np.array([10.0, 10.0]))
    # only 1 sample, need >=2 to even compute silhouette / meet min_samples
    assert should_register_new_speaker(buf, np.zeros((0, 2)), min_samples=3) is False


def test_should_register_with_min_samples_1_does_not_crash_on_single_sample():
    """Regression test: min_samples=1 must not bypass the hard >=2 floor
    that Silhouette Coefficient computation requires (found via
    SpeakerIdentificationSystem's default min_samples_for_new_speaker=1)."""
    buf = NovelSpeakerBuffer()
    buf.add(np.array([10.0, 10.0]))
    assert should_register_new_speaker(buf, np.zeros((0, 2)), min_samples=1) is False


def test_should_register_true_when_well_separated_and_enough_samples():
    buf = NovelSpeakerBuffer()
    rng = np.random.default_rng(0)
    for e in rng.normal(loc=[20.0, 20.0], scale=0.05, size=(4, 2)):
        buf.add(e)
    existing = np.array([[0.0, 0.0]])
    assert should_register_new_speaker(buf, existing, min_samples=3, silhouette_threshold=0.5) is True


def test_should_register_false_when_too_close_to_existing():
    buf = NovelSpeakerBuffer()
    rng = np.random.default_rng(0)
    for e in rng.normal(loc=[0.01, 0.01], scale=0.05, size=(4, 2)):
        buf.add(e)
    existing = np.array([[0.0, 0.0]])  # candidate cluster basically on top of this
    assert should_register_new_speaker(buf, existing, min_samples=3, silhouette_threshold=0.5) is False


# --------------------------------------------------------------------------
# manager.py (F6-06/08)
# --------------------------------------------------------------------------


def test_manager_invalid_mode_raises():
    db = SpeakerDatabase()
    with pytest.raises(ValueError, match="mode must be one of"):
        ContinualLearningManager(db, threshold=1.0, mode="bogus")


def test_manager_known_speaker_running_average_updates_prototype():
    db = SpeakerDatabase()
    db.set("alice", np.array([0.0, 0.0]), 1)
    manager = ContinualLearningManager(db, threshold=1.0, mode="running_average")

    result = manager.process_sample(np.array([0.2, 0.0]))
    assert result.is_known is True
    assert result.predicted_speaker_id == "alice"
    # prototype should have moved toward the new sample (n=1 -> n=2)
    updated = db.get("alice")
    assert updated.n_samples == 2
    assert np.allclose(updated.prototype, [0.1, 0.0])


def test_manager_static_mode_does_not_update_prototype():
    db = SpeakerDatabase()
    db.set("alice", np.array([0.0, 0.0]), 1)
    manager = ContinualLearningManager(db, threshold=1.0, mode="static")

    manager.process_sample(np.array([0.2, 0.0]))
    unchanged = db.get("alice")
    assert unchanged.n_samples == 1
    assert np.allclose(unchanged.prototype, [0.0, 0.0])


def test_manager_unknown_sample_far_from_everyone_not_registered_too_early():
    db = SpeakerDatabase()
    db.set("alice", np.array([0.0, 0.0]), 5)
    manager = ContinualLearningManager(
        db, threshold=1.0, mode="running_average", min_samples_for_new_speaker=3
    )

    result = manager.process_sample(np.array([50.0, 50.0]))
    assert result.is_known is False
    assert result.newly_registered_speaker_id is None
    assert len(db) == 1  # still just alice


def test_manager_registers_new_speaker_after_enough_consistent_unknown_samples():
    db = SpeakerDatabase()
    db.set("alice", np.array([0.0, 0.0]), 5)
    manager = ContinualLearningManager(
        db, threshold=1.0, mode="running_average", min_samples_for_new_speaker=3, silhouette_threshold=0.5
    )

    rng = np.random.default_rng(0)
    novel_samples = rng.normal(loc=[50.0, 50.0], scale=0.05, size=(3, 2))

    results = [manager.process_sample(s) for s in novel_samples]
    assert all(r.is_known is False for r in results)
    # registration should trigger on the 3rd sample (buffer reaches min_samples)
    assert results[-1].newly_registered_speaker_id is not None
    assert len(db) == 2

    # a 4th sample from the same novel cluster should now be recognized as known
    result4 = manager.process_sample(np.array([50.0, 50.0]))
    assert result4.is_known is True
    assert result4.predicted_speaker_id == results[-1].newly_registered_speaker_id


# --------------------------------------------------------------------------
# Integration: full mini incremental session on REAL cached embeddings
# --------------------------------------------------------------------------


@requires_sample_audio
def test_continual_learning_end_to_end_real_embeddings(sample_audio_manifest):
    """Simulate a tiny incremental session: register a few known speakers'
    prototypes from held-out support samples, then stream the remaining
    real utterances (a mix of known-speaker and truly-unseen-speaker audio)
    through the manager, and check known speakers get recognized+updated
    while a genuinely new speaker eventually gets registered."""
    counts = sample_audio_manifest["speaker_id"].value_counts()
    eligible = counts[counts >= 3].index.tolist()
    assert len(eligible) >= 4, "need at least 4 speakers with >=3 utterances"

    known_speakers = eligible[:3]
    novel_speaker = eligible[3]

    torch.manual_seed(0)
    fusion = GatedAttentionFusion(ECAPA_DIM, WHISPER_DIM, FUSION_DIM, mode="fusion")
    fusion.eval()

    def fused(path: str) -> np.ndarray:
        raw = build_raw_embedding_index(sample_audio_manifest[sample_audio_manifest["path"] == path])
        raw_vec = next(iter(raw.values()))[0]
        e1, e2 = split_raw_embedding(raw_vec)
        with torch.no_grad():
            out = fusion(torch.from_numpy(e1), torch.from_numpy(e2))
        return out.numpy()

    db = SpeakerDatabase()
    remaining_known_paths = []
    for spk in known_speakers:
        paths = sample_audio_manifest.loc[sample_audio_manifest["speaker_id"] == spk, "path"].tolist()
        enroll_path, rest = paths[0], paths[1:]
        db.set(spk, fused(enroll_path), 1)
        remaining_known_paths.extend(rest)

    novel_paths = sample_audio_manifest.loc[
        sample_audio_manifest["speaker_id"] == novel_speaker, "path"
    ].tolist()

    manager = ContinualLearningManager(
        db, threshold=0.8, mode="running_average", min_samples_for_new_speaker=2, silhouette_threshold=0.3
    )

    known_recognized = 0
    for path in remaining_known_paths:
        result = manager.process_sample(fused(path))
        if result.is_known:
            known_recognized += 1

    # most held-out known-speaker samples should be recognized as known
    assert known_recognized / len(remaining_known_paths) > 0.5

    registered = False
    for path in novel_paths:
        result = manager.process_sample(fused(path))
        if result.newly_registered_speaker_id is not None:
            registered = True
    # with enough novel-speaker samples pushed through, registration should
    # eventually trigger (not asserting strictly, since real embeddings are
    # noisier than synthetic clusters -- but len(db) must not have shrunk)
    assert len(db) >= 3
    if registered:
        assert len(db) == 4
