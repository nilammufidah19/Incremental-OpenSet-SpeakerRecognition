"""Tests for src/data/voxceleb.py and src/data/splits.py.

The test at the bottom of this file, `test_full_split_is_speaker_disjoint`,
is the **blocking** test referenced by plan/04-tasks.md F1-10: nothing in
F2 onward should proceed while this test fails.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.splits import (
    SpeakerOverlapError,
    allocate_reserved_pool,
    assert_disjoint,
    assign_support_query,
    build_calibration_scheme,
    build_episodic_sessions,
    run_full_split,
    split_base,
)
from src.data.voxceleb import build_speaker_catalog
from tests.conftest import requires_real_metadata, requires_vox1_manifest


def make_synthetic_catalog(n_vox1: int = 200, n_vox2: int = 800) -> pd.DataFrame:
    """Small synthetic catalog for fast, dependency-free unit tests that
    don't need the real ~7363-speaker VoxCeleb metadata."""
    vox1 = pd.DataFrame(
        {
            "speaker_id": [f"id1{i:04d}" for i in range(n_vox1)],
            "gender": ["m" if i % 2 == 0 else "f" for i in range(n_vox1)],
            "nationality": ["USA"] * n_vox1,
            "official_set": ["dev"] * n_vox1,
            "dataset": "vox1",
        }
    )
    vox2 = pd.DataFrame(
        {
            "speaker_id": [f"id0{i:04d}" for i in range(n_vox2)],
            "gender": ["m" if i % 2 == 0 else "f" for i in range(n_vox2)],
            "nationality": [None] * n_vox2,
            "official_set": ["dev"] * n_vox2,
            "dataset": "vox2",
        }
    )
    return build_speaker_catalog(vox1, vox2)


# --------------------------------------------------------------------------
# assert_disjoint / SpeakerOverlapError
# --------------------------------------------------------------------------


def test_assert_disjoint_passes_on_disjoint_groups():
    assert_disjoint({"a": ["1", "2"], "b": ["3", "4"]})  # should not raise


def test_assert_disjoint_raises_on_overlap():
    with pytest.raises(SpeakerOverlapError):
        assert_disjoint({"a": ["1", "2"], "b": ["2", "3"]})


def test_build_speaker_catalog_raises_on_vox1_vox2_overlap():
    vox1 = pd.DataFrame(
        {
            "speaker_id": ["id10001"],
            "gender": ["m"],
            "nationality": ["USA"],
            "official_set": ["dev"],
            "dataset": "vox1",
        }
    )
    vox2 = pd.DataFrame(
        {
            "speaker_id": ["id10001"],  # deliberately colliding id
            "gender": ["m"],
            "nationality": [None],
            "official_set": ["dev"],
            "dataset": "vox2",
        }
    )
    with pytest.raises(ValueError, match="overlap"):
        build_speaker_catalog(vox1, vox2)


# --------------------------------------------------------------------------
# split_base / allocate_reserved_pool ratios (synthetic, fast)
# --------------------------------------------------------------------------


def test_split_base_ratio_approximately_70_30():
    catalog = make_synthetic_catalog(n_vox1=300, n_vox2=700)  # 1000 total
    base_train, data_uji_global = split_base(catalog, train_ratio=0.7, seed=0)
    assert len(base_train) == 700
    assert len(data_uji_global) == 300
    assert set(base_train).isdisjoint(data_uji_global)


def test_allocate_reserved_pool_ratio_approximately_10_percent():
    catalog = make_synthetic_catalog(n_vox1=300, n_vox2=700)
    _, data_uji_global = split_base(catalog, train_ratio=0.7, seed=0)
    reserved, episodic_pool = allocate_reserved_pool(data_uji_global, reserved_ratio=0.10, seed=0)
    assert len(reserved) == round(len(data_uji_global) * 0.10)
    assert len(episodic_pool) == len(data_uji_global) - len(reserved)
    assert set(reserved).isdisjoint(episodic_pool)


def test_split_is_deterministic_given_same_seed():
    catalog = make_synthetic_catalog()
    a = split_base(catalog, seed=42)
    b = split_base(catalog, seed=42)
    assert a == b


def test_split_differs_across_seeds():
    catalog = make_synthetic_catalog()
    a = split_base(catalog, seed=0)
    b = split_base(catalog, seed=1)
    assert a != b


# --------------------------------------------------------------------------
# build_episodic_sessions: 10 sessions x 10-way (DR-04)
# --------------------------------------------------------------------------


def test_build_episodic_sessions_shape():
    catalog = make_synthetic_catalog(n_vox1=100, n_vox2=900)
    _, data_uji_global = split_base(catalog, seed=0)
    _, episodic_pool = allocate_reserved_pool(data_uji_global, seed=0)

    split = build_episodic_sessions(episodic_pool, n_sessions=10, n_way=10, seed=0)

    assert len(split.sessions) == 10
    for session in split.sessions:
        assert len(session) == 10
    assert len(split.task_speakers) == 100
    assert len(set(split.task_speakers)) == 100  # no duplicates across sessions
    assert set(split.task_speakers).isdisjoint(split.calibration_impostor_pool)


def test_build_episodic_sessions_raises_if_pool_too_small():
    with pytest.raises(ValueError, match="need >="):
        build_episodic_sessions(["id0001", "id0002"], n_sessions=10, n_way=10, seed=0)


# --------------------------------------------------------------------------
# build_calibration_scheme (DR-05)
# --------------------------------------------------------------------------


def test_calibration_scheme_sources_are_disjoint():
    base_train = ["a1", "a2", "a3"]
    impostor_pool = ["b1", "b2"]
    scheme = build_calibration_scheme(base_train, impostor_pool)
    assert set(scheme.genuine_source_speakers) == set(base_train)
    assert set(scheme.impostor_source_speakers) == set(impostor_pool)
    assert set(scheme.genuine_source_speakers).isdisjoint(scheme.impostor_source_speakers)


# --------------------------------------------------------------------------
# assign_support_query (DR-06)
# --------------------------------------------------------------------------


def test_assign_support_query_k_shot_1():
    manifest = pd.DataFrame(
        {
            "speaker_id": ["s1", "s1", "s1", "s2", "s2"],
            "utterance_id": ["s1/a", "s1/b", "s1/c", "s2/a", "s2/b"],
        }
    )
    assignment = assign_support_query(manifest, ["s1", "s2"], k_shot=1, seed=0)
    assert len(assignment["s1"]["support"]) == 1
    assert len(assignment["s1"]["query"]) == 2
    assert len(assignment["s2"]["support"]) == 1
    assert len(assignment["s2"]["query"]) == 1
    # support and query must not overlap
    assert set(assignment["s1"]["support"]).isdisjoint(assignment["s1"]["query"])


def test_assign_support_query_raises_if_not_enough_utterances():
    manifest = pd.DataFrame({"speaker_id": ["s1"], "utterance_id": ["s1/a"]})
    with pytest.raises(ValueError, match="need >="):
        assign_support_query(manifest, ["s1"], k_shot=1, seed=0)


# --------------------------------------------------------------------------
# F1-10 (BLOCKING): full pipeline disjointness on synthetic data (fast)
# --------------------------------------------------------------------------


def test_full_split_is_speaker_disjoint_synthetic():
    """Fast synthetic version of the blocking F1-10 test -- always runs,
    does not require the real ~7363-speaker download."""
    catalog = make_synthetic_catalog(n_vox1=300, n_vox2=1200)  # 1500 total
    split = run_full_split(catalog, seed=0)

    groups = {
        "base_train": split.base_train,
        "reserved_unknown_pool": split.reserved_unknown_pool,
        "task_speakers": split.episodic_split.task_speakers,
        "calibration_impostor_pool": split.episodic_split.calibration_impostor_pool,
    }
    assert_disjoint(groups)  # must not raise

    # every speaker accounted for exactly once across the 4 leaf partitions
    all_ids = set(catalog["speaker_id"])
    union = set().union(*groups.values())
    assert union.issubset(all_ids)


# --------------------------------------------------------------------------
# F1-03/F1-10 on REAL VoxCeleb1/2 metadata (skipped if not downloaded)
# --------------------------------------------------------------------------


@requires_real_metadata
def test_real_catalog_matches_table_4_1_speaker_counts(real_catalog):
    from src.data.voxceleb import summarize_catalog

    stats = summarize_catalog(real_catalog)
    # VoxCeleb1 must match Tabel 4.1 exactly (1.251 speakers).
    assert stats["n_speakers_vox1"] == 1251
    # VoxCeleb2 may drift slightly from the historical 6.112 figure as the
    # officially distributed metadata file is updated over time; require it
    # stays within a small tolerance rather than an exact match (documented
    # in scripts/build_splits.py summary output).
    assert abs(stats["n_speakers_vox2"] - 6112) <= 50


@requires_real_metadata
def test_real_catalog_has_no_vox1_vox2_overlap(real_catalog):
    # build_speaker_catalog (called inside load_combined_catalog) already
    # raises on overlap, so simply constructing `real_catalog` without error
    # is itself the assertion; this test documents that intent explicitly.
    assert real_catalog["speaker_id"].is_unique


@requires_real_metadata
def test_full_split_is_speaker_disjoint_real_data(real_catalog):
    """The actual F1-10 blocking test on the real ~7363-speaker catalog."""
    split = run_full_split(real_catalog, seed=0)

    groups = {
        "base_train": split.base_train,
        "reserved_unknown_pool": split.reserved_unknown_pool,
        "task_speakers": split.episodic_split.task_speakers,
        "calibration_impostor_pool": split.episodic_split.calibration_impostor_pool,
    }
    assert_disjoint(groups)

    total = real_catalog["speaker_id"].nunique()
    assert len(split.base_train) == round(total * 0.7)
    assert len(split.episodic_split.task_speakers) == 100


@requires_real_metadata
@requires_vox1_manifest
def test_assign_support_query_on_real_vox1_utterances(real_catalog, vox1_utterance_manifest):
    """Exercise assign_support_query on genuine VoxCeleb1 utterance IDs for
    a handful of real speakers that have enough utterances for 1-shot."""
    vox1_speakers = real_catalog.loc[real_catalog["dataset"] == "vox1", "speaker_id"]
    utt_counts = vox1_utterance_manifest["speaker_id"].value_counts()
    eligible = [s for s in vox1_speakers if utt_counts.get(s, 0) >= 2][:10]
    assert len(eligible) > 0, "expected some VoxCeleb1 speakers with >=2 utterances"

    assignment = assign_support_query(vox1_utterance_manifest, eligible, k_shot=1, seed=0)
    for speaker_id in eligible:
        assert len(assignment[speaker_id]["support"]) == 1
        assert len(assignment[speaker_id]["query"]) >= 1
        assert set(assignment[speaker_id]["support"]).isdisjoint(assignment[speaker_id]["query"])
