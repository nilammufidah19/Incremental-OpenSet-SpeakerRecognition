"""Speaker-disjoint data splitting for the FSCIL-style evaluation protocol.

Ref: plan/03-architecture.md (data flow), plan/04-tasks.md F1-05..F1-09,
plan/02-requirements.md DR-02..DR-06, proposal Bab 4.3.

Partition hierarchy produced by `run_full_split`:

    combined catalog (~7363 speakers, vox1+vox2, speaker-disjoint)
    |
    +-- base_train (70%)              -> "Data Latih Awal"
    |
    +-- data_uji_global (30%)         -> "Data Uji Global"
         |
         +-- reserved_unknown_pool (10% of data_uji_global) -> "data simpanan"
         |
         +-- episodic_pool (remaining 90%)
              |
              +-- task_speakers (100 speakers: 10 sessions x 10-way)
              |
              +-- calibration_impostor_pool (episodic_pool - task_speakers)

Every function here is deterministic given a seed and operates purely on
speaker ID lists (no audio I/O), so it runs today even though the raw
VoxCeleb audio has not been downloaded -- see scripts/download_voxceleb.py
for the (separate, credential-gated) audio acquisition step.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import pandas as pd


class SpeakerOverlapError(ValueError):
    """Raised when two supposedly disjoint speaker-id groups overlap."""


def assert_disjoint(named_groups: dict[str, list[str]]) -> None:
    """Assert every pair of named speaker-id groups is disjoint.

    This is the single choke point used both internally by `run_full_split`
    and by the blocking test in tests/test_data_splits.py (F1-10) -- there
    must be exactly one implementation of "what counts as leakage".
    """
    names = list(named_groups)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = set(named_groups[a]) & set(named_groups[b])
            if overlap:
                raise SpeakerOverlapError(
                    f"Speaker-id leakage between '{a}' and '{b}': "
                    f"{len(overlap)} overlapping id(s), e.g. {sorted(overlap)[:5]}"
                )


def split_base(
    catalog: pd.DataFrame, train_ratio: float = 0.7, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Level Global (Base Split): 70% Data Latih Awal / 30% Data Uji Global.

    [DR-02] Split is performed on the full speaker list (already verified
    speaker-disjoint across vox1/vox2 in voxceleb.build_speaker_catalog).
    """
    speaker_ids = sorted(catalog["speaker_id"].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(speaker_ids)

    n_train = round(len(speaker_ids) * train_ratio)
    base_train = sorted(speaker_ids[:n_train])
    data_uji_global = sorted(speaker_ids[n_train:])
    return base_train, data_uji_global


def allocate_reserved_pool(
    data_uji_global: list[str], reserved_ratio: float = 0.10, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Carve out the 10% 'data simpanan' unknown pool from Data Uji Global.

    [DR-03] Returns (reserved_unknown_pool, episodic_pool).
    """
    speaker_ids = sorted(data_uji_global)
    rng = random.Random(seed + 1)  # distinct stream from split_base
    rng.shuffle(speaker_ids)

    n_reserved = round(len(speaker_ids) * reserved_ratio)
    reserved_unknown_pool = sorted(speaker_ids[:n_reserved])
    episodic_pool = sorted(speaker_ids[n_reserved:])
    return reserved_unknown_pool, episodic_pool


def split_reserved_pool_halves(
    reserved_speaker_ids: list[str], seed: int = 0
) -> tuple[list[str], list[str]]:
    """Experiment 3: deterministically split the reserved_unknown_pool into
    (validation_half, detection_half).

    validation_half -- speakers used ONLY to build the validation FSCIL task
        on which threshold hyperparameters (target_frr, AS-Norm params) are
        selected and locked (scripts/exp3_validation_sweep.py), per the
        anti-overfit protocol of docs/experiment-3.md section 7.
    detection_half -- speakers used ONLY as real unknown queries for the
        open-set detection metrics (EER/AUROC/TAR@FAR) of exp3c.

    Keeping the two roles speaker-disjoint (and both disjoint from
    task_speakers/base_train by construction of the reserved pool) is the
    single leakage guard for Experiment 3; both consumers must call THIS
    function rather than re-deriving the halves.
    """
    ids = sorted(set(reserved_speaker_ids))
    rng = random.Random(seed + 3)  # distinct stream from the other split fns
    rng.shuffle(ids)
    mid = len(ids) // 2
    return sorted(ids[:mid]), sorted(ids[mid:])


@dataclass
class EpisodicSplit:
    sessions: list[list[str]]  # 10 sessions, each a list of 10 speaker ids
    task_speakers: list[str] = field(init=False)
    calibration_impostor_pool: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.task_speakers = sorted(s for session in self.sessions for s in session)


def build_episodic_sessions(
    episodic_pool: list[str],
    n_sessions: int = 10,
    n_way: int = 10,
    seed: int = 0,
) -> EpisodicSplit:
    """Level Task/Episodic Split: draw `n_sessions x n_way` speakers for the
    FSCIL-style incremental sessions; the remainder of `episodic_pool`
    becomes the impostor pool used later for open-set threshold calibration.

    [DR-04] Adapts Tao et al. (2020) FSCIL protocol: 10 sessions x 10-way.
    """
    n_needed = n_sessions * n_way
    if len(episodic_pool) < n_needed:
        raise ValueError(
            f"episodic_pool has only {len(episodic_pool)} speakers, "
            f"need >= {n_needed} for {n_sessions} sessions x {n_way}-way"
        )

    speaker_ids = sorted(episodic_pool)
    rng = random.Random(seed + 2)  # distinct stream from base/reserved splits
    rng.shuffle(speaker_ids)

    selected = speaker_ids[:n_needed]
    sessions = [
        sorted(selected[i * n_way : (i + 1) * n_way]) for i in range(n_sessions)
    ]
    calibration_impostor_pool = sorted(speaker_ids[n_needed:])

    split = EpisodicSplit(sessions=sessions)
    split.calibration_impostor_pool = calibration_impostor_pool
    return split


def assign_support_query(
    utterance_manifest: pd.DataFrame,
    speaker_ids: list[str],
    k_shot: int = 1,
    seed: int = 0,
    max_query_per_speaker: int | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Assign support/query utterances per speaker for one N-way K-shot task.

    [DR-06] Support set: k_shot utterances/speaker (adaptation data). Query
    set: remaining utterances (optionally capped at `max_query_per_speaker`
    to bound episodic-testing cost -- proposal does not mandate a cap, so
    default is "use everything else").

    `utterance_manifest` must have columns 'speaker_id' and 'utterance_id'
    (see voxceleb.load_vox1_utterance_manifest for a real source, or any
    equivalent manifest once VoxCeleb2 audio/file-lists are acquired).

    Raises
    ------
    ValueError
        If a speaker has fewer than `k_shot + 1` utterances available (no
        utterance would be left for the query set).
    """
    rng = random.Random(seed + 3)
    assignment: dict[str, dict[str, list[str]]] = {}

    for speaker_id in speaker_ids:
        utt_ids = sorted(
            utterance_manifest.loc[
                utterance_manifest["speaker_id"] == speaker_id, "utterance_id"
            ].tolist()
        )
        if len(utt_ids) < k_shot + 1:
            raise ValueError(
                f"Speaker {speaker_id} has only {len(utt_ids)} utterance(s), "
                f"need >= {k_shot + 1} for {k_shot}-shot support + non-empty query"
            )
        rng.shuffle(utt_ids)
        support = sorted(utt_ids[:k_shot])
        query = sorted(utt_ids[k_shot:])
        if max_query_per_speaker is not None:
            query = query[:max_query_per_speaker]
        assignment[speaker_id] = {"support": support, "query": query}

    return assignment


@dataclass
class CalibrationScheme:
    """Speaker pools used to build genuine/impostor pairs for EER threshold
    calibration [DR-05]. Actual embedding-distance pair construction happens
    downstream (F5-06/F5-07) once the backbone+fusion model exists; this
    dataclass only fixes *which speakers* are eligible for each role, so
    that choice is made once, here, and is provably disjoint from the
    episodic task/reserved pools.
    """

    genuine_source_speakers: list[str]  # from base_train (Data Latih Awal)
    impostor_source_speakers: list[str]  # from calibration_impostor_pool


def build_calibration_scheme(
    base_train: list[str], calibration_impostor_pool: list[str]
) -> CalibrationScheme:
    """[DR-05] Genuine pairs come from Data Latih Awal; impostor pairs come
    from the leftover Data Uji Global speakers not used in episodic sessions
    or the reserved unknown pool (see build_episodic_sessions)."""
    return CalibrationScheme(
        genuine_source_speakers=sorted(base_train),
        impostor_source_speakers=sorted(calibration_impostor_pool),
    )


@dataclass
class FullSplit:
    base_train: list[str]
    data_uji_global: list[str]
    reserved_unknown_pool: list[str]
    episodic_pool: list[str]
    episodic_split: EpisodicSplit
    calibration: CalibrationScheme
    seed: int


def run_full_split(
    catalog: pd.DataFrame,
    seed: int = 0,
    train_ratio: float = 0.7,
    reserved_ratio: float = 0.10,
    n_sessions: int = 10,
    n_way: int = 10,
) -> FullSplit:
    """Orchestrate the entire speaker-level split hierarchy in one call and
    verify (via assert_disjoint) that every resulting partition is mutually
    speaker-disjoint before returning -- this is the function
    scripts/build_splits.py calls, and the function F1-10's test drives.
    """
    base_train, data_uji_global = split_base(catalog, train_ratio=train_ratio, seed=seed)
    reserved_unknown_pool, episodic_pool = allocate_reserved_pool(
        data_uji_global, reserved_ratio=reserved_ratio, seed=seed
    )
    episodic_split = build_episodic_sessions(
        episodic_pool, n_sessions=n_sessions, n_way=n_way, seed=seed
    )
    calibration = build_calibration_scheme(
        base_train, episodic_split.calibration_impostor_pool
    )

    assert_disjoint(
        {
            "base_train": base_train,
            "reserved_unknown_pool": reserved_unknown_pool,
            "task_speakers": episodic_split.task_speakers,
            "calibration_impostor_pool": episodic_split.calibration_impostor_pool,
        }
    )

    return FullSplit(
        base_train=base_train,
        data_uji_global=data_uji_global,
        reserved_unknown_pool=reserved_unknown_pool,
        episodic_pool=episodic_pool,
        episodic_split=episodic_split,
        calibration=calibration,
        seed=seed,
    )
