"""FSCIL evaluation harness (F8-05/06/07) [4.8].

Runs the adapted Few-Shot Class-Incremental Learning protocol (Tao et al.,
2020): `n_sessions` incremental sessions, each introducing `n_way` new
speakers with `k_shot` enrollment samples. After each session, the system
is re-tested on the query sets of *every* task introduced so far (not just
the newest one) -- this is what makes Average Accuracy (Pers. 3.14) and
Forgetting Measure (Pers. 3.15) meaningful.

Designed to run against the real `task_speakers`/`episodic_sessions` from
data/splits/full_split.json once real audio is available for them (see
scripts/fetch_capped_eval_audio.py) -- session/speaker assignment comes
from the caller, this module only drives enroll -> query -> score.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from src.system import SpeakerIdentificationSystem


@dataclass
class FSCILResult:
    """Everything one FSCIL run produces (Experiment 3c extension).

    open_set : dict[session][task] -> open-set accuracy (original metric:
        correct iff accepted by the threshold AND identified correctly).
    closed_set : dict[session][task] -> closed-set identification accuracy
        (top-1 nearest prototype, rejection ignored) -- the apple-to-apple
        number against closed-set baselines. Computed from the SAME
        process()/score() calls as open_set, so reporting it changes no
        behaviour.
    genuine_final_distances : decision scores (min distance, normalized if
        AS-Norm is active) of every final-session genuine query.
    unknown_final_distances : decision scores of the held-out UNKNOWN
        queries against the final database (state-change-free `score()`
        path); empty if no unknown queries were supplied.
    """

    open_set: dict[int, dict[int, float]]
    closed_set: dict[int, dict[int, float]]
    genuine_final_distances: list[float] = field(default_factory=list)
    unknown_final_distances: list[float] = field(default_factory=list)


def run_fscil_detailed(
    system: SpeakerIdentificationSystem,
    sessions: list[list[str]],
    speaker_audio_paths: dict[str, list[str | Path]],
    k_shot: int = 1,
    n_query: int = 5,
    seed: int = 0,
    unknown_query_paths: list[str | Path] | None = None,
) -> FSCILResult:
    """Run the full FSCIL protocol (see run_fscil for the base contract).

    Experiment 3c additions -- both OBSERVE-ONLY (identical enroll/process
    sequence, so `open_set` matches run_fscil exactly for a given seed):
      * closed-set identification accuracy per (session, task);
      * after the final session, every path in `unknown_query_paths` is
        scored via the state-change-free `system.score()` so open-set
        detection (EER/AUROC/TAR@FAR) can be computed against real unknown
        speakers without corrupting the database.

    Each task's query set is fed through the stateful `system.process()`
    exactly once -- at the session where that task is newly introduced
    (i == t) -- simulating those held-out utterances arriving once as
    real incoming traffic, which is what lets "running_average" mode
    (B2) actually adapt. Re-checking an *older* task's query set at every
    later session (i < t, needed for the Average Accuracy / Forgetting
    Measure tables) goes through the state-free `system.score()` instead,
    so measuring old-task accuracy doesn't repeatedly re-feed the same
    handful of query files back into that speaker's running-average
    prototype once per remaining session.
    """
    rng = random.Random(seed)
    task_query_paths: dict[int, dict[str, list[str | Path]]] = {}
    open_set: dict[int, dict[int, float]] = {}
    closed_set: dict[int, dict[int, float]] = {}
    genuine_final: list[float] = []

    final_t = len(sessions) - 1
    for t, session_speakers in enumerate(sessions):
        task_query_paths[t] = {}
        for speaker_id in session_speakers:
            paths = list(speaker_audio_paths[speaker_id])
            rng.shuffle(paths)
            support = paths[:k_shot]
            query = paths[k_shot : k_shot + n_query]
            if not support:
                raise ValueError(f"Speaker {speaker_id} has no audio paths for enrollment")
            system.enroll(speaker_id, support)
            task_query_paths[t][speaker_id] = query

        open_set[t] = {}
        closed_set[t] = {}
        for i in range(t + 1):
            correct, closed_correct, total = 0, 0, 0
            for speaker_id, query_paths in task_query_paths[i].items():
                for query_path in query_paths:
                    # Only the newly-introduced task's queries may mutate
                    # state (once); older tasks are re-checked state-free so
                    # the same held-out audio isn't re-processed into the
                    # prototype at every subsequent session.
                    result = system.process(query_path) if i == t else system.score(query_path)
                    total += 1
                    if result.is_known and result.predicted_speaker_id == speaker_id:
                        correct += 1
                    if result.nearest_speaker_id == speaker_id:
                        closed_correct += 1
                    if t == final_t:
                        genuine_final.append(result.min_distance)
            open_set[t][i] = (correct / total) if total > 0 else float("nan")
            closed_set[t][i] = (closed_correct / total) if total > 0 else float("nan")

    unknown_final: list[float] = []
    for path in unknown_query_paths or []:
        unknown_final.append(system.score(path).min_distance)

    return FSCILResult(
        open_set=open_set,
        closed_set=closed_set,
        genuine_final_distances=genuine_final,
        unknown_final_distances=unknown_final,
    )


def run_fscil(
    system: SpeakerIdentificationSystem,
    sessions: list[list[str]],
    speaker_audio_paths: dict[str, list[str | Path]],
    k_shot: int = 1,
    n_query: int = 5,
    seed: int = 0,
) -> dict[int, dict[int, float]]:
    """Run the full FSCIL protocol.

    Parameters
    ----------
    sessions : list of `n_sessions` lists, each containing `n_way` speaker_ids
        (e.g. `full_split.json["episodic_sessions"]`).
    speaker_audio_paths : speaker_id -> list of that speaker's audio file
        paths (must have >= k_shot + 1 paths per speaker for a non-empty
        query set; entries beyond k_shot + n_query are unused).

    Returns
    -------
    session_task_accuracy : dict[session_idx -> dict[task_idx -> accuracy]]
        `session_task_accuracy[t][i]` = accuracy on task i's query set,
        using the system's state right after session t's enrollment
        (only defined for i <= t). Feed directly into
        src/evaluation/metrics.py's average_accuracy/forgetting_measure.
    """
    return run_fscil_detailed(
        system, sessions, speaker_audio_paths, k_shot=k_shot, n_query=n_query, seed=seed
    ).open_set
