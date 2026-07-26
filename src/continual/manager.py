"""Continual Learning Manager (F6-06) [3.8, 4.1, 4.7].

Integrates the speaker database (F6-01), prototype update rule (F6-02/03),
and novel-speaker buffer+validation (F6-04/05) into the single entry point
called once per incoming sample during an incremental session:

    manager.process_sample(embedding) -> ContinualProcessResult

`mode` (F6-08, feature flag for ablation study B, Tabel 4.4):
  - "running_average": known-speaker prototypes are updated via the
    weighted running average (Pers. 4.5-4.6) on every recognized sample --
    the proposed system (B2).
  - "static": prototypes are computed once at base training and never
    updated again -- the ablation baseline (B1). Novel-speaker
    registration still happens in both modes (that's a separate mechanism
    from *updating* known speakers), since Tabel 4.4 only ablates the
    update rule, not open-set registration itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.continual.novel_speaker import (
    DEFAULT_SILHOUETTE_THRESHOLD,
    NovelSpeakerBuffer,
    should_register_new_speaker,
)
from src.continual.prototype_update import initialize_prototype, update_prototype
from src.continual.speaker_database import SpeakerDatabase

VALID_MODES = ("running_average", "static")


@dataclass
class ContinualProcessResult:
    is_known: bool
    predicted_speaker_id: str | None
    min_distance: float
    newly_registered_speaker_id: str | None = None
    # Experiment 3c (exp3c_dualmetric): the nearest registered speaker
    # regardless of the accept/reject decision -- what a closed-set
    # (no-rejection) system would have predicted. Lets the harness report
    # closed-set identification accuracy alongside the open-set metric
    # without changing any decision behaviour.
    nearest_speaker_id: str | None = None


class ContinualLearningManager:
    def __init__(
        self,
        database: SpeakerDatabase,
        threshold: float,
        mode: str = "running_average",
        min_samples_for_new_speaker: int = 1,
        silhouette_threshold: float = DEFAULT_SILHOUETTE_THRESHOLD,
        new_speaker_id_fn=None,
        score_normalizer=None,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        self.database = database
        self.threshold = threshold
        self.mode = mode
        self.min_samples_for_new_speaker = min_samples_for_new_speaker
        self.silhouette_threshold = silhouette_threshold
        # Experiment 3b (exp3b_asnorm): optional ASNorm instance. When set,
        # the accept/reject decision (and the reported min_distance) happens
        # on cohort-normalized scores instead of raw Euclidean distances;
        # `threshold` must then have been calibrated in the same normalized
        # space (calibrate_threshold(..., score_normalizer=...)). Prototype
        # updates and novelty registration still operate on raw embeddings.
        self.score_normalizer = score_normalizer
        self._new_speaker_id_fn = new_speaker_id_fn or self._default_new_speaker_id
        self._unknown_buffer = NovelSpeakerBuffer()
        self._n_registered_via_novelty = 0

    def _default_new_speaker_id(self) -> str:
        self._n_registered_via_novelty += 1
        return f"novel_speaker_{self._n_registered_via_novelty:04d}"

    def _nearest_known(self, embedding: np.ndarray) -> tuple[str | None, float]:
        if len(self.database) == 0:
            return None, float("inf")
        prototypes = self.database.prototypes_matrix()
        speaker_ids = self.database.speaker_ids
        if self.score_normalizer is not None:
            distances = self.score_normalizer.normalize(embedding[None, :], prototypes)[0]
        else:
            distances = np.linalg.norm(prototypes - embedding, axis=1)
        idx = int(np.argmin(distances))
        return speaker_ids[idx], float(distances[idx])

    def score_sample(self, embedding: np.ndarray) -> ContinualProcessResult:
        """Open-set decision WITHOUT any state change (no prototype update, no
        novelty buffering/registration). Used by the Experiment 3c detection
        path to score unknown queries against the final database without
        letting those queries alter it."""
        predicted_speaker_id, min_distance = self._nearest_known(embedding)
        is_known = predicted_speaker_id is not None and min_distance < self.threshold
        return ContinualProcessResult(
            is_known=is_known,
            predicted_speaker_id=predicted_speaker_id if is_known else None,
            min_distance=min_distance,
            nearest_speaker_id=predicted_speaker_id,
        )

    def process_sample(self, embedding: np.ndarray) -> ContinualProcessResult:
        """Run one open-set decision + continual-learning update step."""
        predicted_speaker_id, min_distance = self._nearest_known(embedding)
        is_known = predicted_speaker_id is not None and min_distance < self.threshold

        if is_known:
            if self.mode == "running_average":
                record = self.database.get(predicted_speaker_id)
                new_prototype, new_n = update_prototype(
                    record.prototype, record.n_samples, np.expand_dims(embedding, axis=0)
                )
                self.database.set(predicted_speaker_id, new_prototype, new_n)
            # mode == "static": prototypes are frozen, no update (F6-08 / B1)
            return ContinualProcessResult(
                is_known=True, predicted_speaker_id=predicted_speaker_id, min_distance=min_distance,
                nearest_speaker_id=predicted_speaker_id,
            )

        # Unknown: accumulate into buffer, check registration gate (F6-04/05)
        self._unknown_buffer.add(embedding)
        existing_prototypes = self.database.prototypes_matrix()
        newly_registered_id = None
        if should_register_new_speaker(
            self._unknown_buffer,
            existing_prototypes,
            self.min_samples_for_new_speaker,
            self.silhouette_threshold,
        ):
            prototype, n = initialize_prototype(self._unknown_buffer.stacked())
            newly_registered_id = self._new_speaker_id_fn()
            self.database.set(newly_registered_id, prototype, n)
            self._unknown_buffer.clear()

        return ContinualProcessResult(
            is_known=False,
            predicted_speaker_id=None,
            min_distance=min_distance,
            newly_registered_speaker_id=newly_registered_id,
            nearest_speaker_id=predicted_speaker_id,
        )
