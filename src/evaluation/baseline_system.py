"""Baseline systems for F10 capability-ladder comparison [4.10, Tabel 4.5].

`SingleBackboneSystem` mirrors src/system.py's SpeakerIdentificationSystem
but operates on ONE raw backbone embedding directly -- no Gated Attention
Fusion -- for baselines that don't use the proposed multi-backbone
architecture at all (x-vector+PLDA, ECAPA-TDNN standard).

PLDA-lite: a full EM-trained PLDA backend is a substantial undertaking on
its own; we approximate it with LDA-whitened cosine scoring (fit an LDA
projection on base_train speaker labels, then score in the whitened space)
-- a standard, well-understood approximation of what PLDA does
(class-discriminative whitening followed by simple distance scoring). This
simplification is stated explicitly here and in plan/05-evaluation-plan.md
rather than silently presented as a full PLDA reimplementation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.continual.manager import ContinualLearningManager, ContinualProcessResult
from src.continual.prototype_update import initialize_prototype
from src.continual.speaker_database import SpeakerDatabase
from src.features.cache import get_or_compute_embedding

CLOSED_SET_THRESHOLD = 1e9  # effectively never rejects as "unknown" -- see module docstring


class SingleBackboneSystem:
    def __init__(
        self,
        backbone: str,
        threshold: float = CLOSED_SET_THRESHOLD,
        continual_mode: str = "static",
        whiten_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        min_samples_for_new_speaker: int = 1,
        silhouette_threshold: float = 0.5,
    ) -> None:
        self.backbone = backbone
        self.threshold = threshold
        self.whiten_fn = whiten_fn
        self.database = SpeakerDatabase()
        self.manager = ContinualLearningManager(
            self.database,
            threshold=threshold,
            mode=continual_mode,
            min_samples_for_new_speaker=min_samples_for_new_speaker,
            silhouette_threshold=silhouette_threshold,
        )

    def embed(self, audio_path: str | Path) -> np.ndarray:
        emb = get_or_compute_embedding(audio_path, self.backbone)
        if self.whiten_fn is not None:
            emb = self.whiten_fn(emb)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
        return emb

    def enroll(self, speaker_id: str, audio_paths: list[str | Path]) -> None:
        embeddings = np.stack([self.embed(p) for p in audio_paths])
        prototype, n = initialize_prototype(embeddings)
        self.database.set(speaker_id, prototype, n)

    def process(self, audio_path: str | Path) -> ContinualProcessResult:
        embedding = self.embed(audio_path)
        return self.manager.process_sample(embedding)

    def score(self, audio_path: str | Path) -> ContinualProcessResult:
        """No-state-change decision (Experiment 3c detection path), mirroring
        SpeakerIdentificationSystem.score so the harness can treat both alike."""
        embedding = self.embed(audio_path)
        return self.manager.score_sample(embedding)


def fit_lda_whitener(
    embeddings_by_speaker: dict[str, list[np.ndarray]], n_components: int | None = None
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit an LDA projection on (embedding, speaker_id) pairs, return a
    callable that projects a single embedding into the whitened space
    ("PLDA-lite" scoring backend for the x-vector baseline, F10-01)."""
    X, y = [], []
    for speaker_id, embeddings in embeddings_by_speaker.items():
        for e in embeddings:
            X.append(e)
            y.append(speaker_id)
    X = np.stack(X)

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(X, y)

    def whiten(embedding: np.ndarray) -> np.ndarray:
        return lda.transform(embedding.reshape(1, -1)).flatten().astype(np.float32)

    return whiten
