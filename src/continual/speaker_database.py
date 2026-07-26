"""Speaker database (F6-01) [3.8, 4.1].

Stores exactly `(speaker_id -> (c_k, n_k))` per Bab 3.8/4.7 -- the running
prototype and its cumulative sample count, never raw embedding history
(that's the whole point of the running-average update rule in
prototype_update.py: O(1) memory per speaker regardless of how many
utterances they've contributed over time).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SpeakerRecord:
    prototype: np.ndarray  # c_k
    n_samples: int         # n_k


class SpeakerDatabase:
    def __init__(self) -> None:
        self._records: dict[str, SpeakerRecord] = {}

    def has(self, speaker_id: str) -> bool:
        return speaker_id in self._records

    def get(self, speaker_id: str) -> SpeakerRecord:
        return self._records[speaker_id]

    def set(self, speaker_id: str, prototype: np.ndarray, n_samples: int) -> None:
        self._records[speaker_id] = SpeakerRecord(prototype=np.asarray(prototype, dtype=np.float32), n_samples=n_samples)

    def remove(self, speaker_id: str) -> None:
        del self._records[speaker_id]

    @property
    def speaker_ids(self) -> list[str]:
        return list(self._records.keys())

    def prototypes_matrix(self) -> np.ndarray:
        """Stacked prototypes in `speaker_ids` order -- ready for
        cdist/euclidean_distance against a batch of query embeddings."""
        if not self._records:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([self._records[s].prototype for s in self.speaker_ids])

    def __len__(self) -> int:
        return len(self._records)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        speaker_ids = self.speaker_ids
        prototypes = self.prototypes_matrix()
        n_samples = np.array([self._records[s].n_samples for s in speaker_ids], dtype=np.int64)
        np.savez(
            path,
            speaker_ids=np.array(speaker_ids, dtype=object),
            prototypes=prototypes,
            n_samples=n_samples,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SpeakerDatabase":
        data = np.load(path, allow_pickle=True)
        db = cls()
        speaker_ids = data["speaker_ids"].tolist()
        prototypes = data["prototypes"]
        n_samples = data["n_samples"]
        for i, speaker_id in enumerate(speaker_ids):
            db.set(speaker_id, prototypes[i], int(n_samples[i]))
        return db
