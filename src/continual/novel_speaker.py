"""Novel speaker detection: buffer + Silhouette Coefficient validation
(F6-04/05) [4.7, Pers. 4.7].

A single unknown sample is never enough to register a new prototype (could
be noise/a one-off misdetection). Two conditions must hold simultaneously:

  1. The unknown-sample buffer has accumulated >= `min_samples` items
     (F6-04).
  2. Those buffered samples are mutually compact AND well-separated from
     every existing prototype, measured via the mean Silhouette
     Coefficient (Rousseeuw, 1987) of the candidate cluster against the
     nearest existing prototype (F6-05, Pers. 4.7):

         s(i) = (b(i) - a(i)) / max(a(i), b(i))

     a(i) = mean distance from sample i to the other buffered samples
            (cluster compactness)
     b(i) = distance from sample i to the nearest *existing* prototype
            (separation from known speakers)

Only when both hold is the buffer promoted to a new prototype (via
prototype_update.initialize_prototype) and cleared.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import cdist

DEFAULT_SILHOUETTE_THRESHOLD = 0.5


@dataclass
class NovelSpeakerBuffer:
    embeddings: list[np.ndarray] = field(default_factory=list)

    def add(self, embedding: np.ndarray) -> None:
        self.embeddings.append(embedding)

    def size(self) -> int:
        return len(self.embeddings)

    def clear(self) -> None:
        self.embeddings = []

    def stacked(self) -> np.ndarray:
        return np.stack(self.embeddings)


def mean_silhouette_against_existing(
    candidate_embeddings: np.ndarray, existing_prototypes: np.ndarray
) -> float:
    """Pers. 4.7, averaged over the candidate cluster.

    If there are no existing prototypes yet (first speaker(s) ever), the
    candidate cluster is trivially "separated" from nothing -- returns 1.0
    (maximally well-separated) so the very first speakers can still be
    registered.
    """
    n = len(candidate_embeddings)
    if n < 2:
        raise ValueError("Need >= 2 candidate embeddings to compute intra-cluster compactness a(i)")

    if len(existing_prototypes) == 0:
        return 1.0

    pairwise = cdist(candidate_embeddings, candidate_embeddings)
    a = pairwise.sum(axis=1) / (n - 1)  # exclude self-distance (always 0)
    b = cdist(candidate_embeddings, existing_prototypes).min(axis=1)
    s = (b - a) / np.maximum(a, b)
    return float(s.mean())


def should_register_new_speaker(
    buffer: NovelSpeakerBuffer,
    existing_prototypes: np.ndarray,
    min_samples: int,
    silhouette_threshold: float = DEFAULT_SILHOUETTE_THRESHOLD,
) -> bool:
    """F6-04+F6-05 combined gate: both the sample-count floor AND the
    Silhouette Coefficient threshold must be satisfied.

    The effective floor is `max(min_samples, 2)`: the Silhouette
    Coefficient's intra-cluster term a(i) is mathematically undefined for a
    single-sample "cluster" (no other member to measure compactness
    against), so fewer than 2 buffered samples can never pass this gate
    regardless of how low `min_samples` is configured.
    """
    effective_min_samples = max(min_samples, 2)
    if buffer.size() < effective_min_samples:
        return False
    mean_s = mean_silhouette_against_existing(buffer.stacked(), existing_prototypes)
    return mean_s >= silhouette_threshold
