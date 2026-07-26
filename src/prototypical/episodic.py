"""Episodic N-way K-shot sampling (F5-01) [3.6, 3.7, 4.6].

Operates on an in-memory index `embeddings_by_speaker: dict[speaker_id ->
list[np.ndarray]]` (e.g. built from src/features/cache.py's precomputed
embeddings) rather than raw audio, since episodic training resamples
support/query splits every episode and re-running the backbone+preprocessing
pipeline per episode would be wastefully slow for a frozen backbone.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class Episode:
    support_embeddings: np.ndarray  # (n_way * k_shot, dim)
    support_labels: np.ndarray      # (n_way * k_shot,), values in [0, n_way)
    query_embeddings: np.ndarray    # (n_way * n_query, dim)
    query_labels: np.ndarray        # (n_way * n_query,), values in [0, n_way)
    speaker_ids: list[str]          # speaker_ids[label] -> original speaker_id


def sample_episode(
    embeddings_by_speaker: dict[str, list[np.ndarray]],
    n_way: int,
    k_shot: int,
    n_query: int,
    seed: int,
) -> Episode:
    """Sample one N-way K-shot episode.

    Only speakers with >= k_shot + n_query available embeddings are
    eligible, so support and query sets never overlap for a given speaker.

    Raises
    ------
    ValueError
        If fewer than `n_way` speakers have enough samples.
    """
    rng = random.Random(seed)

    eligible = [
        spk for spk, embs in embeddings_by_speaker.items()
        if len(embs) >= k_shot + n_query
    ]
    if len(eligible) < n_way:
        raise ValueError(
            f"Only {len(eligible)} speakers have >= {k_shot + n_query} samples, "
            f"need >= {n_way} for {n_way}-way sampling"
        )

    chosen_speakers = rng.sample(eligible, n_way)

    support_embs, support_labels = [], []
    query_embs, query_labels = [], []

    for label, spk in enumerate(chosen_speakers):
        samples = list(embeddings_by_speaker[spk])
        rng.shuffle(samples)
        support = samples[:k_shot]
        query = samples[k_shot : k_shot + n_query]

        support_embs.extend(support)
        support_labels.extend([label] * len(support))
        query_embs.extend(query)
        query_labels.extend([label] * len(query))

    return Episode(
        support_embeddings=np.stack(support_embs),
        support_labels=np.array(support_labels, dtype=np.int64),
        query_embeddings=np.stack(query_embs),
        query_labels=np.array(query_labels, dtype=np.int64),
        speaker_ids=chosen_speakers,
    )
