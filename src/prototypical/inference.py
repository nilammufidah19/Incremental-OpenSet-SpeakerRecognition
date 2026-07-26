"""Open-set inference path (F5-08) [4.6 poin 3].

Applies the fixed, pre-calibrated distance threshold (F5-06) to decide
between "known speaker" (identified as the nearest prototype) and "unknown
speaker" (no prototype close enough) -- Gambar 4.1's
"Distance-Based Prototype Classification" -> "Known/Unknown Speaker" step.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.prototypical.classifier import predict


@dataclass
class IdentificationResult:
    is_known: bool
    predicted_speaker_id: str | None  # None when is_known is False
    min_distance: float


def identify(
    query_embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    speaker_ids: list[str],
    threshold: float,
) -> list[IdentificationResult]:
    """query_embeddings: (Q, dim) fused embeddings. prototypes: (n_known, dim),
    row i corresponds to speaker_ids[i]. Returns one IdentificationResult per query row.
    """
    predicted_idx, min_distance = predict(query_embeddings, prototypes)

    results = []
    for idx, dist in zip(predicted_idx.tolist(), min_distance.tolist()):
        if dist < threshold:
            results.append(
                IdentificationResult(is_known=True, predicted_speaker_id=speaker_ids[idx], min_distance=dist)
            )
        else:
            results.append(IdentificationResult(is_known=False, predicted_speaker_id=None, min_distance=dist))
    return results
