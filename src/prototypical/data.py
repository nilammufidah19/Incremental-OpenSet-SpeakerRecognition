"""Build the per-speaker raw-embedding index consumed by episodic sampling
(F5-01) and training (F5-04).

Each "embedding" in the index is actually the concatenation of the two
*raw, frozen backbone* embeddings `[ecapa_emb (192) ; whisper_emb (512)]`,
not yet fused. This lets episodic training still backpropagate into the
trainable Gated Attention Fusion layer (src/models/fusion.py): the raw
pair is looked up from cache (frozen, no grad needed for the backbones
themselves), split back into its two halves, and only then passed through
`fusion_model(...)` with gradients enabled -- see src/prototypical/train.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.cache import get_or_compute_embedding

REPO_ROOT = Path(__file__).resolve().parents[2]

ECAPA_DIM = 192
WHISPER_DIM = 512


def build_raw_embedding_index(
    manifest_df: pd.DataFrame, whisper_backbone: str = "whisper"
) -> dict[str, list[np.ndarray]]:
    """For every utterance row in `manifest_df` (must have 'speaker_id' and
    'path' columns), fetch (or compute) its cached ECAPA + Whisper
    embeddings, concatenate them, and group by speaker_id.

    `whisper_backbone` selects which Whisper variant to concatenate: "whisper"
    (default, middle layer) or "whisper_l4" (Experiment 2's layer-4 variant).
    Both are 512-d, so the split point (ECAPA_DIM) is unchanged."""
    index: dict[str, list[np.ndarray]] = {}
    for _, row in manifest_df.iterrows():
        path = REPO_ROOT / row["path"]
        e1 = get_or_compute_embedding(path, "ecapa")
        e2 = get_or_compute_embedding(path, whisper_backbone)
        combined = np.concatenate([e1, e2]).astype(np.float32)
        index.setdefault(row["speaker_id"], []).append(combined)
    return index


def split_raw_embedding(raw: "np.ndarray | object") -> tuple:
    """Split a concatenated [ecapa; second_backbone] tensor/array back into
    its two parts. Works on both numpy arrays and torch tensors (duck-typed
    via slicing, which both support identically).

    The second part is everything after ECAPA_DIM -- identical to the
    original fixed [192:704] slice for Whisper-based rows (512-d), and
    correct for Experiment 5's variable-dim candidates (e.g. WavLM 768-d,
    total 960)."""
    return raw[..., :ECAPA_DIM], raw[..., ECAPA_DIM:]
