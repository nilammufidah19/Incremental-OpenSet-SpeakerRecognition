"""Embedding cache (F3-06) [NFR-04].

Precomputing and caching backbone embeddings avoids recomputing them on
every ablation/baseline/statistical-testing run (F9-F11) when the backbone
is frozen -- only the fusion + prototypical network head change between
those runs, so the expensive ECAPA/Whisper forward passes only need to
happen once per utterance per backbone.

Cache key: sha1(resolve(utterance_path)) -- stable across relative/absolute
and OS path-separator/drive-letter-case differences, and independent of any
particular split/session assignment, so the same cached embedding is reused
regardless of which speaker-split experiment is consuming it.
"""
from __future__ import annotations

import hashlib
from functools import partial
from pathlib import Path

import numpy as np

from src.models import ecapa, redimnet, whisper_encoder, xvector
from src.preprocessing.pipeline import preprocess_audio

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "cache" / "embeddings"

# whisper-base has 6 encoder layers (7 hidden states incl. input embeddings);
# Experiment 2's layer sweep found layer 4 (fraction 4/6) more speaker-
# discriminative than the middle layer (fraction 0.5 -> layer 3) used by the
# default "whisper" backbone. "whisper_l4" caches into its own namespace so the
# two variants coexist and never collide (see src/experiments.py, exp2a).
WHISPER_L4_FRACTION = 4.0 / 6.0

BACKBONE_MODES = {
    "ecapa": "ecapa_inference",
    "whisper": "whisper_inference",
    "whisper_l4": "whisper_inference",
    "xvector": "ecapa_inference",  # same duration-handling needs as ECAPA (variable-length, native pooling)
    # Experiment 5 candidate second backbone (docs/experiment-5.md): frozen
    # ReDimNet-b2 (ft_lm, vox2) via torch.hub -- raw variable-length 16 kHz
    # waveform input, so it shares ECAPA's duration handling. Own namespace,
    # so all existing caches/tags are untouched.
    "redimnet_b2": "ecapa_inference",
    # Experiment 6 F6-1 (docs/experiment-6-plan.md): training-free
    # Whisper-PMFA-style readout -- same frozen whisper-base encoder and the
    # same 30s-window duration handling as "whisper", only the READOUT
    # differs (4 layers x mean+std instead of 1 layer x mean). Own namespace,
    # so "whisper" and "whisper_l4" are untouched and every pre-exp6 result
    # stays reproducible.
    #
    # NOTE: unlike every other backbone here, this cache stores RAW,
    # UN-NORMALIZED statistics -- see whisper_encoder.extract_embedding_pmfa
    # for why (normalization and anti-anisotropy post-processing are exactly
    # what F6-1 measures, and both are recoverable from the raw vector).
    "whisper_pmfa": "whisper_inference",
    # F6-1 follow-up: ALL six encoder blocks (6 x 2 x 512 = 6144-d), so any
    # layer subset is a slice rather than a recompute. Same forward pass cost
    # as whisper_pmfa; also stores RAW statistics.
    "whisper_pmfa_all": "whisper_inference",
}
BACKBONE_EXTRACTORS = {
    "ecapa": (ecapa.extract_embedding, ecapa.extract_embedding_windows),
    "whisper": (whisper_encoder.extract_embedding, whisper_encoder.extract_embedding_windows),
    "whisper_l4": (
        partial(whisper_encoder.extract_embedding, layer_fraction=WHISPER_L4_FRACTION),
        partial(whisper_encoder.extract_embedding_windows, layer_fraction=WHISPER_L4_FRACTION),
    ),
    "xvector": (xvector.extract_embedding, xvector.extract_embedding_windows),
    "redimnet_b2": (redimnet.extract_embedding, redimnet.extract_embedding_windows),
    "whisper_pmfa": (
        whisper_encoder.extract_embedding_pmfa,
        whisper_encoder.extract_embedding_pmfa_windows,
    ),
    "whisper_pmfa_all": (
        partial(whisper_encoder.extract_embedding_pmfa,
                layers=whisper_encoder.PMFA_ALL_LAYERS),
        partial(whisper_encoder.extract_embedding_pmfa_windows,
                layers=whisper_encoder.PMFA_ALL_LAYERS),
    ),
}


def _cache_key(utterance_path: str | Path) -> str:
    # .resolve() canonicalizes relative -> absolute and (on Windows) recases
    # the drive letter and path components to match the filesystem, so
    # relative/absolute and c:/C: variants of the same file collapse to one
    # key. This is a no-op for paths already in that canonical form, so it
    # does not invalidate existing cache entries built from REPO_ROOT-
    # absolute paths (see docs/../memory note on cache-keying pitfalls).
    normalized = Path(utterance_path).resolve().as_posix()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _cache_path(utterance_path: str | Path, backbone: str) -> Path:
    return CACHE_DIR / backbone / f"{_cache_key(utterance_path)}.npy"


def get_or_compute_embedding(
    utterance_path: str | Path, backbone: str, force: bool = False
) -> np.ndarray:
    """Return the cached embedding for `utterance_path` under `backbone`
    ("ecapa" or "whisper"), computing (and caching) it if not already present.
    """
    if backbone not in BACKBONE_MODES:
        raise ValueError(f"backbone must be one of {list(BACKBONE_MODES)}, got {backbone!r}")

    path = _cache_path(utterance_path, backbone)
    if path.exists() and not force:
        return np.load(path)

    mode = BACKBONE_MODES[backbone]
    single_fn, windows_fn = BACKBONE_EXTRACTORS[backbone]
    windows = preprocess_audio(utterance_path, mode=mode)
    embedding = windows_fn(windows) if len(windows) > 1 else single_fn(windows[0])

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embedding)
    return embedding


def is_cached(utterance_path: str | Path, backbone: str) -> bool:
    return _cache_path(utterance_path, backbone).exists()
