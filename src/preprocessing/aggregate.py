"""Speech segment aggregation (F2-04) [4.1, 4.4 poin 2].

Concatenates the speech segments detected by VAD into one continuous,
silence-free waveform ("speech segment aggregation" in Gambar 4.1).
"""
from __future__ import annotations

import warnings

import numpy as np


def aggregate_speech_segments(
    waveform: np.ndarray, segments: list[tuple[int, int]]
) -> np.ndarray:
    """Concatenate `segments` (start_sample, end_sample) slices of `waveform`
    into a single contiguous array.

    If VAD found no speech segments at all (`segments` empty), falls back to
    returning the original waveform unchanged with a warning, rather than an
    empty array that would break every downstream step -- this should be rare
    on genuine speech recordings and signals a VAD threshold worth revisiting
    if it happens often in practice.
    """
    if not segments:
        warnings.warn(
            "aggregate_speech_segments: no speech segments detected; "
            "falling back to the original waveform unchanged."
        )
        return waveform
    aggregated = np.concatenate([waveform[start:end] for start, end in segments])
    if aggregated.size == 0:
        # every segment was zero-length (start == end) -- same "nothing to
        # work with" case as the empty-`segments` branch above, just reached
        # via a different route.
        warnings.warn(
            "aggregate_speech_segments: all detected segments were zero-length; "
            "falling back to the original waveform unchanged."
        )
        return waveform
    return aggregated
