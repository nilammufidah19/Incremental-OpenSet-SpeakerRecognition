"""Loudness normalization (F2-05) [4.4 poin 3].

Standardizes waveform loudness to -20 dB LUFS so the model isn't affected
by inter-recording volume variation, per proposal Bab 4.4.
"""
from __future__ import annotations

import numpy as np
import pyloudnorm as pyln

TARGET_LUFS = -20.0


def normalize_loudness(waveform: np.ndarray, sr: int, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    """Normalize integrated loudness to target_lufs (default -20 dB LUFS).

    Falls back to returning the waveform unchanged if it is silent/too short
    for ITU-R BS.1770 block-based measurement (pyloudnorm raises/returns
    -inf in that case) -- there's no loudness to normalize on pure silence.
    """
    if waveform.size == 0 or not np.any(waveform):
        return waveform

    meter = pyln.Meter(sr)
    try:
        current_lufs = meter.integrated_loudness(waveform)
    except (ValueError, ZeroDivisionError):
        return waveform

    if not np.isfinite(current_lufs):
        return waveform

    normalized = pyln.normalize.loudness(waveform, current_lufs, target_lufs)
    # Guard against clipping introduced by normalization gain.
    peak = np.max(np.abs(normalized))
    if peak > 1.0:
        normalized = normalized / peak
    return normalized.astype(np.float32)
