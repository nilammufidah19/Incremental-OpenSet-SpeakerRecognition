"""Basic noise reduction, applied pre-VAD (F2-02) [4.4 poin implisit, 1.1].

Uses spectral-gating noise reduction (`noisereduce`, a standard, lightweight
implementation of the classic spectral-subtraction approach) rather than the
adversarial multi-task VAD architecture described in Bab 3.1 (Larsen et al.,
2022) -- reimplementing and training that adversarial model is out of scope
for this stage (batasan masalah B2: VAD here targets contextual segmentation,
not a research reproduction of a competing VAD paper). Silero VAD (F2-03)
is the actual voice/non-voice decision-maker; this step only cleans up the
signal beforehand so VAD's decision is more reliable on noisy recordings.
"""
from __future__ import annotations

import numpy as np
import noisereduce as nr


def reduce_noise(waveform: np.ndarray, sr: int) -> np.ndarray:
    """Apply stationary spectral-gating noise reduction.

    Returns the input unchanged for degenerate (empty/near-silent/very
    short) signals, since noisereduce's STFT-based estimation needs a
    minimum number of samples and has nothing meaningful to do on silence.
    """
    if waveform.size < sr // 10:  # shorter than 100ms: not enough for STFT noise profiling
        return waveform
    if not np.any(waveform):
        return waveform

    reduced = nr.reduce_noise(y=waveform, sr=sr, stationary=True)
    return reduced.astype(np.float32)
