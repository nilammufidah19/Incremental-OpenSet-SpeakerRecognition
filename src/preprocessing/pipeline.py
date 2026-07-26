"""End-to-end preprocessing pipeline (F2-09/10), orchestrating Gambar 4.1's
"Preprocessing Audio" -> "VAD" -> "Standardisasi Durasi Audio" chain.

    preprocess_audio(path, mode) -> list[np.ndarray]

`mode` selects the duration-standardization strategy (see
src/preprocessing/duration.py): "ecapa_train", "ecapa_inference",
"whisper_train", or "whisper_inference". The return value is always a list
of one-or-more fixed-role waveforms (more than one only when the audio was
long enough to be sliding-windowed), ready to be fed to the corresponding
backbone in F3.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.preprocessing.aggregate import aggregate_speech_segments
from src.preprocessing.denoise import reduce_noise
from src.preprocessing.duration import standardize_duration
from src.preprocessing.io import TARGET_SR, load_and_resample
from src.preprocessing.loudness import normalize_loudness
from src.preprocessing.vad import detect_speech_segments


def preprocess_audio(
    path: str | Path,
    mode: str,
    apply_denoise: bool = True,
) -> list[np.ndarray]:
    """Run the full preprocessing pipeline on a single audio file.

    Steps: load+resample -> [noise reduction] -> VAD -> segment aggregation
    -> loudness normalization -> duration standardization (`mode`).

    `apply_denoise=False` is provided for debugging/ablation (e.g. to check
    how much the noise-reduction step actually changes VAD's segment
    boundaries on a given recording); production use should leave it True.
    """
    waveform = load_and_resample(path, target_sr=TARGET_SR)

    if apply_denoise:
        waveform = reduce_noise(waveform, TARGET_SR)

    segments = detect_speech_segments(waveform, TARGET_SR)
    waveform = aggregate_speech_segments(waveform, segments)

    waveform = normalize_loudness(waveform, TARGET_SR)

    return standardize_duration(waveform, TARGET_SR, mode)
