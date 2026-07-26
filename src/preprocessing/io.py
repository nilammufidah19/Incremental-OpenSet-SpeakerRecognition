"""Audio loading and resampling (F2-01) [4.4 poin implisit, 4.5 Tabel 4.2].

All downstream preprocessing/feature-extraction modules assume mono, float32
waveforms at TARGET_SR=16000 Hz, matching Tabel 4.2's sample rate and the
VoxCeleb1/2 native sample rate (so most files pass through resample()
unchanged; resampling only kicks in for off-spec input, e.g. sample audio
recorded at a different rate).
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

TARGET_SR = 16000


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load an audio file as a float32 mono waveform at its native sample rate.

    Uses librosa rather than calling soundfile directly: librosa tries
    soundfile first (fast path for WAV/FLAC) and falls back to
    audioread+ffmpeg for anything libsndfile can't decode -- notably the
    AAC-in-MP4 (.m4a) utterances VoxCeleb2 is distributed in (see
    src/data/remote_zip_audio.py), which raise
    `soundfile.LibsndfileError: Format not recognised` if read directly.
    """
    data, sr = librosa.load(str(path), sr=None, mono=True)
    return data.astype(np.float32), sr


def resample(waveform: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Resample waveform to target_sr. No-op if already at target_sr."""
    if orig_sr == target_sr:
        return waveform
    return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)


def load_and_resample(path: str | Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Convenience: load_audio + resample in one call."""
    waveform, sr = load_audio(path)
    return resample(waveform, sr, target_sr)
