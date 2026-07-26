"""Voice Activity Detection via Silero VAD (F2-03) [4.1, 4.4 poin 1].

Detects speech segments so downstream aggregation (F2-04) can discard
silence/non-speech and keep only the parts of the recording that actually
carry speaker-discriminative information.
"""
from __future__ import annotations

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

_MODEL = None  # lazy-loaded singleton; loading the model is expensive, calls are not


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_silero_vad()
    return _MODEL


def detect_speech_segments(
    waveform: np.ndarray,
    sr: int = 16000,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
) -> list[tuple[int, int]]:
    """Return speech segments as a list of (start_sample, end_sample).

    Silero VAD only supports 8kHz/16kHz; callers must resample first
    (see src/preprocessing/io.py, which standardizes everything to 16kHz).
    """
    if sr not in (8000, 16000):
        raise ValueError(f"Silero VAD requires sr in (8000, 16000), got {sr}")
    if waveform.size == 0:
        return []

    model = get_model()
    audio_tensor = torch.from_numpy(np.ascontiguousarray(waveform)).float()
    timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=sr,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        return_seconds=False,
    )
    return [(int(t["start"]), int(t["end"])) for t in timestamps]
