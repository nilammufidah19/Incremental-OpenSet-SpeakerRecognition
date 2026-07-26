"""Tests for src/preprocessing/* (F2)."""
from __future__ import annotations

import numpy as np
import pyloudnorm as pyln
import pytest

from src.preprocessing.aggregate import aggregate_speech_segments
from src.preprocessing.denoise import reduce_noise
from src.preprocessing.duration import (
    ECAPA_TRAIN_MIN_SEC,
    MAX_ECAPA_INFERENCE_SEC,
    SLIDING_WINDOW_SEC,
    WHISPER_FIXED_SEC,
    crop,
    pad_circular,
    pad_silence,
    sliding_windows,
    standardize_duration,
)
from src.preprocessing.io import TARGET_SR, load_and_resample, load_audio, resample
from src.preprocessing.loudness import normalize_loudness
from src.preprocessing.pipeline import preprocess_audio
from src.preprocessing.vad import detect_speech_segments
from tests.conftest import REPO_ROOT, requires_sample_audio


def synth_tone(duration_s: float, sr: int = 16000, freq: float = 220.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(duration_s * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --------------------------------------------------------------------------
# io.py
# --------------------------------------------------------------------------


def test_resample_is_noop_at_target_sr():
    wf = synth_tone(1.0, sr=16000)
    out = resample(wf, orig_sr=16000, target_sr=16000)
    assert out is wf  # exact no-op, not just equal


def test_resample_changes_length_proportionally():
    wf = synth_tone(1.0, sr=8000)
    out = resample(wf, orig_sr=8000, target_sr=16000)
    assert abs(len(out) - 16000) <= 2  # allow tiny rounding from resampler


@requires_sample_audio
def test_load_and_resample_real_file(sample_audio_manifest):
    path = REPO_ROOT / sample_audio_manifest.iloc[0]["path"]
    waveform, sr = load_audio(path)
    assert sr == 16000  # VoxCeleb1 native rate
    assert waveform.ndim == 1
    assert waveform.dtype == np.float32

    resampled = load_and_resample(path)
    assert len(resampled) == len(waveform)  # already at target, no-op


# --------------------------------------------------------------------------
# loudness.py
# --------------------------------------------------------------------------


def test_normalize_loudness_on_silence_is_noop():
    silence = np.zeros(16000, dtype=np.float32)
    out = normalize_loudness(silence, 16000)
    assert np.array_equal(out, silence)


@requires_sample_audio
def test_normalize_loudness_reaches_target_on_real_audio(sample_audio_manifest):
    path = REPO_ROOT / sample_audio_manifest.iloc[0]["path"]
    waveform = load_and_resample(path)
    normalized = normalize_loudness(waveform, TARGET_SR, target_lufs=-20.0)

    meter = pyln.Meter(TARGET_SR)
    resulting_lufs = meter.integrated_loudness(normalized)
    assert abs(resulting_lufs - (-20.0)) < 0.5  # pyloudnorm should hit this precisely
    assert np.max(np.abs(normalized)) <= 1.0  # no clipping introduced


# --------------------------------------------------------------------------
# denoise.py
# --------------------------------------------------------------------------


def test_reduce_noise_preserves_length():
    wf = synth_tone(2.0)
    out = reduce_noise(wf, TARGET_SR)
    assert len(out) == len(wf)
    assert np.all(np.isfinite(out))


def test_reduce_noise_short_signal_passthrough():
    wf = synth_tone(0.05)  # 50ms, below the 100ms floor
    out = reduce_noise(wf, TARGET_SR)
    assert np.array_equal(out, wf)


# --------------------------------------------------------------------------
# vad.py
# --------------------------------------------------------------------------


@requires_sample_audio
def test_detect_speech_segments_on_real_speech(sample_audio_manifest):
    path = REPO_ROOT / sample_audio_manifest.iloc[0]["path"]
    waveform = load_and_resample(path)
    segments = detect_speech_segments(waveform, TARGET_SR)

    assert len(segments) >= 1
    for start, end in segments:
        assert 0 <= start < end <= len(waveform)

    total_speech_samples = sum(end - start for start, end in segments)
    # real VoxCeleb clips are curated speech, so most of the clip should be
    # detected as speech, not silence
    assert total_speech_samples / len(waveform) > 0.5


def test_detect_speech_segments_rejects_bad_sample_rate():
    wf = synth_tone(1.0)
    with pytest.raises(ValueError, match="8000, 16000"):
        detect_speech_segments(wf, sr=44100)


# --------------------------------------------------------------------------
# aggregate.py
# --------------------------------------------------------------------------


def test_aggregate_speech_segments_basic():
    wf = np.arange(100, dtype=np.float32)
    segments = [(0, 10), (50, 60)]
    out = aggregate_speech_segments(wf, segments)
    assert len(out) == 20
    assert np.array_equal(out[:10], wf[0:10])
    assert np.array_equal(out[10:20], wf[50:60])


def test_aggregate_speech_segments_empty_falls_back_with_warning():
    wf = np.ones(50, dtype=np.float32)
    with pytest.warns(UserWarning, match="no speech segments"):
        out = aggregate_speech_segments(wf, [])
    assert np.array_equal(out, wf)


# --------------------------------------------------------------------------
# duration.py
# --------------------------------------------------------------------------


def test_pad_circular_short_audio_reaches_target():
    wf = synth_tone(1.0)  # 1s
    out = pad_circular(wf, TARGET_SR, ECAPA_TRAIN_MIN_SEC)
    assert len(out) == int(ECAPA_TRAIN_MIN_SEC * TARGET_SR)
    # circular: first second should equal the original tile
    assert np.allclose(out[: len(wf)], wf)


def test_pad_circular_long_audio_unchanged():
    wf = synth_tone(5.0)
    out = pad_circular(wf, TARGET_SR, ECAPA_TRAIN_MIN_SEC)
    assert len(out) == len(wf)
    assert np.array_equal(out, wf)


def test_pad_silence_pads_with_zeros():
    wf = synth_tone(1.0)
    out = pad_silence(wf, TARGET_SR, 3.0)
    assert len(out) == int(3.0 * TARGET_SR)
    assert np.array_equal(out[: len(wf)], wf)
    assert np.all(out[len(wf) :] == 0.0)


def test_crop_truncates_to_exact_length():
    wf = synth_tone(5.0)
    out = crop(wf, TARGET_SR, 3.0)
    assert len(out) == int(3.0 * TARGET_SR)


def test_sliding_windows_short_audio_single_padded_window():
    wf = synth_tone(5.0)
    windows = sliding_windows(wf, TARGET_SR, window_sec=30.0, hop_sec=30.0)
    assert len(windows) == 1
    assert len(windows[0]) == int(30.0 * TARGET_SR)


def test_sliding_windows_long_audio_multiple_windows():
    wf = synth_tone(65.0)  # spans 3 non-overlapping 30s windows (last one padded)
    windows = sliding_windows(wf, TARGET_SR, window_sec=30.0, hop_sec=30.0)
    assert len(windows) == 3
    for w in windows:
        assert len(w) == int(30.0 * TARGET_SR)


def test_standardize_duration_ecapa_train_pads_short():
    wf = synth_tone(1.0)
    out = standardize_duration(wf, TARGET_SR, "ecapa_train")
    assert len(out) == 1
    assert len(out[0]) == int(ECAPA_TRAIN_MIN_SEC * TARGET_SR)


def test_standardize_duration_ecapa_inference_passthrough_short():
    wf = synth_tone(5.0)
    out = standardize_duration(wf, TARGET_SR, "ecapa_inference")
    assert len(out) == 1
    assert len(out[0]) == len(wf)  # untouched, no padding/cropping


def test_standardize_duration_ecapa_inference_windows_long():
    wf = synth_tone(MAX_ECAPA_INFERENCE_SEC + 5.0)
    out = standardize_duration(wf, TARGET_SR, "ecapa_inference")
    assert len(out) >= 2


def test_standardize_duration_whisper_train_always_exact_30s():
    for dur in (1.0, 30.0, 45.0):
        wf = synth_tone(dur)
        out = standardize_duration(wf, TARGET_SR, "whisper_train")
        assert len(out) == 1
        assert len(out[0]) == int(WHISPER_FIXED_SEC * TARGET_SR)


def test_standardize_duration_whisper_inference_windows_any_length():
    short = standardize_duration(synth_tone(5.0), TARGET_SR, "whisper_inference")
    long = standardize_duration(synth_tone(65.0), TARGET_SR, "whisper_inference")
    assert len(short) == 1 and len(short[0]) == int(SLIDING_WINDOW_SEC * TARGET_SR)
    assert len(long) == 3


def test_standardize_duration_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown mode"):
        standardize_duration(synth_tone(1.0), TARGET_SR, "bogus_mode")


# --------------------------------------------------------------------------
# pipeline.py (F2-09/10 end-to-end on REAL audio)
# --------------------------------------------------------------------------


@requires_sample_audio
@pytest.mark.parametrize(
    "mode,expected_sec",
    [
        ("ecapa_train", None),  # variable (>= ECAPA_TRAIN_MIN_SEC)
        ("ecapa_inference", None),  # variable, single window (real samples are short)
        ("whisper_train", WHISPER_FIXED_SEC),
        ("whisper_inference", SLIDING_WINDOW_SEC),
    ],
)
def test_preprocess_audio_end_to_end_real_file(sample_audio_manifest, mode, expected_sec):
    path = REPO_ROOT / sample_audio_manifest.iloc[0]["path"]
    windows = preprocess_audio(path, mode=mode)

    assert len(windows) >= 1
    for w in windows:
        assert w.dtype == np.float32
        assert np.all(np.isfinite(w))
        if expected_sec is not None:
            assert len(w) == int(expected_sec * TARGET_SR)
        else:
            assert len(w) >= int(ECAPA_TRAIN_MIN_SEC * TARGET_SR) - 1


@requires_sample_audio
def test_preprocess_audio_no_denoise_option_runs(sample_audio_manifest):
    path = REPO_ROOT / sample_audio_manifest.iloc[0]["path"]
    windows = preprocess_audio(path, mode="ecapa_train", apply_denoise=False)
    assert len(windows) == 1
    assert np.all(np.isfinite(windows[0]))
