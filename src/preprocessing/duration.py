"""Duration standardization (F2-06/07/08) [4.4 poin 4].

Implements the padding-not-truncation strategy from Bab 4.4: circular
padding for short ECAPA-TDNN training clips, silence padding for Whisper's
architecturally-fixed 30s window, and sliding-window chunking (with
embedding-level averaging left to the caller, see F3) for long audio at
inference/enrollment time.

Four standardization modes, matching how each backbone is actually used:
  - "ecapa_train":      pad circular to >=3s; longer clips pass through
                         unmodified (ECAPA's attentive statistics pooling
                         natively handles variable-length input).
  - "ecapa_inference":  full-length pass-through, UNLESS longer than
                         MAX_ECAPA_INFERENCE_SEC, in which case sliding
                         windows are used to bound compute cost.
  - "whisper_train":    padded/cropped to exactly WHISPER_FIXED_SEC (30s).
                         Whisper's encoder architecture hard-requires
                         exactly 3000 mel frames (30s @ the Tabel 4.2 hop
                         length); unlike every other case in this module,
                         audio longer than 30s here IS cropped (to the
                         first 30s) rather than preserved in full -- a
                         necessary consequence of Whisper's fixed
                         positional embeddings, not a preprocessing choice.
  - "whisper_inference": always sliding-windowed into fixed 30s chunks
                         (short audio becomes one silence-padded window;
                         long audio becomes several), since Whisper simply
                         cannot accept a non-30s input at all.

Every mode returns a **list** of waveforms (length 1 unless
sliding-windowed), so callers always average one-or-more window embeddings
uniformly regardless of mode.
"""
from __future__ import annotations

import numpy as np

ECAPA_TRAIN_MIN_SEC = 3.0
WHISPER_FIXED_SEC = 30.0
MAX_ECAPA_INFERENCE_SEC = 30.0
SLIDING_WINDOW_SEC = 30.0
SLIDING_HOP_SEC = 30.0  # non-overlapping by default


def pad_circular(waveform: np.ndarray, sr: int, target_sec: float) -> np.ndarray:
    """Pad by circular repetition (never crop) until >= target_sec long."""
    target_len = int(round(target_sec * sr))
    if len(waveform) >= target_len:
        return waveform
    if len(waveform) == 0:
        return np.zeros(target_len, dtype=np.float32)
    n_repeats = int(np.ceil(target_len / len(waveform)))
    return np.tile(waveform, n_repeats)[:target_len].astype(np.float32)


def pad_silence(waveform: np.ndarray, sr: int, target_sec: float) -> np.ndarray:
    """Pad with trailing zeros (never crop) until >= target_sec long."""
    target_len = int(round(target_sec * sr))
    if len(waveform) >= target_len:
        return waveform
    pad_len = target_len - len(waveform)
    return np.concatenate([waveform, np.zeros(pad_len, dtype=np.float32)]).astype(np.float32)


def crop(waveform: np.ndarray, sr: int, target_sec: float) -> np.ndarray:
    """Explicit, clearly-named crop -- used only where architecturally
    mandatory (Whisper's fixed 30s input), never as a general-purpose
    duration-standardization step."""
    target_len = int(round(target_sec * sr))
    return waveform[:target_len]


def sliding_windows(
    waveform: np.ndarray,
    sr: int,
    window_sec: float = SLIDING_WINDOW_SEC,
    hop_sec: float = SLIDING_HOP_SEC,
) -> list[np.ndarray]:
    """Split into (optionally overlapping) fixed-length windows; the final
    window is silence-padded up to window_sec if the audio doesn't divide
    evenly. A waveform shorter than window_sec becomes a single padded
    window."""
    window_len = int(round(window_sec * sr))
    hop_len = int(round(hop_sec * sr))

    if len(waveform) <= window_len:
        return [pad_silence(waveform, sr, window_sec)]

    windows = []
    start = 0
    while True:
        end = start + window_len
        chunk = waveform[start:end]
        if len(chunk) < window_len:
            chunk = pad_silence(chunk, sr, window_sec)
        windows.append(chunk)
        if end >= len(waveform):
            break
        start += hop_len
    return windows


def standardize_duration(waveform: np.ndarray, sr: int, mode: str) -> list[np.ndarray]:
    """Dispatch to the appropriate strategy for `mode` (see module docstring)."""
    if mode == "ecapa_train":
        return [pad_circular(waveform, sr, ECAPA_TRAIN_MIN_SEC)]
    if mode == "ecapa_inference":
        if len(waveform) / sr <= MAX_ECAPA_INFERENCE_SEC:
            return [waveform]
        return sliding_windows(waveform, sr, SLIDING_WINDOW_SEC, SLIDING_HOP_SEC)
    if mode == "whisper_train":
        padded = pad_silence(waveform, sr, WHISPER_FIXED_SEC)
        return [crop(padded, sr, WHISPER_FIXED_SEC)]
    if mode == "whisper_inference":
        return sliding_windows(waveform, sr, SLIDING_WINDOW_SEC, SLIDING_HOP_SEC)
    raise ValueError(
        f"unknown mode {mode!r}, expected one of: "
        "ecapa_train, ecapa_inference, whisper_train, whisper_inference"
    )
