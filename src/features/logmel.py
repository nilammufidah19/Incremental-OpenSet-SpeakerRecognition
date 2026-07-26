"""Log-Mel Spectrogram front-end (F3-01/02) [3.2, 4.5, Tabel 4.2].

Shared front-end parameters for both backbones, per Tabel 4.2: 16 kHz,
80 Mel filters, 25 ms window (400 samples), 10 ms hop (160 samples).
Only the window function differs: Hamming for ECAPA-TDNN, Hann for Whisper
(matching OpenAI Whisper's own `log_mel_spectrogram` convention, whose
N_FFT/HOP_LENGTH/N_MELS happen to be numerically identical to Tabel 4.2).

NOTE on actual backbone usage: this module exists as a standalone,
independently-testable implementation of the shared front-end described in
Bab 4.5/Gambar 4.1, for documentation/ablation/analysis purposes. The actual
pretrained backbones (F3-03/04) each recompute log-Mel features internally
via their own library's native feature extractor (SpeechBrain for
ECAPA-TDNN, HuggingFace `WhisperFeatureExtractor` for Whisper) rather than
consuming this module's output directly -- both of those already implement
parameters numerically matching Tabel 4.2, and reusing their exact
internal implementation avoids subtly mismatching the normalization/scaling
each pretrained model's weights were actually trained on (e.g. Whisper
applies a specific log-clamp + (log+4)/4 rescaling that must match exactly
for the pretrained encoder to behave as expected).
"""
from __future__ import annotations

import numpy as np
import torch
import torchaudio

SAMPLE_RATE = 16000
N_MELS = 80
WIN_LENGTH = int(round(0.025 * SAMPLE_RATE))  # 400 samples (25 ms)
HOP_LENGTH = int(round(0.010 * SAMPLE_RATE))  # 160 samples (10 ms)
N_FFT = WIN_LENGTH

_WINDOW_FNS = {
    "hamming": torch.hamming_window,
    "hann": torch.hann_window,
}


def compute_logmel(
    waveform: np.ndarray | torch.Tensor,
    sr: int = SAMPLE_RATE,
    window: str = "hamming",
    n_mels: int = N_MELS,
) -> torch.Tensor:
    """Compute a log-Mel spectrogram matching Tabel 4.2.

    Parameters
    ----------
    waveform : 1-D array/tensor of audio samples at `sr` Hz.
    sr : sample rate; must be SAMPLE_RATE (16000) to match Tabel 4.2 exactly.
    window : "hamming" (ECAPA-TDNN path) or "hann" (Whisper path).

    Returns
    -------
    torch.Tensor of shape (n_mels, n_frames).
    """
    if sr != SAMPLE_RATE:
        raise ValueError(f"compute_logmel expects sr={SAMPLE_RATE} (Tabel 4.2), got {sr}")
    if window not in _WINDOW_FNS:
        raise ValueError(f"window must be one of {list(_WINDOW_FNS)}, got {window!r}")

    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # [1, T]

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=n_mels,
        window_fn=_WINDOW_FNS[window],
        power=2.0,
    )
    mel = mel_transform(waveform)  # [1, n_mels, n_frames]
    log_mel = torch.log(mel.clamp(min=1e-10))
    return log_mel.squeeze(0)  # [n_mels, n_frames]
