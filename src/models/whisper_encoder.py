"""Whisper encoder speaker embedding extractor (F3-04) [3.3, 4.5].

Uses a pretrained Whisper **encoder only** (no decoder/ASR head) as a fixed
feature extractor, matching plan/03-architecture.md's frozen-backbone
design. HuggingFace's own `WhisperFeatureExtractor` computes the log-Mel
input (16 kHz, 80 mel, 25 ms/10 ms, Hann, padded/cropped to exactly 30s) --
this is numerically Tabel 4.2-compliant and, crucially, matches exactly
what the pretrained encoder weights were trained on, so we use it directly
rather than src/features/logmel.py's standalone implementation (see that
module's docstring for the reasoning).

Pooling: mean-pooling over the time axis of a **middle** encoder layer
(default: layer at 50% depth, e.g. layer 3-of-6 for whisper-base), producing
a fixed `d_model`-sized embedding (384/512/768/1024/1280 for
tiny/base/small/medium/large -- matches the "512-1280 dimensi tergantung
ukuran model" range in Bab 3.4/4.5).

Empirically validated (see plan/04-tasks.md F3-05 notes): pooling the
*final* encoder layer -- the naive choice -- gave same-speaker cosine
similarity *lower* than different-speaker similarity on real VoxCeleb1
audio (checked across 6 speakers / 15 cross-pairs). Whisper's last layer is
specialized for its ASR pretraining objective (content/phonetics), which
washes out speaker identity; middle layers retain much more of it. This
matches the motivation behind Whisper-SV/Whisper-PMFA's multi-layer
aggregation cited in Bab 2 -- a single middle layer is a lighter-weight
stand-in for that here, since the backbone stays frozen until F5's episodic
training exists to justify a learned aggregation instead.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperModel

DEFAULT_MODEL_NAME = "openai/whisper-base"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_MODEL = None
_FEATURE_EXTRACTOR = None
_MODEL_NAME = None
_DEVICE = None


def get_model(model_name: str = DEFAULT_MODEL_NAME, device: str = DEFAULT_DEVICE):
    global _MODEL, _FEATURE_EXTRACTOR, _MODEL_NAME, _DEVICE
    if _MODEL is None or _MODEL_NAME != model_name or _DEVICE != device:
        _FEATURE_EXTRACTOR = WhisperFeatureExtractor.from_pretrained(model_name)
        _MODEL = WhisperModel.from_pretrained(model_name).to(device)
        _MODEL.eval()
        _MODEL_NAME = model_name
        _DEVICE = device
    return _MODEL, _FEATURE_EXTRACTOR


def embedding_dim(model_name: str = DEFAULT_MODEL_NAME) -> int:
    model, _ = get_model(model_name)
    return model.config.d_model


def _valid_encoder_frames(waveform: np.ndarray, n_encoder_frames: int) -> int:
    """How many of `n_encoder_frames` correspond to REAL (non-padded) audio.

    src/preprocessing/duration.py's pad_silence()/sliding_windows() always
    grow a short window up to exactly 30s by appending TRAILING exact-0.0
    samples -- never by modifying the front. Real captured/processed audio
    essentially never contains a long run of bit-exact 0.0 samples, so the
    index of the last nonzero sample reliably locates the pad boundary. The
    fraction of real samples maps linearly onto the encoder's time axis
    (fixed 2x conv-stride downsampling from the fixed-length mel input), so
    this needs no access to feature_extractor/model internals.

    A waveform with no padding at all (e.g. a genuinely exactly-30s or
    cropped-to-30s clip) has its last sample nonzero (or is all-zero, an
    edge case handled below), so this is a no-op and pooling is unchanged.
    """
    nonzero = np.flatnonzero(waveform != 0.0)
    if len(nonzero) == 0:
        return n_encoder_frames  # all-silence input: nothing to mask against
    valid_samples = int(nonzero[-1]) + 1
    frac = valid_samples / len(waveform)
    return max(1, min(n_encoder_frames, round(n_encoder_frames * frac)))


def extract_embedding(
    waveform: np.ndarray,
    sr: int = 16000,
    model_name: str = DEFAULT_MODEL_NAME,
    layer_fraction: float = 0.5,
    device: str = DEFAULT_DEVICE,
) -> np.ndarray:
    """Extract an L2-normalized speaker embedding from a single 30s-window
    waveform (see src/preprocessing/duration.py: "whisper_train"/
    "whisper_inference" modes already produce exactly-30s windows).

    `layer_fraction` selects which encoder hidden-state layer to mean-pool
    (0.0 = input embeddings, 1.0 = final layer); default 0.5 (middle depth)
    -- see module docstring for why the final layer is a poor default.

    Pooling excludes encoder frames that correspond to trailing silence
    padding (see _valid_encoder_frames): most VoxCeleb-style utterances are
    well under 30s, so an unweighted mean over the full window would let
    mostly-silent padding dominate the pooled embedding.
    """
    if sr != 16000:
        raise ValueError(f"Whisper expects sr=16000, got {sr}")

    model, feature_extractor = get_model(model_name, device)
    inputs = feature_extractor(waveform, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        hidden_states = model.encoder(
            input_features, output_hidden_states=True
        ).hidden_states  # tuple of (n_layers + 1) tensors, each [1, T, d_model]
        layer_idx = round(layer_fraction * (len(hidden_states) - 1))
        layer = hidden_states[layer_idx]
        n_valid = _valid_encoder_frames(waveform, layer.shape[1])
        pooled = layer[:, :n_valid, :].mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)

    norm = np.linalg.norm(pooled)
    if norm > 0:
        pooled = pooled / norm
    return pooled


def extract_embedding_windows(
    windows: list[np.ndarray],
    sr: int = 16000,
    model_name: str = DEFAULT_MODEL_NAME,
    layer_fraction: float = 0.5,
    device: str = DEFAULT_DEVICE,
) -> np.ndarray:
    """Extract and average embeddings across multiple 30s windows (long
    audio at inference, always sliding-windowed per duration.py)."""
    embeddings = np.stack(
        [extract_embedding(w, sr, model_name, layer_fraction, device) for w in windows]
    )
    mean_emb = embeddings.mean(axis=0)
    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm
    return mean_emb.astype(np.float32)
