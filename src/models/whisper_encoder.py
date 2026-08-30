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


# --------------------------------------------------------------------------- #
# Experiment 6 F6-1: training-free Whisper-PMFA-style readout                  #
# --------------------------------------------------------------------------- #
# Zhao et al., "Whisper-PMFA: Partial Multi-Scale Feature Aggregation for
# Speaker Verification using Whisper Models" (Interspeech 2024) shows the
# middle-to-later encoder blocks carry the most speaker information and that
# aggregating a SUBSET of them beats any single layer. Their full recipe also
# trains an ECAPA-style backend with AAM-softmax + LoRA, which this thesis
# cannot use (training-free constraint, docs/experiment-6-plan.md section 2),
# so F6-1 keeps only the parts that are pure arithmetic over a frozen encoder:
# multi-layer aggregation, and mean AND standard-deviation pooling
# (Okabe et al., Interspeech 2018 -- statistics pooling captures within-
# utterance variation that a mean discards).
PMFA_LAYERS = (3, 4, 5, 6)  # hidden_states indices; whisper-base has 6 blocks

# F6-1 follow-up (30 Aug 2026). The {3,4,5,6} choice mirrored Whisper-PMFA's
# "middle-to-later blocks", but that paper selects from a 32-block encoder;
# on whisper-base's 6 blocks the same set spans 50-100% of the depth, which is
# deeper than the paper's analogue. The G6.1 sweep then found quality falling
# MONOTONICALLY with depth (layer 3 > 4 > 5 > 6), so the informative region is
# shallower than anything cached -- and layers 1-2 had never been measured at
# all. Caching ALL blocks costs the same single forward pass as caching four,
# so this namespace ends the question permanently: every layer subset becomes
# a slice, and no future layer study needs GPU time again.
PMFA_ALL_LAYERS = (1, 2, 3, 4, 5, 6)


def extract_embedding_pmfa(
    waveform: np.ndarray,
    sr: int = 16000,
    model_name: str = DEFAULT_MODEL_NAME,
    layers: tuple[int, ...] = PMFA_LAYERS,
    device: str = DEFAULT_DEVICE,
) -> np.ndarray:
    """Multi-layer masked mean+std pooling -> len(layers) * 2 * d_model dims
    (4096 for whisper-base with 4 layers).

    Layout is [mean_L1, std_L1, mean_L2, std_L2, ...] in `layers` order, so a
    consumer can slice out any single layer or statistic without recomputing.

    IMPORTANT -- this returns a RAW, UN-NORMALIZED vector, unlike every other
    extractor in this package. That is deliberate:

      * Transformer layers differ by an order of magnitude in activation
        scale, so a plain concatenation is dominated by whichever layer has
        the largest norm. Whether to equalize them (per-layer L2), and
        whether to apply anti-anisotropy post-processing (ABTT / whitening,
        fit on base_train), is exactly what F6-1 has to MEASURE.
      * Every such choice is a deterministic function of these raw statistics,
        so caching raw keeps all of them reachable. Caching a normalized
        vector would bake one choice in and force a ~45 min GPU recompute per
        ablation -- and would make the feature-flag rule "default = old
        behaviour" unenforceable, since the default would live inside the
        cache files.

    Consumers that need unit norm get it for free: ScoreFusionEmbed.forward
    L2-normalizes both halves itself.

    Padding is masked exactly as in extract_embedding (see
    _valid_encoder_frames); with std pooling this matters even more than with
    mean pooling, because trailing silence would otherwise look like genuine
    within-utterance variation.
    """
    if sr != 16000:
        raise ValueError(f"Whisper expects sr=16000, got {sr}")

    model, feature_extractor = get_model(model_name, device)
    inputs = feature_extractor(waveform, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        hidden_states = model.encoder(
            input_features, output_hidden_states=True
        ).hidden_states
        max_idx = len(hidden_states) - 1
        bad = [i for i in layers if not 0 <= i <= max_idx]
        if bad:
            raise ValueError(
                f"layers {bad} out of range for {model_name}: valid 0..{max_idx}"
            )

        n_valid = _valid_encoder_frames(waveform, hidden_states[layers[0]].shape[1])
        parts = []
        for idx in layers:
            frames = hidden_states[idx][:, :n_valid, :]
            mean = frames.mean(dim=1).squeeze(0)
            # unbiased=False: with n_valid == 1 the unbiased estimator is NaN,
            # and short clips are exactly where this must not blow up.
            std = frames.std(dim=1, unbiased=False).squeeze(0)
            parts.append(mean)
            parts.append(std)
        pooled = torch.cat(parts).cpu().numpy().astype(np.float32)

    return pooled


def extract_embedding_pmfa_windows(
    windows: list[np.ndarray],
    sr: int = 16000,
    model_name: str = DEFAULT_MODEL_NAME,
    layers: tuple[int, ...] = PMFA_LAYERS,
    device: str = DEFAULT_DEVICE,
) -> np.ndarray:
    """Average the raw PMFA statistics across sliding windows.

    No normalization here either, for the reasons in extract_embedding_pmfa --
    and note this differs from extract_embedding_windows, which normalizes the
    averaged embedding.
    """
    embeddings = np.stack(
        [extract_embedding_pmfa(w, sr, model_name, layers, device) for w in windows]
    )
    return embeddings.mean(axis=0).astype(np.float32)
