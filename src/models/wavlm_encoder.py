"""WavLM encoder speaker embedding extractor (Experiment 5, Fase 0b).

Uses a pretrained, FROZEN WavLM (microsoft/wavlm-base-plus) as the candidate
second backbone replacing Whisper (docs/experiment-5.md). Motivation:

  * WavLM is the strongest self-supervised model for speaker tasks on the
    SUPERB benchmark (Chen et al., IEEE JSTSP 2022, arXiv:2110.13900) --
    unlike Whisper, whose ASR objective washes out speaker identity
    (Experiment 2/4 negative results; Whisper-SV arXiv:2407.10048).
  * Its pretraining data (Libri-Light + GigaSpeech + VoxPopuli, 94k hours)
    contains NO VoxCeleb, so it is leakage-free w.r.t. this project's
    evaluation speakers -- unlike VoxCeleb2-dev-trained speaker models
    (ReDimNet/CAM++/WeSpeaker): the Fase 0a audit found 80/100 task
    speakers inside VoxCeleb2-dev (experiments/exp5_leakage_audit.json).
  * Zero new dependencies: loaded via the already-installed `transformers`.

Interface mirrors src/models/whisper_encoder.py exactly (singleton model,
`extract_embedding` / `extract_embedding_windows`, `layer_fraction` knob),
so src/features/cache.py can register layer variants with functools.partial
the same way it does for "whisper_l4".

Input: raw 16 kHz waveform, variable length ("ecapa_inference" duration
mode -- WavLM is a wav2vec2-family model and takes native-length audio, no
30s padding needed). Pooling: mean over time of one encoder hidden state,
selected by `layer_fraction` (0.0 = CNN feature-extractor output, 1.0 =
final transformer layer). Speaker information concentrates in EARLY
transformer layers (SUPERB SID layer analyses; DiariZen reports layers 1-3
dominating for speaker tasks), the opposite end from Whisper -- the exact
layer is locked by a validation-half sweep (scripts/exp5_wavlm_layer_sweep
.py), never on task speakers.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoFeatureExtractor, WavLMModel

DEFAULT_MODEL_NAME = "microsoft/wavlm-base-plus"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_MODEL = None
_FEATURE_EXTRACTOR = None
_MODEL_NAME = None
_DEVICE = None


def get_model(model_name: str = DEFAULT_MODEL_NAME, device: str = DEFAULT_DEVICE):
    global _MODEL, _FEATURE_EXTRACTOR, _MODEL_NAME, _DEVICE
    if _MODEL is None or _MODEL_NAME != model_name or _DEVICE != device:
        _FEATURE_EXTRACTOR = AutoFeatureExtractor.from_pretrained(model_name)
        _MODEL = WavLMModel.from_pretrained(model_name).to(device)
        _MODEL.eval()
        _MODEL_NAME = model_name
        _DEVICE = device
    return _MODEL, _FEATURE_EXTRACTOR


def embedding_dim(model_name: str = DEFAULT_MODEL_NAME) -> int:
    model, _ = get_model(model_name)
    return model.config.hidden_size  # 768 for base-plus


def num_hidden_states(model_name: str = DEFAULT_MODEL_NAME) -> int:
    """Number of entries in `hidden_states` (n_transformer_layers + 1)."""
    model, _ = get_model(model_name)
    return model.config.num_hidden_layers + 1  # 13 for base-plus


def extract_all_layer_embeddings(
    waveform: np.ndarray,
    sr: int = 16000,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = DEFAULT_DEVICE,
) -> np.ndarray:
    """One forward pass -> L2-normalized mean-pooled embedding of EVERY
    hidden state: (n_layers + 1, hidden_size). Used by the layer sweep so
    each utterance is embedded once, not once per candidate layer."""
    if sr != 16000:
        raise ValueError(f"WavLM expects sr=16000, got {sr}")
    model, feature_extractor = get_model(model_name, device)
    inputs = feature_extractor(waveform, sampling_rate=sr, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        hidden_states = model(input_values, output_hidden_states=True).hidden_states
        pooled = torch.stack([h.mean(dim=1).squeeze(0) for h in hidden_states])

    pooled = pooled.cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled / np.maximum(norms, 1e-12)


def extract_embedding(
    waveform: np.ndarray,
    sr: int = 16000,
    model_name: str = DEFAULT_MODEL_NAME,
    layer_fraction: float = 0.25,
    device: str = DEFAULT_DEVICE,
) -> np.ndarray:
    """L2-normalized speaker embedding from one variable-length waveform.

    `layer_fraction` selects the hidden state to mean-pool, as a fraction of
    depth (0.0 = CNN output, 1.0 = last transformer layer). The production
    value is locked by the Experiment 5 validation sweep and baked into the
    backbone registration in src/features/cache.py.
    """
    if sr != 16000:
        raise ValueError(f"WavLM expects sr=16000, got {sr}")
    model, feature_extractor = get_model(model_name, device)
    inputs = feature_extractor(waveform, sampling_rate=sr, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        hidden_states = model(input_values, output_hidden_states=True).hidden_states
        layer_idx = round(layer_fraction * (len(hidden_states) - 1))
        pooled = hidden_states[layer_idx].mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)

    norm = np.linalg.norm(pooled)
    if norm > 0:
        pooled = pooled / norm
    return pooled


def extract_embedding_windows(
    windows: list[np.ndarray],
    sr: int = 16000,
    model_name: str = DEFAULT_MODEL_NAME,
    layer_fraction: float = 0.25,
    device: str = DEFAULT_DEVICE,
) -> np.ndarray:
    """Average embeddings across sliding windows (only used when audio is
    longer than MAX_ECAPA_INFERENCE_SEC -- see duration.py)."""
    embeddings = np.stack(
        [extract_embedding(w, sr, model_name, layer_fraction, device) for w in windows]
    )
    mean_emb = embeddings.mean(axis=0)
    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm
    return mean_emb.astype(np.float32)
