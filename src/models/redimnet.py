"""ReDimNet speaker embedding extractor (Experiment 5, kandidat #2).

Frozen pretrained ReDimNet (Yakovlev et al., Interspeech 2024,
arXiv:2407.18223) as the candidate second backbone replacing Whisper.
Chosen after candidate #1 (WavLM-base-plus, leakage-free) FAILED gate G5.1:
frozen mean-pooled WavLM peaked at 0.2925 standalone on the validation
snapshot (best layer 5 of 12; experiments/exp5_wavlm_layer_sweep.json) --
far below the 0.70 bar -- because SSL models need a trained backend head,
which the thesis' training-free constraint forbids. ReDimNet is a dedicated
speaker-verification architecture (reshape 1D<->2D topology -- maximal
architectural contrast to ECAPA's 1D-TDNN), so its raw cosine space IS its
speaker space.

Leakage caveat (REPORTED, Fase 0a audit, experiments/exp5_leakage_audit
.json): ReDimNet is trained on VoxCeleb2-dev, which contains 80/100 task
speakers / 173/221 reserved-pool speakers. The incumbent ECAPA (SpeechBrain
spkrec-ecapa-voxceleb) is likewise VoxCeleb-trained, so internal comparisons
(A1/A2/A3, baselines) remain like-for-like; the overlap is disclosed in
docs/experiment-5.*.

Loaded via torch.hub from the official IDRnD/ReDimNet repo (weights from
its "latest" GitHub release; the hub code + checkpoint are cached under
~/.cache/torch/hub after the first call). Model takes a raw 16 kHz waveform
tensor (B, T) -- its Mel frontend is inside forward() -- and returns a
192-d embedding; we L2-normalize so Euclidean == monotone cosine, matching
every other backbone in src/features/cache.py.

Interface mirrors src/models/whisper_encoder.py so cache.py can register it
like any other backbone. Duration mode: "ecapa_inference" (variable-length
pass-through, sliding windows only for very long audio).
"""
from __future__ import annotations

import numpy as np
import torch

DEFAULT_MODEL_NAME = "b2"
DEFAULT_TRAIN_TYPE = "ft_lm"   # large-margin fine-tuned release (best EER)
DEFAULT_DATASET = "vox2"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_MODEL = None
_KEY = None


def get_model(
    model_name: str = DEFAULT_MODEL_NAME,
    train_type: str = DEFAULT_TRAIN_TYPE,
    dataset: str = DEFAULT_DATASET,
    device: str = DEFAULT_DEVICE,
):
    global _MODEL, _KEY
    key = (model_name, train_type, dataset, device)
    if _MODEL is None or _KEY != key:
        model = torch.hub.load(
            "IDRnD/ReDimNet", "ReDimNet",
            model_name=model_name, train_type=train_type, dataset=dataset,
            trust_repo=True,
        )
        model.eval()
        _MODEL = model.to(device)
        _KEY = key
    return _MODEL


def embedding_dim() -> int:
    return 192


def extract_embedding(
    waveform: np.ndarray,
    sr: int = 16000,
    device: str = DEFAULT_DEVICE,
) -> np.ndarray:
    """L2-normalized 192-d ReDimNet embedding for one variable-length
    16 kHz waveform."""
    if sr != 16000:
        raise ValueError(f"ReDimNet expects sr=16000, got {sr}")
    model = get_model(device=device)
    wave = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))
    wave = wave.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(wave).squeeze(0).cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def extract_embedding_windows(
    windows: list[np.ndarray],
    sr: int = 16000,
    device: str = DEFAULT_DEVICE,
) -> np.ndarray:
    """Average embeddings across sliding windows (long audio only)."""
    embeddings = np.stack([extract_embedding(w, sr, device) for w in windows])
    mean_emb = embeddings.mean(axis=0)
    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm
    return mean_emb.astype(np.float32)
