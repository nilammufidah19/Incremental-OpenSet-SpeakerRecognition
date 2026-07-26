"""x-vector speaker embedding extractor (F10-01 baseline only) [4.10, Tabel 4.5].

Used exclusively as the backbone for the "x-vector + PLDA" baseline (Snyder
et al., 2018) in the capability-ladder comparison (Bab 4.10) -- the
proposed system itself uses ECAPA-TDNN + Whisper (src/models/ecapa.py,
src/models/whisper_encoder.py), not this module.
"""
from __future__ import annotations

import platform

import numpy as np
import torch

EMBEDDING_DIM = 512
_MODEL = None
_DEVICE = None
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _patch_windows_symlink_issue() -> None:
    """Same Windows symlink workaround as src/models/ecapa.py -- see that
    module's docstring for the full explanation."""
    if platform.system() != "Windows":
        return
    import speechbrain.utils.fetching as sbf

    if getattr(sbf.link_with_strategy, "_windows_copy_patch", False):
        return
    original = sbf.link_with_strategy

    def patched(src, dst, local_strategy):
        if local_strategy == sbf.LocalStrategy.SYMLINK:
            local_strategy = sbf.LocalStrategy.COPY
        return original(src, dst, local_strategy)

    patched._windows_copy_patch = True
    sbf.link_with_strategy = patched


def get_model(device: str = DEFAULT_DEVICE):
    global _MODEL, _DEVICE
    if _MODEL is None or _DEVICE != device:
        _patch_windows_symlink_issue()
        from speechbrain.inference import EncoderClassifier

        _MODEL = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="pretrained_models/spkrec-xvect-voxceleb",
            run_opts={"device": device},
        )
        _MODEL.eval()
        _DEVICE = device
    return _MODEL


def extract_embedding(waveform: np.ndarray, sr: int = 16000, device: str = DEFAULT_DEVICE) -> np.ndarray:
    if sr != 16000:
        raise ValueError(f"x-vector expects sr=16000, got {sr}")
    model = get_model(device)
    wav_tensor = torch.from_numpy(np.ascontiguousarray(waveform)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_batch(wav_tensor)
    emb = emb.squeeze().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def extract_embedding_windows(windows: list[np.ndarray], sr: int = 16000, device: str = DEFAULT_DEVICE) -> np.ndarray:
    embeddings = np.stack([extract_embedding(w, sr, device) for w in windows])
    mean_emb = embeddings.mean(axis=0)
    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm
    return mean_emb.astype(np.float32)
