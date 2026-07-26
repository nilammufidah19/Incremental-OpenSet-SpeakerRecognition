"""ECAPA-TDNN speaker embedding extractor (F3-03) [3.4, 4.5].

Uses the pretrained `speechbrain/spkrec-ecapa-voxceleb` checkpoint as a
fixed feature extractor (see plan/03-architecture.md §11: backbones stay
frozen; only the fusion layer + prototypical network head are trained).
Feeds raw waveform directly -- SpeechBrain's own internal feature pipeline
already matches Tabel 4.2 (16 kHz, 80 mel, 25 ms/10 ms, Hamming), so we
don't re-derive log-Mel ourselves here (see src/features/logmel.py
docstring for why: avoiding any mismatch with what the pretrained weights
actually expect).
"""
from __future__ import annotations

import platform

import numpy as np
import torch

EMBEDDING_DIM = 192
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL = None
_DEVICE = None


def _patch_windows_symlink_issue() -> None:
    """SpeechBrain's checkpoint fetcher defaults to symlinking from the HF
    cache into `savedir`, which raises `OSError: [WinError 1314]` on Windows
    unless Developer Mode / elevated privileges are available. Since we
    cannot (and should not) flip that machine-wide OS setting from an
    automated script, fall back to plain file copies on Windows instead --
    functionally identical, just slightly more disk usage (~80 MB, once).
    """
    if platform.system() != "Windows":
        return

    import speechbrain.utils.fetching as sbf

    if getattr(sbf.link_with_strategy, "_windows_copy_patch", False):
        return  # already patched

    original = sbf.link_with_strategy

    def patched(src, dst, local_strategy):
        if local_strategy == sbf.LocalStrategy.SYMLINK:
            local_strategy = sbf.LocalStrategy.COPY
        return original(src, dst, local_strategy)

    patched._windows_copy_patch = True
    sbf.link_with_strategy = patched


def get_model(device: str = DEFAULT_DEVICE):
    """Lazily load and cache the pretrained ECAPA-TDNN model on `device`."""
    global _MODEL, _DEVICE
    if _MODEL is None or _DEVICE != device:
        _patch_windows_symlink_issue()
        from speechbrain.inference import EncoderClassifier

        _MODEL = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        _MODEL.eval()
        _DEVICE = device
    return _MODEL


def extract_embedding(waveform: np.ndarray, sr: int = 16000, device: str = DEFAULT_DEVICE) -> np.ndarray:
    """Extract a 192-d, L2-normalized speaker embedding from a waveform.

    `waveform` must already be preprocessed (16 kHz mono; see
    src/preprocessing/pipeline.py, mode="ecapa_train"/"ecapa_inference").
    """
    if sr != 16000:
        raise ValueError(f"ECAPA-TDNN expects sr=16000, got {sr}")

    model = get_model(device)
    wav_tensor = torch.from_numpy(np.ascontiguousarray(waveform)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_batch(wav_tensor)  # [1, 1, 192]
    emb = emb.squeeze().cpu().numpy().astype(np.float32)

    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def extract_embedding_windows(
    windows: list[np.ndarray], sr: int = 16000, device: str = DEFAULT_DEVICE
) -> np.ndarray:
    """Extract and average embeddings across multiple windows (F2-08's
    sliding-window inference case): compute one embedding per window, then
    mean-average and re-normalize, per the "agregasi rata-rata embedding
    antar-window" strategy in Bab 4.4."""
    embeddings = np.stack([extract_embedding(w, sr, device) for w in windows])
    mean_emb = embeddings.mean(axis=0)
    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm
    return mean_emb.astype(np.float32)
