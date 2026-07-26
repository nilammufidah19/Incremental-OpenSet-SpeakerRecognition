"""Ablation study configurations (F9) [4.9, Tabel 4.3/4.4].

Group A (fusion contribution, Tabel 4.3): trains three otherwise-identical
GatedAttentionFusion instances that differ only in `mode`
(ecapa_only=A1, whisper_only=A2, fusion=A3), so any accuracy/EER difference
is attributable purely to the fusion mechanism.

Group B (continual learning contribution, Tabel 4.4): reuses the trained
A3 (fusion) model and evaluates it twice via the same FSCIL harness, once
with the SpeakerIdentificationSystem's `continual_mode="static"` (B1) and
once with `"running_average"` (B2) -- so any Average Accuracy/Forgetting
Measure difference is attributable purely to whether prototypes update.
"""
from __future__ import annotations

import torch

from src.models.fusion import FUSION_DIM, GatedAttentionFusion
from src.prototypical.data import ECAPA_DIM, WHISPER_DIM
from src.prototypical.train import train_episodic

FUSION_ABLATION_MODES = {
    "A1_ecapa_only": "ecapa_only",
    "A2_whisper_only": "whisper_only",
    "A3_fusion": "fusion",
}
CONTINUAL_ABLATION_MODES = {
    "B1_static": "static",
    "B2_running_average": "running_average",
}


def train_fusion_for_ablation(
    mode: str,
    embeddings_by_speaker: dict,
    n_way: int,
    k_shot: int,
    n_query: int,
    n_episodes: int,
    seed: int,
    lr: float = 1e-3,
    device: str = "cpu",
    residual_init: bool = False,
    second_dim: int = WHISPER_DIM,
) -> GatedAttentionFusion:
    """Train one fusion-mode variant (A1/A2/A3), holding every other
    hyperparameter identical across variants.

    `residual_init` (see GatedAttentionFusion) starts the projection from a
    raw-ECAPA-preserving init instead of random, so the undertrained head
    cannot destroy ECAPA's native embedding space -- applied identically to
    A1/A2/A3 so the ablation stays fair.

    `n_episodes <= 0` freezes the model at its (residual) init and skips
    episodic training entirely. A training-amount sweep on this data scale
    (71 base speakers) showed episodic fine-tuning MONOTONICALLY degrades
    accuracy -- frozen residual-init fusion scores 0.783 closed / 0.734
    open-set vs 0.276 / 0.187 after 500 episodes -- because 71 speakers is far
    too few to improve on ECAPA's fully-pretrained embedding space without
    overfitting it. Freezing keeps ECAPA's quality as a hard lower bound."""
    torch.manual_seed(seed)
    # `second_dim` (Experiment 5): dimension of the raw index's second half --
    # default WHISPER_DIM keeps every pre-exp5 caller byte-identical; the
    # exp5 harness passes the candidate backbone's dim (e.g. WavLM 768).
    model = GatedAttentionFusion(ECAPA_DIM, second_dim, FUSION_DIM, mode=mode, residual_init=residual_init)
    if n_episodes > 0:
        train_episodic(model, embeddings_by_speaker, n_way, k_shot, n_query, n_episodes, seed=seed, lr=lr, device=device)
    model.to(device)
    return model
