"""Episodic training loop (F5-04/05) [4.6].

Trains the Gated Attention Fusion layer (src/models/fusion.py) end-to-end
via the Prototypical Network objective (Pers. 3.6-3.10). Backbones
(ECAPA-TDNN, Whisper) stay frozen -- their embeddings come from the F3-06
cache and are only concatenated/split here, never recomputed with
gradients -- only `fusion_model`'s parameters receive gradient updates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from src.models.fusion import GatedAttentionFusion
from src.prototypical.classifier import classify_log_probs, negative_log_likelihood, predict
from src.prototypical.data import split_raw_embedding
from src.prototypical.episodic import sample_episode
from src.prototypical.prototype import compute_prototypes


@dataclass
class EpisodeResult:
    loss: float
    accuracy: float


def _run_episode(
    fusion_model: GatedAttentionFusion,
    embeddings_by_speaker: dict,
    n_way: int,
    k_shot: int,
    n_query: int,
    seed: int,
    device: str,
) -> tuple[torch.Tensor, float]:
    episode = sample_episode(embeddings_by_speaker, n_way, k_shot, n_query, seed=seed)

    support_raw = torch.from_numpy(episode.support_embeddings).float().to(device)
    query_raw = torch.from_numpy(episode.query_embeddings).float().to(device)
    support_labels = torch.from_numpy(episode.support_labels).long().to(device)
    query_labels = torch.from_numpy(episode.query_labels).long().to(device)

    support_ecapa, support_whisper = split_raw_embedding(support_raw)
    query_ecapa, query_whisper = split_raw_embedding(query_raw)

    support_fused = fusion_model(support_ecapa, support_whisper)
    query_fused = fusion_model(query_ecapa, query_whisper)

    prototypes = compute_prototypes(support_fused, support_labels, n_way)
    log_probs = classify_log_probs(query_fused, prototypes)
    loss = negative_log_likelihood(log_probs, query_labels)

    with torch.no_grad():
        predicted, _ = predict(query_fused, prototypes)
        accuracy = (predicted == query_labels).float().mean().item()

    return loss, accuracy


def train_episodic(
    fusion_model: GatedAttentionFusion,
    embeddings_by_speaker: dict,
    n_way: int,
    k_shot: int,
    n_query: int,
    n_episodes: int,
    seed: int = 0,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list[EpisodeResult]:
    """Run `n_episodes` of N-way K-shot episodic training (base training,
    F5-05, on Data Latih Awal in the full protocol). Returns per-episode
    (loss, accuracy) so callers can inspect convergence."""
    fusion_model.to(device)
    fusion_model.train()
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=lr)

    results = []
    for i in range(n_episodes):
        loss, accuracy = _run_episode(
            fusion_model, embeddings_by_speaker, n_way, k_shot, n_query, seed=seed + i, device=device
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        results.append(EpisodeResult(loss=loss.item(), accuracy=accuracy))

    return results


@torch.no_grad()
def evaluate_episodic(
    fusion_model: GatedAttentionFusion,
    embeddings_by_speaker: dict,
    n_way: int,
    k_shot: int,
    n_query: int,
    n_episodes: int,
    seed: int = 0,
    device: str = "cpu",
) -> list[EpisodeResult]:
    """Same as train_episodic but no gradient updates -- for validation/test
    accuracy reporting."""
    fusion_model.to(device)
    fusion_model.eval()

    results = []
    for i in range(n_episodes):
        loss, accuracy = _run_episode(
            fusion_model, embeddings_by_speaker, n_way, k_shot, n_query, seed=seed + i, device=device
        )
        results.append(EpisodeResult(loss=loss.item(), accuracy=accuracy))
    return results


def save_checkpoint(fusion_model: GatedAttentionFusion, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": fusion_model.state_dict(), "mode": fusion_model.mode}, path)


def load_checkpoint(path: str | Path, fusion_model: GatedAttentionFusion) -> GatedAttentionFusion:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    fusion_model.load_state_dict(checkpoint["state_dict"])
    return fusion_model
