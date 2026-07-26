#!/usr/bin/env python
"""F5-05: functional base-training trial on real cached embeddings.

IMPORTANT -- scale caveat: the proposal's base training runs on the full
Data Latih Awal (5,156 speakers, Bab 4.3). We currently have real cached
embeddings for only ~112 speakers (40 from data/raw/audio/vox1_sample +
~72 from the first slice of data/raw/audio/base_train_capped -- see
plan/04-tasks.md F1/F3 notes on data acquisition scale). This script
demonstrates and exercises the ACTUAL base-training mechanism end-to-end
(episodic sampling -> fusion -> prototypes -> NLL loss -> backprop) on the
data available now; it is a functional smoke test, not the final,
reportable base-training run described in Bab 4.6/4.8, which needs the
full speaker set once more audio (or the pending VoxCeleb2 download) is
available.

Usage:
    .venv/Scripts/python.exe scripts/run_base_training_trial.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.features.cache import is_cached  # noqa: E402
from src.models.fusion import FUSION_DIM, GatedAttentionFusion  # noqa: E402
from src.prototypical.data import ECAPA_DIM, WHISPER_DIM, build_raw_embedding_index  # noqa: E402
from src.prototypical.train import evaluate_episodic, save_checkpoint, train_episodic  # noqa: E402

MANIFESTS = [
    "data/raw/audio/vox1_sample/manifest.csv",
    "data/raw/audio/base_train_capped/manifest.csv",
]
CHECKPOINT_PATH = REPO_ROOT / "experiments" / "checkpoints" / "fusion_trial.ckpt"

N_WAY = 10
K_SHOT = 1  # one-shot, per proposal's chosen protocol
N_QUERY = 5
N_TRAIN_EPISODES = 500
N_EVAL_EPISODES = 50


def main() -> None:
    frames = []
    for m in MANIFESTS:
        df = pd.read_csv(REPO_ROOT / m)
        frames.append(df)
    manifest = pd.concat(frames, ignore_index=True)

    # Restrict to utterances already cached (F3-06) so this trial runs in
    # minutes, not the ~50+ min it'd take to newly compute embeddings for
    # the full 11.9k-utterance manifest -- see module docstring's scale caveat.
    manifest["_cached"] = manifest["path"].apply(
        lambda p: is_cached(REPO_ROOT / p, "ecapa") and is_cached(REPO_ROOT / p, "whisper")
    )
    n_total = len(manifest)
    manifest = manifest[manifest["_cached"]]
    print(f"{len(manifest)}/{n_total} utterances already cached; using only those for this trial.")

    counts = manifest["speaker_id"].value_counts()
    eligible = counts[counts >= K_SHOT + N_QUERY].index.tolist()
    manifest = manifest[manifest["speaker_id"].isin(eligible)]
    print(f"Building raw embedding index from {len(manifest)} utterances "
          f"across {manifest['speaker_id'].nunique()} eligible speakers "
          f"(>= {K_SHOT + N_QUERY} samples each)...")

    index = build_raw_embedding_index(manifest)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = GatedAttentionFusion(ECAPA_DIM, WHISPER_DIM, FUSION_DIM, mode="fusion")

    print(f"\nEvaluating BEFORE training ({N_EVAL_EPISODES} episodes, "
          f"{N_WAY}-way {K_SHOT}-shot)...")
    before = evaluate_episodic(model, index, N_WAY, K_SHOT, N_QUERY, N_EVAL_EPISODES, seed=9000, device=device)
    acc_before = sum(r.accuracy for r in before) / len(before)
    loss_before = sum(r.loss for r in before) / len(before)
    print(f"  before: mean_accuracy={acc_before:.3f} mean_loss={loss_before:.3f}")

    print(f"\nTraining ({N_TRAIN_EPISODES} episodes)...")
    results = train_episodic(
        model, index, N_WAY, K_SHOT, N_QUERY, N_TRAIN_EPISODES, seed=0, lr=1e-3, device=device
    )
    window = 50
    for i in range(0, len(results), window):
        chunk = results[i : i + window]
        acc = sum(r.accuracy for r in chunk) / len(chunk)
        loss = sum(r.loss for r in chunk) / len(chunk)
        print(f"  episodes {i:4d}-{i+len(chunk):4d}: mean_accuracy={acc:.3f} mean_loss={loss:.3f}")

    print(f"\nEvaluating AFTER training ({N_EVAL_EPISODES} episodes)...")
    after = evaluate_episodic(model, index, N_WAY, K_SHOT, N_QUERY, N_EVAL_EPISODES, seed=9000, device=device)
    acc_after = sum(r.accuracy for r in after) / len(after)
    loss_after = sum(r.loss for r in after) / len(after)
    print(f"  after: mean_accuracy={acc_after:.3f} mean_loss={loss_after:.3f}")

    save_checkpoint(model, CHECKPOINT_PATH)
    print(f"\nSaved checkpoint to {CHECKPOINT_PATH}")
    print(f"\nSummary: accuracy {acc_before:.3f} -> {acc_after:.3f}, "
          f"loss {loss_before:.3f} -> {loss_after:.3f}")


if __name__ == "__main__":
    main()
