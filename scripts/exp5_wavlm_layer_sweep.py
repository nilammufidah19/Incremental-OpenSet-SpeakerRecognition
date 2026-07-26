#!/usr/bin/env python
"""Experiment 5 Fase 0b -- WavLM layer sweep (docs/experiment-5.md).

Locks WHICH WavLM hidden layer to register as the candidate second backbone,
using the validation half of reserved_unknown_pool (the designated tuning
ground, exp3 section 4 discipline) -- task speakers are never touched.

Protocol: the same static 1-shot snapshot as scripts/exp4_complementarity_
asnorm.py (100 validation-half speakers, 1 support + 4 queries each), seed 0
only (the winning layer is then screened properly on 3 seeds via the exp4
script). Every utterance gets ONE WavLM forward pass that yields all 13
hidden states at once (extract_all_layer_embeddings), so the sweep costs no
more than a single-layer precompute of the same audio. Each layer is scored
standalone: raw accuracy and AS-Norm accuracy (cohort = the same 300
base_train utterances as exp4, embedded per layer).

Output: experiments/exp5_wavlm_layer_sweep.json + console table.

Usage:
    .venv/Scripts/python.exe scripts/exp5_wavlm_layer_sweep.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.splits import split_reserved_pool_halves  # noqa: E402
from src.models.wavlm_encoder import extract_all_layer_embeddings  # noqa: E402
from src.preprocessing.pipeline import preprocess_audio  # noqa: E402
from src.prototypical.score_norm import ASNorm  # noqa: E402

# import the shared cohort/task construction from the exp4 script so the
# sweep sees exactly the same utterances
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from exp4_complementarity_asnorm import (  # noqa: E402
    K_SHOT, N_QUERY, N_TASK_SPEAKERS, COHORT_SIZE, TOP_K, build_cohort_paths,
)

OUT_PATH = REPO_ROOT / "experiments" / "exp5_wavlm_layer_sweep.json"
SEED = 0


def all_layer_embed(path: Path, cache: dict) -> np.ndarray:
    """(n_layers+1, 768) all-layer embedding for one utterance (memoized).
    Long audio (> MAX_ECAPA_INFERENCE_SEC) is sliding-windowed by the
    preprocessing pipeline; window embeddings are averaged per layer."""
    key = str(path)
    if key not in cache:
        windows = preprocess_audio(path, mode="ecapa_inference")
        embs = np.stack([extract_all_layer_embeddings(w) for w in windows])  # (W, L, D)
        emb = embs.mean(axis=0)
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        cache[key] = emb.astype(np.float32)
    return cache[key]


def main() -> None:
    t0 = time.time()
    split = json.loads((REPO_ROOT / "data/splits/full_split.json").read_text(encoding="utf-8"))
    validation_speakers, _ = split_reserved_pool_halves(split["reserved_unknown_pool"])
    eval_manifest = pd.read_csv(REPO_ROOT / "data/raw/audio/eval_capped/manifest.csv")
    val_manifest = eval_manifest[eval_manifest["speaker_id"].isin(set(validation_speakers))]
    counts = val_manifest["speaker_id"].value_counts()
    usable = sorted(counts[counts >= K_SHOT + N_QUERY].index.tolist())
    task_speakers = usable[:N_TASK_SPEAKERS]

    rng = random.Random(SEED)
    support_paths, query_paths, labels = [], [], []
    for idx, spk in enumerate(task_speakers):
        paths = [REPO_ROOT / p for p in val_manifest.loc[val_manifest["speaker_id"] == spk, "path"]]
        rng.shuffle(paths)
        support_paths.append(paths[0])
        query_paths.extend(paths[1:1 + N_QUERY])
        labels.extend([idx] * len(paths[1:1 + N_QUERY]))
    labels = np.asarray(labels)
    cohort_paths = build_cohort_paths(COHORT_SIZE, "whisper_l4", seed=0)  # same cohort utts as exp4
    print(f"sweep: {len(task_speakers)} speakers, {len(query_paths)} queries, "
          f"{len(cohort_paths)} cohort utts, seed {SEED}")

    cache: dict = {}
    embed = lambda paths: np.stack([all_layer_embed(p, cache) for p in paths])  # noqa: E731
    t1 = time.time()
    proto_all = embed(support_paths)     # (n_spk, L, D)
    query_all = embed(query_paths)       # (n_q, L, D)
    cohort_all = embed(cohort_paths)     # (n_c, L, D)
    n_layers = proto_all.shape[1]
    print(f"embedded {len(cache)} utterances x {n_layers} layers "
          f"in {(time.time() - t1) / 60:.1f} min")

    results = []
    for layer in range(n_layers):
        p, q, c = proto_all[:, layer], query_all[:, layer], cohort_all[:, layer]
        d_raw = np.linalg.norm(q[:, None, :] - p[None, :, :], axis=-1)
        acc_raw = float((d_raw.argmin(axis=1) == labels).mean())
        z = ASNorm(c, top_k=TOP_K).normalize(q, p)
        acc_asnorm = float((z.argmin(axis=1) == labels).mean())
        results.append({"layer": layer, "layer_fraction": layer / (n_layers - 1),
                        "acc_raw": acc_raw, "acc_asnorm": acc_asnorm})
        print(f"  layer {layer:2d} (frac {layer / (n_layers - 1):.3f}): "
              f"raw {acc_raw:.4f}  asnorm {acc_asnorm:.4f}")

    best = max(results, key=lambda r: r["acc_asnorm"])
    print(f"\nBEST: layer {best['layer']} (fraction {best['layer_fraction']:.3f}) "
          f"asnorm acc {best['acc_asnorm']:.4f} (raw {best['acc_raw']:.4f})")
    print("referensi: whisper_l4 standalone asnorm (protokol sama, seed 0) = 0.2400; "
          "gerbang G5.1 butuh >= 0.70")

    OUT_PATH.write_text(json.dumps({
        "protocol": {"n_speakers": len(task_speakers), "k_shot": K_SHOT, "n_query": N_QUERY,
                     "seed": SEED, "cohort_size": COHORT_SIZE, "top_k": TOP_K,
                     "model": "microsoft/wavlm-base-plus",
                     "source": "validation half snapshot (identical to exp4 script, seed 0)"},
        "results": results,
        "best": best,
        "elapsed_minutes": (time.time() - t0) / 60,
    }, indent=2), encoding="utf-8")
    print(f"saved {OUT_PATH} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
