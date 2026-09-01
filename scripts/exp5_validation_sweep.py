#!/usr/bin/env python
"""Experiment 5 Fase 3 -- FSCIL validation sweep locking the fusion weight w
(docs/experiment-5.md, gate G5.3).

System under test: the exp5b dual-space architecture -- parameter-free
concat embedder [e_ecapa_hat ; e_redimnet_hat] + DualASNorm (per-space
AS-Norm c300/k200, z-score fusion weight w on the ECAPA side) + target-FRR
5% threshold (exp3b's locked operating point) + running-average continual
updates. Exactly the configuration scripts/run_full_evaluation.py runs
under fusion_strategy="score_norm"; only w varies here.

Protocol: identical to scripts/exp3_validation_sweep.py -- FSCIL task built
from the VALIDATION half of reserved_unknown_pool (10 x 10-way, k=1,
n_query=4, 3 seeds); threshold calibrated on base_train genuine +
calibration_impostor_pool; the official task speakers and the detection
half are never touched. The chosen w is then HARD-CODED into the
exp5b tag in src/experiments.py before the single official run.

Gate G5.3: best-w val acc must beat w=1.0 (ECAPA-only in the same dual
machinery) consistently across seeds.

Usage:
    .venv/Scripts/python.exe scripts/exp5_validation_sweep.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.splits import split_reserved_pool_halves  # noqa: E402
from src.evaluation.fscil import run_fscil_detailed  # noqa: E402
from src.evaluation.metrics import average_accuracy  # noqa: E402
from src.features.cache import is_cached  # noqa: E402
from src.models.fusion import ScoreFusionEmbed  # noqa: E402
from src.prototypical.calibration import build_genuine_impostor_distances, find_operating_point  # noqa: E402
from src.prototypical.data import ECAPA_DIM, build_raw_embedding_index, split_raw_embedding  # noqa: E402
from src.prototypical.score_norm import DualASNorm, build_cohort, split_cohort_and_genuine  # noqa: E402
from src.system import SpeakerIdentificationSystem  # noqa: E402
from src.utils.seed import SEED_LIST  # noqa: E402

SPLITS_PATH = REPO_ROOT / "data" / "splits" / "full_split.json"
OUT_PATH = REPO_ROOT / "experiments" / "exp5_validation_sweep.json"

SECOND_BACKBONE = "redimnet_b2"
K_SHOT, N_QUERY, N_WAY, N_SESSIONS = 1, 4, 10, 10
SEEDS = SEED_LIST[:3]
COHORT_SIZE, TOP_K = 300, 200        # exp3b locked AS-Norm params
TARGET_FRR = 0.01                    # exp3b locked operating point (re-locked 2026-07-26)
W_GRID = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.0]


def build_base_train_index() -> dict:
    frames = [
        pd.read_csv(REPO_ROOT / "data/raw/audio/vox1_sample/manifest.csv"),
        pd.read_csv(REPO_ROOT / "data/raw/audio/base_train_capped/manifest.csv"),
    ]
    manifest = pd.concat(frames, ignore_index=True)
    manifest = manifest[manifest["path"].apply(
        lambda p: is_cached(REPO_ROOT / p, "ecapa") and is_cached(REPO_ROOT / p, SECOND_BACKBONE)
    )]
    counts = manifest["speaker_id"].value_counts()
    manifest = manifest[manifest["speaker_id"].isin(counts[counts >= 2].index)]
    print(f"base_train index: {manifest['speaker_id'].nunique()} speakers, "
          f"{len(manifest)} utterances (cached for ecapa+{SECOND_BACKBONE})")
    return build_raw_embedding_index(manifest, whisper_backbone=SECOND_BACKBONE)


def main() -> None:
    t0 = time.time()
    split = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    validation_speakers, detection_speakers = split_reserved_pool_halves(split["reserved_unknown_pool"])
    eval_manifest = pd.read_csv(REPO_ROOT / "data/raw/audio/eval_capped/manifest.csv")
    val_manifest = eval_manifest[eval_manifest["speaker_id"].isin(set(validation_speakers))]
    counts = val_manifest["speaker_id"].value_counts()
    usable = sorted(counts[counts >= K_SHOT + N_QUERY].index.tolist())
    n_task = min(N_SESSIONS * N_WAY, len(usable) - len(usable) % N_WAY)
    task_speakers = usable[:n_task]
    sessions = [task_speakers[i:i + N_WAY] for i in range(0, n_task, N_WAY)]
    speaker_audio_paths = {
        spk: [REPO_ROOT / p for p in val_manifest.loc[val_manifest["speaker_id"] == spk, "path"]]
        for spk in task_speakers
    }
    print(f"validation task: {len(sessions)}x{N_WAY}-way k={K_SHOT} n_query={N_QUERY}; "
          f"detection half untouched: {len(detection_speakers)} speakers")

    base_train_index = build_base_train_index()
    fusion = ScoreFusionEmbed("fusion", weight=0.5).to(device)

    def fuse_fn(raw_list):
        raw = torch.from_numpy(np.stack(raw_list)).float().to(device)
        e, w = split_raw_embedding(raw)
        with torch.no_grad():
            return fusion(e, w).cpu().numpy()

    # Speaker-disjoint halves (score_norm.py::split_cohort_and_genuine): every
    # w in this sweep uses AS-Norm, so the cohort and the calibration genuine
    # trials must never share speakers (see run_full_evaluation.py for the
    # full rationale).
    cohort_pool, calib_genuine_pool = split_cohort_and_genuine(base_train_index, seed=0)
    cohort = build_cohort(cohort_pool, fuse_fn, cohort_size=COHORT_SIZE, seed=0)
    print(f"cohort: {cohort.shape} (concat dim = 192 + {cohort.shape[1] - 192})")

    impostor_ids = set(split["calibration_impostor_pool"])
    impostor_manifest = eval_manifest[eval_manifest["speaker_id"].isin(impostor_ids)]
    impostor_manifest = impostor_manifest[impostor_manifest["path"].apply(
        lambda p: is_cached(REPO_ROOT / p, "ecapa") and is_cached(REPO_ROOT / p, SECOND_BACKBONE)
    )]
    impostor_index = build_raw_embedding_index(impostor_manifest, whisper_backbone=SECOND_BACKBONE)
    print(f"impostor pool: {len(impostor_index)} speakers")

    results = []
    for w in W_GRID:
        normalizer = DualASNorm(cohort, split_dim=ECAPA_DIM, top_k=TOP_K, weight=w)
        genuine_d, impostor_d = build_genuine_impostor_distances(
            fusion, calib_genuine_pool, impostor_index, enrollment_k=1, seed=0,
            device=device, score_normalizer=normalizer,
        )
        op = find_operating_point(genuine_d, impostor_d, "target_frr", target_frr=TARGET_FRR)
        accs = []
        for seed in SEEDS:
            system = SpeakerIdentificationSystem(
                fusion, op.threshold, continual_mode="running_average",
                whisper_backbone=SECOND_BACKBONE, score_normalizer=normalizer,
            )
            detailed = run_fscil_detailed(
                system, sessions, speaker_audio_paths,
                k_shot=K_SHOT, n_query=N_QUERY, seed=seed,
            )
            accs.append(average_accuracy(detailed.open_set, max(detailed.open_set)))
        entry = {"w": w, "threshold": op.threshold, "calibration_eer": op.eer,
                 "val_acc_mean": float(np.mean(accs)), "val_acc_std": float(np.std(accs)),
                 "val_accs": accs}
        results.append(entry)
        print(f"  w={w:4.2f} thr={op.threshold:8.4f} calEER={op.eer:.4f} "
              f"val_acc={entry['val_acc_mean']:.4f}±{entry['val_acc_std']:.4f} {accs}")

    ref = next(r for r in results if r["w"] == 1.0)
    best = max(results, key=lambda r: r["val_acc_mean"])
    per_seed_wins = sum(b > a for b, a in zip(best["val_accs"], ref["val_accs"]))
    gate = best["w"] != 1.0 and best["val_acc_mean"] > ref["val_acc_mean"] and per_seed_wins == len(SEEDS)
    print(f"\nBEST: w={best['w']} val_acc={best['val_acc_mean']:.4f} "
          f"vs ECAPA-only(w=1.0) {ref['val_acc_mean']:.4f} "
          f"(menang di {per_seed_wins}/{len(SEEDS)} seed)")
    print(f"Gate G5.3 (fusi > ECAPA-only konsisten): {'LOLOS' if gate else 'GAGAL'}")

    OUT_PATH.write_text(json.dumps({
        "protocol": {"second_backbone": SECOND_BACKBONE, "n_sessions": len(sessions),
                     "n_way": N_WAY, "k_shot": K_SHOT, "n_query": N_QUERY, "seeds": SEEDS,
                     "cohort_size": COHORT_SIZE, "top_k": TOP_K, "target_frr": TARGET_FRR},
        "results": results, "best": best, "ecapa_only_ref": ref,
        "gate_g53_passed": bool(gate),
        "elapsed_minutes": (time.time() - t0) / 60,
    }, indent=2), encoding="utf-8")
    print(f"saved {OUT_PATH} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
