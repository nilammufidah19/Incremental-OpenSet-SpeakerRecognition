#!/usr/bin/env python
"""Experiment 6 -- FSCIL validation sweep locking the fusion weight w for the
ECAPA + Whisper-PMFA pairing, ahead of the official run.

This is the exp5_validation_sweep.py procedure applied to the second backbone
Experiment 6 actually cares about. It exists because the Whisper verdict in
Experiment 4 was decided on a gate analysis over a validation task and never
went through the official evaluation, so there is no ECAPA+Whisper row in the
format the thesis reports ECAPA+ReDimNet in. Producing that row starts here:
w must be chosen on the VALIDATION half, never on the official task.

Two deliberate differences from exp5_validation_sweep.py:

  * 10 seeds, not 3. F6-3b showed the 3-seed sweep is not powered enough to
    order the arms: at 3 seeds the fused arm sat ABOVE ReDimNet-alone, at 10
    seeds the order flipped. exp5's G5.3 was decided on that unstable
    ordering. See docs/experiment-6.md section 8.
  * The A2 arm (w = 0) is reported explicitly, so "fusion beats the second
    backbone alone" can be read off the sweep instead of only "fusion beats
    ECAPA alone" -- the exp5b trap.

The chosen w is then hard-coded into the exp6 tag in src/experiments.py
before the single official run.

Usage:
    .venv/Scripts/python.exe scripts/exp6_pmfa_validation_sweep.py
    .venv/Scripts/python.exe scripts/exp6_pmfa_validation_sweep.py --second-backbone whisper
"""
from __future__ import annotations

import argparse
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
from src.evaluation.statistics import paired_significance_test  # noqa: E402
from src.features.cache import is_cached  # noqa: E402
from src.models.fusion import ScoreFusionEmbed  # noqa: E402
from src.prototypical.calibration import build_genuine_impostor_distances, find_operating_point  # noqa: E402
from src.prototypical.data import ECAPA_DIM, build_raw_embedding_index, split_raw_embedding  # noqa: E402
from src.prototypical.score_norm import DualASNorm, build_cohort, split_cohort_and_genuine  # noqa: E402
from src.system import SpeakerIdentificationSystem  # noqa: E402
from src.utils.seed import SEED_LIST  # noqa: E402

SPLITS_PATH = REPO_ROOT / "data" / "splits" / "full_split.json"

K_SHOT, N_QUERY, N_WAY, N_SESSIONS = 1, 4, 10, 10
SEEDS = SEED_LIST                    # 10 -- see module docstring
COHORT_SIZE, TOP_K = 300, 200        # exp3b locked AS-Norm params
TARGET_FRR = 0.01                    # exp3b locked operating point
W_GRID = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1, 0.0]


def build_base_train_index(second: str) -> dict:
    frames = [
        pd.read_csv(REPO_ROOT / "data/raw/audio/vox1_sample/manifest.csv"),
        pd.read_csv(REPO_ROOT / "data/raw/audio/base_train_capped/manifest.csv"),
    ]
    manifest = pd.concat(frames, ignore_index=True)
    manifest = manifest[manifest["path"].apply(
        lambda p: is_cached(REPO_ROOT / p, "ecapa") and is_cached(REPO_ROOT / p, second)
    )]
    counts = manifest["speaker_id"].value_counts()
    manifest = manifest[manifest["speaker_id"].isin(counts[counts >= 2].index)]
    print(f"base_train index: {manifest['speaker_id'].nunique()} speakers, "
          f"{len(manifest)} utterances (cached for ecapa+{second})")
    return build_raw_embedding_index(manifest, whisper_backbone=second)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--second-backbone", default="whisper_pmfa")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    second = args.second_backbone
    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "experiments" / f"exp6_validation_sweep_{second}.json"
    )

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
    print(f"second backbone: {second}")
    print(f"validation task: {len(sessions)}x{N_WAY}-way k={K_SHOT} n_query={N_QUERY}; "
          f"detection half untouched: {len(detection_speakers)} speakers")

    base_train_index = build_base_train_index(second)
    fusion = ScoreFusionEmbed("fusion", weight=0.5).to(device)

    def fuse_fn(raw_list):
        raw = torch.from_numpy(np.stack(raw_list)).float().to(device)
        e, w = split_raw_embedding(raw)
        with torch.no_grad():
            return fusion(e, w).cpu().numpy()

    cohort_pool, calib_genuine_pool = split_cohort_and_genuine(base_train_index, seed=0)
    cohort = build_cohort(cohort_pool, fuse_fn, cohort_size=COHORT_SIZE, seed=0)
    print(f"cohort: {cohort.shape} (concat dim = {ECAPA_DIM} + {cohort.shape[1] - ECAPA_DIM})")

    impostor_ids = set(split["calibration_impostor_pool"])
    impostor_manifest = eval_manifest[eval_manifest["speaker_id"].isin(impostor_ids)]
    impostor_manifest = impostor_manifest[impostor_manifest["path"].apply(
        lambda p: is_cached(REPO_ROOT / p, "ecapa") and is_cached(REPO_ROOT / p, second)
    )]
    impostor_index = build_raw_embedding_index(impostor_manifest, whisper_backbone=second)
    print(f"impostor pool: {len(impostor_index)} speakers\n")

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
                whisper_backbone=second, score_normalizer=normalizer,
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
              f"val_acc={entry['val_acc_mean']:.4f}+/-{entry['val_acc_std']:.4f}")

    a1 = next(r for r in results if r["w"] == 1.0)   # ECAPA alone
    a2 = next(r for r in results if r["w"] == 0.0)   # second backbone alone
    fused = [r for r in results if r["w"] not in (0.0, 1.0)]
    best = max(fused, key=lambda r: r["val_acc_mean"])
    best_single = max(a1, a2, key=lambda r: r["val_acc_mean"])

    t_vs_single = paired_significance_test(
        np.asarray(best["val_accs"]), np.asarray(best_single["val_accs"])
    )
    wins = sum(b > s for b, s in zip(best["val_accs"], best_single["val_accs"]))
    gate = best["val_acc_mean"] > best_single["val_acc_mean"] and t_vs_single.p_value < 0.05

    print(f"\nA1 ECAPA-only (w=1.0)      : {a1['val_acc_mean']:.4f}")
    print(f"A2 {second}-only (w=0.0)   : {a2['val_acc_mean']:.4f}")
    print(f"best fused w={best['w']}          : {best['val_acc_mean']:.4f}")
    print(f"fused vs best single ({best_single['w']}): "
          f"delta={best['val_acc_mean'] - best_single['val_acc_mean']:+.4f} "
          f"{t_vs_single.test_used} p={t_vs_single.p_value:.4f} wins {wins}/{len(SEEDS)}")
    print(f"Fusion beats the best SINGLE space: {'YA' if gate else 'TIDAK'}")
    print(f"\n-> w LOCKED for the official run: {best['w']}")

    out_path.write_text(json.dumps({
        "protocol": {"second_backbone": second, "n_sessions": len(sessions),
                     "n_way": N_WAY, "k_shot": K_SHOT, "n_query": N_QUERY, "seeds": SEEDS,
                     "cohort_size": COHORT_SIZE, "top_k": TOP_K, "target_frr": TARGET_FRR},
        "results": results,
        "a1_ecapa_only": a1, "a2_second_only": a2,
        "best_fused": best, "best_single": best_single,
        "fused_vs_best_single": {
            "delta": best["val_acc_mean"] - best_single["val_acc_mean"],
            "test_used": t_vs_single.test_used, "p_value": t_vs_single.p_value,
            "seed_wins": int(wins),
        },
        "fusion_beats_best_single": bool(gate),
        "locked_w": best["w"],
        "elapsed_minutes": (time.time() - t0) / 60,
    }, indent=2), encoding="utf-8")
    print(f"saved {out_path.name} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
