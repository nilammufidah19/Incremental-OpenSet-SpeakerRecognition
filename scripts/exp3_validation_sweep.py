#!/usr/bin/env python
"""Experiment 3 validation sweep (docs/experiment-3.md section 7, anti-overfit).

Selects and LOCKS the Experiment 3 threshold hyperparameters --
`calibration_strategy` / `target_frr` (exp3a) and AS-Norm params (exp3b) --
on a VALIDATION task that is speaker-disjoint from the real task:

  * validation task speakers = the VALIDATION half of reserved_unknown_pool
    (src/data/splits.py::split_reserved_pool_halves); the other half is kept
    untouched as unknown-query material for exp3c's detection metrics;
  * threshold calibration itself uses the same base_train (genuine) +
    calibration_impostor_pool (impostor) data as the official harness;
  * the system under test is exp1's frozen residual-init fusion (the
    Experiment 3 base system) -- nothing is trained.

Protocol note: reserved-pool speakers have exactly 5 cached utterances each,
so the validation FSCIL task uses n_query=4 (vs the official task's 5);
sessions are 10 x 10-way, k_shot=1, identical otherwise.

The selected values are then hard-coded into the exp3a/exp3b/exp3c tags in
src/experiments.py (documented in docs/experiment-3.md), and the official
task run happens exactly once per tag via scripts/run_full_evaluation.py.

Usage:
    .venv/Scripts/python.exe scripts/exp3_validation_sweep.py
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
from src.evaluation.ablation import train_fusion_for_ablation  # noqa: E402
from src.evaluation.fscil import run_fscil_detailed  # noqa: E402
from src.evaluation.metrics import average_accuracy  # noqa: E402
from src.features.cache import is_cached  # noqa: E402
from src.prototypical.calibration import build_genuine_impostor_distances, find_operating_point  # noqa: E402
from src.prototypical.data import build_raw_embedding_index, split_raw_embedding  # noqa: E402
from src.prototypical.score_norm import ASNorm, build_cohort, split_cohort_and_genuine  # noqa: E402
from src.system import SpeakerIdentificationSystem  # noqa: E402
from src.utils.seed import SEED_LIST  # noqa: E402

SPLITS_PATH = REPO_ROOT / "data" / "splits" / "full_split.json"
OUT_PATH = REPO_ROOT / "experiments" / "exp3_validation_sweep.json"

K_SHOT = 1
N_QUERY = 4          # reserved-pool speakers have exactly 5 utterances (1 support + 4 query)
N_WAY = 10
N_SESSIONS = 10
SEEDS = SEED_LIST[:3]

TARGET_FRRS = [0.01, 0.05, 0.10, 0.15]
ASNORM_VARIANTS = [                # (cohort_size, top_k); None = no normalization
    None,
    (300, 100),
    (300, 50),
    (300, 200),
]


def build_base_train_index() -> dict:
    frames = [
        pd.read_csv(REPO_ROOT / "data/raw/audio/vox1_sample/manifest.csv"),
        pd.read_csv(REPO_ROOT / "data/raw/audio/base_train_capped/manifest.csv"),
    ]
    manifest = pd.concat(frames, ignore_index=True)
    manifest["_cached"] = manifest["path"].apply(
        lambda p: is_cached(REPO_ROOT / p, "ecapa") and is_cached(REPO_ROOT / p, "whisper")
    )
    manifest = manifest[manifest["_cached"]]
    counts = manifest["speaker_id"].value_counts()
    eligible = counts[counts >= K_SHOT + N_QUERY + 1].index.tolist()
    manifest = manifest[manifest["speaker_id"].isin(eligible)]
    print(f"base_train index: {manifest['speaker_id'].nunique()} speakers, {len(manifest)} utterances")
    return build_raw_embedding_index(manifest)


def main() -> None:
    t0 = time.time()
    split = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    # ---- validation task from the VALIDATION half of reserved_unknown_pool ----
    validation_speakers, detection_speakers = split_reserved_pool_halves(split["reserved_unknown_pool"])
    eval_manifest = pd.read_csv(REPO_ROOT / "data/raw/audio/eval_capped/manifest.csv")
    val_manifest = eval_manifest[eval_manifest["speaker_id"].isin(set(validation_speakers))]
    counts = val_manifest["speaker_id"].value_counts()
    usable = sorted(counts[counts >= K_SHOT + N_QUERY].index.tolist())
    print(f"validation half: {len(validation_speakers)} speakers, usable (>= {K_SHOT + N_QUERY} utts "
          f"with audio): {len(usable)}; detection half kept untouched: {len(detection_speakers)}")

    n_task = min(N_SESSIONS * N_WAY, len(usable) - len(usable) % N_WAY)
    task_speakers = usable[:n_task]
    sessions = [task_speakers[i:i + N_WAY] for i in range(0, n_task, N_WAY)]
    speaker_audio_paths = {
        spk: [REPO_ROOT / p for p in val_manifest.loc[val_manifest["speaker_id"] == spk, "path"]]
        for spk in task_speakers
    }
    print(f"validation task: {len(sessions)} sessions x {N_WAY}-way, k_shot={K_SHOT}, n_query={N_QUERY}\n")

    # ---- frozen exp1 base system (residual init, 0 episodes) ----
    base_train_index = build_base_train_index()
    fusion = train_fusion_for_ablation(
        "fusion", base_train_index, N_WAY, K_SHOT, N_QUERY, n_episodes=0,
        seed=0, device=device, residual_init=True,
    )

    impostor_ids = set(split["calibration_impostor_pool"])
    impostor_manifest = eval_manifest[eval_manifest["speaker_id"].isin(impostor_ids)]
    impostor_index = build_raw_embedding_index(impostor_manifest)

    def fuse_fn(raw_list):
        raw = torch.from_numpy(np.stack(raw_list)).float().to(device)
        e, w = split_raw_embedding(raw)
        with torch.no_grad():
            return fusion(e, w).cpu().numpy()

    # Speaker-disjoint halves of base_train_index (score_norm.py::
    # split_cohort_and_genuine): AS-Norm cohort and calibration genuine
    # trials must not draw from the same speakers, or a genuine trial's own
    # speaker can land inside its own cohort and bias the swept EER/operating
    # points this script exists to select. Non-AS-Norm ("none") keeps using
    # the full pool, unaffected.
    cohort_pool, calib_genuine_pool = split_cohort_and_genuine(base_train_index, seed=0)

    results = []
    for variant in ASNORM_VARIANTS:
        if variant is None:
            normalizer, norm_name = None, "none"
        else:
            cohort_size, top_k = variant
            cohort = build_cohort(cohort_pool, fuse_fn, cohort_size=cohort_size, seed=0)
            normalizer = ASNorm(cohort, top_k=top_k)
            norm_name = f"asnorm_c{cohort_size}_k{top_k}"

        genuine_index_for_calibration = calib_genuine_pool if normalizer is not None else base_train_index
        # genuine/impostor calibration distances depend only on the normalizer,
        # so build them once per variant and derive every operating point from them
        genuine_d, impostor_d = build_genuine_impostor_distances(
            fusion, genuine_index_for_calibration, impostor_index, enrollment_k=1, seed=0,
            device=device, score_normalizer=normalizer,
        )
        eer_here = find_operating_point(genuine_d, impostor_d, "eer").eer
        print(f"[{norm_name}] calibration EER = {eer_here:.4f}")

        candidates = [("eer", None)] + [("target_frr", f) for f in TARGET_FRRS]
        for strategy, frr in candidates:
            op = find_operating_point(genuine_d, impostor_d, strategy, target_frr=frr or 0.05)
            accs = []
            for seed in SEEDS:
                system = SpeakerIdentificationSystem(
                    fusion, op.threshold, continual_mode="running_average",
                    score_normalizer=normalizer,
                )
                detailed = run_fscil_detailed(
                    system, sessions, speaker_audio_paths,
                    k_shot=K_SHOT, n_query=N_QUERY, seed=seed,
                )
                accs.append(average_accuracy(detailed.open_set, max(detailed.open_set)))
            entry = {
                "score_norm": norm_name,
                "strategy": strategy,
                "target_frr": frr,
                "threshold": op.threshold,
                "calibration_eer": eer_here,
                "far_at_threshold": op.far_at_threshold,
                "frr_at_threshold": op.frr_at_threshold,
                "val_open_set_acc_mean": float(np.mean(accs)),
                "val_open_set_acc_std": float(np.std(accs)),
                "val_accs": accs,
            }
            results.append(entry)
            label = strategy if frr is None else f"frr={frr}"
            print(f"  {norm_name:18s} {label:10s} thr={op.threshold:.4f} "
                  f"val_acc={entry['val_open_set_acc_mean']:.4f}±{entry['val_open_set_acc_std']:.4f}")

    best = max(results, key=lambda r: r["val_open_set_acc_mean"])
    print(f"\nBEST on validation: score_norm={best['score_norm']} strategy={best['strategy']} "
          f"target_frr={best['target_frr']} -> val acc {best['val_open_set_acc_mean']:.4f}")

    OUT_PATH.write_text(json.dumps({
        "protocol": {
            "n_sessions": len(sessions), "n_way": N_WAY, "k_shot": K_SHOT, "n_query": N_QUERY,
            "seeds": SEEDS, "n_validation_speakers": len(task_speakers),
            "validation_source": "reserved_unknown_pool validation half (speaker-disjoint from task)",
        },
        "results": results,
        "best": best,
        "elapsed_minutes": (time.time() - t0) / 60,
    }, indent=2), encoding="utf-8")
    print(f"saved {OUT_PATH} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
