#!/usr/bin/env python
"""Experiment 5b leakage-robustness check (post-hoc, 2026-08-02).

docs/experiment-5.md discloses that 80/100 official task_speakers are
VoxCeleb2-dev speakers -- data ReDimNet-b2 (ft_lm) was pretrained on -- but
the disclosure was never actually tested for impact. This script asks: does
"fusion (A3) > ECAPA-only (A1), significant" survive on ONLY the task
speakers that are NOT in VoxCeleb2-dev (so ReDimNet has never seen them)?

Method: identify task_speakers absent from vox2_meta.csv's "dev" split
(19 VoxCeleb1 + 1 VoxCeleb2-test speaker, all with cached audio in
eval_capped, 10 utterances each -- see data/raw/metadata/vox2_meta.csv).
Run a single static 19-way 1-shot closed-set identification task (K_SHOT=1,
N_QUERY=9, exhausting every cached utterance) over the same 10 seeds used
elsewhere (src.utils.seed.SEED_LIST[:10]), using the EXACT exp5b-locked
machinery: DualASNorm cohort=300/top_k=200 sampled from base_train (a
completely separate split, so this analysis carries no additional leakage
of its own), threshold calibrated at target_frr=0.01 (exp3b/exp5b's locked
operating point). Compares A1 (w=1.0, ECAPA-only), A2 (w=0.0, ReDimNet-
only), A3 (w=0.5, official exp5b weight) via closed-set identification
accuracy (threshold-independent -- argmin correctness only), paired
significance test A3 vs A1 across the 10 seeds.

Usage:
    .venv/Scripts/python.exe scripts/exp5_leakage_robustness.py
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
VOX2_META_PATH = REPO_ROOT / "data" / "raw" / "metadata" / "vox2_meta.csv"
OUT_PATH = REPO_ROOT / "experiments" / "exp5_leakage_robustness.json"

SECOND_BACKBONE = "redimnet_b2"
K_SHOT = 1
N_QUERY = 9          # clean task speakers have exactly 10 cached utterances each
SEEDS = SEED_LIST[:10]
COHORT_SIZE, TOP_K = 300, 200   # exp3b/exp5b locked AS-Norm params
TARGET_FRR = 0.01               # exp3b/exp5b re-locked operating point
# exp6 (2026-08-30): w=0.3 arm added -- the 10-seed re-derived weight now used
# by the best official configuration (exp6_redimnet_fusion_w30). The original
# w=0.5 arm stays untouched so the exp5b-era numbers keep reproducing.
WEIGHTS = {
    "A1_ecapa_only": 1.0,
    "A2_redimnet_only": 0.0,
    "A3_fusion": 0.5,
    "A3_fusion_w30": 0.3,
}


def clean_task_speakers() -> list[str]:
    """task_speakers NOT in VoxCeleb2-dev (ReDimNet's pretraining split)."""
    split = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    meta = pd.read_csv(VOX2_META_PATH)
    meta.columns = [c.strip() for c in meta.columns]
    meta["VoxCeleb2 ID"] = meta["VoxCeleb2 ID"].str.strip()
    meta["Set"] = meta["Set"].str.strip()
    vox2_dev = set(meta.loc[meta["Set"] == "dev", "VoxCeleb2 ID"])
    return [s for s in split["task_speakers"] if s not in vox2_dev]


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
    return build_raw_embedding_index(manifest, whisper_backbone=SECOND_BACKBONE)


def main() -> None:
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))

    candidates = clean_task_speakers()
    eval_manifest = pd.read_csv(REPO_ROOT / "data/raw/audio/eval_capped/manifest.csv")
    task_manifest = eval_manifest[eval_manifest["speaker_id"].isin(candidates)]
    task_manifest = task_manifest[task_manifest["path"].apply(
        lambda p: is_cached(REPO_ROOT / p, "ecapa") and is_cached(REPO_ROOT / p, SECOND_BACKBONE)
    )]
    counts = task_manifest["speaker_id"].value_counts()
    task_speakers = sorted(counts[counts >= K_SHOT + N_QUERY].index.tolist())
    speaker_audio_paths = {
        spk: [REPO_ROOT / p for p in task_manifest.loc[task_manifest["speaker_id"] == spk, "path"]]
        for spk in task_speakers
    }
    print(f"clean (non-vox2-dev) task_speakers: {len(candidates)} candidates, "
          f"{len(task_speakers)} usable (>= {K_SHOT + N_QUERY} cached utterances)")
    sessions = [task_speakers]  # single static session -- no incremental/continual component

    base_train_index = build_base_train_index()
    fusion = ScoreFusionEmbed("fusion", weight=0.5).to(device)

    def fuse_fn(raw_list):
        raw = torch.from_numpy(np.stack(raw_list)).float().to(device)
        e, r = split_raw_embedding(raw)
        with torch.no_grad():
            return fusion(e, r).cpu().numpy()

    cohort_pool, calib_genuine_pool = split_cohort_and_genuine(base_train_index, seed=0)
    cohort = build_cohort(cohort_pool, fuse_fn, cohort_size=COHORT_SIZE, seed=0)
    print(f"cohort: {cohort.shape} (from base_train -- disjoint split from task_speakers)")

    impostor_ids = set(split["calibration_impostor_pool"])
    impostor_manifest = eval_manifest[eval_manifest["speaker_id"].isin(impostor_ids)]
    impostor_manifest = impostor_manifest[impostor_manifest["path"].apply(
        lambda p: is_cached(REPO_ROOT / p, "ecapa") and is_cached(REPO_ROOT / p, SECOND_BACKBONE)
    )]
    impostor_index = build_raw_embedding_index(impostor_manifest, whisper_backbone=SECOND_BACKBONE)
    print(f"impostor pool: {len(impostor_index)} speakers\n")

    results: dict[str, dict] = {}
    for arm, w in WEIGHTS.items():
        normalizer = DualASNorm(cohort, split_dim=ECAPA_DIM, top_k=TOP_K, weight=w)
        genuine_d, impostor_d = build_genuine_impostor_distances(
            fusion, calib_genuine_pool, impostor_index, enrollment_k=1, seed=0,
            device=device, score_normalizer=normalizer,
        )
        op = find_operating_point(genuine_d, impostor_d, "target_frr", target_frr=TARGET_FRR)

        open_accs, closed_accs = [], []
        for seed in SEEDS:
            system = SpeakerIdentificationSystem(
                fusion, op.threshold, continual_mode="running_average",
                whisper_backbone=SECOND_BACKBONE, score_normalizer=normalizer,
            )
            detailed = run_fscil_detailed(
                system, sessions, speaker_audio_paths,
                k_shot=K_SHOT, n_query=N_QUERY, seed=seed,
            )
            open_accs.append(average_accuracy(detailed.open_set, 0))
            closed_accs.append(average_accuracy(detailed.closed_set, 0))

        results[arm] = {
            "weight": w, "threshold": op.threshold, "calibration_eer": op.eer,
            "open_acc_mean": float(np.mean(open_accs)), "open_acc_std": float(np.std(open_accs)),
            "closed_acc_mean": float(np.mean(closed_accs)), "closed_acc_std": float(np.std(closed_accs)),
            "closed_accs": closed_accs, "open_accs": open_accs,
        }
        print(f"{arm:18s} w={w:4.2f} closed_acc={np.mean(closed_accs):.4f}+/-{np.std(closed_accs):.4f} "
              f"open_acc={np.mean(open_accs):.4f}+/-{np.std(open_accs):.4f}")

    a3 = np.array(results["A3_fusion"]["closed_accs"])
    a1 = np.array(results["A1_ecapa_only"]["closed_accs"])
    a2 = np.array(results["A2_redimnet_only"]["closed_accs"])
    a3w30 = np.array(results["A3_fusion_w30"]["closed_accs"])
    sig_a3_vs_a1 = paired_significance_test(a3, a1)
    sig_a3_vs_a2 = paired_significance_test(a3, a2)
    sig_w30_vs_a1 = paired_significance_test(a3w30, a1)
    sig_w30_vs_a2 = paired_significance_test(a3w30, a2)
    print(f"A3(w=0.3) vs A1: {sig_w30_vs_a1.test_used}, p={sig_w30_vs_a1.p_value:.4f}, "
          f"significant={sig_w30_vs_a1.p_value < 0.05}")
    print(f"A3(w=0.3) vs A2: {sig_w30_vs_a2.test_used}, p={sig_w30_vs_a2.p_value:.4f}, "
          f"significant={sig_w30_vs_a2.p_value < 0.05}")
    print(f"\nA3 vs A1 (leakage-free subset, n={len(task_speakers)} speakers, "
          f"{len(SEEDS)} seeds): {sig_a3_vs_a1.test_used}, p={sig_a3_vs_a1.p_value:.4f}, "
          f"significant={sig_a3_vs_a1.p_value < 0.05}")
    print(f"A3 vs A2: {sig_a3_vs_a2.test_used}, p={sig_a3_vs_a2.p_value:.4f}, "
          f"significant={sig_a3_vs_a2.p_value < 0.05}")

    OUT_PATH.write_text(json.dumps({
        "protocol": {
            "second_backbone": SECOND_BACKBONE, "n_task_speakers": len(task_speakers),
            "task_speakers": task_speakers, "k_shot": K_SHOT, "n_query": N_QUERY,
            "seeds": SEEDS, "cohort_size": COHORT_SIZE, "top_k": TOP_K,
            "target_frr": TARGET_FRR,
            "note": "single static 19-way session, no continual/incremental component -- "
                    "this checks identification robustness to leakage, not continual learning",
        },
        "results": results,
        "significance": {
            "A3_vs_A1": {"test": sig_a3_vs_a1.test_used, "p_value": sig_a3_vs_a1.p_value,
                         "significant_at_0.05": sig_a3_vs_a1.p_value < 0.05},
            "A3_vs_A2": {"test": sig_a3_vs_a2.test_used, "p_value": sig_a3_vs_a2.p_value,
                         "significant_at_0.05": sig_a3_vs_a2.p_value < 0.05},
            "A3_w30_vs_A1": {"test": sig_w30_vs_a1.test_used, "p_value": sig_w30_vs_a1.p_value,
                             "significant_at_0.05": sig_w30_vs_a1.p_value < 0.05},
            "A3_w30_vs_A2": {"test": sig_w30_vs_a2.test_used, "p_value": sig_w30_vs_a2.p_value,
                             "significant_at_0.05": sig_w30_vs_a2.p_value < 0.05},
        },
        "elapsed_minutes": (time.time() - t0) / 60,
    }, indent=2), encoding="utf-8")
    print(f"\nsaved {OUT_PATH} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
