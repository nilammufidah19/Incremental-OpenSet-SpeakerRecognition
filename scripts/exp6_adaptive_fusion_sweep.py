#!/usr/bin/env python
"""Experiment 6 phase F6-3b -- validation sweep for ZERO-PARAMETER adaptive
score fusion (docs/experiment-6-plan.md, gate G6.3b).

Why this runs BEFORE F6-1 (the expensive Whisper readout):

  exp5b's final numbers were A1 ECAPA-only 0.8332, A2 ReDimNet-only 0.8766,
  A3 fusion 0.8760 -- i.e. A2 >= A3 at p=0.88. The entire gain came from
  swapping the backbone; the fusion MECHANISM contributed nothing. exp4's
  re-audit then measured why: the best CONSTANT weight w buys +0.0025 over
  ECAPA-alone while an oracle picking w per query reaches +0.0333.

  So "does adaptive fusion help at all?" is a question about the fusion
  operator, and it is answerable RIGHT NOW on the ReDimNet cache -- the best
  backbone pair we have, where readout quality is not a confound. It needs
  no new embeddings (zero GPU recompute) and no ruling on F6-3's
  training-free question, because nothing here is fitted.

  If a BETTER operator still cannot beat max(A1, A2) even here, then the
  operator was not what was holding fusion back, and F6-1's premise -- fix
  Whisper's readout and fusion will win -- loses its main support before we
  spend two days on it.

  (Outcome, 10 seeds: exactly that. G6.3b-2 passed, so adaptivity IS worth
  something over a constant weight; G6.3b-1 failed, so the total still does
  not beat ReDimNet alone. See the results section in
  docs/experiment-6-plan.md.)

Protocol: identical to scripts/exp5_validation_sweep.py -- FSCIL task from
the VALIDATION half of reserved_unknown_pool (10 x 10-way, k=1, n_query=4,
3 seeds), threshold calibrated per arm on base_train genuine +
calibration_impostor_pool, AS-Norm c300/k200 over a speaker-disjoint cohort,
target-FRR 1%. The official task speakers and the detection half are never
touched. Only the score-fusion RULE varies between arms.

Two gates, and the second is the one that matters:

  G6.3b-1 (pre-registered in the plan): an adaptive arm beats max(A1, A2) by
    more than one across-seed std, on every seed.
  G6.3b-2 (decisive): an adaptive arm beats A3 with a FIXED w, paired test
    over seeds, p < 0.05.

G6.3b-1 alone proves nothing about adaptivity: on this validation task the
FIXED-w arm already beats max(A1, A2), so an adaptive rule that merely
matched it would still "pass". Only G6.3b-2 isolates what F6-3b claims --
that a per-query weight buys something a constant cannot. This mirrors the
exp5b trap the plan warns about (A3 > A1 passing while A2 >= A3).

Seeds: all 10 are run. SEEDS_PARITY (the first 3) reproduces
exp5_validation_sweep.py bit-for-bit as a harness check; the full 10 give
G6.3b-2 the power that 3 seeds cannot.

Usage:
    .venv/Scripts/python.exe scripts/exp6_adaptive_fusion_sweep.py
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
from src.evaluation.statistics import paired_significance_test  # noqa: E402
from src.features.cache import is_cached  # noqa: E402
from src.models.fusion import ScoreFusionEmbed  # noqa: E402
from src.prototypical.calibration import build_genuine_impostor_distances, find_operating_point  # noqa: E402
from src.prototypical.data import ECAPA_DIM, build_raw_embedding_index, split_raw_embedding  # noqa: E402
from src.prototypical.score_norm import (  # noqa: E402
    AdaptiveDualASNorm,
    DualASNorm,
    build_cohort,
    split_cohort_and_genuine,
)
from src.system import SpeakerIdentificationSystem  # noqa: E402
from src.utils.seed import SEED_LIST  # noqa: E402

SPLITS_PATH = REPO_ROOT / "data" / "splits" / "full_split.json"
OUT_PATH = REPO_ROOT / "experiments" / "exp6_adaptive_fusion_sweep.json"

# Every constant below is exp5b's LOCKED operating point, copied unchanged so
# the only difference between this sweep and exp5's is the fusion rule.
SECOND_BACKBONE = "redimnet_b2"
K_SHOT, N_QUERY, N_WAY, N_SESSIONS = 1, 4, 10, 10
SEEDS = SEED_LIST            # all 10 -- G6.3b-2 needs the power
SEEDS_PARITY = SEED_LIST[:3]  # the subset exp5_validation_sweep.py used
COHORT_SIZE, TOP_K = 300, 200
TARGET_FRR = 0.01
FIXED_W = 0.5  # exp5b's locked fusion weight

ARMS = [
    ("A1_ecapa_only", "fixed", {"weight": 1.0}),
    ("A2_redimnet_only", "fixed", {"weight": 0.0}),
    ("A3_fixed_w", "fixed", {"weight": FIXED_W}),
    ("A3_margin_weighted", "adaptive", {"weight": FIXED_W, "rule": "margin_weighted"}),
    ("A3_margin_select", "adaptive", {"weight": FIXED_W, "rule": "margin_select"}),
]


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


def weight_diagnostics(normalizer, queries: np.ndarray, protos: np.ndarray) -> dict:
    """A rule whose per-query weight is nearly constant is not adaptive, it is
    a badly-chosen fixed w. Report the distribution so the sweep cannot be
    read as "adaptive helped" when the weights never actually moved."""
    w = normalizer.per_query_weight(queries, protos)
    return {
        "mean": float(np.mean(w)), "std": float(np.std(w)),
        "p05": float(np.quantile(w, 0.05)), "p50": float(np.quantile(w, 0.5)),
        "p95": float(np.quantile(w, 0.95)),
        "frac_at_extremes": float(np.mean((w <= 1e-9) | (w >= 1 - 1e-9))),
    }


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
    fusion = ScoreFusionEmbed("fusion", weight=FIXED_W).to(device)

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
        lambda p: is_cached(REPO_ROOT / p, "ecapa") and is_cached(REPO_ROOT / p, SECOND_BACKBONE)
    )]
    impostor_index = build_raw_embedding_index(impostor_manifest, whisper_backbone=SECOND_BACKBONE)
    print(f"impostor pool: {len(impostor_index)} speakers\n")

    results = []
    for name, kind, kwargs in ARMS:
        if kind == "fixed":
            normalizer = DualASNorm(cohort, split_dim=ECAPA_DIM, top_k=TOP_K, **kwargs)
        else:
            normalizer = AdaptiveDualASNorm(cohort, split_dim=ECAPA_DIM, top_k=TOP_K, **kwargs)

        # Threshold is re-calibrated PER ARM: per-query weighting changes the
        # score distribution, so reusing exp5b's threshold would compare arms
        # at different operating points.
        calib_matrices: dict = {}
        genuine_d, impostor_d = build_genuine_impostor_distances(
            fusion, calib_genuine_pool, impostor_index, enrollment_k=1, seed=0,
            device=device, score_normalizer=normalizer, out_matrices=calib_matrices,
        )
        op = find_operating_point(genuine_d, impostor_d, "target_frr", target_frr=TARGET_FRR)
        calib_queries = calib_matrices["queries"]
        calib_protos = calib_matrices["prototypes"]

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

        parity = [a for seed, a in zip(SEEDS, accs) if seed in SEEDS_PARITY]
        entry = {
            "arm": name, "kind": kind, "params": kwargs,
            "threshold": op.threshold, "calibration_eer": op.eer,
            "val_acc_mean": float(np.mean(accs)), "val_acc_std": float(np.std(accs)),
            "val_accs": accs,
            "val_acc_mean_parity3": float(np.mean(parity)), "val_accs_parity3": parity,
        }
        if kind == "adaptive":
            entry["weight_distribution"] = weight_diagnostics(
                normalizer, calib_queries, calib_protos
            )
        results.append(entry)
        wd = entry.get("weight_distribution")
        print(f"  {name:20s} thr={op.threshold:8.4f} calEER={op.eer:.4f} "
              f"val_acc={entry['val_acc_mean']:.4f}+/-{entry['val_acc_std']:.4f} "
              f"(3-seed parity {entry['val_acc_mean_parity3']:.4f})"
              + (f"\n    {'':20s} w: mean={wd['mean']:.3f} std={wd['std']:.3f} "
                 f"p05-p95=[{wd['p05']:.2f},{wd['p95']:.2f}] "
                 f"at-extremes={wd['frac_at_extremes']:.2f}" if wd else ""))

    by_name = {r["arm"]: r for r in results}
    a1, a2 = by_name["A1_ecapa_only"], by_name["A2_redimnet_only"]
    best_single = max(a1, a2, key=lambda r: r["val_acc_mean"])
    adaptive_arms = [r for r in results if r["kind"] == "adaptive"]
    best_adaptive = max(adaptive_arms, key=lambda r: r["val_acc_mean"])

    fixed = by_name["A3_fixed_w"]

    margin = best_adaptive["val_acc_mean"] - best_single["val_acc_mean"]
    seed_wins = sum(b > s for b, s in zip(best_adaptive["val_accs"], best_single["val_accs"]))
    spread = max(best_adaptive["val_acc_std"], best_single["val_acc_std"])
    gate1 = margin > spread and seed_wins == len(SEEDS)

    # G6.3b-2: the decisive one -- adaptive vs a CONSTANT weight, paired over
    # seeds. Every adaptive rule is reported, not just the winner, so the
    # comparison cannot be read off the best-of-two after the fact.
    vs_fixed = {}
    for arm in adaptive_arms:
        test = paired_significance_test(
            np.asarray(arm["val_accs"]), np.asarray(fixed["val_accs"])
        )
        vs_fixed[arm["arm"]] = {
            "delta_vs_fixed": arm["val_acc_mean"] - fixed["val_acc_mean"],
            "test_used": test.test_used, "p_value": test.p_value,
            "seed_wins": int(sum(a > f for a, f in zip(arm["val_accs"], fixed["val_accs"]))),
        }
    best_vs_fixed = vs_fixed[best_adaptive["arm"]]
    gate2 = best_vs_fixed["delta_vs_fixed"] > 0 and best_vs_fixed["p_value"] < 0.05

    print(f"\nbest single-space arm : {best_single['arm']} {best_single['val_acc_mean']:.4f}")
    print(f"A3 fixed w={FIXED_W}       : {fixed['val_acc_mean']:.4f}")
    print(f"best adaptive arm     : {best_adaptive['arm']} {best_adaptive['val_acc_mean']:.4f}")
    print(f"\nG6.3b-1  vs max(A1,A2): margin={margin:+.4f} vs spread {spread:.4f}, "
          f"wins {seed_wins}/{len(SEEDS)} -> {'LOLOS' if gate1 else 'GAGAL'}")
    for arm_name, v in vs_fixed.items():
        print(f"G6.3b-2  {arm_name:20s} vs fixed w: delta={v['delta_vs_fixed']:+.4f} "
              f"{v['test_used']} p={v['p_value']:.4f} wins {v['seed_wins']}/{len(SEEDS)}")
    print(f"G6.3b-2 (adaptivity buys something a constant cannot): "
          f"{'LOLOS' if gate2 else 'GAGAL'}")
    if gate1 and not gate2:
        print("  NOTE: G6.3b-1 passes only because fixed-w fusion ALREADY beats "
              "max(A1,A2) on this task. That is not evidence for adaptivity.")

    OUT_PATH.write_text(json.dumps({
        "protocol": {
            "second_backbone": SECOND_BACKBONE, "n_sessions": len(sessions),
            "n_way": N_WAY, "k_shot": K_SHOT, "n_query": N_QUERY, "seeds": SEEDS,
            "cohort_size": COHORT_SIZE, "top_k": TOP_K, "target_frr": TARGET_FRR,
            "fixed_w_reference": FIXED_W,
            "note": "identical to exp5_validation_sweep.py except the fusion rule",
        },
        "results": results,
        "best_single_space": best_single, "best_adaptive": best_adaptive,
        "a3_fixed_w_reference": fixed,
        "margin_vs_best_single": margin, "across_seed_spread": spread,
        "seed_wins": seed_wins,
        "adaptive_vs_fixed": vs_fixed,
        "gate_g63b_1_vs_best_single_passed": bool(gate1),
        "gate_g63b_2_vs_fixed_w_passed": bool(gate2),
        "elapsed_minutes": (time.time() - t0) / 60,
    }, indent=2), encoding="utf-8")
    print(f"\nsaved {OUT_PATH.name} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
