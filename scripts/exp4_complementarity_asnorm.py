#!/usr/bin/env python
"""Experiment 4 -- fusion decision gate: complementarity ceiling in AS-Norm
space (docs/experiment-4.md).

Every prior fusion conclusion ("Whisper contributes ~0") was reached in RAW
distance space; Experiment 3b then showed AS-Norm changes conclusions reached
in raw space (it flipped B1-vs-B2 and lifted ECAPA 0.732 -> 0.865). This
script re-measures, in the exp3b AS-normalized score space, on the exp3
validation task (anti-overfit half of reserved_unknown_pool -- the official
task speakers and the detection half are never touched):

  4a  the ORACLE complementarity ceiling of ECAPA + Whisper -- the upper
      bound ANY fusion mechanism could reach (gate G4.1: ceiling < +0.02
      over ECAPA-alone closes Whisper fusion for good);
  4b  the accuracy of late score fusion  w*z_ecapa + (1-w)*z_whisper  over
      per-space AS-normalized distances, sweeping w (gate G4.2);
  4c  training-free two-space combination rules for open-set DETECTION
      (EER/AUROC/TAR@1%FAR on leftover validation-half speakers as unknowns;
      gate G4.3).

Analysis-only: no ExperimentConfig, no production code path is touched, so
every registered experiment tag (baseline_v0 ... exp3c) behaves exactly as
before. If gate G4.2 passes, the production tag `exp4b_scorefusion_asnorm`
gets implemented behind its own feature flag -- not before.

Spaces: each backbone's NATIVE space (L2-normalized raw embedding, Euclidean
== monotone cosine -- the ScoreFusionEmbed convention), NOT the gated-fusion
projection, so the Whisper side is measured undamaged (exp1 showed a random
projection destroys a space; residual init only protects the ECAPA side).
Both Whisper variants are analyzed: "whisper" (L3, exp1/exp3's) and
"whisper_l4" (exp2's best).

Protocol per seed (3 seeds, matching the exp3 validation sweep):
  * validation task = first 100 usable validation-half speakers, 1-shot
    support + 4 queries per speaker (same counts as the exp3 sweep task);
  * prototypes are STATIC 1-shot supports (a final-session snapshot without
    continual updates -- the ceiling is a property of the identification
    geometry; continual dynamics are exercised only in official runs);
  * AS-Norm per space: cohort = 300 base_train utterances (round-robin,
    same sampling as exp3b), top_k = 200 (exp3b's locked value).

Usage:
    .venv/Scripts/python.exe scripts/exp4_complementarity_asnorm.py
    # Experiment 5 candidate screening (backbone must be registered in
    # src/features/cache.py and its embeddings ideally precomputed first):
    .venv/Scripts/python.exe scripts/exp4_complementarity_asnorm.py --second-backbone redimnet_b2
"""
from __future__ import annotations

import argparse
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
from src.evaluation.complementarity import (  # noqa: E402
    best_global_weight,
    detection_scores,
    fusion_accuracy,
    l2_normalize,
    linear_fusion_oracle,
    oracle_report,
)
from src.evaluation.metrics import auroc, tar_at_far  # noqa: E402
from src.evaluation.statistics import bootstrap_eer_difference  # noqa: E402
from src.features.cache import get_or_compute_embedding, is_cached  # noqa: E402
from src.prototypical.calibration import find_operating_point  # noqa: E402
from src.prototypical.score_norm import ASNorm  # noqa: E402
from src.utils.seed import SEED_LIST  # noqa: E402

SPLITS_PATH = REPO_ROOT / "data" / "splits" / "full_split.json"
OUT_PATH = REPO_ROOT / "experiments" / "exp4_ceiling_asnorm.json"

K_SHOT = 1
N_QUERY = 4                    # validation-half speakers have exactly 5 utterances
N_TASK_SPEAKERS = 100          # same size as the exp3 validation task (10 x 10-way)
SEEDS = SEED_LIST[:3]          # same seeds as the exp3 validation sweep
WHISPER_VARIANTS = ["whisper_l4", "whisper"]   # exp2's best first, exp1/exp3's for reference
COHORT_SIZE = 300              # exp3b locked AS-Norm params
TOP_K = 200
FUSION_W_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]   # ORIGINAL coarse grid -- kept
                               # verbatim so exp5's G5.1/G5.2 screening numbers
                               # (experiments/exp5_screening_redimnet_b2.json)
                               # stay reproducible from this script. The fine
                               # sweep below is reported ADDITIVELY.
GATE_G41_MIN_CEILING = 0.02    # docs/experiment-4.md section 3
DISAGREE_PENALTY = 0.5

# --- additions of the 2026-08-15 re-audit (docs/experiment-4-reaudit.md) ------
# C5: exp4's 6-point grid never looked below w=0.5 and missed the true optimum
# (w=0.883). The fine grid spans the full interval at 0.01 resolution.
FINE_W_GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.01), 2)
# C7: 3 seeds with no CI decided gates on differences smaller than the seed
# spread. Every detection rule now gets a paired bootstrap CI vs the ECAPA-only
# baseline.
N_BOOTSTRAP = 1000
# C6: the primary unknown set is only 40 utterances (1 query = 2.5% EER). The
# detection HALF of reserved_unknown_pool would be the obvious enlargement, but
# that is exactly the population the OFFICIAL runs score their unknowns on --
# selecting a detection rule there would be test-set leakage. Instead we add a
# second, leak-free panel: base_train speakers held OUT of the AS-Norm cohort.
HOLDOUT_COHORT_SIZE = 300
HOLDOUT_MAX_UNKNOWN = 600
GATE_G41B_MIN_CEILING = 0.02   # same threshold, applied to the *linear* ceiling


def load_base_train_manifest(whisper_backbone: str) -> pd.DataFrame:
    """Cached-in-both-backbones base_train utterances."""
    manifest = pd.concat([
        pd.read_csv(REPO_ROOT / "data/raw/audio/vox1_sample/manifest.csv"),
        pd.read_csv(REPO_ROOT / "data/raw/audio/base_train_capped/manifest.csv"),
    ], ignore_index=True)
    return manifest[manifest["path"].apply(
        lambda p: is_cached(REPO_ROOT / p, "ecapa") and is_cached(REPO_ROOT / p, whisper_backbone)
    )]


def build_cohort_paths(
    cohort_size: int,
    whisper_backbone: str,
    seed: int = 0,
    speakers_allowed: set[str] | None = None,
) -> list[Path]:
    """Round-robin-across-speakers cohort utterance selection from base_train
    (same sampling discipline as score_norm.build_cohort, but at the PATH
    level so both backbone spaces get the *same* cohort utterances).

    `speakers_allowed` restricts the draw to a speaker subset -- used by the
    leak-free holdout panel so the cohort and the unknown queries can never
    share a speaker. Left as None (the default) the behaviour is byte-identical
    to the original, which is what keeps exp5's screening numbers reproducible.
    """
    manifest = load_base_train_manifest(whisper_backbone)
    if speakers_allowed is not None:
        manifest = manifest[manifest["speaker_id"].isin(speakers_allowed)]
    rng = np.random.default_rng(seed)
    per_speaker: dict[str, list[str]] = {}
    for _, row in manifest.iterrows():
        per_speaker.setdefault(row["speaker_id"], []).append(row["path"])
    speakers = sorted(per_speaker)
    for s in speakers:
        rng.shuffle(per_speaker[s])

    picked: list[Path] = []
    round_idx = 0
    while len(picked) < cohort_size:
        added = False
        for s in speakers:
            if round_idx < len(per_speaker[s]):
                picked.append(REPO_ROOT / per_speaker[s][round_idx])
                added = True
                if len(picked) >= cohort_size:
                    break
        if not added:
            break
        round_idx += 1
    if len(picked) < 2:
        raise ValueError("not enough cached base_train utterances for a cohort")
    return picked


def base_train_speaker_halves(
    whisper_backbone: str, seed: int = 0
) -> tuple[set[str], set[str]]:
    """Speaker-disjoint halves of the cached base_train speakers: (cohort side,
    unknown-query side). Same discipline as
    score_norm.split_cohort_and_genuine -- an unknown query must never find its
    own speaker inside the cohort its score is normalized against.
    """
    manifest = load_base_train_manifest(whisper_backbone)
    speakers = sorted(set(manifest["speaker_id"]))
    rng = random.Random(seed + 11)   # distinct stream from splits.py / score_norm
    rng.shuffle(speakers)
    mid = len(speakers) // 2
    return set(speakers[:mid]), set(speakers[mid:])


def holdout_unknown_paths(
    whisper_backbone: str, speakers: set[str], limit: int, seed: int = 0
) -> list[Path]:
    manifest = load_base_train_manifest(whisper_backbone)
    manifest = manifest[manifest["speaker_id"].isin(speakers)]
    paths = sorted(manifest["path"].tolist())
    rng = random.Random(seed + 13)
    rng.shuffle(paths)
    return [REPO_ROOT / p for p in paths[:limit]]


def detection_panel(
    z_e: np.ndarray,
    z_w: np.ndarray,
    zu_e: np.ndarray,
    zu_w: np.ndarray,
    seed: int,
) -> dict:
    """EER/AUROC/TAR for every two-space rule, plus a PAIRED bootstrap CI on
    (rule EER - ECAPA-only EER) -- the same genuine/unknown trials scored by
    two rules, so Bengio & Mariethoz's paired resampling applies (C7)."""
    gen = detection_scores(z_e, z_w, DISAGREE_PENALTY)
    unk = detection_scores(zu_e, zu_w, DISAGREE_PENALTY)
    out = {}
    for rule in gen:
        g, u = gen[rule], unk[rule]
        entry = {
            "eer": find_operating_point(g, u, "eer").eer,
            "auroc": auroc(g, u),
            "tar_at_far1": tar_at_far(g, u, 0.01),
        }
        if rule != "a_only":
            ci = bootstrap_eer_difference(
                g, u, gen["a_only"], unk["a_only"],
                n_bootstrap=N_BOOTSTRAP, seed=seed,
            )
            entry["delta_eer_vs_a_only"] = ci.mean_diff      # negative = rule better
            entry["ci_lower"] = ci.ci_lower
            entry["ci_upper"] = ci.ci_upper
            entry["significant"] = bool(ci.significant)
        out[rule] = entry
    return out


def embed_paths(paths: list[Path], backbone: str) -> np.ndarray:
    """L2-normalized native-space embeddings for a list of utterance paths."""
    return l2_normalize(np.stack([get_or_compute_embedding(p, backbone) for p in paths]))


def summarize(per_seed: list[dict], key_paths: list[tuple[str, ...]]) -> dict:
    """mean/std across seeds for the given nested key paths. Output keys are
    "/"-joined (fusion weights like "0.5" contain dots, so "." can't be the
    path separator)."""
    out = {}
    for path in key_paths:
        vals = []
        for entry in per_seed:
            v = entry
            for part in path:
                v = v[part]
            vals.append(v)
        out["/".join(path)] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                               "per_seed": [float(v) for v in vals]}
    return out


def main(second_backbones: list[str] | None = None, out_path: Path = OUT_PATH) -> None:
    """`second_backbones` overrides WHISPER_VARIANTS -- the Experiment 5
    screening entry point (`--second-backbone <name>`): the same ceiling /
    fusion-sweep / detection analysis, but ECAPA paired with a candidate
    replacement backbone instead of Whisper."""
    variants = second_backbones or WHISPER_VARIANTS
    t0 = time.time()
    split = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))

    # ---- validation task (identical construction to exp3_validation_sweep) ----
    validation_speakers, detection_speakers = split_reserved_pool_halves(split["reserved_unknown_pool"])
    eval_manifest = pd.read_csv(REPO_ROOT / "data/raw/audio/eval_capped/manifest.csv")
    val_manifest = eval_manifest[eval_manifest["speaker_id"].isin(set(validation_speakers))]
    counts = val_manifest["speaker_id"].value_counts()
    usable = sorted(counts[counts >= K_SHOT + N_QUERY].index.tolist())
    task_speakers = usable[:N_TASK_SPEAKERS]
    leftover_speakers = sorted(set(val_manifest["speaker_id"]) - set(task_speakers))
    print(f"validation half: {len(validation_speakers)} speakers, usable {len(usable)}; "
          f"task {len(task_speakers)}, leftover (4c unknowns) {len(leftover_speakers)}; "
          f"detection half untouched: {len(detection_speakers)}")

    speaker_paths = {
        spk: [REPO_ROOT / p for p in val_manifest.loc[val_manifest["speaker_id"] == spk, "path"]]
        for spk in sorted(set(val_manifest["speaker_id"]))
    }
    unknown_paths = [p for spk in leftover_speakers for p in speaker_paths[spk]]
    print(f"4c unknown queries: {len(unknown_paths)} utterances "
          f"(leftover validation-half speakers only; detection half stays pristine)\n")

    results: dict = {"variants": {}}
    for whisper_backbone in variants:
        print(f"=== variant: ecapa + {whisper_backbone} ===")
        cohort_paths = build_cohort_paths(COHORT_SIZE, whisper_backbone, seed=0)
        cohort_e = embed_paths(cohort_paths, "ecapa")
        cohort_w = embed_paths(cohort_paths, whisper_backbone)
        norm_e = ASNorm(cohort_e, top_k=TOP_K)
        norm_w = ASNorm(cohort_w, top_k=TOP_K)

        # --- leak-free holdout panel (C6): cohort and unknown queries drawn
        # from DISJOINT base_train speaker halves, so this panel has its own
        # normalizer. The primary numbers above are untouched.
        coh_spk, unk_spk = base_train_speaker_halves(whisper_backbone, seed=0)
        ho_cohort_paths = build_cohort_paths(
            HOLDOUT_COHORT_SIZE, whisper_backbone, seed=0, speakers_allowed=coh_spk)
        ho_unknown_paths = holdout_unknown_paths(
            whisper_backbone, unk_spk, HOLDOUT_MAX_UNKNOWN, seed=0)
        ho_norm_e = ASNorm(embed_paths(ho_cohort_paths, "ecapa"), top_k=TOP_K)
        ho_norm_w = ASNorm(embed_paths(ho_cohort_paths, whisper_backbone), top_k=TOP_K)
        ho_u_e = embed_paths(ho_unknown_paths, "ecapa")
        ho_u_w = embed_paths(ho_unknown_paths, whisper_backbone)
        print(f"  holdout panel: cohort {len(ho_cohort_paths)} utt / {len(coh_spk)} spk, "
              f"unknowns {len(ho_unknown_paths)} utt / {len(unk_spk)} spk (disjoint)")

        per_seed: list[dict] = []
        rescue_examples: list[dict] = []
        for seed in SEEDS:
            rng = random.Random(seed)

            support_paths, query_paths, labels = [], [], []
            for idx, spk in enumerate(task_speakers):
                paths = list(speaker_paths[spk])
                rng.shuffle(paths)
                support_paths.append(paths[0])
                query_paths.extend(paths[1:1 + N_QUERY])
                labels.extend([idx] * len(paths[1:1 + N_QUERY]))
            labels = np.asarray(labels)

            proto_e = embed_paths(support_paths, "ecapa")
            proto_w = embed_paths(support_paths, whisper_backbone)
            q_e = embed_paths(query_paths, "ecapa")
            q_w = embed_paths(query_paths, whisper_backbone)

            # raw native-space distances (replicates the exp2-style ceiling here)
            d_e_raw = np.linalg.norm(q_e[:, None, :] - proto_e[None, :, :], axis=-1)
            d_w_raw = np.linalg.norm(q_w[:, None, :] - proto_w[None, :, :], axis=-1)
            # AS-normalized per-space distances (the exp3b score space)
            z_e = norm_e.normalize(q_e, proto_e)
            z_w = norm_w.normalize(q_w, proto_w)

            fine_asnorm = best_global_weight(z_e, z_w, labels, FINE_W_GRID)
            fine_raw = best_global_weight(d_e_raw, d_w_raw, labels, FINE_W_GRID)
            entry = {
                "seed": seed,
                "raw": oracle_report(d_e_raw, d_w_raw, labels),
                "asnorm": oracle_report(z_e, z_w, labels),
                "fusion_raw": {str(w): fusion_accuracy(d_e_raw, d_w_raw, labels, w)
                               for w in FUSION_W_GRID},
                "fusion_asnorm": {str(w): fusion_accuracy(z_e, z_w, labels, w)
                                  for w in FUSION_W_GRID},
                # --- re-audit additions (all NEW keys; nothing above changed) ---
                # C1: the ceiling of the score-fusion family actually being swept,
                # as opposed to oracle_report's per-query SELECTION ceiling.
                "linear_oracle_asnorm": linear_fusion_oracle(z_e, z_w, labels),
                "linear_oracle_raw": linear_fusion_oracle(d_e_raw, d_w_raw, labels),
                # C5: full-interval 0.01 sweep (the coarse grid above missed the optimum)
                "fine_asnorm": {"best_w": fine_asnorm["best_w"],
                                "best_acc": fine_asnorm["best_acc"]},
                "fine_raw": {"best_w": fine_raw["best_w"],
                             "best_acc": fine_raw["best_acc"]},
            }

            # 4c: detection on genuine (task queries) vs unknown (leftover speakers)
            u_e = embed_paths(unknown_paths, "ecapa")
            u_w = embed_paths(unknown_paths, whisper_backbone)
            zu_e = norm_e.normalize(u_e, proto_e)
            zu_w = norm_w.normalize(u_w, proto_w)
            # C7: same rules as before, now each with a paired bootstrap CI on
            # (rule EER - a_only EER). The "eer"/"auroc"/"tar_at_far1" keys keep
            # their original meaning, so exp5 screening comparisons still hold.
            entry["detection"] = detection_panel(z_e, z_w, zu_e, zu_w, seed)

            # C6: leak-free enlargement -- same task queries as genuine, but
            # ~600 base_train-holdout unknowns and a speaker-disjoint cohort.
            entry["detection_holdout"] = detection_panel(
                ho_norm_e.normalize(q_e, proto_e),
                ho_norm_w.normalize(q_w, proto_w),
                ho_norm_e.normalize(ho_u_e, proto_e),
                ho_norm_w.normalize(ho_u_w, proto_w),
                seed,
            )
            per_seed.append(entry)

            if seed == SEEDS[0]:   # qualitative sample: Whisper rescues ECAPA
                ok_e = z_e.argmin(axis=1) == labels
                ok_w = z_w.argmin(axis=1) == labels
                for qi in np.flatnonzero(ok_w & ~ok_e)[:15]:
                    rescue_examples.append({
                        "query": str(query_paths[qi].relative_to(REPO_ROOT)),
                        "true_speaker": task_speakers[labels[qi]],
                        "ecapa_pred": task_speakers[int(z_e[qi].argmin())],
                    })

            print(f"  seed {seed}: raw  acc_e={entry['raw']['acc_a']:.4f} "
                  f"acc_w={entry['raw']['acc_b']:.4f} oracle={entry['raw']['acc_oracle']:.4f} "
                  f"(+{entry['raw']['delta_ceiling']:.4f})")
            print(f"          asnorm acc_e={entry['asnorm']['acc_a']:.4f} "
                  f"acc_w={entry['asnorm']['acc_b']:.4f} oracle={entry['asnorm']['acc_oracle']:.4f} "
                  f"(+{entry['asnorm']['delta_ceiling']:.4f})")

        key_paths = ([("raw", m) for m in ["acc_a", "acc_b", "acc_oracle", "delta_ceiling"]]
                     + [("asnorm", m) for m in ["acc_a", "acc_b", "acc_oracle", "delta_ceiling"]]
                     + [("fusion_asnorm", str(w)) for w in FUSION_W_GRID]
                     + [("fusion_raw", str(w)) for w in FUSION_W_GRID]
                     + [("detection", r, m)
                        for r in ["a_only", "b_only", "mean", "max", "a_disagree"]
                        for m in ["eer", "auroc", "tar_at_far1"]]
                     # --- re-audit additions ---
                     + [("linear_oracle_asnorm", m)
                        for m in ["acc_linear_oracle", "delta_linear_ceiling",
                                  "recovered_both_argmin_wrong"]]
                     + [("linear_oracle_raw", m)
                        for m in ["acc_linear_oracle", "delta_linear_ceiling"]]
                     + [("fine_asnorm", m) for m in ["best_w", "best_acc"]]
                     + [("fine_raw", m) for m in ["best_w", "best_acc"]]
                     + [("detection_holdout", r, m)
                        for r in ["a_only", "b_only", "mean", "max", "a_disagree"]
                        for m in ["eer", "auroc", "tar_at_far1"]]
                     + [("detection_holdout", r, m)
                        for r in ["b_only", "mean", "max", "a_disagree"]
                        for m in ["delta_eer_vs_a_only", "ci_lower", "ci_upper"]]
                     + [("detection", r, m)
                        for r in ["b_only", "mean", "max", "a_disagree"]
                        for m in ["delta_eer_vs_a_only", "ci_lower", "ci_upper"]])
        summary = summarize(per_seed, key_paths)

        delta = summary["asnorm/delta_ceiling"]
        gate_g41_closed = delta["mean"] < GATE_G41_MIN_CEILING
        best_w, best_acc = max(
            ((w, summary[f"fusion_asnorm/{w}"]["mean"]) for w in map(str, FUSION_W_GRID)),
            key=lambda t: t[1],
        )
        lin_delta = summary["linear_oracle_asnorm/delta_linear_ceiling"]
        fine_acc = summary["fine_asnorm/best_acc"]
        fine_w = summary["fine_asnorm/best_w"]
        acc_a = summary["asnorm/acc_a"]
        variant_result = {
            "per_seed": per_seed,
            "summary": summary,
            "rescue_examples_seed0": rescue_examples,
            "holdout_panel": {
                "n_cohort_utterances": len(ho_cohort_paths),
                "n_cohort_speakers": len(coh_spk),
                "n_unknown_queries": len(ho_unknown_paths),
                "n_unknown_speakers": len(unk_spk),
            },
            "gates": {
                "G4.1_ceiling_asnorm": {
                    "delta_ceiling_mean": delta["mean"],
                    "threshold": GATE_G41_MIN_CEILING,
                    "fusion_closed": gate_g41_closed,
                    "note": "SELECTION ceiling -- not an upper bound on score "
                            "fusion. See G4.1b.",
                },
                # C1: the gate G4.1 should have been decided on
                "G4.1b_linear_ceiling_asnorm": {
                    "delta_linear_ceiling_mean": lin_delta["mean"],
                    "delta_linear_ceiling_std": lin_delta["std"],
                    "threshold": GATE_G41B_MIN_CEILING,
                    "fusion_closed": lin_delta["mean"] < GATE_G41B_MIN_CEILING,
                    "recovered_both_argmin_wrong":
                        summary["linear_oracle_asnorm/recovered_both_argmin_wrong"]["mean"],
                },
                "G4.2_best_fusion": {
                    "best_w": best_w,
                    "best_fusion_acc": best_acc,
                    "acc_ecapa_alone": acc_a["mean"],
                    "fusion_beats_ecapa": best_acc > acc_a["mean"],
                },
                # C5: same gate, decided on the full-interval 0.01 sweep
                "G4.2b_best_fusion_fine_grid": {
                    "best_w_mean": fine_w["mean"],
                    "best_w_per_seed": fine_w["per_seed"],
                    "best_fusion_acc": fine_acc["mean"],
                    "acc_ecapa_alone": acc_a["mean"],
                    "gain_over_ecapa": fine_acc["mean"] - acc_a["mean"],
                    # a gain smaller than the across-seed spread is not a result
                    "exceeds_seed_noise":
                        (fine_acc["mean"] - acc_a["mean"]) > acc_a["std"],
                },
            },
        }
        results["variants"][whisper_backbone] = variant_result

        print(f"\n  [{whisper_backbone}] ceiling raw   : +{summary['raw/delta_ceiling']['mean']:.4f} "
              f"± {summary['raw/delta_ceiling']['std']:.4f}")
        print(f"  [{whisper_backbone}] ceiling asnorm: +{delta['mean']:.4f} ± {delta['std']:.4f} "
              f"-> G4.1 {'CLOSED (fusion not viable)' if gate_g41_closed else 'OPEN (headroom exists)'}")
        print(f"  [{whisper_backbone}] ceiling asnorm LINEAR-FUSION: "
              f"+{lin_delta['mean']:.4f} ± {lin_delta['std']:.4f} -> G4.1b "
              f"{'CLOSED' if lin_delta['mean'] < GATE_G41B_MIN_CEILING else 'OPEN'} "
              f"(of which both-argmin-wrong: "
              f"{summary['linear_oracle_asnorm/recovered_both_argmin_wrong']['mean']:.4f})")
        print(f"  [{whisper_backbone}] best asnorm fusion (coarse grid): w={best_w} "
              f"acc={best_acc:.4f} vs ecapa-alone {acc_a['mean']:.4f}")
        print(f"  [{whisper_backbone}] best asnorm fusion (fine 0.01 grid) : "
              f"w={fine_w['mean']:.3f} acc={fine_acc['mean']:.4f} "
              f"gain={fine_acc['mean'] - acc_a['mean']:+.4f} "
              f"(seed std {acc_a['std']:.4f}) -> "
              f"{'EXCEEDS noise' if (fine_acc['mean'] - acc_a['mean']) > acc_a['std'] else 'WITHIN noise'}")
        for panel, tag in (("detection", "4c val-leftover"),
                           ("detection_holdout", "4c bt-holdout ")):
            det_base = summary[f"{panel}/a_only/eer"]["mean"]
            for rule in ["b_only", "mean", "max", "a_disagree"]:
                d = summary[f"{panel}/{rule}/delta_eer_vs_a_only"]["mean"]
                lo = summary[f"{panel}/{rule}/ci_lower"]["mean"]
                hi = summary[f"{panel}/{rule}/ci_upper"]["mean"]
                sig = "SIG" if not (lo <= 0.0 <= hi) else "ns "
                print(f"  [{whisper_backbone}] {tag} {rule:10s}: "
                      f"EER {summary[f'{panel}/{rule}/eer']['mean']:.4f} "
                      f"(a_only {det_base:.4f}) dEER {d:+.4f} "
                      f"CI[{lo:+.4f},{hi:+.4f}] {sig}")
        print()

    results["protocol"] = {
        "task": f"{len(task_speakers)} validation-half speakers, k_shot={K_SHOT}, "
                f"n_query={N_QUERY}, static 1-shot prototypes (snapshot)",
        "seeds": SEEDS,
        "asnorm": {"cohort_size": COHORT_SIZE, "top_k": TOP_K},
        "fusion_w_grid": FUSION_W_GRID,
        "fine_w_grid": {"start": 0.0, "stop": 1.0, "step": 0.01},
        "n_bootstrap": N_BOOTSTRAP,
        "n_unknown_queries_4c": len(unknown_paths),
        "unknown_source": "leftover validation-half speakers (detection half untouched)",
        "unknown_source_holdout": (
            "base_train speakers held OUT of the AS-Norm cohort (speaker-disjoint "
            "halves); the reserved_unknown_pool DETECTION half is deliberately NOT "
            "used -- official runs score their unknowns there, so selecting a "
            "detection rule on it would be test-set leakage"
        ),
        # holdout counts depend on which utterances are cached for the 2nd
        # backbone, so they are reported per-variant, not here
        "gate_g41_min_ceiling": GATE_G41_MIN_CEILING,
        "disagree_penalty": DISAGREE_PENALTY,
    }
    results["elapsed_minutes"] = (time.time() - t0) / 60

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {out_path} ({results['elapsed_minutes']:.1f} min)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--second-backbone", action="append", dest="second_backbones",
                        default=None, metavar="NAME",
                        help="candidate second backbone (repeatable; must be registered in "
                             "src/features/cache.py). Default: whisper_l4 + whisper (Experiment 4).")
    parser.add_argument("--out", type=Path, default=OUT_PATH,
                        help="output JSON path (default: experiments/exp4_ceiling_asnorm.json; "
                             "use a different file for Experiment 5 screening runs)")
    args = parser.parse_args()
    main(args.second_backbones, args.out)
