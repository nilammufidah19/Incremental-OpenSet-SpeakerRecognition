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
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.splits import split_reserved_pool_halves  # noqa: E402
from src.evaluation.complementarity import (  # noqa: E402
    detection_scores,
    fusion_accuracy,
    l2_normalize,
    oracle_report,
)
from src.evaluation.metrics import auroc, tar_at_far  # noqa: E402
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
FUSION_W_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
GATE_G41_MIN_CEILING = 0.02    # docs/experiment-4.md section 3
DISAGREE_PENALTY = 0.5


def build_cohort_paths(cohort_size: int, whisper_backbone: str, seed: int = 0) -> list[Path]:
    """Round-robin-across-speakers cohort utterance selection from base_train
    (same sampling discipline as score_norm.build_cohort, but at the PATH
    level so both backbone spaces get the *same* cohort utterances)."""
    frames = [
        pd.read_csv(REPO_ROOT / "data/raw/audio/vox1_sample/manifest.csv"),
        pd.read_csv(REPO_ROOT / "data/raw/audio/base_train_capped/manifest.csv"),
    ]
    manifest = pd.concat(frames, ignore_index=True)
    manifest = manifest[manifest["path"].apply(
        lambda p: is_cached(REPO_ROOT / p, "ecapa") and is_cached(REPO_ROOT / p, whisper_backbone)
    )]
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

        per_seed: list[dict] = []
        rescue_examples: list[dict] = []
        for seed in SEEDS:
            import random as _random
            rng = _random.Random(seed)

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

            entry = {
                "seed": seed,
                "raw": oracle_report(d_e_raw, d_w_raw, labels),
                "asnorm": oracle_report(z_e, z_w, labels),
                "fusion_raw": {str(w): fusion_accuracy(d_e_raw, d_w_raw, labels, w)
                               for w in FUSION_W_GRID},
                "fusion_asnorm": {str(w): fusion_accuracy(z_e, z_w, labels, w)
                                  for w in FUSION_W_GRID},
            }

            # 4c: detection on genuine (task queries) vs unknown (leftover speakers)
            u_e = embed_paths(unknown_paths, "ecapa")
            u_w = embed_paths(unknown_paths, whisper_backbone)
            zu_e = norm_e.normalize(u_e, proto_e)
            zu_w = norm_w.normalize(u_w, proto_w)
            gen_scores = detection_scores(z_e, z_w, DISAGREE_PENALTY)
            unk_scores = detection_scores(zu_e, zu_w, DISAGREE_PENALTY)
            det = {}
            for rule in gen_scores:
                g, u = gen_scores[rule], unk_scores[rule]
                det[rule] = {
                    "eer": find_operating_point(g, u, "eer").eer,
                    "auroc": auroc(g, u),
                    "tar_at_far1": tar_at_far(g, u, 0.01),
                }
            entry["detection"] = det
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
                        for m in ["eer", "auroc", "tar_at_far1"]])
        summary = summarize(per_seed, key_paths)

        delta = summary["asnorm/delta_ceiling"]
        gate_g41_closed = delta["mean"] < GATE_G41_MIN_CEILING
        best_w, best_acc = max(
            ((w, summary[f"fusion_asnorm/{w}"]["mean"]) for w in map(str, FUSION_W_GRID)),
            key=lambda t: t[1],
        )
        variant_result = {
            "per_seed": per_seed,
            "summary": summary,
            "rescue_examples_seed0": rescue_examples,
            "gates": {
                "G4.1_ceiling_asnorm": {
                    "delta_ceiling_mean": delta["mean"],
                    "threshold": GATE_G41_MIN_CEILING,
                    "fusion_closed": gate_g41_closed,
                },
                "G4.2_best_fusion": {
                    "best_w": best_w,
                    "best_fusion_acc": best_acc,
                    "acc_ecapa_alone": summary["asnorm/acc_a"]["mean"],
                    "fusion_beats_ecapa": best_acc > summary["asnorm/acc_a"]["mean"],
                },
            },
        }
        results["variants"][whisper_backbone] = variant_result

        print(f"\n  [{whisper_backbone}] ceiling raw   : +{summary['raw/delta_ceiling']['mean']:.4f} "
              f"± {summary['raw/delta_ceiling']['std']:.4f}")
        print(f"  [{whisper_backbone}] ceiling asnorm: +{delta['mean']:.4f} ± {delta['std']:.4f} "
              f"-> G4.1 {'CLOSED (fusion not viable)' if gate_g41_closed else 'OPEN (headroom exists)'}")
        print(f"  [{whisper_backbone}] best asnorm fusion: w={best_w} acc={best_acc:.4f} "
              f"vs ecapa-alone {summary['asnorm/acc_a']['mean']:.4f}")
        det_base = summary["detection/a_only/eer"]["mean"]
        for rule in ["b_only", "mean", "max", "a_disagree"]:
            print(f"  [{whisper_backbone}] 4c det {rule:10s}: "
                  f"EER {summary[f'detection/{rule}/eer']['mean']:.4f} "
                  f"(ecapa-only {det_base:.4f}), "
                  f"AUROC {summary[f'detection/{rule}/auroc']['mean']:.4f}")
        print()

    results["protocol"] = {
        "task": f"{len(task_speakers)} validation-half speakers, k_shot={K_SHOT}, "
                f"n_query={N_QUERY}, static 1-shot prototypes (snapshot)",
        "seeds": SEEDS,
        "asnorm": {"cohort_size": COHORT_SIZE, "top_k": TOP_K},
        "fusion_w_grid": FUSION_W_GRID,
        "n_unknown_queries_4c": len(unknown_paths),
        "unknown_source": "leftover validation-half speakers (detection half untouched)",
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
