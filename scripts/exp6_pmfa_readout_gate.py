#!/usr/bin/env python
"""Experiment 6 phase F6-1 -- gate G6.1 for the training-free Whisper-PMFA
readout (docs/experiment-6-plan.md).

Question: does a better READOUT of the same frozen Whisper encoder rescue the
Whisper space? Nothing is trained -- multi-layer aggregation and mean+std
pooling are arithmetic, and every post-processing transform is FIT ON
base_train only (speaker-disjoint from the task), the same tier-1 status as
the AS-Norm cohort already in the system.

Gate G6.1: Whisper-alone accuracy >= 0.45 on the exp4 validation task, and
Fisher trace ratio > 1.0 (currently 0.51 -- within-speaker variation exceeds
between-speaker variation, i.e. not a speaker space at all).

Baselines to beat, from the F6-0 re-audit on this exact task/protocol:
    whisper (L3, 512-d)          raw 0.3125, best transform (ABTT k=1) 0.3475
    whisper_l4                   raw 0.2842
    Fisher ratio (whisper L3)    0.51

Comparability: task, fit set, cohort, seeds and transforms are all imported
from scripts/exp4_reaudit_geometry.py rather than re-implemented, so the
numbers land in the same frame as the baselines above. That script is not
modified -- exp4's re-audit stays reproducible exactly as committed.

Two PMFA-specific things this adds:

  1. A per-layer L2 pre-transform. Transformer layers differ several-fold in
     activation scale (measured on real audio: ||.|| from 8.1 to 29.7 across
     the four blocks), so a plain concatenation is dominated by the largest
     block. Equalizing per layer is a candidate fix and costs nothing --
     which is why the cache stores RAW statistics.
  2. A slice ablation. The 4096-d vector is laid out
     [mean_L3, std_L3, mean_L4, std_L4, ...], so mean-only, std-only and
     single-layer variants are free to evaluate. This separates "multi-layer
     helped" from "std pooling helped" instead of reporting one number for
     both, which is what the plan's hypothesis actually needs.

Usage:
    .venv/Scripts/python.exe scripts/exp6_pmfa_readout_gate.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import exp4_reaudit_geometry as geo  # noqa: E402
from src.evaluation.complementarity import l2_normalize  # noqa: E402
from src.models.whisper_encoder import PMFA_ALL_LAYERS, PMFA_LAYERS  # noqa: E402
from src.prototypical.score_norm import ASNorm  # noqa: E402

# Which layers each cached namespace actually holds, in stored order.
BACKBONE_LAYERS = {
    "whisper_pmfa": PMFA_LAYERS,
    "whisper_pmfa_all": PMFA_ALL_LAYERS,
}
LAYERS = PMFA_LAYERS      # rebound in main() from --backbone

D_MODEL = 512                      # whisper-base
BLOCK = 2 * D_MODEL                # [mean; std] per layer
GATE_ACC = 0.45
GATE_FISHER = 1.0


def per_layer_l2(x: np.ndarray) -> np.ndarray:
    """L2-normalize each [mean_L; std_L] block independently, so every layer
    contributes equally regardless of its native activation scale."""
    x = np.asarray(x, dtype=np.float64)
    n_blocks = x.shape[1] // BLOCK
    parts = [l2_normalize(x[:, i * BLOCK:(i + 1) * BLOCK]) for i in range(n_blocks)]
    return np.concatenate(parts, axis=1)


def slice_variant(x: np.ndarray, name: str) -> np.ndarray:
    """Sub-readouts recoverable from the cached raw statistics."""
    n_blocks = x.shape[1] // BLOCK
    if name == "full":
        return x
    if name == "mean_only":
        return np.concatenate([x[:, i * BLOCK:i * BLOCK + D_MODEL] for i in range(n_blocks)], axis=1)
    if name == "std_only":
        return np.concatenate(
            [x[:, i * BLOCK + D_MODEL:(i + 1) * BLOCK] for i in range(n_blocks)], axis=1
        )
    if name.startswith("layer"):  # "layer5" -> one block; "layer34" -> blocks 3 and 4
        wanted = [int(c) for c in name[len("layer"):]]
        idx = [LAYERS.index(l) for l in wanted]
        return np.concatenate([x[:, i * BLOCK:(i + 1) * BLOCK] for i in idx], axis=1)
    raise ValueError(f"unknown slice variant {name!r}")


def build_slices(layers: tuple[int, ...]) -> list[str]:
    """Single layers, then progressively wider windows anchored at the
    shallowest cached block. The windows exist to separate "aggregation helps"
    from "one good layer carries it"."""
    singles = [f"layer{l}" for l in layers]
    windows = ["".join(str(l) for l in layers[:k]) for k in range(2, len(layers) + 1)]
    return ["full", "mean_only", "std_only"] + singles + [f"layer{w}" for w in windows]


SLICES = build_slices(PMFA_LAYERS)


def make_per_layer_whitener(fit_x: np.ndarray, d: int):
    """PCA-whiten each [mean_L; std_L] block INDEPENDENTLY, to `d` dims, then
    concatenate.

    Motivation (measured, not assumed): the cached base_train fit set holds
    1190 utterances, so whitening the 4096-d concatenation estimates a
    covariance of rank <= 1189 in 4096 dimensions -- an ill-posed problem, and
    a direct explanation for the effective rank collapsing to 7.0 there. The
    existing per_layer_l2 equalizes each block's SCALE but not its SHAPE, so
    it cannot fix that.

    Whitening blockwise instead estimates four 1024-d covariances from the
    same 1190 samples and keeps only the leading `d` directions of each, which
    is well-posed for d well under 1190. Everything is still fit on
    base_train only.
    """
    fit_x = np.asarray(fit_x, dtype=np.float64)
    n_blocks = fit_x.shape[1] // BLOCK
    fitted = []
    for i in range(n_blocks):
        block = l2_normalize(fit_x[:, i * BLOCK:(i + 1) * BLOCK])
        mu = block.mean(axis=0)
        xc = block - mu
        cov = (xc.T @ xc) / (len(xc) - 1)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        k = min(d, block.shape[1], len(xc) - 1)
        eps = 1e-3 * evals[0]
        fitted.append((mu, evecs[:, :k], 1.0 / np.sqrt(evals[:k] + eps)))

    def apply(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        parts = []
        for i, (mu, V, s) in enumerate(fitted):
            block = l2_normalize(x[:, i * BLOCK:(i + 1) * BLOCK])
            parts.append(((block - mu) @ V) * s)
        return np.concatenate(parts, axis=1)

    return apply


def evaluate(fit_x, fit_spk, coh_raw, task_emb, key_q, key_p, tag, results):
    """Fit every transform on `fit_x` and score Whisper-alone accuracy."""
    transforms = geo.build_transforms(fit_x, fit_spk)
    geom = geo.geometry(fit_x, fit_spk)
    best = None
    for t in transforms:
        norm = ASNorm(t(coh_raw), top_k=geo.TOP_K)
        accs = []
        for te in task_emb:
            z = norm.normalize(t(te[key_q]), t(te[key_p]))
            accs.append(float((z.argmin(axis=1) == te["lab"]).mean()))
        acc = float(np.mean(accs))
        results.append({"variant": tag, "transform": t.name, "acc": acc,
                        "dim": int(fit_x.shape[1])})
        if best is None or acc > best[1]:
            best = (t.name, acc)
    print(f"  {tag:34s} dim={fit_x.shape[1]:5d}  best={best[0]:16s} acc={best[1]:.4f}  "
          f"fisher={geom['fisher_trace_ratio']:.3f}  eff_rank={geom['eff_rank_participation']:.1f}")
    return {"variant": tag, "best_transform": best[0], "best_acc": best[1], "geometry": geom}


def main() -> None:
    global LAYERS, SLICES, OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="whisper_pmfa", choices=sorted(BACKBONE_LAYERS))
    args = ap.parse_args()
    backbone = args.backbone
    LAYERS = BACKBONE_LAYERS[backbone]
    SLICES = build_slices(LAYERS)
    OUT_PATH = REPO_ROOT / "experiments" / f"exp6_readout_gate_{backbone}.json"
    print(f"backbone: {backbone}  layers={LAYERS}")

    print("loading manifests / fit set / cohort ...", flush=True)
    bt = geo.load_base_train()
    fit_paths, fit_spk = geo.round_robin(bt, geo.FIT_SIZE, seed=1)
    cohort_paths, _ = geo.round_robin(bt, geo.COHORT_SIZE, seed=0)
    print(f"fit set: {len(fit_paths)} utts / {len(set(fit_spk))} speakers; "
          f"cohort {len(cohort_paths)}")

    episodes = geo.build_task()
    summaries, results = [], []

    # ---- reference points: the single-layer readouts, same protocol -------
    print("\n=== A. REFERENCE single-layer readouts ===")
    # NOTE: loop variable is ref_backbone, NOT backbone -- shadowing the
    # --backbone argument here silently made section B load whisper_l4.
    for ref_backbone in ("whisper", "whisper_l4"):
        fit_x = geo.embed(fit_paths, ref_backbone)
        coh = geo.embed(cohort_paths, ref_backbone)
        task_emb = [{"q": geo.embed(q, ref_backbone), "p": geo.embed(s, ref_backbone), "lab": lab}
                    for s, q, lab in episodes]
        summaries.append(evaluate(fit_x, fit_spk, coh, task_emb, "q", "p",
                                  f"{ref_backbone}", results))

    # ---- PMFA: 2 pre-transforms x 7 slices --------------------------------
    print("\n=== B. PMFA readout (4 layers x mean+std, raw cache) ===")
    fit_raw = geo.embed(fit_paths, backbone)
    coh_raw = geo.embed(cohort_paths, backbone)
    task_raw = [{"q": geo.embed(q, backbone), "p": geo.embed(s, backbone), "lab": lab}
                for s, q, lab in episodes]

    # Variants = (pre-transform, slice). per-layer L2 only makes sense on the
    # full concatenation -- once a single block is sliced out there is nothing
    # left to equalize against, so those combinations are skipped rather than
    # silently duplicating the global-L2 rows.
    variants = (
        [("global_l2", sl) for sl in SLICES]
        + [("per_layer_l2", "full")]
        + [(f"per_layer_whiten_d{d}", "full") for d in (64, 128, 192)]
    )

    whiteners = {d: make_per_layer_whitener(fit_raw, d) for d in (64, 128, 192)}

    def prepare(x, pre_name, sl):
        sliced = slice_variant(x, sl)
        if pre_name == "per_layer_l2":
            return per_layer_l2(sliced)
        if pre_name.startswith("per_layer_whiten_d"):
            return whiteners[int(pre_name[len("per_layer_whiten_d"):])](sliced)
        return sliced

    for pre_name, sl in variants:
        f = prepare(fit_raw, pre_name, sl)
        c = prepare(coh_raw, pre_name, sl)
        te = [{"q": prepare(t["q"], pre_name, sl), "p": prepare(t["p"], pre_name, sl),
               "lab": t["lab"]} for t in task_raw]
        summaries.append(evaluate(f, fit_spk, c, te, "q", "p",
                                  f"pmfa/{pre_name}/{sl}", results))

    # ---- gate --------------------------------------------------------------
    pmfa = [s for s in summaries if s["variant"].startswith("pmfa/")]
    best = max(pmfa, key=lambda s: s["best_acc"])
    ref = next(s for s in summaries if s["variant"] == "whisper")
    fisher = best["geometry"]["fisher_trace_ratio"]
    gate = best["best_acc"] >= GATE_ACC and fisher > GATE_FISHER

    print(f"\nbest PMFA variant : {best['variant']} + {best['best_transform']} "
          f"-> acc={best['best_acc']:.4f}, fisher={fisher:.3f}")
    print(f"reference whisper : {ref['best_transform']} -> acc={ref['best_acc']:.4f}, "
          f"fisher={ref['geometry']['fisher_trace_ratio']:.3f}")
    print(f"delta vs single-layer reference: {best['best_acc'] - ref['best_acc']:+.4f}")
    print(f"\nGate G6.1 (acc >= {GATE_ACC} AND fisher > {GATE_FISHER}): "
          f"{'LOLOS' if gate else 'GAGAL'}")
    if not gate:
        why = []
        if best["best_acc"] < GATE_ACC:
            why.append(f"acc {best['best_acc']:.4f} < {GATE_ACC}")
        if fisher <= GATE_FISHER:
            why.append(f"fisher {fisher:.3f} <= {GATE_FISHER}")
        print("  reason: " + "; ".join(why))

    OUT_PATH.write_text(json.dumps({
        "protocol": {
            "task": "exp4 validation task (100 speakers, k=1, n_query=4, seeds [0,1,2])",
            "fit_size": geo.FIT_SIZE, "cohort_size": geo.COHORT_SIZE, "top_k": geo.TOP_K,
            "backbone": backbone, "pmfa_layers": list(LAYERS), "d_model": D_MODEL,
            "note": "task/fit/cohort/transforms imported from exp4_reaudit_geometry.py",
        },
        "gate": {"acc_threshold": GATE_ACC, "fisher_threshold": GATE_FISHER,
                 "passed": bool(gate), "best": best, "single_layer_reference": ref},
        "summaries": summaries,
        "all_transform_results": results,
    }, indent=2), encoding="utf-8")
    print(f"saved {OUT_PATH.name}")


if __name__ == "__main__":
    main()
