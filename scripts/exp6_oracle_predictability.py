#!/usr/bin/env python
"""Experiment 6 -- can a ZERO-PARAMETER rule harvest the fusion headroom, or
is a fitted backend (F6-3, LLR) the only way?

The ceiling analysis established that linear fusion of ECAPA + ReDimNet has
roughly +0.037 of headroom over ReDimNet alone, and that a fixed global weight
harvests under half of it. F6-3b's per-query margin rule recovered only
+0.0050. Two very different explanations fit that:

  (a) the headroom needs information a per-query rule cannot see without
      fitting -- in which case F6-3 (logistic-regression fusion) is the only
      route, and the supervisor's ruling becomes the critical path; or
  (b) margin is simply a weak proxy, and some other parameter-free signal
      predicts which space to trust -- in which case a better tier-2 rule
      exists and needs no ruling from anyone.

This script decides between them WITHOUT building any new rule. For every
query where exactly one space's argmin is correct -- the only queries where a
per-query weight can change the outcome -- it asks how well each candidate
signal separates "trust ECAPA" from "trust the second space", measured as
AUROC.

Reading the result:
  AUROC ~ 0.50  the signal is uninformative; no tier-2 rule built on it can
                work, and (a) holds.
  AUROC ~ 0.65+ the signal carries real information the current rules waste,
                so (b) holds and a better zero-parameter rule is worth building.

Signals are all computable from the two z-score matrices already in hand, so
none of them requires anything fitted beyond the AS-Norm cohort the system
already uses.

Usage:
    .venv/Scripts/python.exe scripts/exp6_oracle_predictability.py
    .venv/Scripts/python.exe scripts/exp6_oracle_predictability.py --second-backbone whisper_best
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
from src.prototypical.score_norm import ASNorm  # noqa: E402


def softmax_entropy(z: np.ndarray) -> np.ndarray:
    """Entropy of softmax(-z) per row. Lower = the space is more decided about
    this query. Unlike the top1-vs-top2 margin this uses the whole score
    distribution, so a query with one clear winner and a long flat tail scores
    differently from one with two close contenders."""
    p = np.exp(-(z - z.min(axis=1, keepdims=True)))
    p /= p.sum(axis=1, keepdims=True)
    return -(p * np.log(p + 1e-12)).sum(axis=1)


def margin(z: np.ndarray) -> np.ndarray:
    part = np.partition(z, 1, axis=1)
    return part[:, 1] - part[:, 0]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """P(score of a positive > score of a negative), ties counted as half."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    vals = np.concatenate([pos, neg])
    for v in np.unique(vals):
        m = vals == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--second-backbone", default="redimnet_b2")
    args = ap.parse_args()
    second = args.second_backbone

    bt = geo.load_base_train()
    cohort_paths, _ = geo.round_robin(bt, geo.COHORT_SIZE, seed=0)
    coh_e = l2_normalize(geo.embed(cohort_paths, "ecapa"))
    coh_b = l2_normalize(geo.embed(cohort_paths, second))
    norm_e, norm_b = ASNorm(coh_e, top_k=geo.TOP_K), ASNorm(coh_b, top_k=geo.TOP_K)

    episodes = geo.build_task()
    print(f"second backbone: {second}; {len(episodes)} episodes")

    sig_names = ["margin_ratio_f63b", "margin_diff", "entropy_diff", "best_z_diff"]
    pooled = {k: [] for k in sig_names}
    pooled_lab = []
    counts = {"total": 0, "both_right": 0, "both_wrong": 0, "decidable": 0}

    for sup, qry, lab in episodes:
        z_e = norm_e.normalize(l2_normalize(geo.embed(qry, "ecapa")),
                               l2_normalize(geo.embed(sup, "ecapa")))
        z_b = norm_b.normalize(l2_normalize(geo.embed(qry, second)),
                               l2_normalize(geo.embed(sup, second)))
        ok_e = z_e.argmin(axis=1) == lab
        ok_b = z_b.argmin(axis=1) == lab

        counts["total"] += len(lab)
        counts["both_right"] += int((ok_e & ok_b).sum())
        counts["both_wrong"] += int((~ok_e & ~ok_b).sum())

        # Only queries where the two spaces disagree can be rescued or ruined
        # by a per-query weight; everywhere else the weight is irrelevant.
        dec = ok_e ^ ok_b
        counts["decidable"] += int(dec.sum())
        if dec.sum() == 0:
            continue

        m_e, m_b = margin(z_e)[dec], margin(z_b)[dec]
        h_e, h_b = softmax_entropy(z_e)[dec], softmax_entropy(z_b)[dec]
        best_e, best_b = z_e.min(axis=1)[dec], z_b.min(axis=1)[dec]

        total = m_e + m_b
        pooled["margin_ratio_f63b"].append(np.where(total > 1e-12, m_e / np.maximum(total, 1e-12), 0.5))
        pooled["margin_diff"].append(m_e - m_b)
        pooled["entropy_diff"].append(h_b - h_e)      # higher = ECAPA more decided
        pooled["best_z_diff"].append(best_b - best_e)  # higher = ECAPA more confident
        pooled_lab.append(ok_e[dec].astype(int))       # 1 = ECAPA is the right space

    labels = np.concatenate(pooled_lab)
    print(f"\nquery: {counts['total']} total | {counts['both_right']} keduanya benar | "
          f"{counts['both_wrong']} keduanya salah | {counts['decidable']} DECIDABLE")
    print(f"  di antara yang decidable: ECAPA benar {int(labels.sum())}, "
          f"{second} benar {int((1 - labels).sum())}")

    print(f"\n{'sinyal (nol-parameter)':26s} {'AUROC':>7s}  tafsir")
    results = {}
    for k in sig_names:
        a = auroc(np.concatenate(pooled[k]), labels)
        results[k] = a
        verdict = ("tidak informatif" if a < 0.55 else
                   "lemah" if a < 0.60 else
                   "sedang" if a < 0.65 else "kuat")
        print(f"{k:26s} {a:7.4f}  {verdict}")

    best = max(results, key=results.get)
    print(f"\nsinyal terbaik: {best} (AUROC {results[best]:.4f})")
    if results[best] < 0.60:
        print("  -> aturan tier-2 (nol parameter) TIDAK punya sinyal yang cukup.")
        print("     Jalur yang tersisa untuk memanen plafon adalah F6-3 (LLR).")
    else:
        print("  -> ada sinyal yang belum dimanfaatkan; aturan tier-2 yang lebih")
        print("     baik layak dibangun dan tidak butuh keputusan pembimbing.")

    out = REPO_ROOT / "experiments" / f"exp6_oracle_predictability_{second}.json"
    out.write_text(json.dumps({
        "second_backbone": second, "counts": counts,
        "n_decidable_pooled": int(len(labels)),
        "auroc": results, "best_signal": best,
        "note": "AUROC of each parameter-free signal for predicting which space "
                "is correct, over queries where exactly one space is correct.",
    }, indent=2), encoding="utf-8")
    print(f"saved {out.name}")


if __name__ == "__main__":
    main()
