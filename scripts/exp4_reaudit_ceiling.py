#!/usr/bin/env python
"""Part 2 of the Experiment-4 re-audit.

(1) Verify ScoreFusionEmbed's weighted-concatenation identity numerically.
(2) Is exp4's `delta_ceiling` really "the upper bound ANY fusion mechanism
    could reach"?  It is the oracle of per-query SELECTION between the two
    spaces.  A weighted SUM can be right on queries where BOTH argmins are
    wrong, so the true ceiling of the linear-score-fusion family is a
    different (and possibly larger) number.  Compute it exactly via the
    per-query feasible-w interval.
(3) Try fusion operators outside the weighted-sum family (rank/RRF, product,
    min, joint whitening of the concatenated space).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(r"c:/Users/MSI/OneDrive/Documents/Nilam/Kuliah MKA/Thesis/Code/Incremental-OpenSet-SpeakerRecognition")
sys.path.insert(0, str(REPO_ROOT))

from src.data.splits import split_reserved_pool_halves  # noqa: E402
from src.evaluation.complementarity import l2_normalize, oracle_report  # noqa: E402
from src.features.cache import get_or_compute_embedding, is_cached  # noqa: E402
from src.models.fusion import ScoreFusionEmbed  # noqa: E402
from src.prototypical.score_norm import ASNorm  # noqa: E402

K_SHOT, N_QUERY, N_TASK = 1, 4, 100
SEEDS = [0, 1, 2]
COHORT_SIZE, TOP_K = 300, 200
SECOND = "whisper"


# ---------------------------------------------------------------- (1) identity
def check_concat_identity() -> dict:
    torch.manual_seed(0)
    e = torch.randn(7, 192)
    w = torch.randn(7, 512)
    out = {}
    for weight in (0.2, 0.4, 0.7):
        emb = ScoreFusionEmbed(weight=weight)
        z = emb(e, w)
        u, v = torch.nn.functional.normalize(e, dim=-1), torch.nn.functional.normalize(w, dim=-1)
        sq = torch.cdist(z, z) ** 2
        cos_e = 1 - u @ u.T
        cos_w = 1 - v @ v.T
        target = 2 * (weight * cos_e + (1 - weight) * cos_w)
        out[str(weight)] = {
            "max_abs_err": float((sq - target).abs().max()),
            "unit_norm_max_dev": float((z.norm(dim=-1) - 1).abs().max()),
        }
    return out


# ---------------------------------------------------------------- data
def load_base_train():
    m = pd.concat([
        pd.read_csv(REPO_ROOT / "data/raw/audio/vox1_sample/manifest.csv"),
        pd.read_csv(REPO_ROOT / "data/raw/audio/base_train_capped/manifest.csv"),
    ], ignore_index=True)
    return m[m["path"].apply(
        lambda p: is_cached(REPO_ROOT / p, "ecapa") and is_cached(REPO_ROOT / p, SECOND))
    ].reset_index(drop=True)


def round_robin(manifest, n, seed=0):
    rng = np.random.default_rng(seed)
    per = {}
    for _, r in manifest.iterrows():
        per.setdefault(r["speaker_id"], []).append(r["path"])
    spk = sorted(per)
    for s in spk:
        rng.shuffle(per[s])
    picked, i = [], 0
    while len(picked) < n:
        added = False
        for s in spk:
            if i < len(per[s]):
                picked.append(REPO_ROOT / per[s][i]); added = True
                if len(picked) >= n:
                    break
        if not added:
            break
        i += 1
    return picked


def embed(paths, backbone):
    return np.stack([get_or_compute_embedding(p, backbone) for p in paths]).astype(np.float64)


def build_task():
    split = json.loads((REPO_ROOT / "data/splits/full_split.json").read_text(encoding="utf-8"))
    val_spk, _ = split_reserved_pool_halves(split["reserved_unknown_pool"])
    em = pd.read_csv(REPO_ROOT / "data/raw/audio/eval_capped/manifest.csv")
    vm = em[em["speaker_id"].isin(set(val_spk))]
    c = vm["speaker_id"].value_counts()
    task = sorted(c[c >= K_SHOT + N_QUERY].index.tolist())[:N_TASK]
    sp = {s: [REPO_ROOT / p for p in vm.loc[vm["speaker_id"] == s, "path"]] for s in task}
    eps = []
    for seed in SEEDS:
        rng = random.Random(seed)
        sup, qry, lab = [], [], []
        for i, s in enumerate(task):
            p = list(sp[s]); rng.shuffle(p)
            sup.append(p[0]); qry.extend(p[1:1 + N_QUERY]); lab.extend([i] * len(p[1:1 + N_QUERY]))
        eps.append((sup, qry, np.asarray(lab)))
    return eps


# ------------------------------------------- (2) exact linear-fusion ceiling
def linear_fusion_oracle(d_a, d_b, labels):
    """Fraction of queries for which SOME w in [0,1] makes the true class the
    argmin of  w*d_a + (1-w)*d_b.

    For competitor j:  w*(a_t-a_j) + (1-w)*(b_t-b_j) < 0
                    => w*(A_j - B_j) + B_j < 0     with A_j=a_t-a_j, B_j=b_t-b_j
    Each competitor gives a half-line in w; intersect over j, then with [0,1].
    """
    n = len(labels)
    ok = np.zeros(n, dtype=bool)
    lo_all = np.zeros(n)
    hi_all = np.zeros(n)
    for i in range(n):
        t = labels[i]
        A = d_a[i, t] - d_a[i]
        B = d_b[i, t] - d_b[i]
        m = np.ones(len(A), dtype=bool); m[t] = False
        A, B = A[m], B[m]
        C = A - B
        lo, hi = 0.0, 1.0
        feasible = True
        for c, b in zip(C, B):
            if abs(c) < 1e-15:
                if b >= 0:
                    feasible = False; break
            elif c > 0:
                hi = min(hi, -b / c)
            else:
                lo = max(lo, -b / c)
            if lo >= hi:
                feasible = False; break
        ok[i] = feasible and lo < hi
        lo_all[i], hi_all[i] = lo, hi
    return ok, lo_all, hi_all


def best_global_w(d_a, d_b, labels, grid):
    accs = {}
    for w in grid:
        f = w * d_a + (1 - w) * d_b
        accs[w] = float((f.argmin(axis=1) == labels).mean())
    bw = max(accs, key=accs.get)
    return bw, accs[bw], accs


# ------------------------------------------------- (3) non-linear operators
def rank_matrix(d):
    """rank 0 = best (smallest distance), per row."""
    order = np.argsort(d, axis=1)
    r = np.empty_like(order)
    rows = np.arange(d.shape[0])[:, None]
    r[rows, order] = np.arange(d.shape[1])[None, :]
    return r


def other_operators(z_e, z_w, labels):
    out = {}
    re_, rw = rank_matrix(z_e), rank_matrix(z_w)
    out["borda_rank_sum"] = float(((re_ + rw).argmin(axis=1) == labels).mean())
    for k in (10, 60):
        rrf = -(1.0 / (k + 1 + re_) + 1.0 / (k + 1 + rw))
        out[f"rrf_k{k}"] = float((rrf.argmin(axis=1) == labels).mean())
    out["min_rule"] = float((np.minimum(z_e, z_w).argmin(axis=1) == labels).mean())
    out["max_rule"] = float((np.maximum(z_e, z_w).argmin(axis=1) == labels).mean())
    # softmax-probability product / sum rule (classic Kittler et al. combiners)
    def soft(z, T=1.0):
        s = -z / T
        s = s - s.max(axis=1, keepdims=True)
        p = np.exp(s)
        return p / p.sum(axis=1, keepdims=True)
    pe, pw = soft(z_e), soft(z_w)
    out["product_rule"] = float(((pe * pw).argmax(axis=1) == labels).mean())
    out["sum_rule_prob"] = float(((pe + pw).argmax(axis=1) == labels).mean())
    out["weighted_prob_0.9"] = float(((0.9 * pe + 0.1 * pw).argmax(axis=1) == labels).mean())
    return out


def main():
    print("=== (1) ScoreFusionEmbed weighted-concat identity ===")
    print(json.dumps(check_concat_identity(), indent=2))

    bt = load_base_train()
    cohort = round_robin(bt, COHORT_SIZE, seed=0)
    coh_e = l2_normalize(embed(cohort, "ecapa"))
    coh_w = l2_normalize(embed(cohort, SECOND))
    norm_e, norm_w = ASNorm(coh_e, top_k=TOP_K), ASNorm(coh_w, top_k=TOP_K)

    grid = np.round(np.arange(0.0, 1.0001, 0.01), 2)
    agg = {"sel_oracle": [], "lin_oracle": [], "acc_e": [], "acc_w": [],
           "best_w": [], "best_acc": [], "both_wrong_recoverable": []}
    ops_all = []
    for sup, qry, lab in build_task():
        p_e, p_w = l2_normalize(embed(sup, "ecapa")), l2_normalize(embed(sup, SECOND))
        q_e, q_w = l2_normalize(embed(qry, "ecapa")), l2_normalize(embed(qry, SECOND))
        z_e, z_w = norm_e.normalize(q_e, p_e), norm_w.normalize(q_w, p_w)

        rep = oracle_report(z_e, z_w, lab)
        ok_lin, lo, hi = linear_fusion_oracle(z_e, z_w, lab)
        ok_e = z_e.argmin(axis=1) == lab
        ok_w = z_w.argmin(axis=1) == lab
        bw, bacc, accs = best_global_w(z_e, z_w, lab, grid)

        agg["acc_e"].append(rep["acc_a"]); agg["acc_w"].append(rep["acc_b"])
        agg["sel_oracle"].append(rep["acc_oracle"])
        agg["lin_oracle"].append(float(ok_lin.mean()))
        agg["best_w"].append(bw); agg["best_acc"].append(bacc)
        agg["both_wrong_recoverable"].append(float((ok_lin & ~ok_e & ~ok_w).mean()))
        ops_all.append(other_operators(z_e, z_w, lab))

    m = {k: float(np.mean(v)) for k, v in agg.items()}
    print("\n=== (2) ceilings (AS-Norm space, mean over 3 seeds) ===")
    print(f"  ECAPA alone                         : {m['acc_e']:.4f}")
    print(f"  Whisper alone                       : {m['acc_w']:.4f}")
    print(f"  SELECTION oracle  (exp4's ceiling)  : {m['sel_oracle']:.4f}  "
          f"(+{m['sel_oracle'] - m['acc_e']:.4f})")
    print(f"  LINEAR-FUSION oracle (per-query w)  : {m['lin_oracle']:.4f}  "
          f"(+{m['lin_oracle'] - m['acc_e']:.4f})   <-- true ceiling of w*z_e+(1-w)*z_w")
    print(f"    of which BOTH argmins were wrong  : {m['both_wrong_recoverable']:.4f}")
    print(f"  best single GLOBAL w                : w={m['best_w']:.3f} acc={m['best_acc']:.4f} "
          f"(+{m['best_acc'] - m['acc_e']:+.4f})")

    print("\n=== (3) fusion operators outside the weighted-sum family ===")
    keys = ops_all[0].keys()
    for k in keys:
        v = float(np.mean([o[k] for o in ops_all]))
        print(f"  {k:22s} {v:.4f}  ({v - m['acc_e']:+.4f} vs ECAPA-alone)")

    Path(__file__).with_name("exp4_fusion_diag2.json").write_text(
        json.dumps({"ceilings": m, "operators": {k: float(np.mean([o[k] for o in ops_all]))
                                                 for k in keys}}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
