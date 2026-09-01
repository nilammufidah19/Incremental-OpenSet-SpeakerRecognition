#!/usr/bin/env python
"""Re-audit Experiment 4's ECAPA+Whisper fusion: is the second space being
*mis-conditioned* (anisotropy / no centering / no whitening) before it is
concatenated & score-fused?

Everything here is analysis-only; nothing in src/ is modified.

All post-processing transforms are FIT ON base_train ONLY (speaker-disjoint
from the eval/validation-half task speakers), so there is no task leakage.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(r"c:/Users/MSI/OneDrive/Documents/Nilam/Kuliah MKA/Thesis/Code/Incremental-OpenSet-SpeakerRecognition")
sys.path.insert(0, str(REPO_ROOT))

from src.data.splits import split_reserved_pool_halves  # noqa: E402
from src.evaluation.complementarity import fusion_accuracy, l2_normalize, oracle_report  # noqa: E402
from src.features.cache import get_or_compute_embedding, is_cached  # noqa: E402
from src.prototypical.score_norm import ASNorm  # noqa: E402

SPLITS_PATH = REPO_ROOT / "data" / "splits" / "full_split.json"
K_SHOT, N_QUERY, N_TASK_SPEAKERS = 1, 4, 100
SEEDS = [0, 1, 2]
COHORT_SIZE, TOP_K = 300, 200
FIT_SIZE = 4000            # base_train utterances used to FIT centering/whitening/LDA
W_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
SECOND = "whisper"         # whisper_l4 cache is empty on this checkout


# ------------------------------------------------------------------ loading
def load_base_train() -> pd.DataFrame:
    frames = [
        pd.read_csv(REPO_ROOT / "data/raw/audio/vox1_sample/manifest.csv"),
        pd.read_csv(REPO_ROOT / "data/raw/audio/base_train_capped/manifest.csv"),
    ]
    m = pd.concat(frames, ignore_index=True)
    keep = m["path"].apply(
        lambda p: is_cached(REPO_ROOT / p, "ecapa") and is_cached(REPO_ROOT / p, SECOND)
    )
    return m[keep].reset_index(drop=True)


def round_robin(manifest: pd.DataFrame, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    per: dict[str, list[str]] = {}
    for _, row in manifest.iterrows():
        per.setdefault(row["speaker_id"], []).append(row["path"])
    speakers = sorted(per)
    for s in speakers:
        rng.shuffle(per[s])
    picked_p, picked_s, r = [], [], 0
    while len(picked_p) < n:
        added = False
        for s in speakers:
            if r < len(per[s]):
                picked_p.append(REPO_ROOT / per[s][r])
                picked_s.append(s)
                added = True
                if len(picked_p) >= n:
                    break
        if not added:
            break
        r += 1
    return picked_p, np.array(picked_s)


def embed(paths, backbone) -> np.ndarray:
    return np.stack([get_or_compute_embedding(p, backbone) for p in paths]).astype(np.float64)


# ------------------------------------------------------- geometry diagnostics
def geometry(x: np.ndarray, labels: np.ndarray | None = None) -> dict:
    x = np.asarray(x, dtype=np.float64)
    u = l2_normalize(x)
    mean_vec = u.mean(axis=0)
    cos = u @ u.T
    off = cos[~np.eye(len(u), dtype=bool)]
    cov = np.cov(u, rowvar=False)
    ev = np.linalg.eigvalsh(cov)[::-1]
    ev = np.maximum(ev, 0)
    out = {
        "dim": int(x.shape[1]),
        "n": int(x.shape[0]),
        "mean_norm_ratio": float(np.linalg.norm(mean_vec)),   # ||mean of unit vecs||; 0 = isotropic, 1 = collapsed
        "mean_pairwise_cos": float(off.mean()),
        "std_pairwise_cos": float(off.std()),
        "top1_var_share": float(ev[0] / ev.sum()),
        "top10_var_share": float(ev[:10].sum() / ev.sum()),
        "eff_rank_participation": float((ev.sum() ** 2) / (ev ** 2).sum()),
    }
    if labels is not None:
        # Fisher ratio: between-speaker / within-speaker scatter (trace)
        classes = np.unique(labels)
        gm = u.mean(axis=0)
        sw = sb = 0.0
        for c in classes:
            xc = u[labels == c]
            if len(xc) < 2:
                continue
            mc = xc.mean(axis=0)
            sw += ((xc - mc) ** 2).sum()
            sb += len(xc) * ((mc - gm) ** 2).sum()
        out["fisher_trace_ratio"] = float(sb / sw) if sw > 0 else float("nan")
    return out


# ------------------------------------------------------------- transforms
class Transform:
    """Fit on base_train, apply anywhere. Always returns L2-normalized rows."""

    def __init__(self, name, fn):
        self.name, self.fn = name, fn

    def __call__(self, x):
        return l2_normalize(self.fn(np.asarray(x, dtype=np.float64)))


def build_transforms(fit_x: np.ndarray, fit_y: np.ndarray) -> list[Transform]:
    fit_u = l2_normalize(fit_x)
    mu = fit_u.mean(axis=0)
    xc = fit_u - mu
    cov = (xc.T @ xc) / (len(xc) - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]

    def center(x):
        return l2_normalize(x) - mu

    def abtt(k):
        V = evecs[:, :k]
        return lambda x: (lambda c: c - (c @ V) @ V.T)(l2_normalize(x) - mu)

    def whiten(d, eps_frac=1e-3):
        eps = eps_frac * evals[0]
        V = evecs[:, :d]
        s = 1.0 / np.sqrt(evals[:d] + eps)
        return lambda x: ((l2_normalize(x) - mu) @ V) * s

    # LDA (class-aware) on top of centered space -------------------------
    classes = np.unique(fit_y)
    gm = xc.mean(axis=0)
    Sw = np.zeros((xc.shape[1],) * 2)
    Sb = np.zeros_like(Sw)
    for c in classes:
        m = fit_y == c
        if m.sum() < 2:
            continue
        Xc = xc[m]
        mc = Xc.mean(axis=0)
        D = Xc - mc
        Sw += D.T @ D
        d = (mc - gm)[:, None]
        Sb += m.sum() * (d @ d.T)
    Sw /= max(len(xc) - len(classes), 1)
    Sb /= max(len(xc), 1)
    Sw_reg = Sw + 1e-4 * np.trace(Sw) / Sw.shape[0] * np.eye(Sw.shape[0])
    ev_l, evec_l = np.linalg.eig(np.linalg.solve(Sw_reg, Sb))
    ev_l, evec_l = np.real(ev_l), np.real(evec_l)
    o = np.argsort(ev_l)[::-1]
    evec_l = evec_l[:, o]

    def lda(d):
        V = evec_l[:, :d]
        V = V / np.linalg.norm(V, axis=0, keepdims=True)
        return lambda x: (l2_normalize(x) - mu) @ V

    # WCCN: whiten by within-class covariance ---------------------------
    Sw_c = Sw + 1e-3 * np.trace(Sw) / Sw.shape[0] * np.eye(Sw.shape[0])
    L = np.linalg.cholesky(np.linalg.inv(Sw_c))

    def wccn(x):
        return (l2_normalize(x) - mu) @ L

    dim = fit_x.shape[1]
    ts = [
        Transform("raw (current exp4)", lambda x: l2_normalize(x)),
        Transform("center", center),
        Transform("abtt_k1", abtt(1)),
        Transform("abtt_k5", abtt(5)),
        Transform("abtt_k10", abtt(10)),
        Transform("pca_whiten_d64", whiten(min(64, dim))),
        Transform("pca_whiten_d128", whiten(min(128, dim))),
        Transform("pca_whiten_d192", whiten(min(192, dim))),
        Transform("pca_whiten_full", whiten(dim)),
        Transform("wccn", wccn),
        Transform("lda_d64", lda(min(64, dim))),
        Transform("lda_d128", lda(min(128, dim))),
        Transform("lda_d192", lda(min(192, dim))),
    ]
    return ts


# ------------------------------------------------------------------ task
def build_task():
    split = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    validation_speakers, _ = split_reserved_pool_halves(split["reserved_unknown_pool"])
    em = pd.read_csv(REPO_ROOT / "data/raw/audio/eval_capped/manifest.csv")
    vm = em[em["speaker_id"].isin(set(validation_speakers))]
    counts = vm["speaker_id"].value_counts()
    usable = sorted(counts[counts >= K_SHOT + N_QUERY].index.tolist())
    task_speakers = usable[:N_TASK_SPEAKERS]
    speaker_paths = {
        s: [REPO_ROOT / p for p in vm.loc[vm["speaker_id"] == s, "path"]]
        for s in task_speakers
    }
    episodes = []
    for seed in SEEDS:
        rng = random.Random(seed)
        sup, qry, lab = [], [], []
        for idx, s in enumerate(task_speakers):
            paths = list(speaker_paths[s])
            rng.shuffle(paths)
            sup.append(paths[0])
            qry.extend(paths[1:1 + N_QUERY])
            lab.extend([idx] * len(paths[1:1 + N_QUERY]))
        episodes.append((sup, qry, np.asarray(lab)))
    return episodes


def main():
    print("loading manifests / cohort ...", flush=True)
    bt = load_base_train()
    fit_paths, fit_spk = round_robin(bt, FIT_SIZE, seed=1)
    cohort_paths, _ = round_robin(bt, COHORT_SIZE, seed=0)
    print(f"fit set: {len(fit_paths)} utts / {len(set(fit_spk))} speakers; cohort {len(cohort_paths)}")

    fit_e, fit_w = embed(fit_paths, "ecapa"), embed(fit_paths, SECOND)
    coh_e_raw, coh_w_raw = embed(cohort_paths, "ecapa"), embed(cohort_paths, SECOND)

    print("\n=== A. GEOMETRY OF EACH NATIVE SPACE (fit set, base_train) ===")
    ge = geometry(fit_e, fit_spk)
    gw = geometry(fit_w, fit_spk)
    keys = list(ge)
    print(f"{'metric':26s} {'ECAPA':>12s} {SECOND:>12s}")
    for k in keys:
        print(f"{k:26s} {ge[k]:12.4f} {gw[k]:12.4f}")

    episodes = build_task()
    # precompute task embeddings once
    task_emb = []
    for sup, qry, lab in episodes:
        task_emb.append({
            "p_e": embed(sup, "ecapa"), "p_w": embed(sup, SECOND),
            "q_e": embed(qry, "ecapa"), "q_w": embed(qry, SECOND),
            "lab": lab,
        })

    transforms = build_transforms(fit_w, fit_spk)
    ecapa_transforms = build_transforms(fit_e, fit_spk)

    print("\n=== B. SECOND-BACKBONE POST-PROCESSING (ECAPA left as-is) ===")
    print(f"{'transform':22s} {'acc_w':>7s} {'acc_e':>7s} {'oracle':>7s} {'ceil':>7s} "
          f"{'bestw':>6s} {'fus_acc':>8s} {'gain':>7s}")
    rows = []
    norm_e_cache = {}
    for t in transforms:
        accs_w, accs_e, orcs, ceils, fus = [], [], [], [], {str(w): [] for w in W_GRID}
        coh_e = l2_normalize(coh_e_raw)
        coh_w = t(coh_w_raw)
        norm_e = ASNorm(coh_e, top_k=TOP_K)
        norm_w = ASNorm(coh_w, top_k=TOP_K)
        for te in task_emb:
            z_e = norm_e.normalize(l2_normalize(te["q_e"]), l2_normalize(te["p_e"]))
            z_w = norm_w.normalize(t(te["q_w"]), t(te["p_w"]))
            rep = oracle_report(z_e, z_w, te["lab"])
            accs_w.append(rep["acc_b"]); accs_e.append(rep["acc_a"])
            orcs.append(rep["acc_oracle"]); ceils.append(rep["delta_ceiling"])
            for w in W_GRID:
                fus[str(w)].append(fusion_accuracy(z_e, z_w, te["lab"], w))
        best_w, best_acc = max(((w, float(np.mean(v))) for w, v in fus.items()), key=lambda x: x[1])
        row = {
            "transform": t.name, "acc_w": float(np.mean(accs_w)), "acc_e": float(np.mean(accs_e)),
            "oracle": float(np.mean(orcs)), "ceiling": float(np.mean(ceils)),
            "best_w": best_w, "fusion_acc": best_acc,
            "gain": best_acc - float(np.mean(accs_e)),
        }
        rows.append(row)
        print(f"{row['transform']:22s} {row['acc_w']:7.4f} {row['acc_e']:7.4f} "
              f"{row['oracle']:7.4f} {row['ceiling']:+7.4f} {row['best_w']:>6s} "
              f"{row['fusion_acc']:8.4f} {row['gain']:+7.4f}")

    print("\n=== C. SANITY: same transforms applied to ECAPA (alone) ===")
    print(f"{'transform':22s} {'acc_e':>7s}")
    for t in ecapa_transforms:
        coh_e = t(coh_e_raw)
        norm_e = ASNorm(coh_e, top_k=TOP_K)
        a = []
        for te in task_emb:
            z_e = norm_e.normalize(t(te["q_e"]), t(te["p_e"]))
            a.append(float((z_e.argmin(axis=1) == te["lab"]).mean()))
        print(f"{t.name:22s} {np.mean(a):7.4f}")

    out = Path(__file__).with_name("exp4_fusion_diag.json")
    out.write_text(json.dumps({"geometry": {"ecapa": ge, SECOND: gw}, "second_backbone": rows},
                              indent=2), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
