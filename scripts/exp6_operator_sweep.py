#!/usr/bin/env python
"""Experiment 6 -- operator sweep: can a better fusion operator harvest the
measured headroom over ReDimNet-alone?

Context (docs/experiment-6.md section 7b): the linear-fusion ceiling sits
+0.0413 above A2, a fixed weight harvests 41% of it, F6-3b's margin-ratio rule
only +0.0050, yet the margin DIFFERENCE separates "which space is right" at
AUROC 0.8583. Two operators are built on that finding and compared on the
exp5/exp6 validation protocol, IDENTICAL except for the operator:

  * margin_shift (tier 2 standing): w(q) = clip(0.3 + alpha*(m_e - m_b), 0, 1)
    -- src/prototypical/score_norm.py::MarginShiftDualASNorm. alpha is one
    scalar swept HERE on the validation half, the same standing as w itself.
    alpha=0 IS the fixed-w arm.
  * LLR fusion (F6-3, tier 1 -- EXPLORATORY, CONTINGENT ON THE SUPERVISOR'S
    RULING, docs/memo-keputusan-f63.md): logistic regression on calibration
    trials from base_train.
      - llr3:  logit = a0 + a1*z_e + a2*z_b. NOTE a structural fact stated
        BEFORE results: a linear LLR in (z_e, z_b) is a fixed linear
        combination, so its argmin (identification) can at best match the best
        fixed w -- its value is a FITTED weight + calibrated scores. It is run
        as the plan's mandated 3-coefficient first step.
      - llr4:  adds a3 * (m_e(q) - m_b(q)) * (z_e - z_b), the margin-difference
        quality feature; the effective weight then varies per query, which is
        what the headroom requires. Still 4 scalars, fit by IRLS.
    Both are implemented IN THIS SCRIPT, not in src/, so nothing ships into
    the runtime until the ruling.

Fit discipline for the LLR arms: positives are genuine trials (held-out
base_train query vs its own speaker's prototype), negatives are impostor
trials (calibration_impostor_pool query vs its nearest prototype); the
prototype/query construction reuses the same split_cohort_and_genuine halves
the AS-Norm cohort discipline uses, so the coefficients never see task,
validation-half, or detection-half speakers. Coefficients are dumped verbatim
into the artifact (plan F6-3 spec item 6).

Pre-registered gates, in order of what they would mean:
  G-op-A (the thesis claim): best operator arm > A2 ReDimNet-alone,
          paired test over 10 seeds, p < 0.05.
  G-op-B (operator improved): best operator arm > A3 fixed w=0.3, p < 0.05.

Usage:
    .venv/Scripts/python.exe scripts/exp6_operator_sweep.py
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
    DualASNorm,
    MarginShiftDualASNorm,
    build_cohort,
    split_cohort_and_genuine,
)
from src.system import SpeakerIdentificationSystem  # noqa: E402
from src.utils.seed import SEED_LIST  # noqa: E402

SPLITS_PATH = REPO_ROOT / "data" / "splits" / "full_split.json"
OUT_PATH = REPO_ROOT / "experiments" / "exp6_operator_sweep.json"

SECOND_BACKBONE = "redimnet_b2"
K_SHOT, N_QUERY, N_WAY, N_SESSIONS = 1, 4, 10, 10
SEEDS = SEED_LIST                    # 10 -- mandatory since exp6 sec 8
COHORT_SIZE, TOP_K = 300, 200
TARGET_FRR = 0.01
BASE_W = 0.30                        # the 10-seed-locked operating point


# --------------------------------------------------------------------------- #
# LLR fusion -- lives here, not in src/, until the supervisor's ruling.        #
# --------------------------------------------------------------------------- #
class LLRDualASNorm(DualASNorm):
    """Score fusion via fitted log-odds instead of a fixed weight.

    fused(q, p) = -(a0 + a1*z_e + a2*z_b [+ a3*(m_e - m_b)*(z_e - z_b)])

    The negation keeps the system convention (lower = more genuine) since the
    logit is fit with genuine = 1 and genuine trials have LOW z. The offset a0
    shifts every score equally, so identification (argmin) is unaffected by it
    and only the threshold moves -- which is the point of calibration.
    """

    def __init__(self, cohort_embeddings, split_dim, top_k, coefs, use_margin_feature):
        super().__init__(cohort_embeddings, split_dim=split_dim, top_k=top_k, weight=0.5)
        self.coefs = np.asarray(coefs, dtype=np.float64)
        self.use_margin_feature = bool(use_margin_feature)

    @staticmethod
    def _margins(z):
        part = np.partition(z, 1, axis=1)
        return part[:, 1] - part[:, 0]

    def normalize(self, queries, prototypes):
        queries = np.atleast_2d(np.asarray(queries, dtype=np.float64))
        prototypes = np.atleast_2d(np.asarray(prototypes, dtype=np.float64))
        d = self.split_dim
        z_e = self._first.normalize(queries[:, :d], prototypes[:, :d])
        z_b = self._second.normalize(queries[:, d:], prototypes[:, d:])
        a = self.coefs
        logit = a[0] + a[1] * z_e + a[2] * z_b
        if self.use_margin_feature:
            if z_e.shape[1] >= 2:
                dm = (self._margins(z_e) - self._margins(z_b))[:, None]
            else:
                dm = np.zeros((len(z_e), 1))
            logit = logit + a[3] * dm * (z_e - z_b)
        return -logit


def fit_logistic_irls(X: np.ndarray, y: np.ndarray, l2: float = 1e-2,
                      n_iter: int = 50) -> np.ndarray:
    """Prior-weighted logistic regression by IRLS with L2 (plan F6-3 spec 4).
    Class weights balance genuine/impostor so the minority class is not
    swamped; returns [a0, a1, ...]."""
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    prior_w = np.where(y == 1, 0.5 / max(y.mean(), 1e-9),
                       0.5 / max(1 - y.mean(), 1e-9))
    for _ in range(n_iter):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (prior_w * (y - p)) - l2 * w
        s = np.maximum(prior_w * p * (1 - p), 1e-9)
        H = (Xb * s[:, None]).T @ Xb + l2 * np.eye(Xb.shape[1])
        step = np.linalg.solve(H, g)
        w = w + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def build_llr_trials(genuine_index, impostor_index, fuse_fn, norm_e, norm_b,
                     enrollment_k=1, seed=0, use_margin_feature=False):
    """Per-pair training trials for LLR, with explicit speaker alignment.

    Positives: held-out genuine query vs its OWN speaker's prototype.
    Negatives: every query (genuine + impostor) vs its nearest NON-matching
    prototype -- hard negatives, matching how the runtime actually errs.
    """
    rng = np.random.default_rng(seed)
    protos, proto_spk = [], []
    genuine_q, genuine_spk = [], []
    for spk, raws in genuine_index.items():
        if len(raws) < enrollment_k + 1:
            continue
        raws = list(raws)
        rng.shuffle(raws)
        protos.append(fuse_fn(raws[:enrollment_k]).mean(axis=0))
        proto_spk.append(spk)
        genuine_q.extend(raws[enrollment_k:])
        genuine_spk.extend([spk] * (len(raws) - enrollment_k))
    P = np.stack(protos)
    proto_spk = np.array(proto_spk)
    Q_g = fuse_fn(genuine_q)
    Q_i = fuse_fn([r for rs in impostor_index.values() for r in rs])

    d = ECAPA_DIM
    feats, labels = [], []
    for Q, spks in ((Q_g, np.array(genuine_spk)), (Q_i, None)):
        z_e = norm_e.normalize(Q[:, :d], P[:, :d])
        z_b = norm_b.normalize(Q[:, d:], P[:, d:])
        if use_margin_feature:
            dm = (LLRDualASNorm._margins(z_e) - LLRDualASNorm._margins(z_b))
        comb = 0.5 * (z_e + z_b)
        for i in range(len(Q)):
            own = None
            if spks is not None:
                own_idx = np.where(proto_spk == spks[i])[0]
                own = own_idx[0] if len(own_idx) else None
                if own is not None:
                    f = [z_e[i, own], z_b[i, own]]
                    if use_margin_feature:
                        f.append(dm[i] * (z_e[i, own] - z_b[i, own]))
                    feats.append(f); labels.append(1)
            order = np.argsort(comb[i])
            neg = next(j for j in order if own is None or j != own)
            f = [z_e[i, neg], z_b[i, neg]]
            if use_margin_feature:
                f.append(dm[i] * (z_e[i, neg] - z_b[i, neg]))
            feats.append(f); labels.append(0)
    return np.asarray(feats, dtype=np.float64), np.asarray(labels, dtype=np.float64)


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
    print(f"validation task: {len(sessions)}x{N_WAY}-way; detection half untouched: "
          f"{len(detection_speakers)} speakers")

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
    base_train_index = build_raw_embedding_index(manifest, whisper_backbone=SECOND_BACKBONE)
    print(f"base_train index: {manifest['speaker_id'].nunique()} speakers")

    fusion = ScoreFusionEmbed("fusion", weight=BASE_W).to(device)

    def fuse_fn(raw_list):
        raw = torch.from_numpy(np.stack(raw_list)).float().to(device)
        e, w = split_raw_embedding(raw)
        with torch.no_grad():
            return fusion(e, w).cpu().numpy()

    cohort_pool, calib_genuine_pool = split_cohort_and_genuine(base_train_index, seed=0)
    cohort = build_cohort(cohort_pool, fuse_fn, cohort_size=COHORT_SIZE, seed=0)

    impostor_ids = set(split["calibration_impostor_pool"])
    impostor_manifest = eval_manifest[eval_manifest["speaker_id"].isin(impostor_ids)]
    impostor_manifest = impostor_manifest[impostor_manifest["path"].apply(
        lambda p: is_cached(REPO_ROOT / p, "ecapa") and is_cached(REPO_ROOT / p, SECOND_BACKBONE)
    )]
    impostor_index = build_raw_embedding_index(impostor_manifest, whisper_backbone=SECOND_BACKBONE)

    # per-space normalizers shared by LLR fitting
    from src.prototypical.score_norm import ASNorm
    norm_e = ASNorm(cohort[:, :ECAPA_DIM], top_k=TOP_K)
    norm_b = ASNorm(cohort[:, ECAPA_DIM:], top_k=TOP_K)

    # ---- fit both LLR variants on base_train trials ------------------------
    llr_meta = {}
    llr_norms = {}
    for name, use_m in (("llr3", False), ("llr4_margin", True)):
        X, y = build_llr_trials(calib_genuine_pool, impostor_index, fuse_fn,
                                norm_e, norm_b, use_margin_feature=use_m)
        coefs = fit_logistic_irls(X, y)
        llr_meta[name] = {"coefficients": coefs.tolist(),
                          "n_trials": int(len(y)), "n_genuine": int(y.sum()),
                          "implied_fixed_w": float(coefs[1] / (coefs[1] + coefs[2]))
                          if (coefs[1] + coefs[2]) != 0 else None}
        llr_norms[name] = LLRDualASNorm(cohort, ECAPA_DIM, TOP_K, coefs, use_m)
        print(f"{name}: coefs={np.round(coefs, 4).tolist()} "
              f"({len(y)} trials, {int(y.sum())} genuine) "
              f"implied_w={llr_meta[name]['implied_fixed_w']:.3f}" if not use_m else
              f"{name}: coefs={np.round(coefs, 4).tolist()} ({len(y)} trials)")

    ARMS = (
        [("A2_redimnet_only", DualASNorm(cohort, ECAPA_DIM, TOP_K, weight=0.0)),
         ("A3_fixed_w30", DualASNorm(cohort, ECAPA_DIM, TOP_K, weight=BASE_W))]
        + [(f"margin_shift_a{a}", MarginShiftDualASNorm(
                cohort, ECAPA_DIM, TOP_K, weight=BASE_W, shift_scale=a))
           for a in (0.25, 0.5, 1.0, 2.0)]
        + [(n, llr_norms[n]) for n in llr_norms]
    )

    results = []
    for name, normalizer in ARMS:
        genuine_d, impostor_d = build_genuine_impostor_distances(
            fusion, calib_genuine_pool, impostor_index, enrollment_k=1, seed=0,
            device=device, score_normalizer=normalizer,
        )
        op = find_operating_point(genuine_d, impostor_d, "target_frr", target_frr=TARGET_FRR)
        accs = []
        for seed in SEEDS:
            system = SpeakerIdentificationSystem(
                fusion, op.threshold, continual_mode="running_average",
                whisper_backbone=SECOND_BACKBONE, score_normalizer=normalizer,
            )
            detailed = run_fscil_detailed(system, sessions, speaker_audio_paths,
                                          k_shot=K_SHOT, n_query=N_QUERY, seed=seed)
            accs.append(average_accuracy(detailed.open_set, max(detailed.open_set)))
        entry = {"arm": name, "threshold": op.threshold, "calibration_eer": op.eer,
                 "val_acc_mean": float(np.mean(accs)), "val_acc_std": float(np.std(accs)),
                 "val_accs": accs}
        results.append(entry)
        print(f"  {name:20s} thr={op.threshold:8.4f} calEER={op.eer:.4f} "
              f"val_acc={entry['val_acc_mean']:.4f}+/-{entry['val_acc_std']:.4f}")

    by = {r["arm"]: r for r in results}
    a2, fixed = by["A2_redimnet_only"], by["A3_fixed_w30"]
    candidates = [r for r in results if r["arm"] not in ("A2_redimnet_only", "A3_fixed_w30")]
    best = max(candidates, key=lambda r: r["val_acc_mean"])

    tests = {}
    for label, ref in (("vs_A2", a2), ("vs_fixed_w30", fixed)):
        t = paired_significance_test(np.asarray(best["val_accs"]), np.asarray(ref["val_accs"]))
        tests[label] = {"delta": best["val_acc_mean"] - ref["val_acc_mean"],
                        "test": t.test_used, "p_value": t.p_value,
                        "wins": int(sum(b > r for b, r in zip(best["val_accs"], ref["val_accs"])))}
        print(f"\nbest={best['arm']} {label}: delta={tests[label]['delta']:+.4f} "
              f"{t.test_used} p={t.p_value:.4f} wins {tests[label]['wins']}/{len(SEEDS)}")

    gate_a = tests["vs_A2"]["delta"] > 0 and tests["vs_A2"]["p_value"] < 0.05
    gate_b = tests["vs_fixed_w30"]["delta"] > 0 and tests["vs_fixed_w30"]["p_value"] < 0.05
    print(f"G-op-A (best > A2, the thesis claim): {'LOLOS' if gate_a else 'GAGAL'}")
    print(f"G-op-B (best > fixed w=0.3):          {'LOLOS' if gate_b else 'GAGAL'}")

    OUT_PATH.write_text(json.dumps({
        "protocol": {"second_backbone": SECOND_BACKBONE, "base_w": BASE_W,
                     "seeds": SEEDS, "cohort_size": COHORT_SIZE, "top_k": TOP_K,
                     "target_frr": TARGET_FRR,
                     "llr_status": "EXPLORATORY -- contingent on the supervisor's "
                                   "ruling (docs/memo-keputusan-f63.md); implemented "
                                   "in this script only, nothing in src/"},
        "llr_fits": llr_meta,
        "results": results, "best_operator": best, "tests": tests,
        "gate_op_A_vs_A2_passed": bool(gate_a),
        "gate_op_B_vs_fixed_passed": bool(gate_b),
        "elapsed_minutes": (time.time() - t0) / 60,
    }, indent=2), encoding="utf-8")
    print(f"saved {OUT_PATH.name} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
