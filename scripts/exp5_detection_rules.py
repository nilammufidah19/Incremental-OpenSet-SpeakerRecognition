#!/usr/bin/env python
"""Experiment 5 -- post-hoc test of the exp4c two-space detection rules on
the OFFICIAL detection half (530 real unknown queries), gate G4.3 follow-up.

Reads experiments/detection_scores_<tag>.json (written by
scripts/run_full_evaluation.py when ExperimentConfig.dump_detection_scores
is on). That dump stores, per config per rep, the final-session decision
scores of every genuine query and every unknown query, in deterministic
order -- identical order across configs for a given rep (same seed, same
shuffles), which is what makes element-wise score combination valid.

Rules tested (training-free, from src/evaluation/complementarity.py's 4c
analysis): a_only (A1 = ECAPA-space score), b_only (A2 = second-backbone
space), mean, max -- compared against the production A3 score. The exp4c
finding (validation, n=40): "mean" cut EER 0.165 -> 0.151; this script
checks whether that survives on 530 pristine unknowns.

Usage:
    .venv/Scripts/python.exe scripts/exp5_detection_rules.py [tag]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import auroc, tar_at_far  # noqa: E402
from src.prototypical.calibration import find_eer_threshold  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "exp5b_redimnet_fusion"
DUMP_PATH = REPO_ROOT / "experiments" / f"detection_scores_{TAG}.json"
OUT_PATH = REPO_ROOT / "experiments" / f"exp5_detection_rules_{TAG}.json"


def metrics(genuine: np.ndarray, unknown: np.ndarray) -> dict:
    return {
        "eer": find_eer_threshold(genuine, unknown).eer,
        "auroc": auroc(genuine, unknown),
        "tar_at_far1": tar_at_far(genuine, unknown, 0.01),
    }


def main() -> None:
    dump = json.loads(DUMP_PATH.read_text(encoding="utf-8"))
    a1, a2 = dump["A1_ecapa_only"], dump["A2_whisper_only"]
    a3 = dump["proposed_A3_running_average"]
    reps = sorted(a1, key=int)

    per_rule: dict[str, list[dict]] = {}
    for rep in reps:
        g1, u1 = np.array(a1[rep]["genuine"]), np.array(a1[rep]["unknown"])
        g2, u2 = np.array(a2[rep]["genuine"]), np.array(a2[rep]["unknown"])
        g3, u3 = np.array(a3[rep]["genuine"]), np.array(a3[rep]["unknown"])
        assert len(g1) == len(g2) and len(u1) == len(u2), "score arrays misaligned"
        rules = {
            "A3_production": (g3, u3),
            "a_only_A1": (g1, u1),
            "b_only_A2": (g2, u2),
            "mean": (0.5 * (g1 + g2), 0.5 * (u1 + u2)),
            "max": (np.maximum(g1, g2), np.maximum(u1, u2)),
        }
        for name, (g, u) in rules.items():
            per_rule.setdefault(name, []).append(metrics(g, u))

    summary = {}
    print(f"tag={TAG}: {len(reps)} reps, "
          f"{len(a1[reps[0]]['genuine'])} genuine / {len(a1[reps[0]]['unknown'])} unknown per rep\n")
    print(f"{'rule':16s} {'EER':>16s} {'AUROC':>8s} {'TAR@1%FAR':>10s}")
    for name, entries in per_rule.items():
        eers = [e["eer"] for e in entries]
        summary[name] = {
            "eer_mean": float(np.mean(eers)), "eer_std": float(np.std(eers)),
            "auroc_mean": float(np.mean([e["auroc"] for e in entries])),
            "tar_at_far1_mean": float(np.mean([e["tar_at_far1"] for e in entries])),
            "per_rep": entries,
        }
        s = summary[name]
        print(f"{name:16s} {s['eer_mean']:.4f}±{s['eer_std']:.4f} "
              f"{s['auroc_mean']:8.4f} {s['tar_at_far1_mean']:10.4f}")

    base, mean_rule = summary["a_only_A1"], summary["mean"]
    verdict = mean_rule["eer_mean"] < base["eer_mean"] - 0.01
    print(f"\nGate G4.3 pada paruh-deteksi resmi (mean EER < a_only EER - 0.01): "
          f"{'TERKONFIRMASI' if verdict else 'TIDAK terkonfirmasi'}")
    OUT_PATH.write_text(json.dumps({"tag": TAG, "summary": summary,
                                    "g43_confirmed": bool(verdict)}, indent=2), encoding="utf-8")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
