#!/usr/bin/env python
"""Bab 4.11 (proposal) -- bootstrap significance test on EER differences,
executed for Experiment 5b (the first experiment whose harness dumps the
per-rep genuine/unknown decision scores needed as input).

The proposal specifies (p.54): EER cannot be tested with a simple paired
test, so a bootstrap-resampling 95% CI on the EER DIFFERENCE between
configurations is used (Bengio & Mariethoz, 2004); the difference is
significant when 0 lies outside the CI. `src/evaluation/statistics.py::
bootstrap_eer_difference` (F11-05) implements this and is unit-tested, but
no run harness had ever invoked it -- this script closes that gap.

Input : experiments/detection_scores_<tag>.json (per config, per rep:
        final-session genuine scores + 530-unknown scores).
Output: experiments/exp5_bootstrap_eer_<tag>.json + console table.
        Per comparison: bootstrap CI per rep (paired seeds), the count of
        reps whose CI excludes 0, and a pooled-across-reps CI.

Usage:
    .venv/Scripts/python.exe scripts/exp5_bootstrap_eer.py [tag]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Windows console defaults to cp1252; keep output ASCII-safe regardless.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.statistics import bootstrap_eer_difference  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "exp5b_redimnet_fusion"
DUMP_PATH = REPO_ROOT / "experiments" / f"detection_scores_{TAG}.json"
OUT_PATH = REPO_ROOT / "experiments" / f"exp5_bootstrap_eer_{TAG}.json"

# exp6 (2026-08-30): overridable via env. At 1000 resamples the Monte-Carlo
# error on the 97.5% quantile is ~1e-4 -- the same order as the distance
# between the w=0.3 A3-vs-A2 CI bound and zero, so 1000 cannot decide that
# borderline. Default stays 1000 so earlier artifacts reproduce.
N_BOOTSTRAP = int(os.environ.get("N_BOOTSTRAP", "1000"))
PROPOSED = "proposed_A3_running_average"
COMPARISONS = [
    ("proposed_A3_running_average", "A1_ecapa_only"),
    ("proposed_A3_running_average", "A2_whisper_only"),
    ("B1_static", "proposed_A3_running_average"),
    ("proposed_A3_running_average", "ECAPA_standard_baseline"),
    ("proposed_A3_running_average", "ProtoNet_vanilla_baseline"),
    ("proposed_A3_running_average", "xvector_PLDA_baseline"),
]


def arrays(dump: dict, config: str, rep: str) -> tuple[np.ndarray, np.ndarray]:
    entry = dump[config][rep]
    return np.asarray(entry["genuine"]), np.asarray(entry["unknown"])


def main() -> None:
    t0 = time.time()
    dump = json.loads(DUMP_PATH.read_text(encoding="utf-8"))
    reps = sorted(dump[PROPOSED], key=int)
    print(f"tag={TAG}: {len(reps)} reps, n_bootstrap={N_BOOTSTRAP}\n")

    results = {}
    for name_a, name_b in COMPARISONS:
        per_rep = []
        for rep in reps:
            g_a, u_a = arrays(dump, name_a, rep)
            g_b, u_b = arrays(dump, name_b, rep)
            r = bootstrap_eer_difference(g_a, u_a, g_b, u_b,
                                         n_bootstrap=N_BOOTSTRAP, seed=int(rep))
            per_rep.append({"rep": rep, "mean_diff": r.mean_diff,
                            "ci": [r.ci_lower, r.ci_upper], "significant": r.significant})
        pooled = bootstrap_eer_difference(
            np.concatenate([arrays(dump, name_a, rep)[0] for rep in reps]),
            np.concatenate([arrays(dump, name_a, rep)[1] for rep in reps]),
            np.concatenate([arrays(dump, name_b, rep)[0] for rep in reps]),
            np.concatenate([arrays(dump, name_b, rep)[1] for rep in reps]),
            n_bootstrap=N_BOOTSTRAP, seed=0,
        )
        n_sig = sum(e["significant"] for e in per_rep)
        results[f"{name_a}_vs_{name_b}"] = {
            "per_rep": per_rep,
            "n_reps_significant": n_sig,
            "pooled": {"mean_diff": pooled.mean_diff,
                       "ci": [pooled.ci_lower, pooled.ci_upper],
                       "significant": pooled.significant},
        }
        print(f"{name_a} vs {name_b}:")
        print(f"  per-rep signifikan: {n_sig}/{len(reps)}  "
              f"mean dEER={np.mean([e['mean_diff'] for e in per_rep]):+.4f}")
        print(f"  pooled dEER={pooled.mean_diff:+.4f} "
              f"CI95=[{pooled.ci_lower:+.4f}, {pooled.ci_upper:+.4f}] "
              f"-> {'SIGNIFIKAN' if pooled.significant else 'tidak signifikan'}\n")

    OUT_PATH.write_text(json.dumps({
        "tag": TAG, "n_bootstrap": N_BOOTSTRAP, "confidence": 0.95,
        "method": "Bengio & Mariethoz (2004) bootstrap CI on EER difference "
                  "(proposal Bab 4.11; F11-05)",
        "results": results,
        "elapsed_minutes": (time.time() - t0) / 60,
    }, indent=2), encoding="utf-8")
    print(f"saved {OUT_PATH} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
