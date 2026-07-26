"""Lightweight experiment tracking (F0-04, [ER-07]).

Design goal: every ablation-study / baseline-comparison / statistical-testing
run (F9-F11) must persist its *raw*, per-run results -- not just aggregates
-- so mean +/- std (FR-38) and significance tests (F11) can be recomputed
and audited later (NFR-06). This module gives one small, dependency-free
JSON-Lines logger rather than pulling in MLflow/W&B, which is overkill for a
single-author thesis project and adds a service dependency this pipeline
does not need.

Each call to `log_run` appends one JSON record to
`experiments/<experiment_name>.jsonl`. Records are never mutated in place,
only appended, so partial/failed runs never corrupt earlier results.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"


def log_run(
    experiment_name: str,
    config_id: str,
    run_index: int,
    seed: int,
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Append one run's results to `experiments/<experiment_name>.jsonl`.

    Parameters
    ----------
    experiment_name:
        Logical experiment group, e.g. "ablation_fusion", "baseline_comparison".
    config_id:
        Configuration label within the experiment, e.g. "A1", "A2", "A3",
        "B1", "B2", "openFEAT", "proposed_system" (see plan/05-evaluation-plan.md).
    run_index:
        0-based repetition index (0..9), see src/utils/seed.py SEED_LIST.
    seed:
        The actual seed applied for this run (from seed_everything_for_run).
    metrics:
        Flat dict of metric name -> value for this run, e.g.
        {"accuracy": 0.842, "eer": 0.071} or per-session lists for
        Average Accuracy / Forgetting Measure.
    extra:
        Optional free-form metadata (e.g. git commit hash, checkpoint path).

    Returns
    -------
    Path to the JSONL file the record was appended to.
    """
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXPERIMENTS_DIR / f"{experiment_name}.jsonl"

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment_name": experiment_name,
        "config_id": config_id,
        "run_index": run_index,
        "seed": seed,
        "metrics": metrics,
        "extra": extra or {},
    }

    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return out_path


def load_runs(experiment_name: str) -> list[dict[str, Any]]:
    """Load every logged record for an experiment (for aggregation/F11 stats)."""
    path = EXPERIMENTS_DIR / f"{experiment_name}.jsonl"
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
