"""Centralized random-seed management.

Ref: plan/02-requirements.md NFR-01, FR-38 (10 independent repetitions per
configuration for ablation/baseline/statistical testing).

Every experiment entry point must call `set_seed(seed)` once at start, and
multi-run drivers (ablation study, baseline comparison, significance testing)
must iterate over `SEED_LIST` rather than inventing ad-hoc seeds, so that
runs are reproducible and comparable across configurations.
"""
from __future__ import annotations

import os
import random

import numpy as np

# 10 fixed seeds shared by every multi-run experiment (F9/F10/F11).
# Do not change these once experiments have started -- changing them breaks
# reproducibility of already-reported results.
SEED_LIST: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed python, numpy, and (if installed) torch RNGs.

    Parameters
    ----------
    seed:
        Integer seed to apply to all RNGs.
    deterministic_torch:
        If True and torch is importable, also request deterministic
        algorithms. Kept optional because full determinism on GPU can be
        slower; disable only for quick, non-reported exploratory runs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_everything_for_run(run_index: int) -> int:
    """Map a 0-based run index (0..9) to its fixed seed and apply it.

    Returns the seed that was applied, so callers can log it alongside
    results (required for FR-38 mean +/- std reporting and F11 significance
    tests to be traceable back to a specific seed).
    """
    if not (0 <= run_index < len(SEED_LIST)):
        raise ValueError(
            f"run_index must be in [0, {len(SEED_LIST) - 1}], got {run_index}"
        )
    seed = SEED_LIST[run_index]
    set_seed(seed)
    return seed
