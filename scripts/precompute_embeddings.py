#!/usr/bin/env python
"""F3-06: precompute & cache backbone embeddings for real downloaded audio.

Populates data/cache/embeddings/{ecapa,whisper}/<sha1>.npy for every
utterance listed in the given manifest CSV(s) (must have a `path` column
relative to repo root -- see data/raw/audio/vox1_sample/manifest.csv and
data/raw/audio/base_train_capped/manifest.csv).

Usage:
    .venv/Scripts/python.exe scripts/precompute_embeddings.py \\
        --manifest data/raw/audio/vox1_sample/manifest.csv \\
        --manifest data/raw/audio/base_train_capped/manifest.csv \\
        --limit 1000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.features.cache import get_or_compute_embedding, is_cached  # noqa: E402


def main(manifest_paths: list[str], limit: int | None, backbones: list[str]) -> None:
    # Cache keys are sha1(path-string), so the path MUST be spelled exactly the
    # way consumers spell it. scripts/run_full_evaluation.py (and the whole
    # evaluation stack) uses absolute REPO_ROOT-based paths; this script used to
    # pass the manifest's relative strings, which silently populated a parallel,
    # never-reused cache namespace (found during Experiment 3).
    all_paths: list[Path] = []
    for mp in manifest_paths:
        df = pd.read_csv(REPO_ROOT / mp)
        all_paths.extend(REPO_ROOT / p for p in df["path"].tolist())

    if limit is not None:
        all_paths = all_paths[:limit]

    print(f"{len(all_paths)} utterances to process across {backbones}")

    for backbone in backbones:
        n_done, n_skipped, n_failed = 0, 0, 0
        t0 = time.time()
        for i, path in enumerate(all_paths, 1):
            if is_cached(path, backbone):
                n_skipped += 1
                continue
            try:
                get_or_compute_embedding(path, backbone)
                n_done += 1
            except Exception as e:  # noqa: BLE001
                n_failed += 1
                print(f"  [warn] failed {path}: {e}")
            if i % 100 == 0 or i == len(all_paths):
                elapsed = time.time() - t0
                print(f"  [{backbone}] {i}/{len(all_paths)} "
                      f"({n_done} computed, {n_skipped} cached, {n_failed} failed) "
                      f"in {elapsed/60:.1f} min")
        print(f"[{backbone}] done: {n_done} computed, {n_skipped} already cached, {n_failed} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", action="append", required=True, dest="manifests")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backbone", action="append", dest="backbones", default=None)
    args = parser.parse_args()
    main(args.manifests, args.limit, args.backbones or ["ecapa", "whisper"])
