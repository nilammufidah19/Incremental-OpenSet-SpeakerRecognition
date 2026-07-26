#!/usr/bin/env python
"""Download a per-speaker-capped audio subset for base_train (Data Latih
Awal), using the ungated `ProgramComputer/voxceleb` HF mirror -- see
src/data/remote_zip_audio.py for how/why this avoids both the full-corpus
size AND the VGG credential gate.

Sizing (see chat discussion "1/4 dari data train"): cap=14 utterances/speaker
across all 5,156 base_train speakers (868 vox1 + 4,288 vox2) targets
~19.2 GB total, matching the user's ~20 GB budget.

IMPORTANT -- realistic timing: this mirror is only range-readable at
~600-650 ms/file with 4 concurrent workers (benchmarked). At cap=14:
  - vox1 portion (~838 non-test-split dev speakers, ~11.7k files): ~2 hours
  - vox2 portion (~4,288 speakers, ~60k files):                    ~10+ hours
Run --dataset vox1 first (smaller, self-contained); treat --dataset vox2 as
a separate, much longer-running job you kick off deliberately.

Usage:
    .venv/Scripts/python.exe scripts/fetch_capped_base_train_audio.py \\
        --dataset vox1 --cap 14 --n-workers 4
    .venv/Scripts/python.exe scripts/fetch_capped_base_train_audio.py \\
        --dataset vox2 --cap 14 --n-workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.remote_zip_audio import download_capped  # noqa: E402
from src.data.voxceleb import load_vox1_meta  # noqa: E402

SPLITS_PATH = REPO_ROOT / "data" / "splits" / "full_split.json"
DEST_ROOT = REPO_ROOT / "data" / "raw" / "audio" / "base_train_capped"
MANIFEST_PATH = DEST_ROOT / "manifest.csv"


def main(dataset: str, cap: int, n_workers: int, seed: int) -> None:
    split = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    base_train = split["base_train"]

    vox1_meta = load_vox1_meta(REPO_ROOT / "data" / "raw" / "metadata" / "vox1_meta.csv")
    vox1_ids = set(vox1_meta["speaker_id"])

    if dataset == "vox1":
        wanted = [s for s in base_train if s in vox1_ids]
        print(f"base_train vox1 speakers: {len(wanted)}")
        download_capped(
            source_key="vox1_dev_wav",
            wanted_speakers=wanted,
            cap=cap,
            dest_dir=DEST_ROOT,
            manifest_path=MANIFEST_PATH,
            n_workers=n_workers,
            seed=seed,
        )
    elif dataset == "vox2":
        wanted = [s for s in base_train if s not in vox1_ids]
        print(f"base_train vox2 speakers: {len(wanted)}")
        for source_key in ("vox2_aac_1", "vox2_aac_2"):
            download_capped(
                source_key=source_key,
                wanted_speakers=wanted,
                cap=cap,
                dest_dir=DEST_ROOT,
                manifest_path=MANIFEST_PATH,
                n_workers=n_workers,
                seed=seed,
            )
    else:
        raise SystemExit(f"unknown --dataset {dataset!r}, expected vox1 or vox2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["vox1", "vox2"], required=True)
    parser.add_argument("--cap", type=int, default=14)
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.dataset, args.cap, args.n_workers, args.seed)
