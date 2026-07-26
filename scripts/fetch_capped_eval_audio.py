#!/usr/bin/env python
"""Download capped real audio for the evaluation-side partitions that F8's
FSCIL harness actually needs: task_speakers (100, the 10 sessions x 10-way),
reserved_unknown_pool (221), and a bounded sample of calibration_impostor_pool
(the full 1,888 is unnecessary -- calibration only needs enough impostor
diversity to calibrate a stable EER threshold).

Reuses the same ungated-mirror + HTTP-Range mechanism as
scripts/fetch_capped_base_train_audio.py (see src/data/remote_zip_audio.py).

Usage:
    .venv/Scripts/python.exe scripts/fetch_capped_eval_audio.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.remote_zip_audio import download_capped  # noqa: E402
from src.data.voxceleb import load_vox1_meta  # noqa: E402

SPLITS_PATH = REPO_ROOT / "data" / "splits" / "full_split.json"
DEST_ROOT = REPO_ROOT / "data" / "raw" / "audio" / "eval_capped"
MANIFEST_PATH = DEST_ROOT / "manifest.csv"

CALIBRATION_SAMPLE_SIZE = 300  # of 1,888 total -- enough impostor diversity for a stable EER


def fetch_group(name: str, speakers: list[str], cap: int, vox1_ids: set[str]) -> None:
    v1 = [s for s in speakers if s in vox1_ids]
    v2 = [s for s in speakers if s not in vox1_ids]
    print(f"=== {name}: {len(speakers)} speakers ({len(v1)} vox1, {len(v2)} vox2), cap={cap} ===")

    if v1:
        download_capped(
            source_key="vox1_dev_wav",
            wanted_speakers=v1,
            cap=cap,
            dest_dir=DEST_ROOT,
            manifest_path=MANIFEST_PATH,
            n_workers=4,
        )
    if v2:
        for source_key in ("vox2_aac_1", "vox2_aac_2"):
            download_capped(
                source_key=source_key,
                wanted_speakers=v2,
                cap=cap,
                dest_dir=DEST_ROOT,
                manifest_path=MANIFEST_PATH,
                n_workers=4,
            )


def main() -> None:
    split = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    vox1_meta = load_vox1_meta(REPO_ROOT / "data" / "raw" / "metadata" / "vox1_meta.csv")
    vox1_ids = set(vox1_meta["speaker_id"])

    fetch_group("task_speakers", split["task_speakers"], cap=10, vox1_ids=vox1_ids)
    fetch_group("reserved_unknown_pool", split["reserved_unknown_pool"], cap=5, vox1_ids=vox1_ids)

    import random

    rng = random.Random(0)
    calibration_sample = rng.sample(
        split["calibration_impostor_pool"], min(CALIBRATION_SAMPLE_SIZE, len(split["calibration_impostor_pool"]))
    )
    fetch_group("calibration_impostor_sample", calibration_sample, cap=3, vox1_ids=vox1_ids)

    print("\nAll groups done.")


if __name__ == "__main__":
    main()
