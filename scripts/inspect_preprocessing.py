#!/usr/bin/env python
"""F2-10: run the preprocessing pipeline end-to-end on a small real-audio
subset and report before/after stats, plus write a few processed files to
disk so they can be listened to manually.

Usage:
    .venv/Scripts/python.exe scripts/inspect_preprocessing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.io import TARGET_SR, load_and_resample  # noqa: E402
from src.preprocessing.pipeline import preprocess_audio  # noqa: E402

SAMPLE_MANIFEST = REPO_ROOT / "data" / "raw" / "audio" / "vox1_sample" / "manifest.csv"
OUT_DIR = REPO_ROOT / "data" / "processed" / "preprocessing_inspection"
N_FILES = 5


def stats(waveform: np.ndarray, sr: int) -> dict:
    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(waveform) if np.any(waveform) else float("-inf")
    return {
        "duration_s": round(len(waveform) / sr, 2),
        "rms": round(float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))), 4),
        "lufs": round(lufs, 2) if np.isfinite(lufs) else lufs,
        "peak": round(float(np.max(np.abs(waveform))) if waveform.size else 0.0, 4),
    }


def main() -> None:
    import pandas as pd

    manifest = pd.read_csv(SAMPLE_MANIFEST)
    rows = manifest.sample(n=N_FILES, random_state=0).to_dict("records")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'file':40s} {'mode':18s} {'before':40s} {'after':40s}")
    for row in rows:
        path = REPO_ROOT / row["path"]
        before = load_and_resample(path)
        before_stats = stats(before, TARGET_SR)

        for mode in ["ecapa_train", "whisper_train"]:
            windows = preprocess_audio(path, mode=mode)
            after = windows[0]
            after_stats = stats(after, TARGET_SR)
            print(f"{path.name:40s} {mode:18s} {str(before_stats):40s} {str(after_stats):40s}")

            out_name = f"{path.stem}_{mode}.wav"
            sf.write(OUT_DIR / out_name, after, TARGET_SR)

        # also copy the raw (pre-preprocessing) file for A/B listening
        sf.write(OUT_DIR / f"{path.stem}_original.wav", before, TARGET_SR)

    print(f"\nWrote before/after WAV files to {OUT_DIR} for manual listening.")


if __name__ == "__main__":
    main()
