#!/usr/bin/env python
"""F3-07: benchmark ECAPA-TDNN vs Whisper extraction time.

Reports two numbers per backbone:
  - "pure" forward-pass time (already-preprocessed waveform -> embedding)
  - "pipeline" time (raw file -> preprocessing -> embedding), which is what
    scripts/precompute_embeddings.py actually pays per utterance

Usage:
    .venv/Scripts/python.exe scripts/benchmark_backbones.py --n 50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models import ecapa, whisper_encoder  # noqa: E402
from src.preprocessing.pipeline import preprocess_audio  # noqa: E402

MANIFEST = REPO_ROOT / "data" / "raw" / "audio" / "vox1_sample" / "manifest.csv"


def main(n: int) -> None:
    manifest = pd.read_csv(MANIFEST)
    paths = [REPO_ROOT / p for p in manifest["path"].tolist()[:n]]
    print(f"Benchmarking on {len(paths)} real utterances "
          f"(device: ecapa={ecapa.DEFAULT_DEVICE}, whisper={whisper_encoder.DEFAULT_DEVICE})\n")

    # Warm up (model loading, CUDA kernel compilation) so timing reflects
    # steady-state throughput, not one-time initialization cost.
    ecapa.extract_embedding(preprocess_audio(paths[0], mode="ecapa_inference")[0])
    whisper_encoder.extract_embedding(preprocess_audio(paths[0], mode="whisper_inference")[0])

    # --- pipeline (raw file -> preprocessing -> embedding) ---
    t0 = time.time()
    ecapa_windows = [preprocess_audio(p, mode="ecapa_inference") for p in paths]
    for w in ecapa_windows:
        ecapa.extract_embedding(w[0]) if len(w) == 1 else ecapa.extract_embedding_windows(w)
    ecapa_pipeline_s = time.time() - t0

    t0 = time.time()
    whisper_windows = [preprocess_audio(p, mode="whisper_inference") for p in paths]
    for w in whisper_windows:
        whisper_encoder.extract_embedding(w[0]) if len(w) == 1 else whisper_encoder.extract_embedding_windows(w)
    whisper_pipeline_s = time.time() - t0

    # --- pure forward pass (preprocessing already done above, reuse windows) ---
    t0 = time.time()
    for w in ecapa_windows:
        ecapa.extract_embedding(w[0]) if len(w) == 1 else ecapa.extract_embedding_windows(w)
    ecapa_pure_s = time.time() - t0

    t0 = time.time()
    for w in whisper_windows:
        whisper_encoder.extract_embedding(w[0]) if len(w) == 1 else whisper_encoder.extract_embedding_windows(w)
    whisper_pure_s = time.time() - t0

    print(f"{'Backbone':10s} {'pure ms/file':>14s} {'pipeline ms/file':>18s} {'preprocessing overhead':>24s}")
    for name, pure, pipeline in [
        ("ECAPA-TDNN", ecapa_pure_s, ecapa_pipeline_s),
        ("Whisper", whisper_pure_s, whisper_pipeline_s),
    ]:
        pure_ms = pure / n * 1000
        pipeline_ms = pipeline / n * 1000
        overhead_ms = pipeline_ms - pure_ms
        print(f"{name:10s} {pure_ms:14.1f} {pipeline_ms:18.1f} {overhead_ms:24.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    main(args.n)
