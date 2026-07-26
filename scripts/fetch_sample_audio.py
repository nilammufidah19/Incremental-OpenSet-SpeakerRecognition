#!/usr/bin/env python
"""Fetch a SMALL, real-audio sample of VoxCeleb1 for pipeline development
(F2 onward), without needing the full credentialed VGG download.

Source: `asahi417/voxceleb1-test-split` on HuggingFace Hub -- a public,
**ungated** re-hosting of the official VoxCeleb1 *test* split (the 4,874
utterances / ~40 speakers used for the standard Vox1-O verification trial
list). This is genuine VoxCeleb1 audio, not synthetic -- but it is only the
test partition, not the full ~1,251-speaker / 153,516-utterance corpus, so
it must NOT be treated as a substitute for the real F1-01/02 download when
running the actual reported experiments (base training needs Data Latih
Awal, which mostly consists of speakers not in this test-only sample).

Use this only to:
  - develop/smoke-test preprocessing, VAD, feature extraction code (F2/F3)
    against real speech instead of synthetic tones/noise
  - validate that assign_support_query (F1-08) works end-to-end on
    genuine multi-utterance-per-speaker audio

Output layout:
  data/raw/audio/vox1_sample/<speaker_id>/<utterance_filename>.wav
  data/raw/audio/vox1_sample/manifest.csv  (speaker_id, utterance_id, path,
                                              sampling_rate, n_samples)

Usage:
    .venv/Scripts/python.exe scripts/fetch_sample_audio.py \\
        --min-speakers 12 --max-utts-per-speaker 5
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "raw" / "audio" / "vox1_sample"
HF_DATASET = "asahi417/voxceleb1-test-split"


def main(min_speakers: int, max_utts_per_speaker: int, max_rows_scanned: int) -> None:
    from datasets import load_dataset

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(HF_DATASET, split="test", streaming=True)

    per_speaker_count: dict[str, int] = defaultdict(int)
    manifest_rows = []

    n_scanned = 0
    for row in ds:
        n_scanned += 1
        speaker_id = row["speaker_id"]

        if per_speaker_count[speaker_id] >= max_utts_per_speaker:
            if len(per_speaker_count) >= min_speakers and all(
                c >= 2 or c >= max_utts_per_speaker for c in per_speaker_count.values()
            ):
                pass  # keep scanning is fine; just skip saving more for this speaker
            if n_scanned > max_rows_scanned:
                break
            continue

        audio = row["audio"]
        utt_id = row["id"]  # e.g. "id10270+5r0dWxy17C8+00001.wav"
        speaker_dir = OUT_DIR / speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True)
        dest = speaker_dir / utt_id.split("+", 1)[1].replace("+", "_")
        sf.write(dest, audio["array"], audio["sampling_rate"])

        per_speaker_count[speaker_id] += 1
        manifest_rows.append(
            {
                "speaker_id": speaker_id,
                "utterance_id": utt_id,
                "path": str(dest.relative_to(REPO_ROOT)),
                "sampling_rate": audio["sampling_rate"],
                "n_samples": len(audio["array"]),
            }
        )

        n_eligible_speakers = sum(1 for c in per_speaker_count.values() if c >= 2)
        if n_eligible_speakers >= min_speakers and n_scanned > min_speakers * 3:
            # enough speakers with >=2 utterances each (support+query viable)
            break
        if n_scanned > max_rows_scanned:
            print(f"[warn] hit max_rows_scanned={max_rows_scanned} before reaching "
                  f"min_speakers={min_speakers} with >=2 utterances each")
            break

    manifest_path = OUT_DIR / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    n_speakers = len(per_speaker_count)
    n_files = len(manifest_rows)
    total_seconds = sum(r["n_samples"] / r["sampling_rate"] for r in manifest_rows)
    print(f"Saved {n_files} real VoxCeleb1 utterances from {n_speakers} speakers "
          f"({total_seconds:.1f}s total audio) to {OUT_DIR}")
    print(f"Manifest: {manifest_path}")
    print(
        "\nReminder: this is a small SAMPLE from the official VoxCeleb1 test "
        "split (public, ungated HF mirror), not the full corpus. See "
        "scripts/download_voxceleb.py for the full, credentialed F1-01/02 "
        "download once VGG access is granted."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-speakers", type=int, default=12)
    parser.add_argument("--max-utts-per-speaker", type=int, default=5)
    parser.add_argument("--max-rows-scanned", type=int, default=2000)
    args = parser.parse_args()
    main(args.min_speakers, args.max_utts_per_speaker, args.max_rows_scanned)
