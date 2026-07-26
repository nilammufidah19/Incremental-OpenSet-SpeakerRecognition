"""VoxCeleb1/2 metadata loading and speaker-catalog construction.

Ref: plan/04-tasks.md F1-01..F1-04, plan/02-requirements.md DR-01,
proposal Bab 4.3 / Tabel 4.1.

This module only deals with *speaker-level metadata* (official
`vox1_meta.csv` / `vox2_meta.csv`), not raw audio. Raw audio acquisition is
handled separately (see scripts/download_voxceleb.py) because VoxCeleb2 in
particular is gated behind a request-based access agreement with the
dataset owners (Oxford VGG) and the combined corpus is ~2.8k hours -- not
something to be silently bulk-downloaded from an automated agent.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_VOX1_COLUMNS = ["speaker_id", "gender", "nationality", "official_set", "dataset"]
# Includes "nationality" (always None for vox2) even though the source file
# doesn't have it: build_speaker_catalog() concat()s vox1_df/vox2_df and then
# selects CATALOG_COLUMNS (which includes nationality), so vox2's output
# frame must carry that column too for the merge to work.
REQUIRED_VOX2_COLUMNS = ["speaker_id", "gender", "nationality", "official_set", "dataset"]
CATALOG_COLUMNS = ["speaker_id", "dataset", "gender", "nationality", "official_set"]


def load_vox1_meta(path: str | Path) -> pd.DataFrame:
    """Load the official VoxCeleb1 metadata file (tab-separated).

    Expected columns in the source file: 'VoxCeleb1 ID', 'VGGFace1 ID',
    'Gender', 'Nationality', 'Set'.
    """
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame(
        {
            "speaker_id": df["VoxCeleb1 ID"].str.strip(),
            "gender": df["Gender"].str.strip(),
            "nationality": df["Nationality"].str.strip(),
            "official_set": df["Set"].str.strip(),
            "dataset": "vox1",
        }
    )
    return out[REQUIRED_VOX1_COLUMNS]


def load_vox2_meta(path: str | Path) -> pd.DataFrame:
    """Load the official VoxCeleb2 metadata file (comma-separated, CRLF,
    columns padded with stray spaces -- both are quirks of the file as
    distributed by VGG, handled explicitly here rather than assumed away).

    Expected columns: 'VoxCeleb2 ID', 'VGGFace2 ID', 'Gender', 'Set'.
    VoxCeleb2 metadata does not include nationality (unlike VoxCeleb1).
    """
    df = pd.read_csv(path, sep=",", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame(
        {
            "speaker_id": df["VoxCeleb2 ID"].astype(str).str.strip(),
            "gender": df["Gender"].astype(str).str.strip(),
            "official_set": df["Set"].astype(str).str.strip(),
            "dataset": "vox2",
        }
    )
    out["nationality"] = None
    return out[REQUIRED_VOX2_COLUMNS]


def build_speaker_catalog(vox1_df: pd.DataFrame, vox2_df: pd.DataFrame) -> pd.DataFrame:
    """Merge VoxCeleb1 + VoxCeleb2 speaker metadata into one catalog.

    Raises
    ------
    ValueError
        If any speaker_id appears in both datasets (violates the
        speaker-disjoint assumption from proposal Bab 4.3 that this whole
        pipeline depends on -- see also F1-04 and the blocking test in
        tests/test_data_splits.py::test_vox1_vox2_speaker_ids_disjoint).
    """
    overlap = set(vox1_df["speaker_id"]) & set(vox2_df["speaker_id"])
    if overlap:
        raise ValueError(
            f"VoxCeleb1/VoxCeleb2 speaker_id overlap detected ({len(overlap)} ids): "
            f"{sorted(overlap)[:10]}... -- speaker-disjoint assumption violated."
        )

    catalog = pd.concat([vox1_df, vox2_df], ignore_index=True)
    catalog = catalog[CATALOG_COLUMNS]

    duplicated = catalog["speaker_id"].duplicated()
    if duplicated.any():
        raise ValueError(
            f"Duplicate speaker_id rows within combined catalog: "
            f"{catalog.loc[duplicated, 'speaker_id'].tolist()[:10]}..."
        )

    return catalog.reset_index(drop=True)


def load_combined_catalog(
    vox1_meta_path: str | Path, vox2_meta_path: str | Path
) -> pd.DataFrame:
    """Convenience wrapper: load both metadata files and merge (F1-04)."""
    vox1_df = load_vox1_meta(vox1_meta_path)
    vox2_df = load_vox2_meta(vox2_meta_path)
    return build_speaker_catalog(vox1_df, vox2_df)


def load_vox1_utterance_manifest(path: str | Path) -> pd.DataFrame:
    """Parse the official VoxCeleb1 identification split file (`iden_split.txt`).

    This file is a small (~5 MB) plain-text list distributed alongside the
    VoxCeleb1 metadata -- it does *not* require downloading any audio -- and
    gives a genuine utterance-level manifest: one line per utterance,
    formatted as `<official_partition> <speaker_id>/<video_id>/<utt_file>`,
    where official_partition in {1: train, 2: val, 3: test} is VGG's own
    identification-protocol split (distinct from, and not to be confused
    with, this project's own Base/Task split in splits.py).

    Returns a DataFrame with columns: speaker_id, video_id, utterance_file,
    utterance_id (unique per row), vgg_official_partition.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            partition_str, rel_path = line.split(" ", 1)
            speaker_id, video_id, utt_file = rel_path.split("/")
            rows.append(
                {
                    "speaker_id": speaker_id,
                    "video_id": video_id,
                    "utterance_file": utt_file,
                    "utterance_id": rel_path,
                    "vgg_official_partition": int(partition_str),
                }
            )
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError(f"No utterance rows parsed from {path}")
    return manifest


def summarize_catalog(catalog: pd.DataFrame) -> dict:
    """Compute the summary stats used to cross-check against Tabel 4.1.

    Note: the officially published Tabel 4.1 figures (1.251 / 6.112 speakers)
    were computed against a specific snapshot of vox2_meta.csv; the file as
    currently distributed lists a small number of additional VoxCeleb2
    speakers (see plan/05-evaluation-plan.md and F1-03 verification notes).
    This function reports the *actual current* counts rather than forcing
    them to match historical numbers, so discrepancies are visible instead
    of silently hidden.
    """
    per_dataset = catalog.groupby("dataset")["speaker_id"].nunique().to_dict()
    return {
        "n_speakers_total": int(catalog["speaker_id"].nunique()),
        "n_speakers_vox1": int(per_dataset.get("vox1", 0)),
        "n_speakers_vox2": int(per_dataset.get("vox2", 0)),
        "gender_ratio_vox1": catalog.loc[catalog["dataset"] == "vox1", "gender"]
        .value_counts(normalize=True)
        .to_dict(),
        "gender_ratio_vox2": catalog.loc[catalog["dataset"] == "vox2", "gender"]
        .value_counts(normalize=True)
        .to_dict(),
    }
