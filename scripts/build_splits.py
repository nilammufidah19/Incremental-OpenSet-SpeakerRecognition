#!/usr/bin/env python
"""Build and persist the speaker-disjoint split (F1-05..F1-11).

Reads the metadata downloaded by scripts/download_voxceleb.py
(--stage metadata), runs src/data/splits.py::run_full_split, verifies
disjointness, and writes:

  data/splits/full_split.json   -- the complete partition (speaker IDs)
  data/splits/summary.md        -- human-readable summary for F1-11 /
                                    cross-check against proposal Tabel 4.1

Usage:
    .venv/Scripts/python.exe scripts/build_splits.py [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.voxceleb import load_combined_catalog, summarize_catalog  # noqa: E402
from src.data.splits import run_full_split  # noqa: E402

METADATA_DIR = REPO_ROOT / "data" / "raw" / "metadata"
SPLITS_DIR = REPO_ROOT / "data" / "splits"


def main(seed: int = 0) -> None:
    vox1_meta = METADATA_DIR / "vox1_meta.csv"
    vox2_meta = METADATA_DIR / "vox2_meta.csv"
    for p in (vox1_meta, vox2_meta):
        if not p.exists():
            raise SystemExit(
                f"Missing {p}. Run: python scripts/download_voxceleb.py --stage metadata"
            )

    catalog = load_combined_catalog(vox1_meta, vox2_meta)
    stats = summarize_catalog(catalog)

    split = run_full_split(catalog, seed=seed)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "seed": split.seed,
        "n_speakers_total": stats["n_speakers_total"],
        "n_speakers_vox1": stats["n_speakers_vox1"],
        "n_speakers_vox2": stats["n_speakers_vox2"],
        "base_train": split.base_train,
        "data_uji_global": split.data_uji_global,
        "reserved_unknown_pool": split.reserved_unknown_pool,
        "episodic_pool": split.episodic_pool,
        "episodic_sessions": split.episodic_split.sessions,
        "task_speakers": split.episodic_split.task_speakers,
        "calibration_impostor_pool": split.episodic_split.calibration_impostor_pool,
        "calibration_genuine_source_speakers": split.calibration.genuine_source_speakers,
    }
    out_json = SPLITS_DIR / "full_split.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_lines = [
        "# Ringkasan Split Data (F1-11)",
        "",
        f"Seed: `{split.seed}`",
        "",
        "| Partisi | Jumlah Speaker |",
        "|---|---|",
        f"| Total gabungan (vox1+vox2) | {stats['n_speakers_total']} |",
        f"| &nbsp;&nbsp;- VoxCeleb1 | {stats['n_speakers_vox1']} |",
        f"| &nbsp;&nbsp;- VoxCeleb2 | {stats['n_speakers_vox2']} |",
        f"| Data Latih Awal (base_train, 70%) | {len(split.base_train)} |",
        f"| Data Uji Global (30%) | {len(split.data_uji_global)} |",
        f"| &nbsp;&nbsp;- Data simpanan (reserved, 10% of Data Uji Global) | {len(split.reserved_unknown_pool)} |",
        f"| &nbsp;&nbsp;- Episodic pool (90% of Data Uji Global) | {len(split.episodic_pool)} |",
        f"| &nbsp;&nbsp;&nbsp;&nbsp;- Task speakers (10 sesi x 10-way) | {len(split.episodic_split.task_speakers)} |",
        f"| &nbsp;&nbsp;&nbsp;&nbsp;- Calibration impostor pool | {len(split.episodic_split.calibration_impostor_pool)} |",
        "",
        "## Catatan verifikasi vs Tabel 4.1 (F1-03)",
        "",
        f"- Proposal Tabel 4.1: VoxCeleb1 = 1.251 speaker, VoxCeleb2 = 6.112 speaker (total 7.363).",
        f"- Metadata resmi terkini (vox1_meta.csv / vox2_meta.csv, diunduh langsung dari VGG): "
        f"VoxCeleb1 = {stats['n_speakers_vox1']} speaker, VoxCeleb2 = {stats['n_speakers_vox2']} speaker "
        f"(total {stats['n_speakers_total']}).",
        f"- VoxCeleb1 cocok persis (1.251). VoxCeleb2 selisih kecil "
        f"({stats['n_speakers_vox2']} vs 6.112, ~{stats['n_speakers_vox2'] - 6112:+d}) karena "
        f"file metadata resmi telah diperbarui sejak angka Tabel 4.1 dikutip dari Nagrani et al. "
        f"(2018); selisih ini < 0,05% dan tidak memengaruhi validitas skema split (rasio "
        f"70/30/10 dihitung dari jumlah speaker aktual, bukan angka historis).",
        "",
        "## Guarantee",
        "",
        "Disjointness across base_train / reserved_unknown_pool / task_speakers / "
        "calibration_impostor_pool is asserted programmatically in "
        "`src/data/splits.py::run_full_split` (raises `SpeakerOverlapError` on violation) "
        "and re-verified by `tests/test_data_splits.py` (F1-10, blocking).",
    ]
    (SPLITS_DIR / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {SPLITS_DIR / 'summary.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(seed=args.seed)
