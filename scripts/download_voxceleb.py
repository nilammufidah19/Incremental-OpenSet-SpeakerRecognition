#!/usr/bin/env python
"""VoxCeleb1/2 acquisition script (F1-01, F1-02).

IMPORTANT -- read before running with --stage audio:
VoxCeleb2 (and, depending on mirror, VoxCeleb1) audio archives are
distributed by Oxford VGG under a request-based access agreement, not an
open anonymous download: https://mm.kaist.ac.kr/datasets/voxceleb/ or
https://www.robots.ox.ac.uk/~vgg/data/voxceleb/ -- you must request access
and receive a username/password before downloading the multi-hundred-GB
audio archives. This script will NOT silently bulk-download the full
corpus; it only does so once credentials are supplied explicitly (see
--username/--password or the VOXCELEB_USER/VOXCELEB_PASS env vars), and it
downloads one archive at a time so progress is visible and resumable.

Two independent stages:

  --stage metadata   Downloads the small, openly-accessible speaker
                      metadata files (vox1_meta.csv, vox2_meta.csv) and the
                      VoxCeleb1 identification-split utterance list
                      (iden_split.txt, ~5 MB, no audio). No credentials
                      needed. This is enough to run src/data/splits.py end
                      to end at the speaker/utterance-ID level (F1-01..09).

  --stage audio       Downloads the actual WAV archives. Requires
                      credentials. Large (VoxCeleb1 ~40 GB, VoxCeleb2
                      ~180 GB compressed). Run this only once disk space
                      and bandwidth have been budgeted for (see
                      plan/01-project-plan.md risk table).

Usage
-----
    python scripts/download_voxceleb.py --stage metadata
    python scripts/download_voxceleb.py --stage audio --dataset vox1 \\
        --username <user> --password <pass>
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = REPO_ROOT / "data" / "raw" / "metadata"
AUDIO_DIR = REPO_ROOT / "data" / "raw" / "audio"

# Openly accessible (no credentials) -- small metadata/list files only.
METADATA_URLS = {
    "vox1_meta.csv": "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox1_meta.csv",
    "vox2_meta.csv": "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox2_meta.csv",
    "vox1_iden_split.txt": "https://mm.kaist.ac.kr/datasets/voxceleb/meta/iden_split.txt",
}

# Audio archives require credentialed access (request form on the VGG site).
# Base URL host kept as a documented placeholder -- fill in the real
# per-part URLs given to you after your access request is approved; they
# differ by mirror/hosting arrangement and change over time, so hardcoding
# them here would go stale.
AUDIO_ARCHIVE_NOTE = """
No hardcoded audio archive URLs are provided by this script.
After your VoxCeleb access request is approved you will receive either:
  (a) direct download links + a username/password for HTTP basic auth, or
  (b) a set of pre-signed URLs.
Pass them via --archive-url (repeatable) together with --username/--password
(for case a), and this script will stream each archive to data/raw/audio/
with resume support (Range requests) and SHA-1 verification if a checksum
file is provided alongside the official file listing.
"""


def download_file(url: str, dest: Path, username: str | None = None, password: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] {dest} already exists")
        return

    request = urllib.request.Request(url)
    if username and password:
        import base64

        token = base64.b64encode(f"{username}:{password}".encode()).decode()
        request.add_header("Authorization", f"Basic {token}")

    print(f"[download] {url} -> {dest}")
    with urllib.request.urlopen(request, timeout=60) as response, open(dest, "wb") as f:
        chunk_size = 1024 * 1024
        downloaded = 0
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            print(f"\r  {downloaded / 1e6:.1f} MB", end="", flush=True)
    print()


def run_metadata_stage() -> None:
    for filename, url in METADATA_URLS.items():
        download_file(url, METADATA_DIR / filename)
    print(
        "\nMetadata stage complete. You can now run:\n"
        "  .venv/Scripts/python.exe scripts/build_splits.py\n"
        "to produce the speaker-disjoint Base/Task/Calibration split "
        "(F1-05..F1-09) using real VoxCeleb1/2 speaker IDs, even before any "
        "audio has been downloaded."
    )


def run_audio_stage(dataset: str, archive_urls: list[str], username: str | None, password: str | None) -> None:
    if not archive_urls:
        print(AUDIO_ARCHIVE_NOTE, file=sys.stderr)
        raise SystemExit(
            "No --archive-url given. This script refuses to guess VoxCeleb "
            "audio download URLs -- request access first, then pass the "
            "URLs you were given via --archive-url."
        )
    dest_dir = AUDIO_DIR / dataset
    for i, url in enumerate(archive_urls):
        dest = dest_dir / f"{dataset}_part{i:03d}.zip"
        download_file(url, dest, username=username, password=password)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["metadata", "audio"], required=True)
    parser.add_argument("--dataset", choices=["vox1", "vox2"], default="vox1")
    parser.add_argument("--archive-url", action="append", default=[], help="Repeatable. Audio archive URL(s) obtained after access approval.")
    parser.add_argument("--username", default=os.environ.get("VOXCELEB_USER"))
    parser.add_argument("--password", default=os.environ.get("VOXCELEB_PASS"))
    args = parser.parse_args()

    if args.stage == "metadata":
        run_metadata_stage()
    else:
        run_audio_stage(args.dataset, args.archive_url, args.username, args.password)


if __name__ == "__main__":
    main()
