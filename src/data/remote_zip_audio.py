"""Speaker-capped audio extraction from large remote ZIP archives, without
downloading the full archive.

Context: the official VoxCeleb1/2 audio distribution is gated behind a
request-based access agreement with Oxford VGG (see
scripts/download_voxceleb.py). While researching a smaller sample for
pipeline development (F1/F2), we found that `ProgramComputer/voxceleb` on
HuggingFace Hub re-hosts the exact same official archives, **ungated** and
under the same CC BY 4.0 license as the original dataset:

    vox1/vox1_dev_wav.zip   (~32.6 GB, WAV, 838+ dev speakers)
    vox2/vox2_aac_1.zip     (~50.0 GB, AAC/M4A, part of dev speakers)
    vox2/vox2_aac_2.zip     (~27.5 GB, AAC/M4A, remaining dev speakers)

These are served over HTTPS with `Accept-Ranges: bytes`, which means a ZIP
reader can fetch just the central directory (a few MB at the end of the
file) plus the individual byte ranges of the specific entries it wants --
via `remotezip.RemoteZip` -- without ever pulling the full multi-GB archive.

This module implements exactly that: given a set of wanted speaker IDs and
a per-speaker cap (see plan/04-tasks.md "1/4 data train" sizing discussion),
it lists the remote archive's entries once (cached to disk), groups them by
speaker, deterministically samples up to `cap` entries per speaker, and
downloads only those -- writing a resumable CSV manifest as it goes.

Measured throughput against this mirror (see chat log / commit history):
~600-650 ms per file with 4 concurrent workers; higher concurrency (8+) did
not help and occasionally caused read timeouts, so 4 is the default.
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests
from remotezip import RemoteZip

REPO_ROOT = Path(__file__).resolve().parents[2]
ZIP_INDEX_CACHE_DIR = REPO_ROOT / "data" / "raw" / "audio" / "_zip_index"


def _strip_extended_prefix(path_str: str) -> str:
    """Strip Windows' `\\\\?\\` extended-length-path prefix, if present."""
    if path_str.startswith("\\\\?\\"):
        return path_str[4:]
    return path_str


def _relpath_to_repo_root(path: Path) -> str:
    """os.path.relpath instead of Path.relative_to(REPO_ROOT): on Windows,
    Path.resolve() sometimes returns an extended-length path (`\\\\?\\C:\\...`)
    depending on the specific path's length/characters, which breaks both
    relative_to's strict subpath check AND plain os.path.relpath (which
    raises "path is on mount '\\\\?\\C:', start on mount 'C:'" when only one
    side has the prefix). Stripping the prefix from both sides first makes
    the comparison prefix-agnostic.

    If `path` is on a different drive than REPO_ROOT entirely (e.g. the
    dataset was downloaded to a secondary drive), os.path.relpath has no
    valid relative form and raises ValueError -- fall back to the absolute
    path string instead of crashing mid-download. Every consumer joins
    manifest paths via `REPO_ROOT / p`, and pathlib's `/` operator returns
    the right operand unchanged when it's already absolute, so an absolute
    fallback value here still round-trips correctly everywhere else.
    """
    p = _strip_extended_prefix(str(path))
    root = _strip_extended_prefix(str(REPO_ROOT))
    try:
        return os.path.relpath(p, root)
    except ValueError:
        return p


HF_BASE = "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main"


@dataclass(frozen=True)
class RemoteZipSource:
    name: str  # cache key, e.g. "vox1_dev_wav"
    url: str
    entry_regex: str  # must have named groups 'speaker_id' and (optionally) 'rest'


SOURCES: dict[str, RemoteZipSource] = {
    "vox1_dev_wav": RemoteZipSource(
        name="vox1_dev_wav",
        url=f"{HF_BASE}/vox1/vox1_dev_wav.zip",
        entry_regex=r"^wav/(?P<speaker_id>id\d+)/.+\.wav$",
    ),
    "vox2_aac_1": RemoteZipSource(
        name="vox2_aac_1",
        url=f"{HF_BASE}/vox2/vox2_aac_1.zip",
        entry_regex=r"^(?:dev/)?aac/(?P<speaker_id>id\d+)/.+\.m4a$",
    ),
    "vox2_aac_2": RemoteZipSource(
        name="vox2_aac_2",
        url=f"{HF_BASE}/vox2/vox2_aac_2.zip",
        entry_regex=r"^(?:dev/)?aac/(?P<speaker_id>id\d+)/.+\.m4a$",
    ),
}


def resolve_cdn_url(hf_url: str) -> str:
    """Follow the HF redirect once to get the direct CDN URL -- reusing this
    avoids paying the redirect-hop latency on every single range request,
    which matters a lot when making thousands of them."""
    resp = requests.head(hf_url, allow_redirects=True, timeout=30)
    resp.raise_for_status()
    return resp.url


def list_entries_by_speaker(source: RemoteZipSource, use_cache: bool = True) -> dict[str, list[str]]:
    """Read the remote ZIP's central directory once and group entry names by
    speaker_id. Cached to data/raw/audio/_zip_index/<source>.json because the
    central-directory read itself takes several seconds against a 30-50 GB
    remote file, and we don't want to pay that repeatedly across resumed runs.
    """
    cache_path = ZIP_INDEX_CACHE_DIR / f"{source.name}.json"
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    cdn_url = resolve_cdn_url(source.url)
    rz = RemoteZip(cdn_url, timeout=60)
    pattern = re.compile(source.entry_regex)

    by_speaker: dict[str, list[str]] = {}
    for name in rz.namelist():
        m = pattern.match(name)
        if m:
            by_speaker.setdefault(m.group("speaker_id"), []).append(name)

    ZIP_INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(by_speaker), encoding="utf-8")
    return by_speaker


def select_capped_entries(
    entries_by_speaker: dict[str, list[str]],
    wanted_speakers: list[str],
    cap: int,
    seed: int = 0,
) -> dict[str, list[str]]:
    """Deterministically sample up to `cap` entries per speaker (of those
    actually present in this archive -- a given speaker may live in a
    different archive, e.g. vox2_aac_1 vs vox2_aac_2)."""
    rng = random.Random(seed)
    selected: dict[str, list[str]] = {}
    for speaker_id in wanted_speakers:
        candidates = entries_by_speaker.get(speaker_id)
        if not candidates:
            continue
        candidates = sorted(candidates)
        rng.shuffle(candidates)
        selected[speaker_id] = candidates[:cap]
    return selected


def download_capped(
    source_key: str,
    wanted_speakers: list[str],
    cap: int,
    dest_dir: Path,
    manifest_path: Path,
    n_workers: int = 4,
    seed: int = 0,
) -> None:
    """Extract up to `cap` utterances per speaker from `source_key` for every
    speaker in `wanted_speakers` that is actually present in that archive,
    writing files under dest_dir/<source_key>/<speaker_id>/<basename>.
    Resumable: files already on disk (non-empty) are skipped.

    The manifest is (re)built from a directory scan at the end of the run
    (see `rebuild_manifest_from_disk`) rather than written incrementally
    while downloads are in flight. An earlier incremental-write version of
    this function lost its manifest rows (ended up with only the header)
    on a multi-hour run inside this OneDrive-synced repo folder -- the
    actual downloaded files were all intact, only the long-held open file
    handle to manifest.csv got clobbered, almost certainly by OneDrive
    trying to sync the file while it was being written to over ~1 hour.
    Scanning the directory afterwards sidesteps that failure mode entirely
    and is idempotent/self-healing regardless of what happens mid-run.
    """
    source = SOURCES[source_key]
    print(f"[{source_key}] reading remote central directory (cached after first run)...")
    entries_by_speaker = list_entries_by_speaker(source)
    print(f"[{source_key}] {len(entries_by_speaker)} speakers available in this archive")

    selected = select_capped_entries(entries_by_speaker, wanted_speakers, cap, seed=seed)
    total_planned = sum(len(v) for v in selected.values())
    print(f"[{source_key}] {len(selected)}/{len(wanted_speakers)} requested speakers found; "
          f"{total_planned} files planned (cap={cap}/speaker)")

    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    cdn_url = resolve_cdn_url(source.url)
    tls = threading.local()

    def get_rz() -> RemoteZip:
        if not hasattr(tls, "rz"):
            tls.rz = RemoteZip(cdn_url, timeout=60)
        return tls.rz

    def local_path_for(entry_name: str, speaker_id: str) -> Path:
        basename = entry_name.split("/")[-1]
        parent = entry_name.rsplit("/", 2)[-2] if entry_name.count("/") >= 2 else "root"
        return dest_dir / speaker_id / f"{parent}_{basename}"

    jobs = [
        (speaker_id, entry_name)
        for speaker_id, entries in selected.items()
        for entry_name in entries
    ]

    def fetch_one(job: tuple[str, str]) -> tuple[str, str, Path, int] | None:
        speaker_id, entry_name = job
        dest = local_path_for(entry_name, speaker_id).resolve()
        if dest.exists() and dest.stat().st_size > 0:
            return None  # already done (resumability)
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = get_rz().read(entry_name)
        dest.write_bytes(data)
        return speaker_id, entry_name, dest, len(data)

    n_done, n_skipped, n_failed = 0, 0, 0
    completed_rows: list[tuple[str, str, str, str, int]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(fetch_one, job): job for job in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            speaker_id, entry_name = futures[fut]
            try:
                result = fut.result()
            except Exception as e:  # noqa: BLE001 -- log and continue, don't abort the whole run
                n_failed += 1
                print(f"  [warn] failed {speaker_id}/{entry_name}: {e}")
                continue
            if result is None:
                n_skipped += 1
                continue
            spk, name, dest, n_bytes = result
            completed_rows.append((source_key, spk, name, _relpath_to_repo_root(dest), n_bytes))
            n_done += 1
            if i % 50 == 0 or i == len(jobs):
                elapsed = time.time() - t0
                print(f"  [{source_key}] {i}/{len(jobs)} processed "
                      f"({n_done} downloaded, {n_skipped} skipped, {n_failed} failed) "
                      f"in {elapsed/60:.1f} min")

    # Single short-lived manifest write at the end (in-memory rows -> disk in
    # one shot), with a directory-scan rebuild as a self-healing fallback if
    # this write is somehow interrupted (e.g. OneDrive sync -- see docstring).
    try:
        _write_manifest_rows(manifest_path, completed_rows, append=manifest_path.exists())
    except OSError as e:
        print(f"  [warn] manifest write failed ({e}); rebuilding from disk instead")
    finally:
        rebuild_manifest_from_disk(dest_dir, source_key, manifest_path)

    print(f"[{source_key}] done: {n_done} downloaded, {n_skipped} already present, "
          f"{n_failed} failed, manifest -> {manifest_path}")


def _write_manifest_rows(
    manifest_path: Path, rows: list[tuple[str, str, str, str, int]], append: bool
) -> None:
    mode = "a" if append else "w"
    write_header = not (append and manifest_path.exists() and manifest_path.stat().st_size > 0)
    with manifest_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["source", "speaker_id", "entry_name", "path", "n_bytes"])
        writer.writerows(rows)


def rebuild_manifest_from_disk(dest_dir: Path, source_key: str, manifest_path: Path) -> None:
    """Regenerate manifest_path from what's actually present under
    dest_dir/<speaker_id>/*.{wav,m4a} -- idempotent and self-healing, used
    as the final step of download_capped regardless of whether the
    in-memory incremental write above succeeded.

    Merge-aware: dest_dir/manifest_path can be shared across multiple
    sources (e.g. vox2_aac_1 and vox2_aac_2 both extracting into the same
    base_train_capped/ directory) -- files already recorded in an existing
    manifest keep their original `source`/`entry_name`; only files not yet
    recorded get tagged with the *current* `source_key` being processed, and
    an empty `entry_name` (unrecoverable from a bare directory scan).
    """
    known: dict[str, tuple[str, str]] = {}  # path -> (source, entry_name)
    if manifest_path.exists():
        with manifest_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                known[row["path"]] = (row["source"], row["entry_name"])

    files = sorted(p for p in dest_dir.rglob("*") if p.is_file() and p.suffix in (".wav", ".m4a"))
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "speaker_id", "entry_name", "path", "n_bytes"])
        for p in files:
            rel_path = _relpath_to_repo_root(p)
            speaker_id = p.parent.name
            src, entry_name = known.get(rel_path, (source_key, ""))
            writer.writerow([src, speaker_id, entry_name, rel_path, p.stat().st_size])
