#!/usr/bin/env python
"""Experiment 6 -- materialize the CHOSEN Whisper readout into its own cache
namespace, so the official evaluation can run on it unchanged.

Why this exists. The G6.1 gate compares many (slice, transform) pairs, but the
official run needs exactly one, applied identically everywhere a second-
backbone embedding is read: the AS-Norm cohort, the calibration trials, the
FSCIL prototypes, and every query. Threading a fitted transform through all
those call sites invites the two failure modes that matter most here --
applying it in one place and not another, or fitting it twice on different
data. Writing the transformed vectors into a namespace of their own removes
both: downstream code just reads a cache, exactly as it does for ReDimNet.

This is NOT the thing the whisper_pmfa docstring warns against. That warning
is about baking a choice behind a ~55 min GPU recompute. Here the raw cache
stays, and regenerating a different choice costs seconds of CPU.

Leakage discipline: the transform is fit on base_train utterances only, drawn
with the same round-robin and seed the gate uses, so the fitted statistics
never see task or reserved-pool speakers. This is tier 1 in
docs/experiment-6-plan.md section 2 -- the same standing as the AS-Norm cohort
and the calibration threshold already in the system.

Usage:
    .venv/Scripts/python.exe scripts/exp6_materialize_whisper_space.py \
        --source whisper_pmfa_all --slice layer3 --transform pca_whiten_full \
        --out whisper_best
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import exp4_reaudit_geometry as geo  # noqa: E402
import exp6_pmfa_readout_gate as gate  # noqa: E402
from src.features.cache import CACHE_DIR, _cache_path  # noqa: E402

MANIFESTS = [
    "data/raw/audio/vox1_sample/manifest.csv",
    "data/raw/audio/base_train_capped/manifest.csv",
    "data/raw/audio/eval_capped/manifest.csv",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="whisper_pmfa_all",
                    choices=sorted(gate.BACKBONE_LAYERS))
    ap.add_argument("--slice", dest="slice_name", default="full")
    ap.add_argument("--pre", default="none",
                    help="pre-transform applied before the slice's transform: "
                         "none | per_layer_l2 | per_layer_whiten_d<K>")
    ap.add_argument("--transform", default="wccn")
    ap.add_argument("--out", default="whisper_best")
    args = ap.parse_args()

    gate.LAYERS = gate.BACKBONE_LAYERS[args.source]
    t0 = time.time()

    out_dir = CACHE_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- fit the transform on base_train, exactly as the gate does ----------
    bt = geo.load_base_train()
    fit_paths, fit_spk = geo.round_robin(bt, geo.FIT_SIZE, seed=1)
    fit_source = geo.embed(fit_paths, args.source)

    # The pre-transform is fitted too (per_layer_whiten), so it must be built
    # from the base_train fit set ONCE and then reused for every utterance --
    # refitting per batch would silently give each batch its own space.
    if args.pre == "none":
        pre = lambda x: gate.slice_variant(x, args.slice_name)  # noqa: E731
    elif args.pre == "per_layer_l2":
        pre = lambda x: gate.per_layer_l2(gate.slice_variant(x, args.slice_name))  # noqa: E731
    elif args.pre.startswith("per_layer_whiten_d"):
        d = int(args.pre[len("per_layer_whiten_d"):])
        whitener = gate.make_per_layer_whitener(
            gate.slice_variant(fit_source, args.slice_name), d
        )
        pre = lambda x: whitener(gate.slice_variant(x, args.slice_name))  # noqa: E731
    else:
        raise SystemExit(f"unknown --pre {args.pre!r}")

    fit_raw = pre(fit_source)
    print(f"fit set: {len(fit_paths)} utts / {len(set(fit_spk))} speakers, "
          f"slice={args.slice_name} pre={args.pre} dim={fit_raw.shape[1]}")

    transforms = {t.name: t for t in geo.build_transforms(fit_raw, fit_spk)}
    if args.transform not in transforms:
        raise SystemExit(f"unknown transform {args.transform!r}; "
                         f"available: {sorted(transforms)}")
    transform = transforms[args.transform]

    probe = transform(fit_raw[:2])
    print(f"transform={args.transform}: {fit_raw.shape[1]} -> {probe.shape[1]} dims")

    # ---- apply to every cached utterance ------------------------------------
    manifest = pd.concat([pd.read_csv(REPO_ROOT / m) for m in MANIFESTS], ignore_index=True)
    paths = [REPO_ROOT / p for p in manifest["path"]]
    print(f"materializing {len(paths)} utterances -> cache/{args.out}/ ...")

    written = 0
    batch = 512
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        out = transform(pre(geo.embed(chunk, args.source))).astype(np.float32)
        for path, vec in zip(chunk, out):
            np.save(_cache_path(path, args.out), vec)
            written += 1
        if (i // batch) % 5 == 0:
            print(f"  {written}/{len(paths)}", flush=True)

    meta = {
        "source_backbone": args.source, "slice": args.slice_name,
        "pre_transform": args.pre, "transform": args.transform,
        "source_dim": int(fit_raw.shape[1]), "output_dim": int(probe.shape[1]),
        "fit_set": {"n_utterances": len(fit_paths),
                    "n_speakers": int(len(set(fit_spk))),
                    "pool": "base_train", "seed": 1},
        "n_written": written,
        "elapsed_minutes": (time.time() - t0) / 60,
    }
    (REPO_ROOT / "experiments" / f"materialized_{args.out}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nwrote {written} vectors of dim {probe.shape[1]} "
          f"({(time.time() - t0) / 60:.1f} min)")
    print(f"provenance saved to experiments/materialized_{args.out}.json")


if __name__ == "__main__":
    main()
