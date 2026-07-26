#!/usr/bin/env python
"""Diagnostic: decompose the proposed system's FSCIL errors into
(a) false rejections (nearest prototype was the correct speaker, but
distance exceeded the calibrated threshold) vs (b) genuine confusions
(nearest prototype was a different speaker entirely, threshold-independent).

This directly tests the two hypotheses raised for F10's baseline-outperforms
-proposed finding:
  H1: closed-set Accuracy structurally favors systems that never reject.
  H2: small-scale training (71 speakers) limits embedding generalization.

Usage:
    .venv/Scripts/python.exe scripts/diagnose_accuracy_gap.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.fusion import FUSION_DIM, GatedAttentionFusion  # noqa: E402
from src.prototypical.data import ECAPA_DIM, WHISPER_DIM  # noqa: E402
from src.system import SpeakerIdentificationSystem  # noqa: E402

SPLITS_PATH = REPO_ROOT / "data" / "splits" / "full_split.json"
CKPT_PATH = REPO_ROOT / "experiments" / "checkpoints" / "fusion_A3_full_eval.ckpt"
SUMMARY_PATH = REPO_ROOT / "experiments" / "full_evaluation_summary.json"
EVAL_MANIFEST = REPO_ROOT / "data" / "raw" / "audio" / "eval_capped" / "manifest.csv"

K_SHOT, N_QUERY = 1, 5


def main() -> None:
    split = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    threshold = summary["calibration_threshold"]
    print(f"calibrated threshold = {threshold:.4f} (calibration EER = {summary['calibration_eer']:.4f})\n")

    manifest = pd.read_csv(EVAL_MANIFEST)
    sessions = split["episodic_sessions"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fusion_model = GatedAttentionFusion(ECAPA_DIM, WHISPER_DIM, FUSION_DIM, mode="fusion")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    fusion_model.load_state_dict(ckpt["state_dict"])
    fusion_model.to(device)

    system = SpeakerIdentificationSystem(
        fusion_model=fusion_model, threshold=threshold, continual_mode="running_average"
    )

    rng = random.Random(0)
    task_query_paths: dict[str, list[Path]] = {}

    n_correct = 0
    n_false_reject = 0   # nearest prototype correct, but rejected as unknown
    n_confusion = 0       # nearest prototype wrong (regardless of accept/reject)
    n_total = 0
    genuine_own_distances = []   # distance from query to its OWN (correct) prototype
    genuine_nearest_other = []   # distance from query to nearest OTHER (wrong) prototype

    for session_speakers in sessions:
        for speaker_id in session_speakers:
            paths = manifest.loc[manifest["speaker_id"] == speaker_id, "path"].tolist()
            paths = [REPO_ROOT / p for p in paths]
            if len(paths) < K_SHOT + 1:
                continue
            rng.shuffle(paths)
            support, query = paths[:K_SHOT], paths[K_SHOT : K_SHOT + N_QUERY]
            system.enroll(speaker_id, support)
            task_query_paths[speaker_id] = query

    # second pass: query every enrolled speaker's held-out utterances against
    # the FULL final database (all 10 sessions enrolled), recording diagnostics
    all_speaker_ids = system.database.speaker_ids
    prototypes = system.database.prototypes_matrix()

    for speaker_id, query_paths in task_query_paths.items():
        if speaker_id not in all_speaker_ids:
            continue
        own_idx = all_speaker_ids.index(speaker_id)
        for query_path in query_paths:
            embedding = system.embed(query_path)
            distances = np.linalg.norm(prototypes - embedding, axis=1)
            nearest_idx = int(np.argmin(distances))
            nearest_dist = distances[nearest_idx]
            own_dist = distances[own_idx]

            other_mask = np.ones(len(distances), dtype=bool)
            other_mask[own_idx] = False
            nearest_other_dist = distances[other_mask].min() if other_mask.any() else float("nan")

            genuine_own_distances.append(float(own_dist))
            genuine_nearest_other.append(float(nearest_other_dist))

            n_total += 1
            nearest_is_correct = nearest_idx == own_idx
            is_accepted = nearest_dist < threshold

            if nearest_is_correct and is_accepted:
                n_correct += 1
            elif nearest_is_correct and not is_accepted:
                n_false_reject += 1
            else:
                n_confusion += 1

    print(f"n_total queries: {n_total}")
    print(f"  correct (nearest=own AND accepted):        {n_correct} ({n_correct/n_total:.3f})")
    print(f"  false-rejected (nearest=own BUT rejected):  {n_false_reject} ({n_false_reject/n_total:.3f})")
    print(f"  confusion (nearest != own, any threshold):  {n_confusion} ({n_confusion/n_total:.3f})")
    print()
    print(f"  closed-set-equivalent accuracy (nearest=own regardless of threshold): "
          f"{(n_correct+n_false_reject)/n_total:.3f}")
    print(f"  actual open-set accuracy (current F10 metric):                       "
          f"{n_correct/n_total:.3f}")
    print()
    print(f"mean distance query->OWN prototype:        {np.mean(genuine_own_distances):.4f} "
          f"(std {np.std(genuine_own_distances):.4f})")
    print(f"mean distance query->NEAREST OTHER prototype: {np.mean(genuine_nearest_other):.4f} "
          f"(std {np.std(genuine_nearest_other):.4f})")
    print(f"threshold: {threshold:.4f}")
    print(f"fraction of genuine own-distances >= threshold (i.e. would be rejected): "
          f"{np.mean(np.array(genuine_own_distances) >= threshold):.3f}")


if __name__ == "__main__":
    main()
