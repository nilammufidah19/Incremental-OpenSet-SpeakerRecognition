"""Open-set threshold calibration (F5-06/07) [4.6 poin 4].

Computes the fixed EER-based distance threshold that separates "known
speaker" from "unknown speaker" decisions (F5-08), using genuine/impostor
pairs built the same way the actual open-set decision will be made at
inference: min-distance from a query embedding to the *nearest* registered
prototype.

  - genuine distances:  held-out samples of speakers who DO have a
                        registered prototype (their nearest prototype
                        should be their own -> small distance)
  - impostor distances: samples of speakers who have NO registered
                        prototype at all (nearest prototype is necessarily
                        someone else's -> should be large)

Per Bab 4.6, this calibration must run once (on data disjoint from the
episodic task/reserved pools) and the resulting threshold is then reused,
fixed, throughout the FSCIL evaluation -- see F5-07's leakage test and
plan/02-requirements.md DR-05.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.models.fusion import GatedAttentionFusion
from src.prototypical.data import split_raw_embedding


@dataclass
class CalibrationResult:
    threshold: float
    eer: float
    thresholds_swept: np.ndarray
    far_curve: np.ndarray
    frr_curve: np.ndarray
    genuine_distances: np.ndarray
    impostor_distances: np.ndarray
    # Experiment 3a (exp3a_lowfrr): which operating-point strategy produced
    # `threshold`, and the achieved FAR/FRR there. Defaults keep the original
    # EER-only contract for every pre-Experiment-3 caller.
    strategy: str = "eer"
    target_frr: float | None = None
    far_at_threshold: float | None = None
    frr_at_threshold: float | None = None


def _fuse_all(fusion_model: GatedAttentionFusion, raw_embeddings: list[np.ndarray], device: str) -> torch.Tensor:
    raw = torch.from_numpy(np.stack(raw_embeddings)).float().to(device)
    ecapa_part, whisper_part = split_raw_embedding(raw)
    with torch.no_grad():
        return fusion_model(ecapa_part, whisper_part)


def build_genuine_impostor_distances(
    fusion_model: GatedAttentionFusion,
    genuine_index: dict[str, list[np.ndarray]],
    impostor_index: dict[str, list[np.ndarray]],
    enrollment_k: int = 1,
    seed: int = 0,
    device: str = "cpu",
    score_normalizer=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build genuine/impostor min-distance-to-nearest-prototype arrays.

    `genuine_index`/`impostor_index`: speaker_id -> list of raw
    [ecapa;whisper] embeddings (see src/prototypical/data.py). Speakers in
    `genuine_index` must have at least `enrollment_k + 1` samples (some
    held out from prototype building, to test against). Every sample of
    every speaker in `impostor_index` is used as an impostor query.

    `score_normalizer` (Experiment 3b, exp3b_asnorm): optional
    src/prototypical/score_norm.py::ASNorm. When given, the query->prototype
    distance matrix is normalized against the cohort BEFORE the min is taken,
    so the calibrated threshold lives in the same normalized-score space the
    runtime decision (ContinualLearningManager) uses.
    """
    fusion_model.eval()
    rng = np.random.default_rng(seed)

    prototypes = []
    genuine_query_raw = []
    for speaker_id, raw_list in genuine_index.items():
        if len(raw_list) < enrollment_k + 1:
            continue
        raw_list = list(raw_list)
        rng.shuffle(raw_list)
        enrollment, held_out = raw_list[:enrollment_k], raw_list[enrollment_k:]

        fused_enrollment = _fuse_all(fusion_model, enrollment, device)
        prototypes.append(fused_enrollment.mean(dim=0))
        genuine_query_raw.extend(held_out)

    if not prototypes:
        raise ValueError(
            f"No genuine speaker had >= {enrollment_k + 1} samples; "
            "cannot build prototypes and held-out queries."
        )
    prototype_matrix = torch.stack(prototypes)  # (n_genuine_speakers, dim)

    impostor_query_raw = [raw for raws in impostor_index.values() for raw in raws]
    if not impostor_query_raw:
        raise ValueError("impostor_index has no samples at all.")

    genuine_fused = _fuse_all(fusion_model, genuine_query_raw, device)
    impostor_fused = _fuse_all(fusion_model, impostor_query_raw, device)

    if score_normalizer is not None:
        proto_np = prototype_matrix.cpu().numpy()
        genuine_dists = score_normalizer.normalize(genuine_fused.cpu().numpy(), proto_np).min(axis=-1)
        impostor_dists = score_normalizer.normalize(impostor_fused.cpu().numpy(), proto_np).min(axis=-1)
        return genuine_dists, impostor_dists

    genuine_dists = torch.cdist(genuine_fused, prototype_matrix, p=2).min(dim=-1).values
    impostor_dists = torch.cdist(impostor_fused, prototype_matrix, p=2).min(dim=-1).values
    return genuine_dists.cpu().numpy(), impostor_dists.cpu().numpy()


def compute_far_frr(
    genuine_distances: np.ndarray, impostor_distances: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """FAR(theta) = P(impostor distance < theta)  [Pers. 3.12]
    FRR(theta) = P(genuine distance >= theta)     [Pers. 3.13]"""
    far = np.array([(impostor_distances < t).mean() for t in thresholds])
    frr = np.array([(genuine_distances >= t).mean() for t in thresholds])
    return far, frr


def find_eer_threshold(
    genuine_distances: np.ndarray, impostor_distances: np.ndarray, n_thresholds: int = 1000
) -> CalibrationResult:
    """Sweep thresholds across the full observed distance range and return
    the one where FAR and FRR are closest (Equal Error Rate point)."""
    all_distances = np.concatenate([genuine_distances, impostor_distances])
    thresholds = np.linspace(all_distances.min(), all_distances.max(), n_thresholds)
    far, frr = compute_far_frr(genuine_distances, impostor_distances, thresholds)

    eer_idx = int(np.argmin(np.abs(far - frr)))
    eer = float((far[eer_idx] + frr[eer_idx]) / 2)
    threshold = float(thresholds[eer_idx])

    return CalibrationResult(
        threshold=threshold,
        eer=eer,
        thresholds_swept=thresholds,
        far_curve=far,
        frr_curve=frr,
        genuine_distances=genuine_distances,
        impostor_distances=impostor_distances,
    )


def find_operating_point(
    genuine_distances: np.ndarray,
    impostor_distances: np.ndarray,
    strategy: str = "eer",
    target_frr: float = 0.05,
    n_thresholds: int = 1000,
) -> CalibrationResult:
    """Experiment 3a (exp3a_lowfrr): choose the operating threshold.

    strategy="eer"        -- original behaviour (find_eer_threshold): the
                             threshold where FAR = FRR.
    strategy="target_frr" -- low-FRR operating point: the threshold is the
                             (1 - target_frr) quantile of the GENUINE
                             distances, i.e. it false-rejects ~target_frr of
                             genuine queries. Motivated by the Experiment 2
                             threshold study: the FSCIL metric has no unknown
                             queries, so every rejection at the EER point is
                             a pure loss; a permissive threshold that keeps
                             ~95% of genuine queries scored open@p95 > open@EER
                             for every fusion weight tested.

    The EER is always computed and reported (it is a property of the score
    distributions, independent of the operating point chosen).
    """
    if strategy == "eer":
        return find_eer_threshold(genuine_distances, impostor_distances, n_thresholds)
    if strategy != "target_frr":
        raise ValueError(f"strategy must be 'eer' or 'target_frr', got {strategy!r}")
    if not (0.0 < target_frr < 1.0):
        raise ValueError(f"target_frr must be in (0, 1), got {target_frr}")

    eer_result = find_eer_threshold(genuine_distances, impostor_distances, n_thresholds)
    threshold = float(np.quantile(genuine_distances, 1.0 - target_frr))
    far_at = float((impostor_distances < threshold).mean())
    frr_at = float((genuine_distances >= threshold).mean())

    return CalibrationResult(
        threshold=threshold,
        eer=eer_result.eer,
        thresholds_swept=eer_result.thresholds_swept,
        far_curve=eer_result.far_curve,
        frr_curve=eer_result.frr_curve,
        genuine_distances=genuine_distances,
        impostor_distances=impostor_distances,
        strategy="target_frr",
        target_frr=target_frr,
        far_at_threshold=far_at,
        frr_at_threshold=frr_at,
    )


def calibrate_threshold(
    fusion_model: GatedAttentionFusion,
    genuine_index: dict[str, list[np.ndarray]],
    impostor_index: dict[str, list[np.ndarray]],
    enrollment_k: int = 1,
    seed: int = 0,
    device: str = "cpu",
    n_thresholds: int = 1000,
    strategy: str = "eer",
    target_frr: float = 0.05,
    score_normalizer=None,
) -> CalibrationResult:
    """End-to-end F5-06: build genuine/impostor distances, sweep, return the
    fixed operating threshold.

    Defaults reproduce the original EER calibration exactly; `strategy`/
    `target_frr` (Experiment 3a) and `score_normalizer` (Experiment 3b) are
    the feature-flagged extensions driven by ExperimentConfig.
    """
    genuine_d, impostor_d = build_genuine_impostor_distances(
        fusion_model, genuine_index, impostor_index, enrollment_k, seed, device,
        score_normalizer=score_normalizer,
    )
    return find_operating_point(genuine_d, impostor_d, strategy, target_frr, n_thresholds)
