#!/usr/bin/env python
"""F7-02/03 + F8-08/09 + F9 + F10 + F11: end-to-end functional-scale run.

IMPORTANT -- scale caveat (consistent with every prior phase in this
project): this runs the REAL mechanism (training, FSCIL evaluation,
ablation, baseline comparison, statistical testing) on REAL VoxCeleb1/2
audio for the REAL `task_speakers`/`episodic_sessions` defined in
data/splits/full_split.json -- but at reduced repetition count (N_REPS,
default 5 instead of the proposal's 10) and reduced base-training episode
count, to fit a single working session. Every number this script prints is
genuine (not fabricated/simulated), just computed at a smaller compute
budget than the full thesis-reportable run would use.

Usage:
    .venv/Scripts/python.exe scripts/run_full_evaluation.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.experiments import active as active_experiment  # noqa: E402
from src.data.splits import split_reserved_pool_halves  # noqa: E402
from src.evaluation.ablation import FUSION_ABLATION_MODES, train_fusion_for_ablation  # noqa: E402
from src.evaluation.baseline_system import CLOSED_SET_THRESHOLD, SingleBackboneSystem, fit_lda_whitener  # noqa: E402
from src.evaluation.fscil import run_fscil_detailed  # noqa: E402
from src.evaluation.metrics import auroc, average_accuracy, forgetting_measure, tar_at_far  # noqa: E402
from src.evaluation.statistics import bonferroni_correction, paired_significance_test  # noqa: E402
from src.features.cache import is_cached  # noqa: E402
from src.models.fusion import FUSION_DIM, GatedAttentionFusion, ScoreFusionEmbed  # noqa: E402
from src.prototypical.calibration import calibrate_threshold, find_eer_threshold  # noqa: E402
from src.prototypical.data import ECAPA_DIM, WHISPER_DIM, build_raw_embedding_index, split_raw_embedding  # noqa: E402
from src.prototypical.score_norm import ASNorm, DualASNorm, build_cohort, split_cohort_and_genuine  # noqa: E402
from src.system import SpeakerIdentificationSystem  # noqa: E402
from src.utils.seed import SEED_LIST  # noqa: E402
from src.utils.tracking import log_run  # noqa: E402

SPLITS_PATH = REPO_ROOT / "data" / "splits" / "full_split.json"
CHECKPOINT_DIR = REPO_ROOT / "experiments" / "checkpoints"
RESULTS_PATH = REPO_ROOT / "experiments" / "full_evaluation_summary.json"

# All experiment-varying knobs come from the active experiment tag
# (src/experiments.py). Flip ACTIVE_EXPERIMENT there -- or set the
# ACTIVE_EXPERIMENT env var -- to switch between Experiment 1 (frozen
# residual-init) and baseline_v0 (random-init, 500 episodes), etc.
EXP = active_experiment()
N_WAY, K_SHOT, N_QUERY = EXP.n_way, EXP.k_shot, EXP.n_query
N_TRAIN_EPISODES = EXP.n_train_episodes
RESIDUAL_INIT = EXP.residual_init
N_REPS = EXP.n_reps


def build_base_train_index() -> dict:
    frames = [
        pd.read_csv(REPO_ROOT / "data/raw/audio/vox1_sample/manifest.csv"),
        pd.read_csv(REPO_ROOT / "data/raw/audio/base_train_capped/manifest.csv"),
    ]
    manifest = pd.concat(frames, ignore_index=True)
    # Experiment 5 (score_norm strategy): filter by the ACTUAL second backbone's
    # cache; every pre-exp5 strategy keeps the original "whisper" check so old
    # tags stay bit-identical.
    second_cache = EXP.whisper_backbone if EXP.fusion_strategy == "score_norm" else "whisper"
    manifest["_cached"] = manifest["path"].apply(
        lambda p: is_cached(REPO_ROOT / p, "ecapa") and is_cached(REPO_ROOT / p, second_cache)
    )
    manifest = manifest[manifest["_cached"]]
    counts = manifest["speaker_id"].value_counts()
    eligible = counts[counts >= K_SHOT + N_QUERY].index.tolist()
    manifest = manifest[manifest["speaker_id"].isin(eligible)]
    print(f"base_train index: {manifest['speaker_id'].nunique()} speakers, {len(manifest)} utterances (cached-only)")
    return build_raw_embedding_index(manifest, whisper_backbone=EXP.whisper_backbone)


def build_speaker_audio_paths(manifest: pd.DataFrame, wanted_speakers: list[str]) -> dict[str, list[Path]]:
    out = {}
    for spk in wanted_speakers:
        rows = manifest.loc[manifest["speaker_id"] == spk, "path"].tolist()
        if rows:
            out[spk] = [REPO_ROOT / p for p in rows]
    return out


def main() -> None:
    t_start = time.time()
    split = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"ACTIVE EXPERIMENT: {EXP.id} -- {EXP.title}")
    print(f"device={device}, N_REPS={N_REPS}, N_TRAIN_EPISODES={N_TRAIN_EPISODES}, "
          f"RESIDUAL_INIT={RESIDUAL_INIT}, continual_mode={EXP.continual_mode}\n")

    # ---- F7-02: base "training" of A1/A2/A3 (F9 group A) ----
    base_train_index = build_base_train_index()
    fusion_models: dict = {}
    if EXP.fusion_strategy == "score":
        # Experiment 2a: no trainable fusion at all -- A1/A2/A3 are parameter-free
        # score-fusion embedders (ecapa_only=w1, whisper_only=w0, fusion=w).
        for name, mode in FUSION_ABLATION_MODES.items():
            print(f"building {name} (mode={mode}, score-fusion w_ecapa={EXP.score_fusion_weight})...")
            fusion_models[name] = ScoreFusionEmbed(mode, weight=EXP.score_fusion_weight).to(device)
    elif EXP.fusion_strategy == "score_norm":
        # Experiment 5 (exp5b): ONE parameter-free dual-space embedder for every
        # arm -- [e_ecapa_hat ; e_2nd_hat] -- and the ablation happens purely in
        # the DualASNorm weight (A3=w, A1=1.0, A2=0.0; see make_normalizer).
        # The 0.5/0.5 concat scaling is irrelevant: per-half constant scale
        # cancels in the z-scores (tests/test_experiment5.py).
        print(f"building dual-space embedder (second backbone={EXP.whisper_backbone}, "
              f"norm-fusion w_ecapa={EXP.score_fusion_weight})...")
        for name in FUSION_ABLATION_MODES:
            fusion_models[name] = ScoreFusionEmbed("fusion", weight=0.5).to(device)
    else:
        for name, mode in FUSION_ABLATION_MODES.items():
            print(f"training {name} (mode={mode})...")
            fusion_models[name] = train_fusion_for_ablation(
                mode, base_train_index, N_WAY, K_SHOT, N_QUERY, N_TRAIN_EPISODES,
                seed=0, device=device, residual_init=RESIDUAL_INIT,
            )
    print()

    # ---- F7-03: checkpoint the proposed system's fusion (A3) ----
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": fusion_models["A3_fusion"].state_dict(), "mode": "fusion"},
        CHECKPOINT_DIR / "fusion_A3_full_eval.ckpt",
    )

    # ---- F5-06/F1-09: calibrate threshold (genuine=base_train, impostor=calibration sample) ----
    eval_manifest = pd.read_csv(REPO_ROOT / "data/raw/audio/eval_capped/manifest.csv")
    impostor_ids = set(split["calibration_impostor_pool"])
    impostor_manifest = eval_manifest[eval_manifest["speaker_id"].isin(impostor_ids)]
    print(f"calibration impostor pool available: {impostor_manifest['speaker_id'].nunique()} speakers")
    impostor_index = build_raw_embedding_index(impostor_manifest, whisper_backbone=EXP.whisper_backbone)

    # Experiment 3b (exp3b_asnorm): per-fusion-model AS-Norm normalizer. The
    # cohort lives in the FUSED embedding space, so each fusion model needs its
    # own cohort matrix (same underlying base_train utterances, seed-fixed).
    #
    # cohort_pool/calib_genuine_pool: speaker-disjoint halves of base_train_index
    # (src/prototypical/score_norm.py::split_cohort_and_genuine). Building the
    # AS-Norm cohort and the calibration genuine trials from the SAME speaker
    # pool let a genuine trial's own speaker sit inside its own cohort --
    # biasing the calibrated threshold. Only AS-Norm calibration uses the split
    # halves (see calibrate_for below); every non-AS-Norm experiment keeps
    # using the full base_train_index, so those results stay bit-identical.
    cohort_pool, calib_genuine_pool = split_cohort_and_genuine(base_train_index, seed=0)

    def make_fuse_fn(fusion_model):
        def fuse(raw_list):
            raw = torch.from_numpy(np.stack(raw_list)).float().to(device)
            ecapa_part, whisper_part = split_raw_embedding(raw)
            with torch.no_grad():
                return fusion_model(ecapa_part, whisper_part).cpu().numpy()
        return fuse

    def make_normalizer(fusion_model, dual_weight=None):
        if EXP.score_norm != "asnorm":
            return None
        cohort = build_cohort(
            cohort_pool, make_fuse_fn(fusion_model),
            cohort_size=EXP.asnorm_cohort_size, seed=0,
        )
        # Experiment 5 (score_norm): per-space AS-Norm over the concat halves,
        # fused at the z-score level with `dual_weight` on the ECAPA side.
        if EXP.fusion_strategy == "score_norm" and dual_weight is not None:
            return DualASNorm(cohort, split_dim=ECAPA_DIM,
                              top_k=EXP.asnorm_top_k, weight=dual_weight)
        return ASNorm(cohort, top_k=EXP.asnorm_top_k)

    # DualASNorm weight per ablation arm (score_norm strategy only; None
    # elsewhere keeps the single-space ASNorm path bit-identical).
    if EXP.fusion_strategy == "score_norm":
        dual_weights = {"A3_fusion": EXP.score_fusion_weight,
                        "A1_ecapa_only": 1.0, "A2_whisper_only": 0.0}
    else:
        dual_weights = {"A3_fusion": None, "A1_ecapa_only": None, "A2_whisper_only": None}

    # Experiment 3a (exp3a_lowfrr): operating-point strategy comes from the
    # experiment config ("eer" reproduces the original behaviour exactly).
    # Experiment 3 P4 fix (per_config_calibration): every fusion model gets its
    # OWN calibrated threshold+normalizer; the pre-exp3 harness calibrated only
    # on A3 and reused that threshold for A1/A2, which broke those ablations
    # whenever the fused distance scale changed (Experiment 2's A1=0.069 artefact).
    def calibrate_for(fusion_model, dual_weight=None):
        normalizer = make_normalizer(fusion_model, dual_weight=dual_weight)
        # AS-Norm active -> genuine trials must come from the speaker-disjoint
        # half NOT used to build the cohort (see cohort_pool/calib_genuine_pool
        # above); otherwise (raw-distance calibration) use the full pool, same
        # as always.
        genuine_for_calibration = calib_genuine_pool if normalizer is not None else base_train_index
        result = calibrate_threshold(
            fusion_model, genuine_for_calibration, impostor_index, enrollment_k=1, seed=0,
            device=device, strategy=EXP.calibration_strategy, target_frr=EXP.target_frr,
            score_normalizer=normalizer,
        )
        return result, normalizer

    calib, norm_a3 = calibrate_for(fusion_models["A3_fusion"], dual_weights["A3_fusion"])
    print(f"calibrated threshold(A3)={calib.threshold:.4f} "
          f"(strategy={calib.strategy}, FAR={calib.far_at_threshold}, FRR={calib.frr_at_threshold}), "
          f"calibration EER={calib.eer:.4f}")
    if EXP.per_config_calibration:
        calib_a1, norm_a1 = calibrate_for(fusion_models["A1_ecapa_only"], dual_weights["A1_ecapa_only"])
        calib_a2, norm_a2 = calibrate_for(fusion_models["A2_whisper_only"], dual_weights["A2_whisper_only"])
        print(f"calibrated threshold(A1)={calib_a1.threshold:.4f} (EER={calib_a1.eer:.4f}), "
              f"threshold(A2)={calib_a2.threshold:.4f} (EER={calib_a2.eer:.4f})")
    else:
        calib_a1, norm_a1 = calib, norm_a3
        calib_a2, norm_a2 = calib, norm_a3
    print()

    # ---- F8-05: task_speakers / episodic_sessions with REAL audio ----
    task_manifest = eval_manifest[eval_manifest["speaker_id"].isin(split["task_speakers"])]
    speaker_audio_paths = build_speaker_audio_paths(task_manifest, split["task_speakers"])
    sessions = [[s for s in session if s in speaker_audio_paths] for session in split["episodic_sessions"]]
    sessions = [s for s in sessions if len(s) > 0]
    n_available_task_speakers = sum(len(s) for s in sessions)
    print(f"task_speakers with real audio available: {n_available_task_speakers}/100 "
          f"across {len(sessions)}/10 sessions\n")

    # ---- F9 + F10: define all configurations ----
    wb = EXP.whisper_backbone

    def make_proposed():
        return SpeakerIdentificationSystem(fusion_models["A3_fusion"], calib.threshold, continual_mode=EXP.continual_mode, whisper_backbone=wb, score_normalizer=norm_a3)

    def make_b1_static():
        return SpeakerIdentificationSystem(fusion_models["A3_fusion"], calib.threshold, continual_mode="static", whisper_backbone=wb, score_normalizer=norm_a3)

    def make_a1():
        return SpeakerIdentificationSystem(fusion_models["A1_ecapa_only"], calib_a1.threshold, continual_mode="running_average", whisper_backbone=wb, score_normalizer=norm_a1)

    def make_a2():
        return SpeakerIdentificationSystem(fusion_models["A2_whisper_only"], calib_a2.threshold, continual_mode="running_average", whisper_backbone=wb, score_normalizer=norm_a2)

    def make_ecapa_standard():
        return SingleBackboneSystem(backbone="ecapa", threshold=CLOSED_SET_THRESHOLD, continual_mode="static")

    # ProtoNet-vanilla baseline is a genuinely-trained prototypical net from a
    # RANDOM init (no residual/ECAPA-preservation) -- it must stay a separate,
    # weaker reference point, so it trains its own model rather than reusing the
    # frozen A1 (which would collapse it into the strong ECAPA baseline).
    print("training ProtoNet-vanilla baseline (random init, episodic)...")
    # second_dim: 512 (Whisper) for every pre-exp5 tag; derived from the index
    # so score_norm runs with a wider second backbone (WavLM/ReDimNet) don't
    # crash the gated projection this baseline trains.
    _sample_raw = next(iter(base_train_index.values()))[0]
    protonet_vanilla_model = train_fusion_for_ablation(
        "ecapa_only", base_train_index, N_WAY, K_SHOT, N_QUERY, 500,
        seed=0, device=device, residual_init=False,
        second_dim=_sample_raw.shape[-1] - ECAPA_DIM,
    )

    def make_protonet_vanilla():
        # whisper_backbone=wb: the model was trained on the active second
        # backbone's raw index (see second_dim above), so inference must fetch
        # the same embeddings -- identical to before for every pre-exp5 tag
        # (wb="whisper") and dimension-correct for exp5 (redimnet_b2, 192-d).
        return SpeakerIdentificationSystem(protonet_vanilla_model, threshold=CLOSED_SET_THRESHOLD, continual_mode="static", whisper_backbone=wb)

    configs = {
        "proposed_A3_running_average": make_proposed,
        "B1_static": make_b1_static,
        "A1_ecapa_only": make_a1,
        "A2_whisper_only": make_a2,
        "ECAPA_standard_baseline": make_ecapa_standard,
        "ProtoNet_vanilla_baseline": make_protonet_vanilla,
    }

    # x-vector + PLDA baseline needs an LDA whitener fit on base_train x-vector embeddings
    print("fitting x-vector + PLDA-lite baseline...")
    xvector_fit_manifest = pd.concat([
        pd.read_csv(REPO_ROOT / "data/raw/audio/vox1_sample/manifest.csv"),
        pd.read_csv(REPO_ROOT / "data/raw/audio/base_train_capped/manifest.csv"),
    ], ignore_index=True)
    counts = xvector_fit_manifest["speaker_id"].value_counts()
    xvector_eligible = counts[counts >= 2].index.tolist()[:150]  # bounded for time
    from src.features.cache import get_or_compute_embedding
    xvec_by_speaker: dict[str, list[np.ndarray]] = {}
    for spk in xvector_eligible:
        paths = xvector_fit_manifest.loc[xvector_fit_manifest["speaker_id"] == spk, "path"].tolist()[:3]
        xvec_by_speaker[spk] = [get_or_compute_embedding(REPO_ROOT / p, "xvector") for p in paths]
    whitener = fit_lda_whitener(xvec_by_speaker)

    def make_xvector_plda():
        return SingleBackboneSystem(
            backbone="xvector", threshold=CLOSED_SET_THRESHOLD, continual_mode="static", whiten_fn=whitener
        )

    configs["xvector_PLDA_baseline"] = make_xvector_plda
    print("done fitting whitener.\n")

    # ---- Experiment 3c (exp3c/report_open_set_detection): real unknown queries ----
    # Unknown speakers come from the DETECTION half of reserved_unknown_pool
    # (the other half is reserved for the validation sweep that locked
    # target_frr/AS-Norm params -- see src/data/splits.py::split_reserved_pool_halves).
    unknown_query_paths: list = []
    if EXP.report_open_set_detection:
        _, detection_speakers = split_reserved_pool_halves(split["reserved_unknown_pool"])
        unknown_manifest = eval_manifest[eval_manifest["speaker_id"].isin(set(detection_speakers))]
        unknown_query_paths = [REPO_ROOT / p for p in unknown_manifest["path"].tolist()]
        print(f"open-set detection: {unknown_manifest['speaker_id'].nunique()} unknown speakers, "
              f"{len(unknown_query_paths)} unknown queries\n")

    # ---- run every config N_REPS times through the real FSCIL harness ----
    summary = {}
    detection_scores_dump: dict = {}   # Experiment 5 (dump_detection_scores)
    for config_name, factory in configs.items():
        accs, fms, closed_accs = [], [], []
        det_eers, det_aurocs, det_tars = [], [], []
        for rep in range(N_REPS):
            # SEED_LIST holds 10 entries (values 0..9), so for every run to
            # date SEED_LIST[rep] == rep. Extending past 10 (the exp6 n=55
            # pre-registered power run) continues the same identity sequence:
            # bit-identical for rep < 10, deterministic beyond.
            seed = SEED_LIST[rep] if rep < len(SEED_LIST) else rep
            system = factory()
            detailed = run_fscil_detailed(
                system, sessions, speaker_audio_paths, k_shot=K_SHOT, n_query=N_QUERY,
                seed=seed, unknown_query_paths=unknown_query_paths,
            )
            result = detailed.open_set
            final_session = max(result.keys())
            acc = average_accuracy(result, final_session)
            accs.append(acc)
            closed_acc = average_accuracy(detailed.closed_set, final_session)
            closed_accs.append(closed_acc)
            fm = None
            if final_session > 0:
                fm = forgetting_measure(result, final_session)
                fms.append(fm)
            det = {}
            if detailed.unknown_final_distances:
                genuine_d = np.array(detailed.genuine_final_distances)
                unknown_d = np.array(detailed.unknown_final_distances)
                det = {
                    "detection_eer": find_eer_threshold(genuine_d, unknown_d).eer,
                    "detection_auroc": auroc(genuine_d, unknown_d),
                    "detection_tar_at_far1": tar_at_far(genuine_d, unknown_d, 0.01),
                }
                det_eers.append(det["detection_eer"])
                det_aurocs.append(det["detection_auroc"])
                det_tars.append(det["detection_tar_at_far1"])
                if EXP.dump_detection_scores:
                    detection_scores_dump.setdefault(config_name, {})[str(rep)] = {
                        "genuine": [float(x) for x in genuine_d],
                        "unknown": [float(x) for x in unknown_d],
                    }
            log_run(
                "full_evaluation", config_name, rep, seed,
                {"average_accuracy": acc, "forgetting_measure": fm,
                 "closed_set_accuracy": closed_acc, "n_sessions": final_session + 1, **det},
            )
        summary[config_name] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_forgetting": float(np.mean(fms)) if fms else None,
            "accs": accs,
            "mean_closed_set_accuracy": float(np.mean(closed_accs)),
            "std_closed_set_accuracy": float(np.std(closed_accs)),
            "closed_accs": closed_accs,
        }
        if det_eers:
            summary[config_name].update({
                "mean_detection_eer": float(np.mean(det_eers)),
                "mean_detection_auroc": float(np.mean(det_aurocs)),
                "mean_detection_tar_at_far1": float(np.mean(det_tars)),
            })
        line = (f"{config_name:30s} acc={summary[config_name]['mean_accuracy']:.3f}"
                f"±{summary[config_name]['std_accuracy']:.3f}"
                f"  closed={summary[config_name]['mean_closed_set_accuracy']:.3f}"
                f"  forgetting={summary[config_name]['mean_forgetting']}")
        if det_eers:
            line += (f"  detEER={np.mean(det_eers):.3f} AUROC={np.mean(det_aurocs):.3f}"
                     f" TAR@1%FAR={np.mean(det_tars):.3f}")
        print(line)

    # ---- F11: statistical significance for key comparisons ----
    comparisons = [
        ("proposed_A3_running_average", "A1_ecapa_only"),
        ("proposed_A3_running_average", "A2_whisper_only"),
        ("B1_static", "proposed_A3_running_average"),
        ("proposed_A3_running_average", "ECAPA_standard_baseline"),
        ("proposed_A3_running_average", "ProtoNet_vanilla_baseline"),
        ("proposed_A3_running_average", "xvector_PLDA_baseline"),
    ]
    alpha_corrected = bonferroni_correction(0.05, len(comparisons))
    print(f"\nBonferroni-corrected alpha ({len(comparisons)} comparisons): {alpha_corrected:.5f}\n")

    stats_results = {}
    for name_a, name_b in comparisons:
        test = paired_significance_test(np.array(summary[name_a]["accs"]), np.array(summary[name_b]["accs"]))
        significant = test.p_value < alpha_corrected
        stats_results[f"{name_a}_vs_{name_b}"] = {
            "test_used": test.test_used, "p_value": test.p_value, "significant": significant,
        }
        print(f"{name_a} vs {name_b}: {test.test_used}, p={test.p_value:.4f}, "
              f"significant={significant}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "experiment_id": EXP.id, "experiment_title": EXP.title,
        "residual_init": RESIDUAL_INIT, "continual_mode": EXP.continual_mode,
        "summary": summary, "statistics": stats_results, "calibration_threshold": calib.threshold,
        "calibration_eer": calib.eer, "n_reps": N_REPS, "n_train_episodes": N_TRAIN_EPISODES,
        "n_task_speakers_available": n_available_task_speakers, "n_sessions_available": len(sessions),
        # Experiment 3 provenance (feature-flag values actually used this run)
        "calibration_strategy": EXP.calibration_strategy,
        "target_frr": EXP.target_frr if EXP.calibration_strategy == "target_frr" else None,
        "calibration_far_at_threshold": calib.far_at_threshold,
        "calibration_frr_at_threshold": calib.frr_at_threshold,
        "per_config_calibration": EXP.per_config_calibration,
        "calibration_threshold_a1": calib_a1.threshold, "calibration_threshold_a2": calib_a2.threshold,
        "score_norm": EXP.score_norm,
        "asnorm_cohort_size": EXP.asnorm_cohort_size if EXP.score_norm == "asnorm" else None,
        "asnorm_top_k": EXP.asnorm_top_k if EXP.score_norm == "asnorm" else None,
        "n_unknown_queries": len(unknown_query_paths),
        "elapsed_minutes": (time.time() - t_start) / 60,
    }, indent=2), encoding="utf-8")
    # per-experiment copy so a later run can't clobber this experiment's numbers
    # (Experiment 2's run overwrote Experiment 1's summary -- see docs/experiment-2.md section 8)
    per_exp_path = RESULTS_PATH.with_name(f"full_evaluation_summary_{EXP.id}.json")
    per_exp_path.write_text(RESULTS_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    if EXP.dump_detection_scores and detection_scores_dump:
        dump_path = RESULTS_PATH.with_name(f"detection_scores_{EXP.id}.json")
        dump_path.write_text(json.dumps(detection_scores_dump), encoding="utf-8")
        print(f"Saved per-rep detection scores to {dump_path.name}")
    print(f"\nSaved summary to {RESULTS_PATH} (+ {per_exp_path.name})")
    print(f"Total elapsed: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
