"""Experiment registry / feature-flag switch.

Every reportable configuration of the system is registered here as an
`ExperimentConfig` with a stable string id. The *active* experiment is
selected by `ACTIVE_EXPERIMENT` (overridable at run time via the
`ACTIVE_EXPERIMENT` environment variable), and `scripts/run_full_evaluation.py`
reads its knobs from `active()` instead of hard-coding them.

To run a given experiment you therefore only flip the tag -- either edit
`ACTIVE_EXPERIMENT` below, or run e.g. (PowerShell):

    $env:ACTIVE_EXPERIMENT = "baseline_v0"; .venv/Scripts/python.exe scripts/run_full_evaluation.py

Adding a new experiment (Experiment 3, 4, ...) is a matter of appending one
more `ExperimentConfig` to `EXPERIMENTS` and pointing `ACTIVE_EXPERIMENT` at
it -- nothing else in the pipeline needs to change.

See docs/ for the full write-up of each experiment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    id: str
    title: str
    description: str

    # --- fusion / training knobs (the actual variables between experiments) ---
    residual_init: bool          # start fusion from ECAPA-preserving identity init?
    n_train_episodes: int        # episodic fine-tuning episodes (0 = frozen)
    continual_mode: str          # "running_average" (proposed) | "static"

    # --- fusion strategy (Experiment 2; extended by Experiment 5) ---
    fusion_strategy: str = "embedding"   # "embedding" (gated fusion) | "score" (exp2a raw score fusion)
                                         # | "score_norm" (exp5b: per-space AS-Norm score fusion via
                                         #   DualASNorm -- requires score_norm="asnorm")
    score_fusion_weight: float = 0.4     # weight on ECAPA: "score" -> w*d_ecapa+(1-w)*d_2nd (raw);
                                         # "score_norm" -> w*z_ecapa+(1-w)*z_2nd (z-score space)
    whisper_backbone: str = "whisper"    # second-backbone cache name: "whisper" (middle layer),
                                         # "whisper_l4" (exp2), "redimnet_b2" (exp5, 192-d)

    # --- threshold operating point (Experiment 3a) ---
    calibration_strategy: str = "eer"   # "eer" (original) | "target_frr" (low-FRR, exp3a)
    target_frr: float = 0.05            # only used when calibration_strategy == "target_frr"
    per_config_calibration: bool = False  # exp3 P4 fix: calibrate threshold per fusion model
                                          # (A1/A2/A3) instead of sharing A3's threshold

    # --- score normalization (Experiment 3b) ---
    score_norm: str = "none"            # "none" (original) | "asnorm" (exp3b)
    asnorm_cohort_size: int = 300       # cohort utterances sampled from base_train
    asnorm_top_k: int = 100             # adaptive top-K cohort stats per query/prototype

    # --- dual-metric reporting (Experiment 3c; reporting-only, no behaviour change) ---
    report_open_set_detection: bool = False  # unknown-query detection (EER/AUROC/TAR@FAR)
                                             # + closed-set identification accuracy

    # --- detection-score dump (Experiment 5; reporting-only) ---
    dump_detection_scores: bool = False  # save per-rep genuine/unknown decision scores per config
                                         # (experiments/detection_scores_<tag>.json) so two-space
                                         # detection rules (exp4c "mean") can be tested post-hoc

    # --- evaluation protocol (kept constant across experiments unless noted) ---
    n_way: int = 10
    k_shot: int = 1
    n_query: int = 5
    n_reps: int = 5              # proposal target is 10; reduced for functional-scale runs
    lr: float = 1e-3


EXPERIMENTS: dict[str, ExperimentConfig] = {
    # --------------------------------------------------------------------- #
    # Experiment 0 (baseline_v0): the ORIGINAL configuration, kept so the   #
    # regression that motivated Experiment 1 stays reproducible. Random     #
    # (non-residual) fusion init + 500 episodes of episodic fine-tuning.    #
    # Reported result: proposed system Accuracy ~= 0.235 (BELOW baselines). #
    # --------------------------------------------------------------------- #
    "baseline_v0": ExperimentConfig(
        id="baseline_v0",
        title="Baseline v0 -- random-init fusion, 500 episodes",
        description=(
            "Original configuration. Gated Attention Fusion trained from a random "
            "init for 500 episodes. Episodic fine-tuning on only 71 base speakers "
            "overfits and degrades ECAPA's pretrained embedding space -> proposed "
            "system 0.235, below every closed-set baseline."
        ),
        residual_init=False,
        n_train_episodes=500,
        continual_mode="running_average",
    ),
    # --------------------------------------------------------------------- #
    # Experiment 1 (exp1_frozen_residual): the FIX. ECAPA-preserving        #
    # residual init + frozen fusion (0 training episodes). A training-amount #
    # sweep showed episodic fine-tuning monotonically hurts at this data     #
    # scale, so we freeze and keep ECAPA quality as a hard lower bound.      #
    # Reported result: proposed system Accuracy ~= 0.731 (beats every        #
    # few-shot/incremental baseline; near closed-set ECAPA ceiling 0.788).   #
    # --------------------------------------------------------------------- #
    "exp1_frozen_residual": ExperimentConfig(
        id="exp1_frozen_residual",
        title="Experiment 1 -- frozen residual-init fusion",
        description=(
            "ECAPA-preserving residual init + frozen fusion (0 episodes). Preserves "
            "ECAPA's fully-pretrained embedding space instead of destroying it via "
            "undertrained episodic fine-tuning. Proposed system 0.731, near-zero "
            "forgetting (0.0004); continual update still helps (B2 > B1)."
        ),
        residual_init=True,
        n_train_episodes=0,
        continual_mode="running_average",
    ),
    # --------------------------------------------------------------------- #
    # Experiment 2a (exp2a_scorefusion_L4): make Whisper actually contribute. #
    # Keeps Experiment 1's frozen, training-free, 1-shot regime but replaces  #
    # the gated EMBEDDING fusion (which captured ~0 of Whisper, A3==A1) with  #
    # SCORE-level fusion over each backbone's native cosine distance, and     #
    # upgrades the Whisper feature to encoder layer 4 (more speaker-          #
    # discriminative than the middle layer). Exploration: score fusion        #
    # captures ~+1.4% closed-set that gated fusion missed; oracle ceiling     #
    # rises to +4.0% with L4. w=0.4 on ECAPA was best in the sweep.           #
    # --------------------------------------------------------------------- #
    "exp2a_scorefusion_L4": ExperimentConfig(
        id="exp2a_scorefusion_L4",
        title="Experiment 2a -- Whisper L4 + score-level fusion",
        description=(
            "Frozen, training-free, 1-shot (as exp1) but fusion happens at the "
            "distance level: score = 0.4*d_ecapa + 0.6*d_whisper, with Whisper "
            "taken from encoder layer 4. Makes the fusion contribution measurable "
            "(A3 != A1) instead of ECAPA-dominated."
        ),
        residual_init=True,
        n_train_episodes=0,
        continual_mode="running_average",
        fusion_strategy="score",
        score_fusion_weight=0.4,
        whisper_backbone="whisper_l4",
    ),
    # --------------------------------------------------------------------- #
    # Experiment 3 (docs/experiment-3.md): fix HOW and WHERE the open-set    #
    # threshold is chosen. Base system identical to exp1 (frozen residual    #
    # fusion, 1-shot, training-free); only calibration/scoring/reporting     #
    # change, each behind its own config field so 3a/3b/3c stack cleanly.    #
    # All three also enable per_config_calibration (P4 fix: Experiment 2's   #
    # harness calibrated ONE threshold on A3 and reused it for A1/A2, which  #
    # invalidated those ablation numbers whenever the distance scale moved). #
    # target_frr / AS-Norm params are locked on a VALIDATION task built from #
    # reserved_unknown_pool speakers (disjoint from task_speakers), never on #
    # the task itself -- see scripts/exp3_validation_sweep.py.               #
    # --------------------------------------------------------------------- #
    "exp3a_lowfrr": ExperimentConfig(
        id="exp3a_lowfrr",
        title="Experiment 3a -- low-FRR operating point (target-FRR calibration)",
        description=(
            "exp1_frozen_residual + threshold chosen at a low-false-reject "
            "operating point (quantile of genuine calibration distances) instead "
            "of the EER point. The FSCIL metric has no unknown queries, so every "
            "EER-point rejection is a pure loss; Experiment 2's threshold study "
            "showed open@p95 > open@EER for every fusion weight."
        ),
        residual_init=True,
        n_train_episodes=0,
        continual_mode="running_average",
        calibration_strategy="target_frr",
        # LOCKED on the validation task (scripts/exp3_validation_sweep.py,
        # experiments/exp3_validation_sweep.json): among no-normalization
        # candidates, target_frr=0.01 gave val acc 0.7550 vs 0.7492 @EER.
        target_frr=0.01,
        per_config_calibration=True,
        report_open_set_detection=True,
    ),
    "exp3b_asnorm": ExperimentConfig(
        id="exp3b_asnorm",
        title="Experiment 3b -- exp3a + adaptive score normalization (AS-Norm)",
        description=(
            "exp3a_lowfrr + AS-Norm: query->prototype distances are z-normalized "
            "against top-K statistics of a base_train cohort before the threshold "
            "decision (src/prototypical/score_norm.py). Standard speaker-"
            "verification technique targeting the poor genuine/impostor "
            "separation (EER ~0.22) diagnosed in Experiment 2."
        ),
        residual_init=True,
        n_train_episodes=0,
        continual_mode="running_average",
        calibration_strategy="target_frr",
        # LOCKED on the validation task (experiments/exp3_validation_sweep.json):
        # asnorm cohort=300/top_k=200 + target_frr=0.05 was the overall best
        # (val acc 0.8867; top_k 50/100 gave 0.8767/0.8758; EER point 0.8842).
        target_frr=0.05,
        per_config_calibration=True,
        score_norm="asnorm",
        asnorm_cohort_size=300,
        asnorm_top_k=200,
        report_open_set_detection=True,
    ),
    "exp3c_dualmetric": ExperimentConfig(
        id="exp3c_dualmetric",
        title="Experiment 3c -- best-of-3a/3b + dual-metric reporting",
        description=(
            "Reporting configuration for the thesis claim: closed-set "
            "identification accuracy (apple-to-apple vs the closed-set ECAPA "
            "baseline) AND open-set detection quality (EER/AUROC/TAR@FAR=1% "
            "against real unknown speakers from reserved_unknown_pool) reported "
            "side by side. System behaviour = exp3b_asnorm (the validation-best "
            "of 3a/3b), so its official run is numerically identical to exp3b's "
            "-- this tag exists as the named reporting view of that run."
        ),
        residual_init=True,
        n_train_episodes=0,
        continual_mode="running_average",
        calibration_strategy="target_frr",
        target_frr=0.05,  # = exp3b (validation-best)
        per_config_calibration=True,
        score_norm="asnorm",  # validation: asnorm (0.8867) > none (0.7550)
        asnorm_cohort_size=300,
        asnorm_top_k=200,
        report_open_set_detection=True,
    ),
    # --------------------------------------------------------------------- #
    # Experiment 5 (docs/experiment-5.md): replace Whisper with a strong,    #
    # architecturally-heterogeneous second backbone. Candidate selection was #
    # gated: WavLM-base-plus (leakage-free) FAILED G5.1 (frozen mean-pool    #
    # standalone 0.2925 < 0.70; experiments/exp5_wavlm_layer_sweep.json);    #
    # ReDimNet-b2 ft_lm PASSED screening (standalone 0.878, oracle ceiling   #
    # +0.069, snapshot fusion 0.890 > both single backbones;                 #
    # experiments/exp5_screening_redimnet_b2.json). Leakage: ReDimNet is     #
    # VoxCeleb2-dev-trained and 80/100 task speakers are in vox2-dev --      #
    # DISCLOSED (experiments/exp5_leakage_audit.json); ECAPA SpeechBrain is  #
    # likewise VoxCeleb-trained, so internal comparisons stay like-for-like. #
    # Fusion: dual-space concat embedder + DualASNorm z-score fusion         #
    # (w*z_ecapa + (1-w)*z_redimnet); per-arm normalizer weight makes        #
    # A1 (w=1) / A2 (w=0) / A3 (w) share one embedder. score_fusion_weight  #
    # LOCKED on the validation FSCIL sweep (exp5_validation_sweep.json):    #
    # w=0.3 = best genuine-fusion weight (val 0.9183 vs ECAPA-only 0.8875,  #
    # 3/3 seeds; statistically tied with ReDimNet-only 0.9200 -- the A2 arm #
    # of the official run keeps that comparison transparent).               #
    # --------------------------------------------------------------------- #
    "exp5b_redimnet_fusion": ExperimentConfig(
        id="exp5b_redimnet_fusion",
        title="Experiment 5b -- ECAPA + ReDimNet dual-space AS-Norm score fusion",
        description=(
            "Whisper replaced by frozen ReDimNet-b2 (ft_lm, vox2; Interspeech "
            "2024) as the second backbone. Score = 0.3*z_ecapa + 0.7*z_redimnet "
            "with per-space AS-Norm (c300/k200) over a shared base_train cohort, "
            "target-FRR 5% threshold, per-config calibration, running-average "
            "continual updates. Training-free, strict 1-shot, all exp3 "
            "protocol discipline retained."
        ),
        residual_init=True,
        n_train_episodes=0,
        continual_mode="running_average",
        fusion_strategy="score_norm",
        score_fusion_weight=0.3,
        whisper_backbone="redimnet_b2",
        calibration_strategy="target_frr",
        target_frr=0.05,
        per_config_calibration=True,
        score_norm="asnorm",
        asnorm_cohort_size=300,
        asnorm_top_k=200,
        report_open_set_detection=True,
        dump_detection_scores=True,
    ),
}

# The active experiment tag. Flip this (or set the ACTIVE_EXPERIMENT env var)
# to switch which configuration scripts/run_full_evaluation.py executes.
# exp5b_redimnet_fusion is the current headline configuration (0.908 +/- 0.019,
# A3 > A1 significant at p=0.0041 for the first time, detection EER 0.075 --
# see docs/experiment-5.md); exp3b_asnorm (0.865) remains fully reproducible
# by flipping this tag back.
ACTIVE_EXPERIMENT: str = os.environ.get("ACTIVE_EXPERIMENT", "exp5b_redimnet_fusion")


def active() -> ExperimentConfig:
    """Return the currently-selected ExperimentConfig."""
    if ACTIVE_EXPERIMENT not in EXPERIMENTS:
        raise KeyError(
            f"ACTIVE_EXPERIMENT={ACTIVE_EXPERIMENT!r} is not registered; "
            f"known experiments: {sorted(EXPERIMENTS)}"
        )
    return EXPERIMENTS[ACTIVE_EXPERIMENT]
