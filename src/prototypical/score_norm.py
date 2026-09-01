"""Adaptive Score Normalization -- AS-Norm (Experiment 3b, exp3b_asnorm).

Standard technique from speaker verification (Matejka et al., 2017; Cumani
et al., 2011): instead of thresholding the raw query->prototype score, the
score is z-normalized against the statistics of a COHORT of non-task
utterances, adaptively restricted to the top-K most competitive cohort
members for each side. This tightens the genuine/impostor separation
(lower EER) and makes one global threshold transfer better across speakers
-- exactly the weakness Experiment 2 diagnosed (EER ~ 0.22 at 1-shot).

This project scores with *distances* (lower = more genuine), not
similarities, so "most competitive cohort members" = the K SMALLEST cohort
distances, and the normalized score keeps the distance orientation
(lower = accept), which lets the existing threshold/decision machinery in
ContinualLearningManager and calibration.py work unchanged:

    d_norm(q, p) = 1/2 * [ (d(q,p) - mu_topK(q->cohort)) / sigma_topK(q->cohort)
                         + (d(q,p) - mu_topK(p->cohort)) / sigma_topK(p->cohort) ]

The first term is the query-side (t-norm-like) part, the second the
prototype/enrollment-side (z-norm-like) part; averaging both is the
symmetric "S-norm" variant, made adaptive by the top-K selection.

Leakage guard (docs/experiment-3.md section 7 / 9): the cohort MUST come
from non-task data. scripts/run_full_evaluation.py builds it from
base_train utterances (the same pool already used to fit calibration
genuine statistics), never from task/episodic speakers.
"""
from __future__ import annotations

import random

import numpy as np


def _pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix between rows of `a` (n, d) and `b` (m, d)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    sq = (a * a).sum(axis=1)[:, None] + (b * b).sum(axis=1)[None, :] - 2.0 * (a @ b.T)
    return np.sqrt(np.maximum(sq, 0.0))


class ASNorm:
    """Adaptive symmetric score normalization over a fixed cohort.

    Parameters
    ----------
    cohort_embeddings : (n_cohort, dim) array of FUSED embeddings of
        non-task utterances (same embedding space as the prototypes).
    top_k : size of the adaptive subset: statistics are computed over the
        `top_k` smallest distances to the cohort (the most impostor-like
        competitors). Clamped to the cohort size.
    """

    def __init__(self, cohort_embeddings: np.ndarray, top_k: int = 100) -> None:
        cohort = np.asarray(cohort_embeddings, dtype=np.float64)
        if cohort.ndim != 2 or len(cohort) < 2:
            raise ValueError(f"cohort must be (n>=2, dim), got shape {cohort.shape}")
        self.cohort = cohort
        self.top_k = int(min(top_k, len(cohort)))

    def _topk_stats(self, dists_to_cohort: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-row mean/std of the top_k smallest cohort distances.
        dists_to_cohort: (n, n_cohort) -> mu (n,), sigma (n,)."""
        k = self.top_k
        part = np.partition(dists_to_cohort, k - 1, axis=1)[:, :k]
        mu = part.mean(axis=1)
        sigma = part.std(axis=1)
        # degenerate cohort slice (all-equal distances) must not divide by 0
        sigma = np.where(sigma < 1e-8, 1e-8, sigma)
        return mu, sigma

    def normalize(self, queries: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
        """Return the normalized (n_queries, n_prototypes) distance matrix.

        Lower is still "more genuine"; values are in cohort-sigma units and
        typically negative for genuine trials.
        """
        queries = np.atleast_2d(np.asarray(queries, dtype=np.float64))
        prototypes = np.atleast_2d(np.asarray(prototypes, dtype=np.float64))

        raw = _pairwise_dist(queries, prototypes)                 # (n, m)
        mu_q, sig_q = self._topk_stats(_pairwise_dist(queries, self.cohort))
        mu_p, sig_p = self._topk_stats(_pairwise_dist(prototypes, self.cohort))

        query_side = (raw - mu_q[:, None]) / sig_q[:, None]
        proto_side = (raw - mu_p[None, :]) / sig_p[None, :]
        return 0.5 * (query_side + proto_side)


class DualASNorm:
    """Two-space AS-Norm score fusion (Experiment 5, exp5b -- feature-flagged
    behind ExperimentConfig.fusion_strategy == "score_norm").

    Embeddings are CONCATENATIONS of two per-backbone unit-norm halves
    (ScoreFusionEmbed's output, e.g. [e_ecapa_hat ; e_wavlm_hat]); this
    normalizer splits every matrix at `split_dim`, AS-normalizes each half in
    its OWN space against the matching half of the cohort, and returns the
    weighted sum of the two z-score matrices:

        z(q, p) = w * z_first(q, p) + (1 - w) * z_second(q, p)

    Both terms are in cohort-sigma units, which is what makes the weighted
    sum well-posed -- Experiment 2's raw-distance fusion had no such
    guarantee, and Experiment 4 showed the fusion must happen at the score
    level because AS-Norm statistics depend on the query/prototype rows
    (the exp2 weighted-concatenation trick no longer applies).

    weight=1.0 reduces to single-space AS-Norm on the first half (the A1
    ablation), weight=0.0 to the second half (A2) -- so one embedder serves
    every ablation arm and only the normalizer weight changes. Any constant
    per-half scaling of the inputs (e.g. ScoreFusionEmbed's sqrt(w) factors)
    cancels out in the z-scores (see tests/test_experiment5.py).

    Orientation is unchanged: lower = more genuine, so the existing
    threshold/argmin machinery works untouched.
    """

    def __init__(
        self,
        cohort_embeddings: np.ndarray,
        split_dim: int,
        top_k: int = 100,
        weight: float = 0.5,
    ) -> None:
        cohort = np.asarray(cohort_embeddings, dtype=np.float64)
        if cohort.ndim != 2 or not (0 < split_dim < cohort.shape[1]):
            raise ValueError(
                f"cohort must be 2-D with 0 < split_dim < dim, got shape "
                f"{cohort.shape}, split_dim={split_dim}"
            )
        if not (0.0 <= weight <= 1.0):
            raise ValueError(f"weight must be in [0, 1], got {weight}")
        self.split_dim = int(split_dim)
        self.weight = float(weight)
        self.cohort = cohort  # kept for round-tripping (e.g. SpeakerIdentificationSystem.save/load)
        self.top_k = int(top_k)
        self._first = ASNorm(cohort[:, : self.split_dim], top_k=top_k)
        self._second = ASNorm(cohort[:, self.split_dim :], top_k=top_k)

    def normalize(self, queries: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
        queries = np.atleast_2d(np.asarray(queries, dtype=np.float64))
        prototypes = np.atleast_2d(np.asarray(prototypes, dtype=np.float64))
        d = self.split_dim
        z_first = self._first.normalize(queries[:, :d], prototypes[:, :d])
        z_second = self._second.normalize(queries[:, d:], prototypes[:, d:])
        return self.weight * z_first + (1.0 - self.weight) * z_second


def build_cohort(
    embedding_index: dict[str, list[np.ndarray]],
    fuse_fn,
    cohort_size: int = 300,
    seed: int = 0,
) -> np.ndarray:
    """Sample up to `cohort_size` utterances (round-robin across speakers, so
    no single speaker dominates) from a raw-embedding index and fuse them.

    `fuse_fn`: callable mapping a list of raw [ecapa;whisper] embeddings to a
    (n, fused_dim) numpy array -- the same fusion the system under test uses.
    """
    rng = np.random.default_rng(seed)
    speakers = sorted(embedding_index)
    per_speaker = {s: list(embedding_index[s]) for s in speakers}
    for s in speakers:
        rng.shuffle(per_speaker[s])

    picked: list[np.ndarray] = []
    round_idx = 0
    while len(picked) < cohort_size:
        added = False
        for s in speakers:
            if round_idx < len(per_speaker[s]):
                picked.append(per_speaker[s][round_idx])
                added = True
                if len(picked) >= cohort_size:
                    break
        if not added:
            break
        round_idx += 1

    if len(picked) < 2:
        raise ValueError("not enough utterances to build a cohort")
    return fuse_fn(picked)


def split_cohort_and_genuine(
    embedding_index: dict[str, list[np.ndarray]], seed: int = 0
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
    """Speaker-disjoint split of a raw-embedding index into (cohort_pool,
    genuine_pool), so a calibration run's genuine trials and its AS-Norm
    cohort never draw from the same speakers.

    Without this guard, calling build_cohort(pool, ...) and
    calibrate_threshold(model, genuine_index=pool, ...) on the SAME `pool`
    (as every AS-Norm-enabled experiment did before this fix) lets a
    genuine calibration trial's own speaker also sit in the cohort --
    build_cohort's round-robin sampling picks one utterance per speaker, so
    any genuine speaker within the first `cohort_size` ids is virtually
    guaranteed to be its own cohort member. ASNorm._topk_stats then likely
    includes that same-speaker cohort utterance (near-zero distance) among
    the top-K used for mu_q/sigma_q, pulling mu_q down and making the
    normalized genuine score look artificially more genuine than it would
    against a true impostor-only cohort -- biasing the calibrated threshold.

    Callers should keep the ORIGINAL (unsplit) index for anything that
    isn't part of an AS-Norm calibration run (e.g. base-model training),
    so non-AS-Norm experiment results stay bit-identical.
    """
    speakers = sorted(embedding_index)
    rng = random.Random(seed + 7)  # distinct stream from splits.py's seed(+0..+3)
    rng.shuffle(speakers)
    mid = len(speakers) // 2
    cohort_speakers, genuine_speakers = speakers[:mid], speakers[mid:]
    cohort_pool = {s: embedding_index[s] for s in cohort_speakers}
    genuine_pool = {s: embedding_index[s] for s in genuine_speakers}
    return cohort_pool, genuine_pool


class AdaptiveDualASNorm(DualASNorm):
    """Zero-parameter adaptive two-space score fusion (Experiment 6, F6-3b).

    DualASNorm combines the two per-space z-score matrices with a single
    weight `w` that is CONSTANT over every query. Experiment 4's re-audit
    measured the cost of that: the best global w buys +0.0025 over
    ECAPA-alone, while an oracle that picks w PER QUERY reaches +0.0333 --
    a 13x gap that no amount of sweeping a constant can reach.

    The obvious way to close it is to FIT a fusion backend (F6-3, logistic
    regression / BOSARIS-style LLR). This class is the alternative that
    needs no ruling on the thesis' training-free constraint, because it fits
    NOTHING: the per-query weight is read off the two z-score matrices that
    have already been computed.

        margin_i(q) = (second smallest z_i(q, .)) - (smallest z_i(q, .))

    A space with a large margin has one clearly-closest prototype; a space
    with a small margin is undecided between its top two. So margin is used
    directly as that space's confidence on that query:

        "margin_weighted"  w(q) = margin_1(q) / (margin_1(q) + margin_2(q))
        "margin_select"    w(q) = 1 if margin_1(q) >= margin_2(q) else 0

    `margin_select` is the practical form of Experiment 4's SELECTION oracle
    (which used the true label); `margin_weighted` is its soft version and
    stays inside the weighted-sum family whose ceiling is +0.0333.

    Using the RATIO rather than an absolute margin matters: the number of
    prototypes differs between threshold calibration (~55, all base_train
    genuine speakers) and FSCIL session 1 (10, growing per session), and
    margins grow with prototype count. A ratio of two margins measured on
    the same query against the same prototype set is insensitive to that.

    Degenerate cases fall back to `weight` (the constant DualASNorm
    behaviour): fewer than 2 prototypes, so no margin exists, or both
    margins ~0, so the ratio is undefined. This keeps the class a strict
    generalization -- with 1 prototype it IS DualASNorm.

    Orientation is unchanged (lower = more genuine), so the calibration and
    threshold machinery works untouched. Note the threshold must still be
    calibrated against THIS normalizer: per-query weighting changes the
    score distribution, and every arm in the F6-3b sweep re-runs
    build_genuine_impostor_distances for exactly that reason.

    Training-free status: tier 2 in docs/experiment-6-plan.md section 2 --
    no gradients, and nothing fit on base_train either. It is admissible
    under the strictest reading of the constraint, which is the whole point
    of running it before F6-3.
    """

    VALID_RULES = ("margin_weighted", "margin_select")

    def __init__(
        self,
        cohort_embeddings: np.ndarray,
        split_dim: int,
        top_k: int = 100,
        weight: float = 0.5,
        rule: str = "margin_weighted",
    ) -> None:
        super().__init__(cohort_embeddings, split_dim=split_dim, top_k=top_k, weight=weight)
        if rule not in self.VALID_RULES:
            raise ValueError(f"rule must be one of {self.VALID_RULES}, got {rule!r}")
        self.rule = rule

    @staticmethod
    def _margins(z: np.ndarray) -> np.ndarray:
        """Per-row gap between the two smallest entries of `z` (n_q, n_p)."""
        part = np.partition(z, 1, axis=1)
        return part[:, 1] - part[:, 0]

    def per_query_weight(self, queries: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
        """The w(q) this rule would use -- exposed so the sweep can report the
        weight distribution instead of treating the rule as a black box."""
        queries = np.atleast_2d(np.asarray(queries, dtype=np.float64))
        prototypes = np.atleast_2d(np.asarray(prototypes, dtype=np.float64))
        d = self.split_dim
        z_first = self._first.normalize(queries[:, :d], prototypes[:, :d])
        z_second = self._second.normalize(queries[:, d:], prototypes[:, d:])
        return self._weights_from(z_first, z_second)

    def _weights_from(self, z_first: np.ndarray, z_second: np.ndarray) -> np.ndarray:
        n_proto = z_first.shape[1]
        if n_proto < 2:
            return np.full(z_first.shape[0], self.weight)

        m_first = self._margins(z_first)
        m_second = self._margins(z_second)
        total = m_first + m_second

        if self.rule == "margin_select":
            w = (m_first >= m_second).astype(np.float64)
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                w = m_first / total
        # both spaces undecided -> no evidence either way, use the constant
        return np.where(total > 1e-12, w, self.weight)

    def normalize(self, queries: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
        queries = np.atleast_2d(np.asarray(queries, dtype=np.float64))
        prototypes = np.atleast_2d(np.asarray(prototypes, dtype=np.float64))
        d = self.split_dim
        z_first = self._first.normalize(queries[:, :d], prototypes[:, :d])
        z_second = self._second.normalize(queries[:, d:], prototypes[:, d:])
        w = self._weights_from(z_first, z_second)[:, None]
        return w * z_first + (1.0 - w) * z_second


class MarginShiftDualASNorm(AdaptiveDualASNorm):
    """Experiment 6 follow-up to F6-3b: per-query weight from the margin
    DIFFERENCE, shifted off a base weight.

        w(q) = clip(weight + shift_scale * (margin_first(q) - margin_second(q)), 0, 1)

    Why this exists -- the oracle-predictability diagnostic
    (experiments/exp6_oracle_predictability_redimnet_b2.json) showed F6-3b's
    margin-RATIO rule under-harvests for two design reasons:

      1. The ratio is symmetric around 0.5, but the decidable-query population
         is not: the second space is right ~2.5x more often (83 vs 33). A rule
         anchored at 0.5 rescues a few first-space queries while damaging more
         second-space ones. Here the anchor is `weight` -- the validation-locked
         base weight (0.3 for ECAPA+ReDimNet), so the rule only DEVIATES from
         the well-chosen operating point when the margins give a reason.
      2. The margin DIFFERENCE separates "which space is right" better than
         the ratio (AUROC 0.8583 vs 0.8383), and margins are z-score
         (cohort-sigma) units, so the difference is scale-meaningful across
         queries while the ratio throws that scale away.

    `shift_scale` is a single scalar chosen on the validation half -- the same
    standing as `weight` itself (tier 1 of the training-free taxonomy; nothing
    is fitted by gradient, and nothing touches embeddings). shift_scale=0
    reduces EXACTLY to constant-weight DualASNorm, so this is again a strict
    generalization, and the sweep's zero point doubles as the A3-fixed arm.

    Degenerate cases (fewer than 2 prototypes) fall back to the base weight
    via the parent's convention.
    """

    def __init__(
        self,
        cohort_embeddings: np.ndarray,
        split_dim: int,
        top_k: int = 100,
        weight: float = 0.5,
        shift_scale: float = 1.0,
    ) -> None:
        # rule name only matters for the parent's validation; margins are
        # computed the same way, the combination differs in _weights_from.
        super().__init__(cohort_embeddings, split_dim=split_dim, top_k=top_k,
                         weight=weight, rule="margin_weighted")
        self.shift_scale = float(shift_scale)

    def _weights_from(self, z_first: np.ndarray, z_second: np.ndarray) -> np.ndarray:
        if z_first.shape[1] < 2:
            return np.full(z_first.shape[0], self.weight)
        m_first = self._margins(z_first)
        m_second = self._margins(z_second)
        w = self.weight + self.shift_scale * (m_first - m_second)
        return np.clip(w, 0.0, 1.0)
