"""Unit tests for Experiment 5's feature-flagged additions: DualASNorm
(two-space normalized score fusion), the generalized raw-embedding split,
and the dual-dim ablation trainer. Every default keeps pre-exp5 behaviour
byte-identical -- these tests pin both the new behaviour and that guarantee."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation.ablation import train_fusion_for_ablation
from src.prototypical.data import ECAPA_DIM, WHISPER_DIM, split_raw_embedding
from src.prototypical.score_norm import ASNorm, DualASNorm


def _rand(shape, seed):
    return np.random.default_rng(seed).normal(size=shape)


def test_split_raw_embedding_backward_identical_for_whisper_rows():
    raw = _rand((5, ECAPA_DIM + WHISPER_DIM), 0)
    first, second = split_raw_embedding(raw)
    assert first.shape == (5, ECAPA_DIM)
    assert second.shape == (5, WHISPER_DIM)
    np.testing.assert_array_equal(second, raw[:, ECAPA_DIM:ECAPA_DIM + WHISPER_DIM])


def test_split_raw_embedding_handles_wider_second_backbone():
    raw = _rand((3, ECAPA_DIM + 768), 1)  # WavLM base-plus concat
    first, second = split_raw_embedding(raw)
    assert first.shape == (3, ECAPA_DIM)
    assert second.shape == (3, 768)


def test_dual_asnorm_weight_extremes_match_single_space_asnorm():
    d1, d2 = 8, 12
    cohort = _rand((60, d1 + d2), 2)
    queries = _rand((5, d1 + d2), 3)
    protos = _rand((4, d1 + d2), 4)

    z_first = ASNorm(cohort[:, :d1], top_k=20).normalize(queries[:, :d1], protos[:, :d1])
    z_second = ASNorm(cohort[:, d1:], top_k=20).normalize(queries[:, d1:], protos[:, d1:])

    dual_w1 = DualASNorm(cohort, split_dim=d1, top_k=20, weight=1.0)
    dual_w0 = DualASNorm(cohort, split_dim=d1, top_k=20, weight=0.0)
    np.testing.assert_allclose(dual_w1.normalize(queries, protos), z_first, atol=1e-12)
    np.testing.assert_allclose(dual_w0.normalize(queries, protos), z_second, atol=1e-12)

    dual_mid = DualASNorm(cohort, split_dim=d1, top_k=20, weight=0.7)
    np.testing.assert_allclose(
        dual_mid.normalize(queries, protos), 0.7 * z_first + 0.3 * z_second, atol=1e-12
    )


def test_dual_asnorm_invariant_to_per_half_constant_scaling():
    """ScoreFusionEmbed emits sqrt(w)-scaled halves; DualASNorm must give the
    same z-scores regardless -- the property that lets ONE embedder serve
    every ablation arm while only the normalizer weight changes."""
    d1, d2 = 6, 10
    cohort = _rand((40, d1 + d2), 5)
    queries = _rand((3, d1 + d2), 6)
    protos = _rand((4, d1 + d2), 7)

    def scale(x, a, b):
        out = x.copy()
        out[:, :d1] *= a
        out[:, d1:] *= b
        return out

    base = DualASNorm(cohort, d1, top_k=15, weight=0.6).normalize(queries, protos)
    scaled = DualASNorm(scale(cohort, 0.3, 9.0), d1, top_k=15, weight=0.6).normalize(
        scale(queries, 0.3, 9.0), scale(protos, 0.3, 9.0)
    )
    np.testing.assert_allclose(base, scaled, atol=1e-10)


def test_dual_asnorm_orientation_lower_is_genuine():
    """A query lying exactly on a prototype must get a smaller normalized
    score against that prototype than a far-away query does."""
    rng = np.random.default_rng(8)
    d1 = 5
    cohort = rng.normal(size=(50, d1 + 5))
    proto = rng.normal(size=(1, d1 + 5))
    near = proto.copy()
    far = proto + 10.0
    dual = DualASNorm(cohort, d1, top_k=20, weight=0.5)
    assert dual.normalize(near, proto)[0, 0] < dual.normalize(far, proto)[0, 0]


def test_dual_asnorm_validates_inputs():
    cohort = _rand((30, 10), 9)
    with pytest.raises(ValueError):
        DualASNorm(cohort, split_dim=10, top_k=5)      # split at/after last dim
    with pytest.raises(ValueError):
        DualASNorm(cohort, split_dim=4, top_k=5, weight=1.5)


def test_train_fusion_for_ablation_accepts_wider_second_dim():
    torch.manual_seed(0)
    index = {
        f"spk{i}": [np.random.default_rng(i).normal(size=ECAPA_DIM + 768).astype(np.float32)
                    for _ in range(2)]
        for i in range(3)
    }
    model = train_fusion_for_ablation(
        "ecapa_only", index, n_way=2, k_shot=1, n_query=1, n_episodes=0,
        seed=0, residual_init=False, second_dim=768,
    )
    raw = torch.from_numpy(np.stack(index["spk0"])).float()
    e, w = raw[..., :ECAPA_DIM], raw[..., ECAPA_DIM:]
    out = model(e, w)
    assert out.shape[-1] == 256  # FUSION_DIM, no matmul crash on 768-d input
