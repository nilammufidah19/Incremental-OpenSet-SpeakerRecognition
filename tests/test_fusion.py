"""Tests for src/models/fusion.py (F4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.features.cache import get_or_compute_embedding
from src.models.fusion import FUSION_DIM, GatedAttentionFusion
from tests.conftest import REPO_ROOT, requires_sample_audio

ECAPA_DIM = 192
WHISPER_DIM = 512


def test_fusion_output_shape_and_normalization():
    torch.manual_seed(0)
    model = GatedAttentionFusion(ECAPA_DIM, WHISPER_DIM, FUSION_DIM, mode="fusion")
    e1 = torch.randn(ECAPA_DIM)
    e2 = torch.randn(WHISPER_DIM)
    out = model(e1, e2)

    assert out.shape == (FUSION_DIM,)
    assert abs(torch.linalg.norm(out).item() - 1.0) < 1e-5


def test_fusion_supports_batched_input():
    torch.manual_seed(0)
    model = GatedAttentionFusion(ECAPA_DIM, WHISPER_DIM, FUSION_DIM, mode="fusion")
    batch = 8
    e1 = torch.randn(batch, ECAPA_DIM)
    e2 = torch.randn(batch, WHISPER_DIM)
    out = model(e1, e2)

    assert out.shape == (batch, FUSION_DIM)
    norms = torch.linalg.norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones(batch), atol=1e-5)


def test_fusion_gate_values_in_unit_interval():
    torch.manual_seed(0)
    model = GatedAttentionFusion(ECAPA_DIM, WHISPER_DIM, FUSION_DIM, mode="fusion")
    e1 = torch.randn(4, ECAPA_DIM)
    e2 = torch.randn(4, WHISPER_DIM)

    e1p = model.proj_ecapa(e1)
    e2p = model.proj_whisper(e2)
    gate = torch.sigmoid(model.gate(torch.cat([e1p, e2p], dim=-1)))

    assert torch.all(gate >= 0.0) and torch.all(gate <= 1.0)


def test_fusion_mode_ecapa_only_ignores_whisper_embedding():
    torch.manual_seed(0)
    model = GatedAttentionFusion(ECAPA_DIM, WHISPER_DIM, FUSION_DIM, mode="ecapa_only")
    e1 = torch.randn(ECAPA_DIM)
    out_a = model(e1, torch.randn(WHISPER_DIM))
    out_b = model(e1, torch.randn(WHISPER_DIM) * 100)  # very different whisper input
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_fusion_mode_whisper_only_ignores_ecapa_embedding():
    torch.manual_seed(0)
    model = GatedAttentionFusion(ECAPA_DIM, WHISPER_DIM, FUSION_DIM, mode="whisper_only")
    e2 = torch.randn(WHISPER_DIM)
    out_a = model(torch.randn(ECAPA_DIM), e2)
    out_b = model(torch.randn(ECAPA_DIM) * 100, e2)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_fusion_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be one of"):
        GatedAttentionFusion(ECAPA_DIM, WHISPER_DIM, FUSION_DIM, mode="bogus")


def test_fusion_mode_changes_output_relative_to_projections():
    """The three modes should generally disagree (fusion isn't secretly
    always equal to one branch) -- sanity check the gate actually mixes."""
    torch.manual_seed(1)
    fusion = GatedAttentionFusion(ECAPA_DIM, WHISPER_DIM, FUSION_DIM, mode="fusion")
    ecapa_only = GatedAttentionFusion(ECAPA_DIM, WHISPER_DIM, FUSION_DIM, mode="ecapa_only")
    ecapa_only.load_state_dict(fusion.state_dict(), strict=False)

    e1 = torch.randn(ECAPA_DIM)
    e2 = torch.randn(WHISPER_DIM)
    out_fusion = fusion(e1, e2)
    out_ecapa_only = ecapa_only(e1, e2)
    assert not torch.allclose(out_fusion, out_ecapa_only, atol=1e-3)


# --------------------------------------------------------------------------
# Integration: real cached embeddings (F3-06) through fusion (F4)
# --------------------------------------------------------------------------


@requires_sample_audio
def test_fusion_on_real_embeddings_preserves_speaker_discriminability(sample_audio_manifest):
    """Feed real ECAPA+Whisper embeddings (from F3's cache) through a
    freshly (randomly) initialized fusion layer, and check same-speaker
    pairs are still closer than different-speaker pairs on average -- a
    random untrained gate shouldn't *destroy* the discriminative signal
    already present in the two backbones (F3-05)."""
    counts = sample_audio_manifest["speaker_id"].value_counts()
    eligible = counts[counts >= 2].index.tolist()[:6]

    torch.manual_seed(0)
    model = GatedAttentionFusion(ECAPA_DIM, WHISPER_DIM, FUSION_DIM, mode="fusion")
    model.eval()

    def fused_embedding(path: str) -> np.ndarray:
        full_path = REPO_ROOT / path
        e1 = get_or_compute_embedding(full_path, "ecapa")
        e2 = get_or_compute_embedding(full_path, "whisper")
        with torch.no_grad():
            out = model(torch.from_numpy(e1), torch.from_numpy(e2))
        return out.numpy()

    same_sims, diff_sims = [], []
    embs = {}
    for spk in eligible:
        paths = sample_audio_manifest.loc[sample_audio_manifest["speaker_id"] == spk, "path"].tolist()[:2]
        embs[spk] = [fused_embedding(p) for p in paths]

    for i, spk in enumerate(eligible):
        same_sims.append(float(np.dot(embs[spk][0], embs[spk][1])))
        for other in eligible[i + 1 :]:
            diff_sims.append(float(np.dot(embs[spk][0], embs[other][0])))

    assert np.mean(same_sims) > np.mean(diff_sims), (
        f"mean same-speaker sim {np.mean(same_sims):.4f} should exceed "
        f"mean different-speaker sim {np.mean(diff_sims):.4f}"
    )
