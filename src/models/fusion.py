"""Gated Attention Fusion (F4) [3.5, Pers. 4.1-4.3].

    e'_i = W_i @ e_i + b_i                for i in {ECAPA, Whisper}   (4.1)
    g    = sigmoid(W_g @ [e'_1 ; e'_2] + b_g)                         (4.2)
    e_fusion = g ⊙ e'_1 + (1 - g) ⊙ e'_2                              (4.3)

All parameters (W_1, b_1, W_2, b_2, W_g, b_g) are trainable and meant to be
learned end-to-end together with the prototypical network head (F5) --
see plan/03-architecture.md §5 for why a trainable gate (vs a fixed weight)
matters: backbone contribution should adapt per-sample (e.g. Whisper more
reliable on noisy audio, ECAPA more reliable on clean audio).

`mode` implements the ablation-study feature flag from F4-06 / Tabel 4.3:
  - "fusion"       -> full gated attention fusion (A3, the proposed system)
  - "ecapa_only"   -> bypass fusion, use only the projected ECAPA embedding (A1)
  - "whisper_only" -> bypass fusion, use only the projected Whisper embedding (A2)
The projection layers stay identical across modes so A1/A2/A3 differ *only*
in whether fusion is applied, isolating fusion's contribution as intended.

`residual_init` (default False, opt-in): initialize the layer so its output
starts equal to the *raw pretrained ECAPA* embedding rather than a random
projection. Diagnostics (scripts/diagnose_accuracy_gap.py) showed a randomly
initialized, undertrained projection *destroys* ECAPA's native discriminative
space (A1 collapsed to ~16% vs raw ECAPA's ~79%). With residual init the
system *starts* at ECAPA's quality and training can only refine it: proj_ecapa
begins as identity, and the gate is biased so `g -> 1` (ECAPA-dominant), so
Whisper enters as a small trainable residual instead of overwhelming ECAPA.
Default is kept False so A1/A2/A3 remain a fair random-init ablation unless the
experiment explicitly opts in.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

FUSION_DIM = 256
VALID_MODES = ("fusion", "ecapa_only", "whisper_only")


class GatedAttentionFusion(nn.Module):
    def __init__(
        self,
        ecapa_dim: int = 192,
        whisper_dim: int = 512,
        fusion_dim: int = FUSION_DIM,
        mode: str = "fusion",
        residual_init: bool = False,
    ) -> None:
        super().__init__()
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        self.mode = mode
        self.fusion_dim = fusion_dim
        self.residual_init = residual_init

        self.proj_ecapa = nn.Linear(ecapa_dim, fusion_dim)      # W_1, b_1 (Pers. 4.1)
        self.proj_whisper = nn.Linear(whisper_dim, fusion_dim)  # W_2, b_2 (Pers. 4.1)
        self.gate = nn.Linear(fusion_dim * 2, fusion_dim)       # W_g, b_g (Pers. 4.2)

        if residual_init:
            self._apply_residual_init(ecapa_dim, fusion_dim)

    # NOTE: ScoreFusionEmbed (below) is a *parameter-free* alternative to this
    # trainable gated fusion, used by Experiment 2a.

    @torch.no_grad()
    def _apply_residual_init(self, ecapa_dim: int, fusion_dim: int) -> None:
        """Start from raw-ECAPA-preserving weights (see class docstring).

        proj_ecapa becomes a (partial) identity so e'_1 carries the raw ECAPA
        vector unchanged (padded with zeros if fusion_dim > ecapa_dim); the gate
        starts input-independent with g = sigmoid(+4) ~= 0.982, so fusion output
        begins ECAPA-dominant. proj_whisper is left at its default random init so
        Whisper stays trainable (and so whisper_only/A2 is unaffected).
        """
        d = min(ecapa_dim, fusion_dim)
        self.proj_ecapa.weight.zero_()
        self.proj_ecapa.weight[:d, :d] = torch.eye(d)
        self.proj_ecapa.bias.zero_()
        # gate input-independent at init, biased hard toward the ECAPA branch
        self.gate.weight.zero_()
        self.gate.bias.fill_(4.0)

    def forward(self, ecapa_emb: torch.Tensor, whisper_emb: torch.Tensor) -> torch.Tensor:
        """ecapa_emb: (..., ecapa_dim), whisper_emb: (..., whisper_dim) ->
        (..., fusion_dim), L2-normalized."""
        e1 = self.proj_ecapa(ecapa_emb)     # e'_1, Pers. 4.1
        e2 = self.proj_whisper(whisper_emb)  # e'_2, Pers. 4.1

        if self.mode == "ecapa_only":
            fused = e1
        elif self.mode == "whisper_only":
            fused = e2
        else:  # "fusion"
            gate_input = torch.cat([e1, e2], dim=-1)
            g = torch.sigmoid(self.gate(gate_input))  # Pers. 4.2
            fused = g * e1 + (1 - g) * e2              # Pers. 4.3

        return F.normalize(fused, p=2, dim=-1)


class ScoreFusionEmbed(nn.Module):
    """Score-level (late) fusion as a drop-in fusion module (Experiment 2a).

    Instead of learning a joint embedding, we combine the two backbones at the
    *distance* level: for a query and a speaker prototype,

        score = w * d_ecapa + (1 - w) * d_whisper      (w = weight on ECAPA)

    with d_* the cosine distance in each backbone's own (native, undamaged)
    space. This is realized WITHOUT any new machinery by emitting the weighted
    concatenation of the two L2-normalized embeddings:

        out = [ sqrt(w) * e_ecapa_hat ; sqrt(1 - w) * e_whisper_hat ]

    Squared Euclidean distance between two such vectors equals
    2 * (w * cos_dist_ecapa + (1 - w) * cos_dist_whisper), i.e. exactly the
    score fusion above (monotone) -- so the existing prototype-distance system,
    threshold calibration, and FSCIL harness all work unchanged. The output is
    already unit-norm (w + (1 - w) = 1), and the module is parameter-free (a
    single frozen anchor keeps `.parameters()`/`.to(device)` working like any
    other fusion module). Ablation modes map to fixed weights:
    ecapa_only -> w=1, whisper_only -> w=0, fusion -> the configured weight.
    """

    def __init__(self, mode: str = "fusion", weight: float = 0.4) -> None:
        super().__init__()
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        self.mode = mode
        if mode == "ecapa_only":
            self.weight = 1.0
        elif mode == "whisper_only":
            self.weight = 0.0
        else:
            self.weight = float(weight)
        # keeps .parameters() non-empty so device detection / .to() behave like
        # a normal fusion module (see src/system.py::embed).
        self._anchor = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, ecapa_emb: torch.Tensor, whisper_emb: torch.Tensor) -> torch.Tensor:
        u = F.normalize(ecapa_emb, p=2, dim=-1)
        v = F.normalize(whisper_emb, p=2, dim=-1)
        we = self.weight ** 0.5
        ww = (1.0 - self.weight) ** 0.5
        return torch.cat([we * u, ww * v], dim=-1)
