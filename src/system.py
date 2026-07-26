"""Integrated speaker identification system (F7-01) [4.1].

Wires together every component built in F2-F6 into the single pipeline
Gambar 4.1 describes end to end:

    audio path -> preprocessing + backbone (cached, F2/F3)
               -> Gated Attention Fusion (F4)
               -> prototype-distance decision + fixed EER threshold (F5)
               -> continual prototype update / novel-speaker registration (F6)

This is the object F8's evaluation harness drives, and what F7-03's
checkpoint captures (fusion weights + threshold + speaker database).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.continual.manager import ContinualLearningManager, ContinualProcessResult
from src.continual.prototype_update import initialize_prototype
from src.continual.speaker_database import SpeakerDatabase
from src.features.cache import get_or_compute_embedding
from src.models.fusion import FUSION_DIM, GatedAttentionFusion
from src.prototypical.data import ECAPA_DIM, WHISPER_DIM
from src.prototypical.score_norm import ASNorm, DualASNorm


class SpeakerIdentificationSystem:
    def __init__(
        self,
        fusion_model: GatedAttentionFusion,
        threshold: float,
        database: SpeakerDatabase | None = None,
        continual_mode: str = "running_average",
        min_samples_for_new_speaker: int = 1,
        silhouette_threshold: float = 0.5,
        whisper_backbone: str = "whisper",
        score_normalizer=None,
    ) -> None:
        self.fusion_model = fusion_model
        self.fusion_model.eval()
        self.threshold = threshold
        self.whisper_backbone = whisper_backbone
        self.database = database if database is not None else SpeakerDatabase()
        self.continual_mode = continual_mode
        # Experiment 3b (exp3b_asnorm): optional AS-Norm score normalizer;
        # None (default) keeps the original raw-distance behaviour.
        self.score_normalizer = score_normalizer
        self.manager = ContinualLearningManager(
            self.database,
            threshold=threshold,
            mode=continual_mode,
            min_samples_for_new_speaker=min_samples_for_new_speaker,
            silhouette_threshold=silhouette_threshold,
            score_normalizer=score_normalizer,
        )

    def embed(self, audio_path: str | Path) -> np.ndarray:
        """Preprocessing + frozen backbone (cached) + trainable fusion ->
        one fused embedding for `audio_path`."""
        e1 = get_or_compute_embedding(audio_path, "ecapa")
        e2 = get_or_compute_embedding(audio_path, self.whisper_backbone)
        device = next(self.fusion_model.parameters()).device
        with torch.no_grad():
            fused = self.fusion_model(
                torch.from_numpy(e1).to(device), torch.from_numpy(e2).to(device)
            )
        return fused.cpu().numpy()

    def enroll(self, speaker_id: str, audio_paths: list[str | Path]) -> None:
        """Register a brand-new known speaker from one-or-more enrollment
        utterances (Pers. 4.4 initialization)."""
        embeddings = np.stack([self.embed(p) for p in audio_paths])
        prototype, n = initialize_prototype(embeddings)
        self.database.set(speaker_id, prototype, n)

    def process(self, audio_path: str | Path) -> ContinualProcessResult:
        """Full open-set decision + continual-learning update for one
        incoming utterance."""
        embedding = self.embed(audio_path)
        return self.manager.process_sample(embedding)

    def score(self, audio_path: str | Path) -> ContinualProcessResult:
        """Open-set decision WITHOUT any continual-learning state change
        (Experiment 3c detection path for unknown queries)."""
        embedding = self.embed(audio_path)
        return self.manager.score_sample(embedding)

    def save(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.fusion_model.state_dict(), "mode": self.fusion_model.mode},
            dir_path / "fusion.ckpt",
        )
        self.database.save(dir_path / "speaker_database.npz")

        normalizer = self.score_normalizer
        score_normalizer_meta: dict = {"type": "none"}
        if isinstance(normalizer, DualASNorm):
            score_normalizer_meta = {
                "type": "dual_asnorm",
                "top_k": normalizer.top_k,
                "split_dim": normalizer.split_dim,
                "weight": normalizer.weight,
            }
            np.save(dir_path / "score_normalizer_cohort.npy", normalizer.cohort)
        elif isinstance(normalizer, ASNorm):
            score_normalizer_meta = {"type": "asnorm", "top_k": normalizer.top_k}
            np.save(dir_path / "score_normalizer_cohort.npy", normalizer.cohort)

        metadata = {
            "threshold": self.threshold,
            "continual_mode": self.continual_mode,
            "fusion_mode": self.fusion_model.mode,
            "whisper_backbone": self.whisper_backbone,
            "min_samples_for_new_speaker": self.manager.min_samples_for_new_speaker,
            "silhouette_threshold": self.manager.silhouette_threshold,
            "score_normalizer": score_normalizer_meta,
        }
        (dir_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, dir_path: str | Path) -> "SpeakerIdentificationSystem":
        dir_path = Path(dir_path)
        metadata = json.loads((dir_path / "metadata.json").read_text(encoding="utf-8"))

        fusion_model = GatedAttentionFusion(ECAPA_DIM, WHISPER_DIM, FUSION_DIM, mode=metadata["fusion_mode"])
        checkpoint = torch.load(dir_path / "fusion.ckpt", map_location="cpu", weights_only=True)
        fusion_model.load_state_dict(checkpoint["state_dict"])

        database = SpeakerDatabase.load(dir_path / "speaker_database.npz")

        norm_meta = metadata.get("score_normalizer", {"type": "none"})
        score_normalizer = None
        if norm_meta["type"] == "asnorm":
            cohort = np.load(dir_path / "score_normalizer_cohort.npy")
            score_normalizer = ASNorm(cohort, top_k=norm_meta["top_k"])
        elif norm_meta["type"] == "dual_asnorm":
            cohort = np.load(dir_path / "score_normalizer_cohort.npy")
            score_normalizer = DualASNorm(
                cohort, split_dim=norm_meta["split_dim"], top_k=norm_meta["top_k"], weight=norm_meta["weight"]
            )

        return cls(
            fusion_model=fusion_model,
            threshold=metadata["threshold"],
            database=database,
            continual_mode=metadata["continual_mode"],
            whisper_backbone=metadata.get("whisper_backbone", "whisper"),
            min_samples_for_new_speaker=metadata.get("min_samples_for_new_speaker", 1),
            silhouette_threshold=metadata.get("silhouette_threshold", 0.5),
            score_normalizer=score_normalizer,
        )
