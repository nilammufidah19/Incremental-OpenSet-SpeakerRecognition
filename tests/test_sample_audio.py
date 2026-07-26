"""Tests exercising src/data/splits.py against a small REAL-audio sample.

Unlike tests/test_data_splits.py (which only checks utterance-ID *strings*
from iden_split.txt), these tests actually decode WAV bytes fetched by
scripts/fetch_sample_audio.py -- see that script's docstring for provenance
(public, ungated mirror of the official VoxCeleb1 test split; genuine
speech, not synthetic).
"""
from __future__ import annotations

from pathlib import Path

import soundfile as sf

from src.data.splits import assign_support_query
from tests.conftest import REPO_ROOT, requires_sample_audio


@requires_sample_audio
def test_sample_manifest_files_exist_and_are_decodable(sample_audio_manifest):
    assert len(sample_audio_manifest) > 0
    for _, row in sample_audio_manifest.iterrows():
        wav_path = REPO_ROOT / row["path"]
        assert wav_path.exists(), f"missing {wav_path}"
        info = sf.info(wav_path)
        assert info.samplerate == 16000
        assert info.frames > 0


@requires_sample_audio
def test_sample_has_multiple_speakers_with_enough_utterances(sample_audio_manifest):
    counts = sample_audio_manifest["speaker_id"].value_counts()
    eligible = counts[counts >= 2]
    assert len(eligible) >= 10, (
        f"expected >=10 speakers with >=2 utterances in the sample, got {len(eligible)}"
    )


@requires_sample_audio
def test_assign_support_query_end_to_end_on_real_audio(sample_audio_manifest):
    """Run F1-08's assign_support_query on the real sample manifest, then
    actually decode every assigned support/query file to confirm the whole
    speaker -> utterance_id -> WAV chain is genuinely playable audio, not
    just consistent path strings."""
    manifest = sample_audio_manifest.rename(columns={"utterance_id": "utterance_id"}).copy()
    # splits.assign_support_query expects columns speaker_id/utterance_id;
    # here utterance_id must resolve back to a real file, so use the file
    # path itself as the utterance_id.
    manifest["utterance_id"] = manifest["path"]

    speaker_ids = sorted(manifest["speaker_id"].unique())
    assignment = assign_support_query(manifest, speaker_ids, k_shot=1, seed=0)

    for speaker_id in speaker_ids:
        support = assignment[speaker_id]["support"]
        query = assignment[speaker_id]["query"]
        assert len(support) == 1
        assert len(query) >= 1
        assert set(support).isdisjoint(query)

        for utt_path in support + query:
            full_path = REPO_ROOT / utt_path
            data, sr = sf.read(full_path)
            assert sr == 16000
            assert len(data) > 0
