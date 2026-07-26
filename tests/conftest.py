import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

METADATA_DIR = REPO_ROOT / "data" / "raw" / "metadata"

VOX1_META = METADATA_DIR / "vox1_meta.csv"
VOX2_META = METADATA_DIR / "vox2_meta.csv"
VOX1_IDEN_SPLIT = METADATA_DIR / "vox1_iden_split.txt"

requires_real_metadata = pytest.mark.skipif(
    not (VOX1_META.exists() and VOX2_META.exists()),
    reason=(
        "Real VoxCeleb metadata not found under data/raw/metadata/. "
        "Run: python scripts/download_voxceleb.py --stage metadata"
    ),
)

requires_vox1_manifest = pytest.mark.skipif(
    not VOX1_IDEN_SPLIT.exists(),
    reason=(
        "VoxCeleb1 iden_split.txt not found. "
        "Run: python scripts/download_voxceleb.py --stage metadata"
    ),
)

SAMPLE_AUDIO_DIR = REPO_ROOT / "data" / "raw" / "audio" / "vox1_sample"
SAMPLE_AUDIO_MANIFEST = SAMPLE_AUDIO_DIR / "manifest.csv"

requires_sample_audio = pytest.mark.skipif(
    not SAMPLE_AUDIO_MANIFEST.exists(),
    reason=(
        "Real VoxCeleb1 sample audio not found. "
        "Run: python scripts/fetch_sample_audio.py"
    ),
)


@pytest.fixture(scope="session")
def sample_audio_manifest() -> pd.DataFrame:
    if not SAMPLE_AUDIO_MANIFEST.exists():
        pytest.skip("Real VoxCeleb1 sample audio not available")
    return pd.read_csv(SAMPLE_AUDIO_MANIFEST)


@pytest.fixture(scope="session")
def real_catalog() -> pd.DataFrame:
    from src.data.voxceleb import load_combined_catalog

    if not (VOX1_META.exists() and VOX2_META.exists()):
        pytest.skip("Real VoxCeleb metadata not available")
    return load_combined_catalog(VOX1_META, VOX2_META)


@pytest.fixture(scope="session")
def vox1_utterance_manifest() -> pd.DataFrame:
    from src.data.voxceleb import load_vox1_utterance_manifest

    if not VOX1_IDEN_SPLIT.exists():
        pytest.skip("VoxCeleb1 iden_split.txt not available")
    return load_vox1_utterance_manifest(VOX1_IDEN_SPLIT)
