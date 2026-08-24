from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_unfrozen_meb_threshold_prevents_published_track2_seal() -> None:
    assert not (ROOT / "data/phase13/rag/legacy/manifest.json").exists()
