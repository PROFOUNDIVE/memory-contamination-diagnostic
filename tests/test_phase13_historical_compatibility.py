from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Final

import pytest


REPOSITORY_ROOT: Final = Path(__file__).parents[1]
REGISTRY_PATH: Final = (
    REPOSITORY_ROOT / "data/phase13/authority/historical_compatibility_v1.json"
)
CURRENT_V1_CONFIG_PATH: Final = (
    REPOSITORY_ROOT / "configs/phase13/clean_prefix_calibration_v1.yaml"
)
PROVENANCE_MAP_PATH: Final = Path(
    "/home/hyunwoo/gdrive_undergrad_research/PeerJ fast-track/5주차/"
    "2026-08-12_Phase13_Calibration_Evidence_Provenance_and_AgentOps_Map.md"
)
CURRENT_V1_CONFIG_SHA256: Final = (
    "347ed902e959c7d77f19284129c6dd46018d4900a7a6bc590240bb3845cb8734"
)
PROVENANCE_MAP_SHA256: Final = (
    "50b73ba708895cccebc451e0673e85d4aa5945047755fb2780efc8a97acafb19"
)
EXPECTED_IDENTITIES: Final = {
    "historical_execution/implementation_commit": (
        "40b389c3c5035b0054398e3378bcdce55e5afe33"
    ),
    "historical_execution/run_id": "phase13-pre-main-calibration-15usd-rerun1",
    "historical_execution/config/path": "configs/phase13/clean_prefix_calibration_v1.yaml",
    "historical_execution/config/sha256": (
        "c97608f1d6f3bafbcb93a30c711ef979ebccacd8341323b1bfc048a6b35a0040"
    ),
    "historical_execution/request_sha256": (
        "0a8af1c9fc1d9270a4c439670532489b968b5a42f853b7d026e16b0a2b00879e"
    ),
    "historical_execution/authorization_sha256": (
        "c5da3b5e06466fda82b4c34913dbd8e316f16e06520c608bf66fbe0d6e813b9e"
    ),
    "historical_execution/artifacts/rates.json/sha256": (
        "85468516eca29a4c3895a883081435b6b94bae2ec29acf9190ab49de3f5c6645"
    ),
    "historical_execution/artifacts/eligibility.jsonl/sha256": (
        "aab072898f156516e9baaa8617e6c62116b8a93d8c29e3ff452fffc9ea3b277b"
    ),
    "historical_execution/artifacts/calls.jsonl/sha256": (
        "1642931edc995e3f69d44e933620b5f20dce62da03b60bdac7e57034c3cec2b8"
    ),
    "historical_execution/artifacts/accounting.json/sha256": (
        "57f7155f1e78f71c9e75268a37bc5c274a4e81d38f1632c7a0a777546fa46178"
    ),
    "historical_execution/artifacts/artifact_manifest.json/sha256": (
        "59d98125b6721fada1f9f078ccef393fb640cd63736e6b8f74ebb308b5f9cfb9"
    ),
    "historical_execution/sealed_archive/path": (
        "runs/phase13-clean-prefix-calibration-v1/"
        "phase13-pre-main-calibration-15usd-rerun1/"
    ),
    "historical_execution/sealed_archive/availability": "external_reference_unavailable",
    "source_provenance_map/path": str(PROVENANCE_MAP_PATH),
    "source_provenance_map/sha256": PROVENANCE_MAP_SHA256,
}


class HistoricalIdentityMismatch(AssertionError):
    code: Final = "HISTORICAL_IDENTITY_MISMATCH"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registry(path: Path = REGISTRY_PATH) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _value_at(registry: dict[str, object], identity_path: str) -> object:
    value: object = registry
    for key in identity_path.split("/"):
        if not isinstance(value, dict) or key not in value:
            raise HistoricalIdentityMismatch(
                f"{HistoricalIdentityMismatch.code}: missing {identity_path}"
            )
        value = value[key]
    return value


def _assert_historical_identities(registry: dict[str, object]) -> None:
    for identity_path, expected in EXPECTED_IDENTITIES.items():
        if _value_at(registry, identity_path) != expected:
            raise HistoricalIdentityMismatch(
                f"{HistoricalIdentityMismatch.code}: {identity_path}"
            )


def _mutate(registry: dict[str, object], identity_path: str) -> None:
    keys = identity_path.split("/")
    parent: object = registry
    for key in keys[:-1]:
        assert isinstance(parent, dict)
        parent = parent[key]
    assert isinstance(parent, dict)
    parent[keys[-1]] = "mutated"


def test_registry_binds_literal_historical_identity_and_real_source_hashes() -> None:
    assert _sha256(PROVENANCE_MAP_PATH) == PROVENANCE_MAP_SHA256
    assert _sha256(CURRENT_V1_CONFIG_PATH) == CURRENT_V1_CONFIG_SHA256
    assert CURRENT_V1_CONFIG_PATH.stat().st_size == 3990

    registry = _registry()

    _assert_historical_identities(registry)


def test_absent_sealed_archive_is_only_an_unavailable_external_reference() -> None:
    registry = _registry()
    archive_path = REPOSITORY_ROOT / str(
        _value_at(registry, "historical_execution/sealed_archive/path")
    )

    assert not archive_path.exists()
    assert (
        _value_at(registry, "historical_execution/sealed_archive/availability")
        == "external_reference_unavailable"
    )


@pytest.mark.parametrize("identity_path", EXPECTED_IDENTITIES)
def test_each_mutated_identity_is_rejected(
    tmp_path: Path,
    identity_path: str,
) -> None:
    registry = copy.deepcopy(_registry())
    _mutate(registry, identity_path)
    copied_registry = tmp_path / REGISTRY_PATH.name
    copied_registry.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(
        HistoricalIdentityMismatch,
        match=HistoricalIdentityMismatch.code,
    ):
        _assert_historical_identities(_registry(copied_registry))
