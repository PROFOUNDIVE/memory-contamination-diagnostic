from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from memcontam.experiment.phase12.filter_challenge.registry_common import StrictRegistry
from memcontam.experiment.phase12.filter_challenge.registry_manifests import (
    OperationalSuiteRegistry,
    ProbeInventoryRegistry,
)
from memcontam.experiment.phase12.filter_challenge.registry_search import SearchConfig


class BuildArchiveRun(StrictRegistry):
    schema_version: Literal["filter_challenge_build_archive_v1"] = (
        "filter_challenge_build_archive_v1"
    )
    evidence_layer: Literal["build"] = "build"
    scientific_result: Literal[False] = False
    fixture_only: Literal[True] = True
    implementation_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    freeze_id: str
    run_id: str
    search_config_id: str
    search_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_probe_inventory_id: str
    calibration_probe_inventory_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    operational_probe_suite_manifest_id: str
    operational_probe_suite_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_calls_issued: Literal[0] = 0


class ArtifactBinding(StrictRegistry):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    count: Literal[1] = 1


class PublicArtifactManifest(StrictRegistry):
    schema_version: Literal["filter_challenge_build_manifest_v1"] = (
        "filter_challenge_build_manifest_v1"
    )
    status: Literal["completed"] = "completed"
    artifacts: dict[str, ArtifactBinding]


class BuildArchiveSeal(StrictRegistry):
    schema_version: Literal["filter_challenge_build_archive_seal_v1"] = (
        "filter_challenge_build_archive_seal_v1"
    )
    public_artifact_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    implementation_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    freeze_id: str
    run_id: str
    search_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class BuildArchiveReport(StrictRegistry):
    schema_version: Literal["filter_challenge_build_archive_report_v1"] = (
        "filter_challenge_build_archive_report_v1"
    )
    archive_valid: Literal[True] = True
    reason_code: None = None
    implementation_commit: str
    freeze_id: str
    run_id: str
    search_config_id: str
    search_config_hash: str
    calibration_probe_inventory_id: str
    calibration_probe_inventory_manifest_hash: str
    operational_probe_suite_manifest_id: str
    operational_probe_suite_manifest_hash: str
    public_artifact_manifest_hash: str
    archive_seal_hash: str
    provider_calls_issued: Literal[0] = 0


@dataclass(frozen=True, slots=True)
class BuildArchiveRequest:
    search_config: SearchConfig
    inventory: ProbeInventoryRegistry
    suite: OperationalSuiteRegistry
    implementation_commit: str
    freeze_id: str
    run_id: str
    output_root: Path


@dataclass(frozen=True, slots=True)
class ArchiveValidationRequest:
    archive: Path
    expected_implementation_commit: str
    expected_search_config_hash: str


__all__ = (
    "ArchiveValidationRequest",
    "ArtifactBinding",
    "BuildArchiveReport",
    "BuildArchiveRequest",
    "BuildArchiveRun",
    "BuildArchiveSeal",
    "PublicArtifactManifest",
)
