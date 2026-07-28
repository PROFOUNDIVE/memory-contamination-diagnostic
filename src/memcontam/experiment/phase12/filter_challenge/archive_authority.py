from __future__ import annotations

from dataclasses import dataclass

from memcontam.experiment.phase12.filter_challenge.records import (
    FilterChallengeArchive,
    FilterChallengeArchiveError,
)
from memcontam.experiment.phase12.filter_challenge.registry import (
    RegistryClosure,
    validate_registry_closure,
)
from memcontam.experiment.phase12.filter_challenge.registry_common import RegistryValidationError
from memcontam.experiment.phase12.filter_challenge.registry_manifests import (
    OperationalSuiteRegistry,
    ProbeInventoryRegistry,
)
from memcontam.experiment.phase12.filter_challenge.registry_search import SearchConfig


@dataclass(frozen=True, slots=True)
class ArchiveRegistryAuthority:
    search_config: SearchConfig
    inventory: ProbeInventoryRegistry
    suite: OperationalSuiteRegistry

    def registry_closure(self) -> RegistryClosure:
        return validate_registry_closure(self.search_config, self.inventory, self.suite)


def validate_archive_authority(
    archive: FilterChallengeArchive, authority: ArchiveRegistryAuthority
) -> None:
    try:
        closure = authority.registry_closure()
    except RegistryValidationError as error:
        raise FilterChallengeArchiveError(error.code) from error
    suite_ids = {candidate.operational_probe_suite_id for candidate in authority.search_config.suite_candidates}
    coverage_ids = {
        candidate.coverage_contract_id for candidate in authority.search_config.coverage_contract_candidates
    }
    if any(
        set(candidate.strata) != set(authority.search_config.required_strata)
        for candidate in authority.search_config.coverage_contract_candidates
    ):
        raise FilterChallengeArchiveError("REGISTRY_AUTHORITY_MISMATCH")
    for assessment in archive.assessments:
        if (
            assessment.calibration_probe_inventory_id != closure.calibration_probe_inventory_id
            or assessment.calibration_probe_inventory_manifest_hash
            != closure.calibration_probe_inventory_manifest_hash
            or assessment.operational_probe_suite_manifest_hash
            != closure.operational_probe_suite_manifest_hash
            or assessment.probe_id not in authority.inventory.probe_ids
            or assessment.operational_probe_suite_id not in suite_ids
        ):
            raise FilterChallengeArchiveError("REGISTRY_AUTHORITY_MISMATCH")
    for aggregate in archive.candidate_aggregates:
        if (
            aggregate.calibration_probe_inventory_id != closure.calibration_probe_inventory_id
            or aggregate.calibration_probe_inventory_manifest_hash
            != closure.calibration_probe_inventory_manifest_hash
            or aggregate.operational_probe_suite_manifest_hash
            != closure.operational_probe_suite_manifest_hash
            or aggregate.operational_probe_suite_id not in suite_ids
            or aggregate.coverage_contract_id not in coverage_ids
        ):
            raise FilterChallengeArchiveError("REGISTRY_AUTHORITY_MISMATCH")


__all__ = ("ArchiveRegistryAuthority", "validate_archive_authority")
