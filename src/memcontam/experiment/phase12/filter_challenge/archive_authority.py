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
    if any(
        set(candidate.strata) != set(authority.search_config.required_strata)
        for candidate in authority.search_config.coverage_contract_candidates
    ):
        raise FilterChallengeArchiveError("REGISTRY_AUTHORITY_MISMATCH")
    for assessment in archive.assessments:
        selected_suite = next(
            (
                candidate
                for candidate in authority.search_config.suite_candidates
                if candidate.operational_probe_suite_id == assessment.operational_probe_suite_id
            ),
            None,
        )
        if (
            assessment.calibration_probe_inventory_id != closure.calibration_probe_inventory_id
            or assessment.calibration_probe_inventory_manifest_hash
            != closure.calibration_probe_inventory_manifest_hash
            or assessment.operational_probe_suite_manifest_hash
            != closure.operational_probe_suite_manifest_hash
            or selected_suite is None
            or assessment.probe_id not in selected_suite.probe_ids
        ):
            raise FilterChallengeArchiveError("REGISTRY_AUTHORITY_MISMATCH")
    for aggregate in archive.candidate_aggregates:
        selected_suite = next(
            (
                candidate
                for candidate in authority.search_config.suite_candidates
                if candidate.operational_probe_suite_id == aggregate.operational_probe_suite_id
            ),
            None,
        )
        parameters = aggregate.aggregation_parameter_tuple
        kappa = next(
            (candidate for candidate in authority.search_config.kappa_candidates if candidate.kappa_id == parameters[0]),
            None,
        ) if len(parameters) == 2 else None
        coverage = next(
            (
                candidate
                for candidate in authority.search_config.coverage_contract_candidates
                if candidate.coverage_contract_id == parameters[1]
            ),
            None,
        ) if len(parameters) == 2 else None
        rows = tuple(row for row in archive.assessments if row.candidate_entry_id == aggregate.candidate_entry_id)
        witnesses_per_probe = {probe_id: sum(row.probe_id == probe_id for row in rows if row.probe_disposition == "witness") for probe_id in aggregate.witness_probe_ids}
        coverage_satisfied = (
            kappa is not None
            and aggregate.n_strictly_evaluable >= kappa.min_total_evaluable_replicates
            and aggregate.n_distinct_evaluable_probes >= kappa.min_distinct_evaluable_probes
            and aggregate.n_distinct_witness_probes >= kappa.min_distinct_witness_probes
            and all(count >= kappa.min_witness_replicates_per_probe for count in witnesses_per_probe.values())
        )
        if (
            aggregate.calibration_probe_inventory_id != closure.calibration_probe_inventory_id
            or aggregate.calibration_probe_inventory_manifest_hash
            != closure.calibration_probe_inventory_manifest_hash
            or aggregate.operational_probe_suite_manifest_hash
            != closure.operational_probe_suite_manifest_hash
            or selected_suite is None
            or any(probe_id not in selected_suite.probe_ids for probe_id in aggregate.witness_probe_ids)
            or coverage is None
            or aggregate.coverage_contract_id != coverage.coverage_contract_id
            or parameters != (kappa.kappa_id, coverage.coverage_contract_id) if kappa is not None and coverage is not None else True
        ):
            raise FilterChallengeArchiveError("REGISTRY_AUTHORITY_MISMATCH")
        expected_state = (
            ("not_evaluable", "active", "FAIL_OPEN_NOT_EVALUABLE")
            if not coverage_satisfied
            else ("contradicted", "quarantine", "CONTRADICTED")
            if aggregate.n_witness
            else ("not_contradicted", "active", "NOT_CONTRADICTED")
        )
        if (aggregate.assessment_state, aggregate.final_routing_decision, aggregate.final_reason_code) != expected_state:
            raise FilterChallengeArchiveError("AGGREGATE_STATE_INVALID")


__all__ = ("ArchiveRegistryAuthority", "validate_archive_authority")
