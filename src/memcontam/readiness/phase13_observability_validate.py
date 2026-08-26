from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Final

from pydantic import ValidationError

from memcontam.evaluation.phase13_observability import (
    Phase13ObservabilityError,
    aggregate_phase13,
    reconstruct_phase13_trial,
    reconstruct_registered_sequence,
)
from memcontam.evaluation.phase13_aggregate import summarize_sequence_diagnostics
from memcontam.evaluation.phase13_observability_registration import (
    BoundIdentity,
    ObservabilityRegistrationPacket,
)
from memcontam.readiness.phase13_authority_files import AuthorityFileError, read_regular_nofollow
from .phase13_observability_models import (
    ArtifactIdentity,
    Phase13ObservabilityFixture,
    Phase13ObservabilityManifest,
    Phase13ObservabilityReport,
    Phase13Reconstruction,
    TargetSetRegistry,
)


class Phase13ObservabilityValidationError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


_EXPECTED_ARTIFACT_PATHS: Final[dict[str, str]] = {
    "observability_registration_packet": "registration_packet_v1.json",
    "synthetic_contract_fixture": "fixture_v1.json",
    "target_set_registry": "target_set_registry_v1.json",
}
_EXPECTED_IMPLEMENTATION_PATHS: Final[dict[str, str]] = {
    "observability_registration": "src/memcontam/evaluation/phase13_observability_registration.py",
    "observability_sequence": "src/memcontam/evaluation/phase13_observability_sequence.py",
    "observable_contract": "src/memcontam/evaluation/phase13_observability_models.py",
    "trial_and_sequential_reconstruction": "src/memcontam/evaluation/phase13_observability.py",
    "runtime_identity_and_lineage": "src/memcontam/evaluation/phase13_observability_lineage.py",
    "phase13_aggregation": "src/memcontam/evaluation/phase13_aggregate.py",
    "measurement_identity_models": "src/memcontam/readiness/phase13_observability_models.py",
    "measurement_reconstruction_validator": "src/memcontam/readiness/phase13_observability_validate.py",
    "phase13_cli": "src/memcontam/readiness/phase13_cli.py",
    "track1_authority_state": "data/phase13/main/track1_authority_state_sync_checkpoint_v1.json",
    "track2_legacy_rag_seal": "data/phase13/rag/legacy_seal_v1.json",
    "core_main_registry": "src/memcontam/readiness/phase13_execution_contract.py",
}


def validate_phase13_observability_package(
    root: Path,
    repository_root: Path,
    expected_manifest_sha256: str,
) -> Phase13ObservabilityReport:
    try:
        manifest_bytes = read_regular_nofollow(root / "manifest_v1.json")
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if manifest_sha256 != expected_manifest_sha256:
            raise Phase13ObservabilityValidationError("OBSERVABILITY_MANIFEST_HASH_MISMATCH")
        manifest = Phase13ObservabilityManifest.model_validate_json(manifest_bytes)
        _validate_manifest_semantics(manifest)
        _validate_identities(root, root, manifest.artifacts)
        _validate_identities(repository_root, repository_root, manifest.implementations)
        registry = TargetSetRegistry.model_validate_json(
            read_regular_nofollow(root / "target_set_registry_v1.json")
        )
        fixture = Phase13ObservabilityFixture.model_validate_json(
            read_regular_nofollow(root / "fixture_v1.json")
        )
        packet_bytes = read_regular_nofollow(root / "registration_packet_v1.json")
        packet = ObservabilityRegistrationPacket.model_validate_json(packet_bytes)
        if hashlib.sha256(packet_bytes).hexdigest() != manifest.registration_packet_sha256:
            raise Phase13ObservabilityValidationError("OBSERVABILITY_REGISTRATION_HASH_MISMATCH")
        if fixture.registration_packet_sha256 != manifest.registration_packet_sha256:
            raise Phase13ObservabilityValidationError("OBSERVABILITY_FIXTURE_PACKET_MISMATCH")
        for identities in (
            packet.implementation_identities,
            packet.verifier_identities,
            packet.applicability_identities,
        ):
            _validate_identities(repository_root, repository_root, identities)
        if (
            packet.implementation_identities["registration"].model_dump()
            != manifest.implementations["observability_registration"].model_dump()
            or packet.implementation_identities["sequence"].model_dump()
            != manifest.implementations["observability_sequence"].model_dump()
            or packet.implementation_identities["authority_state"].model_dump()
            != manifest.implementations["track1_authority_state"].model_dump()
        ):
            raise Phase13ObservabilityValidationError("OBSERVABILITY_PACKET_MANIFEST_MISMATCH")
        _validate_identities(repository_root, repository_root, {"source_package": registry.source_package_manifest})
        _validate_target_sets(fixture, registry)
        first = reconstruct_fixture(fixture, packet)
        second = reconstruct_fixture(fixture, packet)
        first_hash = _canonical_hash(first)
        second_hash = _canonical_hash(second)
        if first_hash != second_hash or first_hash != manifest.expected_reconstruction_sha256:
            raise Phase13ObservabilityValidationError("OBSERVABILITY_RECONSTRUCTION_MISMATCH")
        _validate_reconstruction(first)
        return Phase13ObservabilityReport(
            manifest_id=manifest.manifest_id,
            manifest_sha256=manifest_sha256,
            evidence_scope=manifest.evidence_scope,
            track2_5_status=manifest.track2_5_status,
            registration_packet_sha256=manifest.registration_packet_sha256,
            reconstruction_sha256=first_hash,
            repeat_reconstruction_sha256=second_hash,
            reconstructed_trial_count=len(fixture.trials),
            target_set_registry_id=registry.registry_id,
            failure_classifier_registry_status=manifest.failure_classifier_registry_status,
            u_t_status=manifest.u_t_status,
            downstream_blockers=manifest.downstream_blockers,
            mr_p4_prerequisite_status=manifest.mr_p4_prerequisite_status,
            mr_p5_handoff_status=manifest.mr_p5_handoff_status,
            mr_p4_closure_claimed=manifest.mr_p4_closure_claimed,
            mr_p5_closure_claimed=manifest.mr_p5_closure_claimed,
            main_execution_authorized=manifest.main_execution_authorized,
            main_a_measured_scientific_execution_count=(
                manifest.main_a_measured_scientific_execution_count
            ),
        )
    except Phase13ObservabilityValidationError:
        raise
    except (AuthorityFileError, OSError) as error:
        raise Phase13ObservabilityValidationError("OBSERVABILITY_ARTIFACT_UNREADABLE") from error
    except ValidationError as error:
        raise Phase13ObservabilityValidationError("OBSERVABILITY_ARTIFACT_INVALID") from error
    except Phase13ObservabilityError as error:
        raise Phase13ObservabilityValidationError("OBSERVABILITY_RECONSTRUCTION_INVALID") from error


def reconstruct_fixture(
    fixture: Phase13ObservabilityFixture,
    packet: ObservabilityRegistrationPacket,
) -> Phase13Reconstruction:
    base_trials = tuple(reconstruct_phase13_trial(trial) for trial in fixture.trials)
    trials = reconstruct_registered_sequence(
        fixture.trials, base_trials, packet.recurrence_lookback_h
    )
    sequence_summary = summarize_sequence_diagnostics(trials)
    aggregate_rows = tuple(
        template.model_copy(
            update={
                "trajectory_seed": seed,
                "concrete_seed_id": concrete_seed_id,
                **(
                    sequence_summary
                    if template.baseline == "bot_style" and template.arm == "contam"
                    else {}
                ),
            }
        )
        for seed, concrete_seed_id in enumerate(fixture.concrete_seed_ids)
        for template in fixture.aggregate_templates
    )
    aggregate = aggregate_phase13(aggregate_rows)
    return Phase13Reconstruction(trials=trials, aggregate=aggregate)


def require_mr_p4_observability(report: Phase13ObservabilityReport) -> None:
    if report.mr_p4_prerequisite_status != "OBSERVABILITY_PREREQUISITE_MET":
        raise Phase13ObservabilityValidationError("OBSERVABILITY_PREREQUISITE_BLOCKED")


def _validate_manifest_semantics(manifest: Phase13ObservabilityManifest) -> None:
    artifact_paths = {name: identity.path for name, identity in manifest.artifacts.items()}
    implementation_paths = {
        name: identity.path for name, identity in manifest.implementations.items()
    }
    if artifact_paths != _EXPECTED_ARTIFACT_PATHS:
        raise Phase13ObservabilityValidationError("OBSERVABILITY_ARTIFACT_BINDING_MISMATCH")
    if implementation_paths != _EXPECTED_IMPLEMENTATION_PATHS:
        raise Phase13ObservabilityValidationError("OBSERVABILITY_IMPLEMENTATION_BINDING_MISMATCH")


def _validate_identities(
    root: Path,
    repository_root: Path,
    identities: Mapping[str, ArtifactIdentity | BoundIdentity],
) -> None:
    if not identities:
        raise Phase13ObservabilityValidationError("OBSERVABILITY_IDENTITY_MISSING")
    for identity in identities.values():
        relative = PurePosixPath(identity.path)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise Phase13ObservabilityValidationError("OBSERVABILITY_PATH_ESCAPE")
        path = root.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(repository_root.resolve()):
            raise Phase13ObservabilityValidationError("OBSERVABILITY_PATH_ESCAPE")
        try:
            raw = read_regular_nofollow(path)
        except AuthorityFileError as error:
            raise Phase13ObservabilityValidationError("OBSERVABILITY_ARTIFACT_UNREADABLE") from error
        if hashlib.sha256(raw).hexdigest() != identity.sha256:
            raise Phase13ObservabilityValidationError("OBSERVABILITY_ARTIFACT_HASH_MISMATCH")


def _validate_target_sets(
    fixture: Phase13ObservabilityFixture,
    registry: TargetSetRegistry,
) -> None:
    registered = {target.target_set_id for target in registry.target_sets}
    if registered != {trial.target_set.target_set_id for trial in fixture.trials}:
        raise Phase13ObservabilityValidationError("OBSERVABILITY_TARGET_SET_MISMATCH")
    if any(
        trial.target_set.source_package_manifest_sha256
        != registry.source_package_manifest.sha256
        for trial in fixture.trials
    ):
        raise Phase13ObservabilityValidationError("OBSERVABILITY_SOURCE_PACKAGE_MISMATCH")


def _validate_reconstruction(payload: Phase13Reconstruction) -> None:
    states = {
        (
            row.target_present_in_store_before_answer.value,
            row.target_retrieved.value,
            row.target_final_context_included.value,
            row.theory_exposure.value,
            row.verified_outcome,
        )
        for row in payload.trials
    }
    required = {
        (True, False, False, False, 1),
        (True, True, False, False, 1),
        (True, True, True, True, 0),
    }
    if not required.issubset(states):
        raise Phase13ObservabilityValidationError("OBSERVABILITY_FIXTURE_COVERAGE_INCOMPLETE")
    anchors = tuple(row for row in payload.trials if row.propagation.value is True)
    recurrents = tuple(
        row
        for row in payload.trials
        if row.generic_recurrence.value is True
        and row.exact_lineage_recurrence.value is True
        and row.exposure_conditioned_recurrence.value is True
        and row.post_eviction_recurrence.value is True
    )
    if len(anchors) != 1 or len(recurrents) != 1:
        raise Phase13ObservabilityValidationError("OBSERVABILITY_SEQUENCE_COVERAGE_INCOMPLETE")
    recurrent = recurrents[0]
    if (
        recurrent.root_retention_duration.value != 3
        or recurrent.root_retention_duration.censoring_status != "OBSERVED_END"
        or recurrent.prompt_retention_duration.value != 1
        or recurrent.prompt_retention_duration.censoring_status != "OBSERVED_END"
        or recurrent.descendant_retention_duration.value != 3
        or recurrent.descendant_retention_duration.censoring_status != "RIGHT_CENSORED"
    ):
        raise Phase13ObservabilityValidationError("OBSERVABILITY_SEQUENCE_COVERAGE_INCOMPLETE")


def _canonical_hash(value: Phase13Reconstruction) -> str:
    raw = json.dumps(value.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "Phase13ObservabilityValidationError",
    "reconstruct_fixture",
    "require_mr_p4_observability",
    "validate_phase13_observability_package",
]
