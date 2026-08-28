from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from memcontam.evaluation.phase13_observability_registration import ObservabilityRegistrationPacket
from memcontam.readiness.phase13_authority_files import AuthorityFileError, read_regular_nofollow
from memcontam.readiness.phase13_capacity_realization import (
    CapacityRealizationError,
    validate_common_capacity_artifact,
)
from memcontam.readiness.phase13_cost_activation import (
    Phase13CostActivationError,
    validate_activated_cost_policy,
)
from memcontam.readiness.phase13_legacy_rag_errors import LegacyRagValidationError
from memcontam.readiness.phase13_legacy_rag_validate import validate_legacy_rag_package
from memcontam.readiness.phase13_main_readiness_authority import (
    validate_package_selection,
    validate_track1,
)
from memcontam.readiness.phase13_main_readiness_models import (
    ArtifactIdentity,
    MainReadinessManifest,
    MainReadinessReport,
    Phase13MainReadinessError,
)
from memcontam.readiness.phase13_main_checkpoint import (
    Phase13MainCheckpointError,
    validate_main_checkpoint_package,
)
from memcontam.readiness.phase13_observability_models import Phase13ObservabilityFixture
from memcontam.readiness.phase13_readiness0 import (
    Phase13Readiness0Error,
    validate_readiness0_request,
)
from memcontam.readiness.phase13_observability_validate import (
    validate_phase13_observability_package,
)
from .phase13_production_observability import (
    ProductionObservabilityError,
    conformance_archive,
    validate_production_archive,
)


def validate_main_readiness(
    root: Path, repository_root: Path, expected_manifest_sha256: str
) -> MainReadinessReport:
    try:
        raw = read_regular_nofollow(root / "manifest_v1.json")
        manifest_sha256 = hashlib.sha256(raw).hexdigest()
        if manifest_sha256 != expected_manifest_sha256:
            raise Phase13MainReadinessError("MR_P4_MANIFEST_HASH_MISMATCH")
        manifest = MainReadinessManifest.model_validate_json(raw)
        _validate_artifacts(repository_root, manifest.artifacts)
        _validate_prerequisites(repository_root, manifest)
        packet_path = repository_root / manifest.artifacts["observability_packet"].path
        fixture_path = repository_root / manifest.artifacts["observability_fixture"].path
        packet_raw = read_regular_nofollow(packet_path)
        packet = ObservabilityRegistrationPacket.model_validate_json(packet_raw)
        fixture = Phase13ObservabilityFixture.model_validate_json(read_regular_nofollow(fixture_path))
        production = validate_production_archive(
            conformance_archive(fixture, hashlib.sha256(packet_raw).hexdigest()),
            packet,
            hashlib.sha256(packet_raw).hexdigest(),
        )
        return MainReadinessReport(
            status=manifest.status,
            manifest_sha256=manifest_sha256,
            execution_template_count=(
                len(manifest.execution_templates.included_task_baseline_pairs)
                * len(manifest.execution_templates.arms)
                + len(manifest.execution_templates.nomem_tasks)
            ),
            level2_interaction_count=sum(
                len(rows) for rows in manifest.level2_interactions.comparators_by_task.values()
            ),
            abstract_seed_slots_per_task=manifest.execution_templates.abstract_seed_slots_per_task,
            H_run=manifest.execution_templates.H_run,
            H_primary=manifest.execution_templates.H_primary,
            synthetic_observability_conformance_status=production.status,
            provider_session_retry_resource_contract_status=(
                manifest.gates.provider_session_retry_resource_contract_status
            ),
            u_t_status=manifest.u_t_status,
            blockers=tuple(
                validate_readiness0_request(
                    repository_root / manifest.artifacts["readiness0_request"].path
                ).external_blockers
            ),
            mr_p4_closure_claimed=manifest.mr_p4_closure_claimed,
            mr_p5_status=manifest.mr_p5_status,
            mr_p6_status=manifest.mr_p6_status,
            main_execution_authorized=manifest.main_execution_authorized,
            main_a_measured_scientific_execution_count=(
                manifest.main_a_measured_scientific_execution_count
            ),
        )
    except Phase13MainReadinessError:
        raise
    except (AuthorityFileError, OSError) as error:
        raise Phase13MainReadinessError("MR_P4_ARTIFACT_UNREADABLE") from error
    except (
        CapacityRealizationError,
        LegacyRagValidationError,
        Phase13CostActivationError,
        Phase13MainCheckpointError,
        Phase13Readiness0Error,
    ) as error:
        raise Phase13MainReadinessError("MR_P4_PREREQUISITE_INVALID") from error
    except ProductionObservabilityError as error:
        raise Phase13MainReadinessError("MR_P4_SYNTHETIC_CONFORMANCE_INVALID") from error
    except ValidationError as error:
        for item in error.errors():
            cause = item.get("ctx", {}).get("error")
            if isinstance(cause, Phase13MainReadinessError):
                raise cause from error
        first = error.errors()[0]
        location = tuple(str(item) for item in first["loc"])
        if "execution_templates" in location:
            raise Phase13MainReadinessError("MR_P4_EXECUTION_CONTRACT_MISMATCH") from error
        raise Phase13MainReadinessError("MR_P4_MANIFEST_INVALID") from error


def _validate_artifacts(root: Path, artifacts: dict[str, ArtifactIdentity]) -> None:
    required = {
        "track1", "package_selection", "capacity", "legacy_rag_seal", "observability_manifest",
        "observability_packet", "observability_fixture", "provider_contract", "openai_client",
        "ordinary_runtime", "production_observability_adapter",
        "production_runtime_join", "production_runtime_evidence",
        "production_runtime_memory", "production_runtime_models",
        "runtime_registry", "recording_client", "logging_schema", "provider_profile",
        "cost_policy", "cost_policy_models", "cost_policy_handoff", "live_branch",
        "main_readiness_validator", "main_readiness_authority", "main_readiness_models",
        "task_seed_orders", "common_checkpoint_registry", "main_checkpoint_validator",
        "activated_cost_policy", "cost_activation_validator", "readiness0_request",
        "readiness0_validator",
    }
    if set(artifacts) != required:
        raise Phase13MainReadinessError("MR_P4_ARTIFACT_SET_MISMATCH")
    for identity in artifacts.values():
        relative = PurePosixPath(identity.path)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise Phase13MainReadinessError("MR_P4_ARTIFACT_PATH_INVALID")
        if hashlib.sha256(read_regular_nofollow(root.joinpath(*relative.parts))).hexdigest() != identity.sha256:
            raise Phase13MainReadinessError("MR_P4_ARTIFACT_HASH_MISMATCH")


def _validate_prerequisites(root: Path, manifest: MainReadinessManifest) -> None:
    validate_track1(read_regular_nofollow(root / manifest.artifacts["track1"].path))
    validate_package_selection(
        read_regular_nofollow(root / manifest.artifacts["package_selection"].path)
    )
    validate_main_checkpoint_package(root / "data/phase13/main/mr_p4", root)
    validate_activated_cost_policy(root)
    validate_readiness0_request(root / manifest.artifacts["readiness0_request"].path)
    validate_common_capacity_artifact(root / manifest.artifacts["capacity"].path, root)
    try:
        legacy = json.loads(
            read_regular_nofollow(root / manifest.artifacts["legacy_rag_seal"].path)
        )
        legacy_manifest_sha256 = legacy["manifest_sha256"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise Phase13MainReadinessError("MR_P4_PREREQUISITE_INVALID") from error
    if not isinstance(legacy_manifest_sha256, str):
        raise Phase13MainReadinessError("MR_P4_PREREQUISITE_INVALID")
    validate_legacy_rag_package(
        root / "data/phase13/rag/legacy", root, legacy_manifest_sha256
    )
    validate_phase13_observability_package(
        root / "data/phase13/observability",
        root,
        manifest.artifacts["observability_manifest"].sha256,
    )


__all__ = [
    "MainReadinessReport",
    "Phase13MainReadinessError",
    "validate_main_readiness",
]
