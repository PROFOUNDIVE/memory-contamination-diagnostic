from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal, assert_never

from .phase13_authority_files import load_authority_v3
from .phase13_v3_builder_inputs import (
    PREFIX, STATIC_PATHS, artifact_raw, binding, first_freeze, parse_artifact,
    phase4_costs, production, resource_closure,
)
from .phase13_v3_builder_models import MRP4Manifest
from .phase13_v3_conformance import ConformanceV3, evaluate_conformance
from .phase13_v3_cost_binding import bind_package_costs
from .phase13_v3_cost_models import FinalOrder, canonical_bytes, digest
from .phase13_v3_entrypoint_models import ExecutionResourceV3, MainAuthorizationV3, MainExecutionPackageV3, MainLiveContractV3
from .phase13_main_resource_contract import RESOURCE_PATHS
from .phase13_v3_publication import ArtifactError, P4_PATHS, P5_PATHS, P6_PATHS, publish_artifacts
from .phase13_v3_resource_files import read_files
from .phase13_v3_runtime_identity import freeze_runtime_identity, validate_runtime_identity
from .phase13_v3_source_closure import freeze_governed, validate_governed

Stage = Literal["mr-p4", "mr-p5", "mr-p6"]


def _phase4_artifacts(manifest: MRP4Manifest, conformance: ConformanceV3) -> tuple[tuple[str, bytes], ...]:
    costs = phase4_costs(manifest)
    models = (manifest.authority, costs.policy, costs.base, costs.witness, conformance)
    predecessors = tuple((path, canonical_bytes(model)) for path, model in zip(P4_PATHS[:-1], models, strict=True))
    frozen = manifest.model_copy(update={"artifacts": tuple(binding(PREFIX + path, raw) for path, raw in predecessors)})
    frozen = frozen.model_copy(update={"closure_hash": digest(frozen, "closure_hash")})
    return (*predecessors, (P4_PATHS[-1], canonical_bytes(frozen)))


def build_mr_p4(repository: Path, authority_root: Path, output: Path, *, governed_source_commit: str) -> MRP4Manifest:
    manifest = MRP4Manifest(authority=load_authority_v3(authority_root),
        governed_source=freeze_governed(repository, governed_source_commit), runtime_identity=freeze_runtime_identity(),
        first_freeze=first_freeze(repository), resources=tuple(row.binding for row in read_files(repository, STATIC_PATHS)), artifacts=())
    artifacts = _phase4_artifacts(manifest, evaluate_conformance(repository, authority_root))
    publish_artifacts(output, artifacts)
    return parse_artifact(artifacts[-1][1], MRP4Manifest)


def validate_mr_p4(repository: Path, authority_root: Path, output: Path) -> MRP4Manifest:
    raw = artifact_raw(output, P4_PATHS[-1])
    manifest = parse_artifact(raw, MRP4Manifest)
    validate_governed(repository, manifest.governed_source)
    validate_runtime_identity(manifest.runtime_identity)
    load_authority_v3(authority_root, manifest.authority)
    if manifest.first_freeze != first_freeze(repository):
        raise ArtifactError("MAIN_MR_P4_FIRST_FREEZE_MISMATCH")
    if manifest.resources != tuple(row.binding for row in read_files(repository, STATIC_PATHS)):
        raise ArtifactError("MAIN_ARTIFACT_BINDING_MISMATCH")
    conformance = evaluate_conformance(repository, authority_root)
    actual = parse_artifact(artifact_raw(output, P4_PATHS[-2]), ConformanceV3)
    if actual != conformance:
        raise ArtifactError("MAIN_PROVIDER_FREE_CONFORMANCE_FAILED")
    expected = _phase4_artifacts(manifest, conformance)
    if any(artifact_raw(output, path) != content for path, content in expected):
        raise ArtifactError("MAIN_ARTIFACT_BINDING_MISMATCH")
    return manifest


def _phase5_artifacts(manifest: MRP4Manifest, output: Path) -> tuple[tuple[str, bytes], ...]:
    costs = phase4_costs(manifest)
    units = production(manifest.first_freeze, manifest.resources)
    rows = {row.path: row for row in (*manifest.resources, *manifest.artifacts)}
    package = MainExecutionPackageV3(schema_version="phase13_main_execution_freeze_v3", identity=manifest.identity,
        status="FROZEN", authority=manifest.authority, runtime_identity=manifest.runtime_identity,
        governed_source=manifest.governed_source, mr_p4_closure=binding(PREFIX + P4_PATHS[-1], canonical_bytes(manifest)),
        resources=tuple(ExecutionResourceV3(role=role, **rows[path].model_dump()) for role, path in RESOURCE_PATHS.items()),
        production=units, measured_main_a_trajectory_count=0,
        final_order=FinalOrder(unit_ids=tuple(unit.unit_id for unit in units), runtime_hash=digest(manifest.runtime_identity),
                              request_hash=costs.base.bindings.request_compiler_hash, tokenizer_hash=costs.base.bindings.tokenizer_hash))
    provisional = bind_package_costs(package, costs)
    attributed = tuple(replace(unit, projected_cost_krw=row.projected_krw)
                       for unit, row in zip(units, provisional.resources.proof.projected_krw, strict=True))
    bound = bind_package_costs(package.model_copy(update={"production": attributed}), costs)
    if bound.resources.proof.totals.gate_result != "PASS":
        raise ArtifactError("MAIN_COST_PROOF_MISMATCH")
    contract = MainLiveContractV3(schema_version="phase13_main_live_contract_v3", identity=manifest.identity,
        package_core_hash=bound.package.package_core_hash, cost_proof_hash=bound.package.cost_proof_hash,
        projected_krw=tuple((row.unit_id, row.projected_krw) for row in bound.resources.proof.projected_krw), contract_hash="0" * 64)
    contract = contract.model_copy(update={"contract_hash": digest(contract, "contract_hash")})
    predecessors = tuple((path, canonical_bytes(model)) for path, model in zip(P5_PATHS[:-1],
                         (bound.resources.complete, bound.resources.proof, contract), strict=True))
    closure = resource_closure((*manifest.resources,
        *(binding(PREFIX + path, artifact_raw(output, path)) for path in P4_PATHS),
        *(binding(PREFIX + path, raw) for path, raw in predecessors)))
    package = bound.package.model_copy(update={"live_contract_hash": contract.contract_hash,
        "generated_closure": closure, "generated_closure_hash": closure.resource_closure_sha256})
    package = package.model_copy(update={"package_hash": digest(package, "package_hash")})
    return (*predecessors, (P5_PATHS[-1], canonical_bytes(package)))


def build_mr_p5(repository: Path, authority_root: Path, output: Path) -> MainExecutionPackageV3:
    manifest = validate_mr_p4(repository, authority_root, output)
    artifacts = _phase5_artifacts(manifest, output)
    publish_artifacts(output, artifacts)
    return parse_artifact(artifacts[-1][1], MainExecutionPackageV3)


def validate_mr_p5(repository: Path, authority_root: Path, output: Path) -> MainExecutionPackageV3:
    raw = artifact_raw(output, P5_PATHS[-1])
    package = parse_artifact(raw, MainExecutionPackageV3)
    manifest = validate_mr_p4(repository, authority_root, output)
    if any(artifact_raw(output, path) != expected for path, expected in _phase5_artifacts(manifest, output)):
        raise ArtifactError("MAIN_COST_PROOF_MISMATCH")
    return package


def _authorization(package: MainExecutionPackageV3) -> MainAuthorizationV3:
    result = MainAuthorizationV3(schema_version="phase13_main_authorization_v3", identity=package.identity,
        authorization_id=package.identity.authorization_id, status="AUTHORIZED_EXECUTION", execution_package_id=package.package_id,
        execution_package_path=PREFIX + P5_PATHS[-1], execution_package_sha256=digest(package), execution_package_hash=package.package_hash,
        authorization_hash="0" * 64, main_a_status="NOT_STARTED", measured_main_a_trajectory_count=0)
    return result.model_copy(update={"authorization_hash": digest(result, "authorization_hash")})


def build_mr_p6(repository: Path, authority_root: Path, output: Path) -> MainAuthorizationV3:
    result = _authorization(validate_mr_p5(repository, authority_root, output))
    publish_artifacts(output, ((P6_PATHS[0], canonical_bytes(result)), (P6_PATHS[1], (digest(result) + "\n").encode())))
    return result


def validate_mr_p6(repository: Path, authority_root: Path, output: Path) -> MainAuthorizationV3:
    raw = artifact_raw(output, P6_PATHS[0])
    authorization = parse_artifact(raw, MainAuthorizationV3)
    expected = _authorization(validate_mr_p5(repository, authority_root, output))
    if authorization != expected or artifact_raw(output, P6_PATHS[1]) != (digest(expected) + "\n").encode():
        raise ArtifactError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    return authorization


def validate_stage(repository: Path, authority_root: Path, output: Path, *, stage: Stage) -> MRP4Manifest | MainExecutionPackageV3 | MainAuthorizationV3:
    match stage:
        case "mr-p4":
            return validate_mr_p4(repository, authority_root, output)
        case "mr-p5":
            return validate_mr_p5(repository, authority_root, output)
        case "mr-p6":
            return validate_mr_p6(repository, authority_root, output)
        case unreachable:
            assert_never(unreachable)


def audit(repository: Path, authority_root: Path, output: Path, *, compare_output_root: Path | None = None) -> None:
    from .phase13_v3_builder_audit import audit_scope
    validate_mr_p6(repository, authority_root, output)
    audit_scope(repository, output, compare_output_root)
