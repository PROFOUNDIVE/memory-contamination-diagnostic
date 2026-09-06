from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from .phase13_authority_files import load_authority_v3
from .phase13_main_execution import validate_main_authorization
from .phase13_main_execution_models import MainAuthorizationReport
from .phase13_v3_authority_models import V3Identity
from .phase13_v3_cost_binding import CostResourcesV3, LiveCosts, MRP4Costs
from .phase13_v3_cost_models import (
    ActivatedPolicyV3,
    BaseCostInputsV3,
    CompleteCostInputsV3,
    CostProofV3,
    CostWitnessV3,
    canonical_bytes,
    digest,
)
from .phase13_v3_entrypoint_models import (
    ExecutionResourceV3,
    MainAuthorizationV3,
    MainExecutionPackageV3,
    MainLiveContractV3,
)
from .phase13_v3_entrypoint_paths import (
    parse_authorization_digest,
    relative_path,
    verify_resource_namespace,
)
from .phase13_v3_resource_files import ValidatedResource, read_files
from .phase13_v3_runtime_identity import validate_runtime_identity


class EntrypointError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class SelectionRequest:
    repository_root: Path
    package_path: Path
    authorization_path: Path
    authority_root: Path | None
    expected_authorization_sha256_file: Path | None
    run_id: str | None = None
    expected_authorization_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class SelectedExecutionV3:
    package: MainExecutionPackageV3
    authorization: MainAuthorizationV3
    package_sha256: str
    authorization_sha256: str
    resources: tuple[ValidatedResource, ...]
    costs: LiveCosts
    lease: ExitStack
    repository_root: Path

    def close(self) -> None:
        self.lease.close()

    def preflight(self, root: Path) -> None:
        validate_runtime_identity(self.package.runtime_identity)
        verify_resource_namespace(root, self.resources)

    def resource_binding(self, role: str) -> ExecutionResourceV3:
        binding = next((row for row in self.package.resources if row.role == role), None)
        if binding is None:
            raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
        return binding

    def resource(self, role: str) -> bytes:
        binding = self.resource_binding(role)
        return next(row.raw for row in self.resources if row.binding.path == binding.path)


def select_execution(
    request: SelectionRequest, command: Literal["validate", "run", "resume", "status"],
) -> SelectedExecutionV3 | MainAuthorizationReport:
    if command in {"run", "resume", "status"} and request.run_id != V3Identity().run_id:
        raise EntrypointError("MAIN_CORRECTED_RUN_ID_MISMATCH")
    with ExitStack() as lease:
        result = _select_with_lease(request, command, lease)
        if isinstance(result, SelectedExecutionV3):
            result.lease.enter_context(lease.pop_all())
        return result


def _select_with_lease(request: SelectionRequest, command: str, lease: ExitStack) -> SelectedExecutionV3 | MainAuthorizationReport:
    package_resource, = read_files(request.repository_root, (
        relative_path(request.repository_root, request.package_path),
    ), lease=lease)
    try:
        raw = json.loads(package_resource.raw)
        schema = raw["schema_version"]
    except (ValueError, KeyError, TypeError) as error:
        raise EntrypointError("MAIN_PACKAGE_VERSION_UNSUPPORTED") from error
    match schema:
        case "phase13_main_execution_freeze_v1" | "phase13_main_execution_freeze_v2":
            if command != "validate":
                raise EntrypointError("MAIN_PACKAGE_VERSION_UNSUPPORTED")
            expected = request.expected_authorization_sha256
            if expected is None:
                raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
            return validate_main_authorization(request.repository_root, request.package_path,
                                               request.authorization_path, expected)
        case "phase13_main_execution_freeze_v3":
            if raw.get("package_id") != V3Identity().package_id:
                raise EntrypointError("MAIN_PACKAGE_VERSION_UNSUPPORTED")
            supplied_identity = raw.get("identity")
            if not isinstance(supplied_identity, dict) or supplied_identity.get("run_id") != V3Identity().run_id:
                raise EntrypointError("MAIN_CORRECTED_RUN_ID_MISMATCH")
            return _select_v3(request, package_resource, lease)
        case _:
            raise EntrypointError("MAIN_PACKAGE_VERSION_UNSUPPORTED")


def _select_v3(request: SelectionRequest, package_resource: ValidatedResource, lease: ExitStack) -> SelectedExecutionV3:
    if request.authority_root is None:
        raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    if request.expected_authorization_sha256_file is None or request.expected_authorization_sha256 is not None:
        raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    sidecar, = read_files(request.repository_root, (
        relative_path(request.repository_root, request.expected_authorization_sha256_file),), lease=lease)
    expected = parse_authorization_digest(sidecar.raw)
    authorization_resource, = read_files(request.repository_root, (
        relative_path(request.repository_root, request.authorization_path),
    ), lease=lease)
    try:
        package = MainExecutionPackageV3.model_validate_json(package_resource.raw)
        authorization = MainAuthorizationV3.model_validate_json(authorization_resource.raw)
    except ValidationError as error:
        raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH") from error
    if (authorization_resource.binding.sha256 != expected
        or authorization.authorization_hash != digest(authorization, "authorization_hash")
        or authorization_resource.raw != canonical_bytes(authorization)
        or package_resource.raw != canonical_bytes(package)
        or authorization.execution_package_sha256 != package_resource.binding.sha256
        or authorization.execution_package_hash != package.package_hash
        or authorization.execution_package_path != package_resource.binding.path
        or package.identity != authorization.identity
        or package.authority != load_authority_v3(request.authority_root)):
        raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    validate_runtime_identity(package.runtime_identity)
    if package.final_order.runtime_hash != digest(package.runtime_identity):
        raise EntrypointError("MAIN_RUNTIME_IDENTITY_DRIFT")
    roles = tuple(row.role for row in package.resources)
    from .phase13_main_resource_contract import RESOURCE_PATHS
    if len(set(roles)) != len(roles) or {row.role: row.path for row in package.resources} != RESOURCE_PATHS:
        raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    cost_root = "data/phase13/main/cost_envelope_v3/"
    additional = (cost_root + "complete_inputs_v3.json", cost_root + "cost_proof_v3.json",
                  "data/phase13/main/main_live_contract_v3.json")
    resources = read_files(request.repository_root, tuple(row.path for row in package.resources) + additional, lease=lease)
    by_path = {row.binding.path: row for row in resources}
    for binding in package.resources:
        actual = by_path[binding.path].binding
        if (actual.size, actual.sha256) != (binding.size, binding.sha256):
            raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    by_role = {row.role: by_path[row.path].raw for row in package.resources}
    try:
        phase4 = MRP4Costs(
            policy=ActivatedPolicyV3.model_validate_json(by_role["activated_policy"]),
            base=BaseCostInputsV3.model_validate_json(by_role["base_inputs"]),
            witness=CostWitnessV3.model_validate_json(by_role["cost_witness"]),
        )
        costs = LiveCosts(package, CostResourcesV3(
            phase4=phase4,
            complete=CompleteCostInputsV3.model_validate_json(by_path[additional[0]].raw),
            proof=CostProofV3.model_validate_json(by_path[additional[1]].raw),
        ))
        contract = MainLiveContractV3.model_validate_json(by_path[additional[2]].raw)
    except (KeyError, ValidationError) as error:
        raise EntrypointError("MAIN_COST_PROOF_MISMATCH") from error
    if (any(by_path[path].raw != canonical_bytes(model) for path, model in (
            (additional[0], costs.resources.complete), (additional[1], costs.resources.proof),
            (additional[2], contract)))
        or any(by_role[role] != canonical_bytes(model) for role, model in (
            ("activated_policy", phase4.policy), ("base_inputs", phase4.base), ("cost_witness", phase4.witness)))
        or phase4.policy.authority != package.authority
        or contract.contract_hash != digest(contract, "contract_hash")
        or package.live_contract_hash != contract.contract_hash
        or contract.package_core_hash != package.package_core_hash
        or contract.cost_proof_hash != package.cost_proof_hash
        or contract.projected_krw != tuple((row.unit_id, row.projected_krw) for row in costs.resources.proof.projected_krw)
        or tuple(row.unit_id for row in package.production) != package.final_order.unit_ids):
        raise EntrypointError("MAIN_COST_PROOF_MISMATCH")
    selected = SelectedExecutionV3(package, authorization, package_resource.binding.sha256,
        authorization_resource.binding.sha256, (package_resource, sidecar, authorization_resource, *resources), costs,
        ExitStack(), request.repository_root)
    from .phase13_main_resource_contract import validate_resource_contract
    try:
        validate_resource_contract(selected)
    except EntrypointError:
        raise
    except (KeyError, IndexError, ValueError) as error:
        raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH") from error
    selected.preflight(request.repository_root)
    return selected
