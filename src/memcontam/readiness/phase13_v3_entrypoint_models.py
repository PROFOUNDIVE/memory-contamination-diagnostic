from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from .phase13_cost_policy_models import Sha256
from .phase13_main_production import ProductionObject
from .phase13_v3_authority_models import AuthoritySnapshotV3, FrozenModel, IdentityComponent, V3Identity
from .phase13_v3_cost_binding import CostBoundPackageV3
from .phase13_v3_resource_files import FileBinding
from .phase13_v3_runtime_identity import RuntimeIdentityV3
from .phase13_v3_source_closure import GovernedInventory, ResourceClosure


class ExecutionResourceV3(FileBinding):
    role: str


class ExecutionPackageError(ValueError):
    def __init__(self) -> None:
        super().__init__("MAIN_AUTHORIZATION_BINDING_MISMATCH")


class MainExecutionPackageV3(CostBoundPackageV3):
    schema_version: Literal["phase13_main_execution_freeze_v3"]
    identity: V3Identity
    status: Literal["FROZEN"]
    authority: AuthoritySnapshotV3
    runtime_identity: RuntimeIdentityV3
    resources: tuple[ExecutionResourceV3, ...]
    production: tuple[ProductionObject, ...]
    tranche_unit_count: Literal[120]
    measured_main_a_trajectory_count: Literal[0]
    governed_source: GovernedInventory | None = None
    mr_p4_closure: FileBinding | None = None
    generated_closure: ResourceClosure | None = None

    @model_validator(mode="after")
    def validate_tranche_boundaries(self) -> Self:
        if self.package_id != self.identity.package_id or self.authority.identity != self.identity:
            raise ExecutionPackageError()
        if len(self.production) >= self.tranche_unit_count and any(
                unit.sequence != sequence or unit.seed != sequence // self.tranche_unit_count
                for sequence, unit in enumerate(self.production)
        ):
            raise ExecutionPackageError()
        return self


class MainAuthorizationV3(FrozenModel):
    schema_version: Literal["phase13_main_authorization_v3"]
    identity: V3Identity
    authorization_id: IdentityComponent
    status: Literal["AUTHORIZED_EXECUTION"]
    execution_package_id: IdentityComponent
    execution_package_path: str
    execution_package_sha256: Sha256
    execution_package_hash: Sha256
    authorization_hash: Sha256
    main_a_status: Literal["NOT_STARTED"]
    measured_main_a_trajectory_count: Literal[0]

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if (self.authorization_id != self.identity.authorization_id
            or self.execution_package_id != self.identity.package_id):
            raise ExecutionPackageError()
        return self


class MainLiveContractV3(FrozenModel):
    schema_version: Literal["phase13_main_live_contract_v3"]
    identity: V3Identity
    package_core_hash: Sha256
    cost_proof_hash: Sha256
    projected_krw: tuple[tuple[str, int], ...]
    contract_hash: Sha256
