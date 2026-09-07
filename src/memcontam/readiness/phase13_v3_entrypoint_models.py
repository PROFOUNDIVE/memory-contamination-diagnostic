from __future__ import annotations

from typing import Literal

from .phase13_cost_policy_models import Sha256
from .phase13_main_production import ProductionObject
from .phase13_v3_authority_models import AuthoritySnapshotV3, FrozenModel, V3Identity
from .phase13_v3_cost_binding import CostBoundPackageV3
from .phase13_v3_resource_files import FileBinding
from .phase13_v3_runtime_identity import RuntimeIdentityV3
from .phase13_v3_source_closure import GovernedInventory, ResourceClosure


class ExecutionResourceV3(FileBinding):
    role: str


class MainExecutionPackageV3(CostBoundPackageV3):
    schema_version: Literal["phase13_main_execution_freeze_v3"]
    identity: V3Identity
    status: Literal["FROZEN"]
    authority: AuthoritySnapshotV3
    runtime_identity: RuntimeIdentityV3
    resources: tuple[ExecutionResourceV3, ...]
    production: tuple[ProductionObject, ...]
    measured_main_a_trajectory_count: Literal[0]
    governed_source: GovernedInventory | None = None
    mr_p4_closure: FileBinding | None = None
    generated_closure: ResourceClosure | None = None


class MainAuthorizationV3(FrozenModel):
    schema_version: Literal["phase13_main_authorization_v3"]
    identity: V3Identity
    authorization_id: Literal["phase13-main-a-corrected-authorized-execution-v3"]
    status: Literal["AUTHORIZED_EXECUTION"]
    execution_package_id: Literal["phase13-main-a-corrected-execution-freeze-v3"]
    execution_package_path: str
    execution_package_sha256: Sha256
    execution_package_hash: Sha256
    authorization_hash: Sha256
    main_a_status: Literal["NOT_STARTED"]
    measured_main_a_trajectory_count: Literal[0]


class MainLiveContractV3(FrozenModel):
    schema_version: Literal["phase13_main_live_contract_v3"]
    identity: V3Identity
    package_core_hash: Sha256
    cost_proof_hash: Sha256
    projected_krw: tuple[tuple[str, int], ...]
    contract_hash: Sha256
