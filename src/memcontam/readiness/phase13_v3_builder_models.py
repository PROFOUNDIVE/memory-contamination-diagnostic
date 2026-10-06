from __future__ import annotations

from typing import Literal

from .phase13_cost_policy_models import Sha256
from .phase13_main_checkpoint import CommonCheckpointRegistry, TaskSeedOrders
from .phase13_v3_authority_models import AuthoritySnapshotV3, FrozenModel, V3Identity
from .phase13_v3_cost_models import CountPricingV1
from .phase13_v3_resource_files import FileBinding
from .phase13_v3_runtime_identity import RuntimeIdentityV3
from .phase13_v3_source_closure import GovernedInventory


class FirstFreeze(FrozenModel):
    concrete_seed_ids: tuple[int, ...]
    orders: TaskSeedOrders
    registry: CommonCheckpointRegistry
    capacity: int


class MRP4Manifest(FrozenModel):
    schema_version: Literal["phase13_mr_p4_local_closure_manifest_v3"] = "phase13_mr_p4_local_closure_manifest_v3"
    identity: V3Identity
    status: Literal["CLOSED"] = "CLOSED"
    authority: AuthoritySnapshotV3
    governed_source: GovernedInventory
    runtime_identity: RuntimeIdentityV3
    count_pricing: CountPricingV1
    first_freeze: FirstFreeze
    resources: tuple[FileBinding, ...]
    artifacts: tuple[FileBinding, ...]
    closure_hash: Sha256 = "0" * 64
