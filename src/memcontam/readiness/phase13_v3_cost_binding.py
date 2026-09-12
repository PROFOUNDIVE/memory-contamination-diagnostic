from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Generic, Self, TypeVar

from pydantic import Field, model_validator

from .phase13_cost_policy_models import Sha256
from .phase13_v3_authority_models import FrozenModel, IdentityComponent
from .phase13_v3_cost import (
    build_proof, freeze_complete, validate_base, validate_complete,
    validate_policy, validate_proof, validate_witness,
)
from .phase13_v3_cost_models import (
    ActivatedPolicyV3, BaseCostInputsV3, CompleteCostInputsV3, CostError,
    CostProofV3, CostWitnessV3, FinalOrder, canonical_bytes, digest,
)

if TYPE_CHECKING:
    from .phase13_main_production import ProductionObject


_CORE_EXCLUDES: Final = frozenset({
    "package_core_hash", "cost_proof_hash", "live_contract_hash",
    "generated_closure_hash", "generated_closure", "package_hash",
})


class MRP4Costs(FrozenModel):
    policy: ActivatedPolicyV3
    base: BaseCostInputsV3
    witness: CostWitnessV3

    @model_validator(mode="after")
    def check_predecessors(self) -> Self:
        validate_phase4_costs(self)
        return self


class CostBoundPackageV3(FrozenModel):
    """Cost-binding fields for the execution package, not execution authorization.

    Concrete package models extend this base; every additional execution field
    participates in the core projection without maintaining a field allowlist.
    """

    package_id: IdentityComponent
    final_order: FinalOrder
    base_inputs_hash: Sha256 = "0" * 64
    witness_hash: Sha256 = "0" * 64
    complete_inputs_hash: Sha256 = "0" * 64
    package_core_hash: Sha256 = "0" * 64
    cost_proof_hash: Sha256 = "0" * 64
    live_contract_hash: Sha256 | None = None
    generated_closure_hash: Sha256 | None = None
    package_hash: Sha256 = "0" * 64


class CostResourcesV3(FrozenModel):
    phase4: MRP4Costs
    complete: CompleteCostInputsV3
    proof: CostProofV3


PackageT = TypeVar("PackageT", bound=CostBoundPackageV3)


@dataclass(frozen=True, slots=True)
class BoundPackageCosts(Generic[PackageT]):
    package: PackageT
    resources: CostResourcesV3


def package_core_hash(package: CostBoundPackageV3) -> str:
    payload = package.model_dump(mode="json", exclude=set(_CORE_EXCLUDES))
    raw = (json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_phase4_costs(phase4: MRP4Costs) -> None:
    validate_policy(phase4.policy, phase4.base.policy.authority)
    if phase4.policy != phase4.base.policy:
        raise CostError("MAIN_COST_PROOF_MISMATCH")
    validate_base(canonical_bytes(phase4.base), phase4.base)
    validate_witness(canonical_bytes(phase4.witness), phase4.base)


def bind_package_costs(package: PackageT, phase4: MRP4Costs) -> BoundPackageCosts[PackageT]:
    validate_phase4_costs(phase4)
    if package.package_id != phase4.policy.authority.identity.package_id:
        raise CostError("MAIN_COST_PROOF_MISMATCH")
    complete = freeze_complete(phase4.base, package.final_order)
    core = package.model_copy(update={
        "base_inputs_hash": phase4.base.base_inputs_hash,
        "witness_hash": phase4.witness.witness_hash,
        "complete_inputs_hash": complete.complete_inputs_hash,
    })
    core_hash = package_core_hash(core)
    proof = build_proof(complete, phase4.witness, core_hash)
    with_proof = core.model_copy(update={
        "package_core_hash": core_hash, "cost_proof_hash": proof.proof_hash,
    })
    sealed = with_proof.model_copy(update={"package_hash": digest(with_proof, "package_hash")})
    return BoundPackageCosts(sealed, CostResourcesV3(phase4=phase4, complete=complete, proof=proof))


def validate_package_costs(package: CostBoundPackageV3, resources: CostResourcesV3) -> None:
    validate_phase4_costs(resources.phase4)
    expected = bind_package_costs(package, resources.phase4)
    validate_complete(canonical_bytes(resources.complete), expected.resources.complete)
    validate_proof(canonical_bytes(resources.proof), expected.resources.complete,
                   expected.package.package_core_hash)
    if package != expected.package:
        raise CostError("MAIN_COST_PROOF_MISMATCH")
    if resources.proof.totals.gate_result != "PASS":
        raise CostError("MAIN_COST_PROOF_MISMATCH")


class TableKey(FrozenModel):
    proof_hash: Sha256
    unit_id: str = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class LiveCosts:
    package: CostBoundPackageV3
    resources: CostResourcesV3

    def __post_init__(self) -> None:
        validate_package_costs(self.package, self.resources)

    def projected(self, package_hash: str, key: TableKey) -> int:
        validate_package_costs(self.package, self.resources)
        if (package_hash, key.proof_hash) != (self.package.package_hash, self.package.cost_proof_hash):
            raise CostError("MAIN_COST_PROOF_MISMATCH")
        for row in self.resources.proof.projected_krw:
            if row.unit_id == key.unit_id:
                return row.projected_krw
        raise CostError("MAIN_COST_PROOF_MISMATCH")


def attribute_v3_projected_cost(
    objects: tuple[ProductionObject, ...], costs: LiveCosts, package_hash: str,
) -> tuple[ProductionObject, ...]:
    if tuple(item.unit_id for item in objects) != costs.package.final_order.unit_ids:
        raise CostError("MAIN_COST_PROOF_MISMATCH")
    return tuple(replace(item, projected_cost_krw=costs.projected(
        package_hash,
        TableKey(proof_hash=costs.package.cost_proof_hash, unit_id=item.unit_id),
    )) for item in objects)
