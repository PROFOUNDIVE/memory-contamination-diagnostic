from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    model_validator,
)

from memcontam.readiness.phase13_authority_files import AuthorityFileError, read_regular_nofollow
from memcontam.readiness.phase13_main_execution_models import MainExecutionFreeze
from memcontam.readiness.phase13_main_production import (
    build_production_objects,
    prefix_stage_call_counts,
    units_sha256,
)


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]


class MainLiveContractError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PrefixRealizationContract(_FrozenModel):
    contract_id: Literal["phase13-main-prefix-realization-v1"]
    dispatch_order_id: Literal["phase13-main-production-object-order-v1"]
    owner_law_id: Literal["phase13-main-prefix-four-consumers-v1"]
    failure_law_id: Literal["phase13-main-prefix-atomic-terminal-fanout-v1"]
    checkpoint_evidence_schema_id: Literal["phase13_main_prefix_checkpoint_v1"]
    realization_count: Literal[230]
    dispatch_order_sha256: Sha256
    ownership_sha256: Sha256
    stage_call_counts: dict[str, int]


class MainCostContract(_FrozenModel):
    cost_envelope_id: Identifier
    cost_envelope_sha256: Sha256
    semantic_calls: int = Field(gt=0)
    cmax_main_krw: int = Field(gt=0)
    core_authorization_gate_krw: int = Field(gt=0)

    @model_validator(mode="after")
    def _within_gate(self) -> MainCostContract:
        if self.cmax_main_krw > self.core_authorization_gate_krw:
            raise MainLiveContractError("MAIN_LIVE_COST_GATE_FAILED")
        return self


class MainLiveContract(_FrozenModel):
    schema_version: Literal["phase13_main_live_contract_v1"]
    authority_sha256: Sha256
    package_id: Literal["phase13-main-a-execution-freeze-v1"]
    checkpoint_registry_sha256: Sha256
    observability_packet_sha256: Sha256
    production_units_sha256: Sha256
    prefix: PrefixRealizationContract
    cost: MainCostContract


def load_main_live_contract(path: Path) -> MainLiveContract:
    try:
        return MainLiveContract.model_validate_json(read_regular_nofollow(path))
    except (AuthorityFileError, OSError, ValidationError, MainLiveContractError) as error:
        raise MainLiveContractError("MAIN_LIVE_CONTRACT_INVALID") from error


def validate_main_live_contract(
    contract: MainLiveContract,
    package: MainExecutionFreeze,
) -> None:
    units = build_production_objects(package)
    prefixes = tuple(unit for unit in units if unit.kind == "CLEAN_PREFIX")
    bindings = {binding.role: binding.sha256 for binding in package.artifacts}
    authorities = {binding.role: binding.sha256 for binding in package.authority}
    dispatch_hash = _canonical_hash([unit.unit_id for unit in prefixes])
    ownership_hash = _canonical_hash(
        [
            [
                prefix.unit_id,
                [unit.unit_id for unit in units if unit.prefix_unit_id == prefix.unit_id],
            ]
            for prefix in prefixes
        ]
    )
    if (
        contract.authority_sha256 != authorities["authority_router"]
        or contract.package_id != package.package_id
        or contract.checkpoint_registry_sha256 != bindings["common_checkpoint_registry"]
        or contract.observability_packet_sha256 != bindings["observability_packet"]
        or contract.production_units_sha256 != units_sha256(units)
        or contract.prefix.realization_count != len(prefixes)
        or contract.prefix.dispatch_order_sha256 != dispatch_hash
        or contract.prefix.ownership_sha256 != ownership_hash
        or contract.prefix.stage_call_counts != prefix_stage_call_counts(units)
        or contract.cost.cost_envelope_id != package.cost_guard.cost_envelope_id
        or contract.cost.cost_envelope_sha256 != package.cost_guard.cost_envelope_sha256
        or contract.cost.semantic_calls != package.cost_guard.semantic_calls
        or contract.cost.cmax_main_krw != package.cost_guard.cmax_main_krw
        or contract.cost.core_authorization_gate_krw
        != package.cost_guard.core_authorization_gate_krw
    ):
        raise MainLiveContractError("MAIN_LIVE_CONTRACT_PACKAGE_MISMATCH")


def _canonical_hash(value: JsonValue) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "MainLiveContract",
    "MainLiveContractError",
    "load_main_live_contract",
    "validate_main_live_contract",
]
