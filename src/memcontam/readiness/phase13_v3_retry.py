from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import ceil

from .phase13_execution_contract import CORE_MAIN_REGISTRY
from .phase13_main_production import ProductionObject, _stages
from .phase13_v3_cost_law import exact_request_cost
from .phase13_v3_cost_models import (
    BaseCostInputsV3,
    CostUnit,
    RequestTokens,
    RetryReservation,
    StageOccurrences,
)
from .phase13_v3_request import RequestKeyV3, Stage

AUTHORITY_TO_STAGE: dict[str, Stage] = dict(zip(
    ("FH_generation", "RAG_generation", "BoT_problem_distillation", "BoT_solve",
     "BoT_thought_distillation", "Reflexion_actor_generation", "Reflexion_reflection",
     "DC_RS_generation", "DC_RS_writer_synthesis", "NoMem_generation"),
    ("full_history_generate", "rag_generate", "bot_problem_distill", "bot_instantiate_solve",
     "bot_thought_distill", "reflexion_generate", "reflexion_reflect", "dc_rs_generate",
     "dc_rs_synthesize", "no_memory_generate"),
    strict=True,
))


@dataclass(frozen=True, slots=True)
class RetryCandidate:
    unit_id: str
    stage_id: str
    dispatch_id: str
    dependency_fanout: int
    rank_hash: str
    reservation_krw: int


def dependency_fanout(production: tuple[ProductionObject, ...], unit_id: str,
                      *, remaining_occurrences: int) -> int:
    return remaining_occurrences + sum(unit.prefix_unit_id == unit_id for unit in production)


def schedule_unit(unit: ProductionObject, cost_unit: CostUnit) -> tuple[tuple[RequestKeyV3, str], ...]:
    trials = 1 if unit.kind == "CLEAN_PREFIX" else CORE_MAIN_REGISTRY.H_run
    native_stages = _stages(unit)
    native_stages = tuple(reversed(native_stages)) if unit.memory_baseline == "dc_rs" else native_stages
    by_native: dict[str, StageOccurrences] = {
        AUTHORITY_TO_STAGE[group.stage_id]: group for group in cost_unit.stages
    }
    if (unit.unit_id != cost_unit.unit_id
        or len(by_native) != len(cost_unit.stages)
        or len(by_native) != len(native_stages)
        or any(by_native.get(stage) is None or by_native[stage].calls != calls for stage, calls in native_stages)):
        raise ValueError("MAIN_RETRY_ENTITLEMENT_INVALID")
    groups = tuple(by_native[stage] for stage, _ in native_stages)
    slots = {group.stage_id: group.calls // trials for group in groups}
    return tuple(
        (RequestKeyV3(parent_id=unit.unit_id, stage=AUTHORITY_TO_STAGE[group.stage_id],
                      ordinal=trial * slots[group.stage_id] + attempt), group.stage_id)
        for trial in range(trials)
        for attempt in range(max(slots.values()))
        for group in groups
        if attempt < slots[group.stage_id]
    )


def allocate_retry_entitlements(
    production: tuple[ProductionObject, ...], base: BaseCostInputsV3,
) -> frozenset[str]:
    return frozenset(row.dispatch_id for row in allocate_retry_reservations(production, base))


def allocate_retry_reservations(
    production: tuple[ProductionObject, ...], base: BaseCostInputsV3,
) -> tuple[RetryReservation, ...]:
    units = {unit.unit_id: unit for unit in production}
    cost_units = {unit.unit_id: unit for unit in base.units}
    if set(units) != set(cost_units):
        raise ValueError("MAIN_RETRY_ENTITLEMENT_INVALID")
    envelopes = {stage.stage_id: stage for stage in base.policy.authority.registry.stages}
    schedules = {unit_id: schedule_unit(units[unit_id], cost_unit)
                 for unit_id, cost_unit in cost_units.items()}

    candidates: list[RetryCandidate] = []
    for unit_id, scheduled_occurrences in schedules.items():
        for position, (key, authority_stage) in enumerate(scheduled_occurrences):
            envelope = envelopes[authority_stage]
            input_cost, output_cost = exact_request_cost(RequestTokens(
                input_tokens=envelope.maximum_input_tokens,
                output_tokens=envelope.maximum_output_tokens,
            ), base.policy.rate_card)
            candidates.append(RetryCandidate(
                unit_id=unit_id,
                stage_id=authority_stage,
                dispatch_id=key.dispatch_id,
                dependency_fanout=dependency_fanout(
                    production, unit_id, remaining_occurrences=len(scheduled_occurrences) - position - 1,
                ),
                rank_hash=hashlib.sha256(
                    b"phase13_retry_entitlement_v1\0" + key.dispatch_id.encode()
                ).hexdigest(),
                reservation_krw=ceil(input_cost) + ceil(output_cost),
            ))
    candidates.sort(key=lambda row: (-row.dependency_fanout, row.rank_hash))
    if len({row.rank_hash for row in candidates}) != len(candidates):
        raise ValueError("MAIN_RETRY_ENTITLEMENT_INVALID")
    remaining: int = base.policy.authority.retry.retry_budget_krw
    selected: list[RetryReservation] = []
    for candidate in candidates:
        if candidate.reservation_krw <= remaining:
            selected.append(RetryReservation(
                unit_id=candidate.unit_id, stage_id=candidate.stage_id,
                dispatch_id=candidate.dispatch_id, reservation_krw=candidate.reservation_krw,
            ))
            remaining -= candidate.reservation_krw
    return tuple(sorted(selected, key=lambda row: row.dispatch_id))
