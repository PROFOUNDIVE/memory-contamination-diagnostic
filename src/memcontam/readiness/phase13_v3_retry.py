from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import ceil

from .phase13_main_production import ProductionObject
from .phase13_v3_cost_law import exact_request_cost
from .phase13_v3_cost_models import BaseCostInputsV3, RequestTokens
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
    dispatch_id: str
    dependency_fanout: int
    rank_hash: str
    reservation_krw: int


def allocate_retry_entitlements(
    production: tuple[ProductionObject, ...], base: BaseCostInputsV3,
) -> frozenset[str]:
    units = {unit.unit_id: unit for unit in production}
    cost_units = {unit.unit_id: unit for unit in base.units}
    if set(units) != set(cost_units):
        raise ValueError("MAIN_RETRY_ENTITLEMENT_INVALID")
    envelopes = {stage.stage_id: stage for stage in base.policy.authority.registry.stages}
    schedules: dict[str, tuple[tuple[RequestKeyV3, str], ...]] = {}
    for unit_id, cost_unit in cost_units.items():
        ordinals: dict[Stage, int] = {}
        occurrences: list[tuple[RequestKeyV3, str]] = []
        for group in cost_unit.stages:
            stage = AUTHORITY_TO_STAGE[group.stage_id]
            start = ordinals.get(stage, 0)
            occurrences.extend(
                (RequestKeyV3(parent_id=unit_id, stage=stage, ordinal=ordinal), group.stage_id)
                for ordinal in range(start, start + group.calls)
            )
            ordinals[stage] = start + group.calls
        schedules[unit_id] = tuple(occurrences)

    child_calls = {
        unit_id: sum(len(schedules[child.unit_id]) for child in production if child.prefix_unit_id == unit_id)
        for unit_id in units
    }
    candidates: list[RetryCandidate] = []
    for unit_id, scheduled_occurrences in schedules.items():
        for position, (key, authority_stage) in enumerate(scheduled_occurrences):
            envelope = envelopes[authority_stage]
            input_cost, output_cost = exact_request_cost(RequestTokens(
                input_tokens=envelope.maximum_input_tokens,
                output_tokens=envelope.maximum_output_tokens,
            ), base.policy.rate_card)
            candidates.append(RetryCandidate(
                dispatch_id=key.dispatch_id,
                dependency_fanout=len(scheduled_occurrences) - position - 1 + child_calls[unit_id],
                rank_hash=hashlib.sha256(
                    b"phase13_retry_entitlement_v1\0" + key.dispatch_id.encode()
                ).hexdigest(),
                reservation_krw=ceil(input_cost) + ceil(output_cost),
            ))
    candidates.sort(key=lambda row: (-row.dependency_fanout, row.rank_hash))
    remaining: int = base.policy.authority.retry.retry_budget_krw
    selected: set[str] = set()
    for candidate in candidates:
        if candidate.reservation_krw <= remaining:
            selected.add(candidate.dispatch_id)
            remaining -= candidate.reservation_krw
    return frozenset(selected)
