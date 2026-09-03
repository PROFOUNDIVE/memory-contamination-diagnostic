from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, replace
from typing import Literal

from memcontam.readiness.phase13_main_execution_models import MainExecutionFreeze


UNIT_IDENTITY_LAW_ID = "phase13-main-a-disjoint-unit-id-v1"
UnitKind = Literal["CLEAN_PREFIX", "MEMORY_BEARING", "NO_MEMORY_SINGLETON"]

_PREFIX_STAGES = {
    "fh_bounded": (("full_history_generate", 1),),
    "rag_frozen": (("rag_generate", 1),),
    "bot_style": (
        ("bot_problem_distill", 1),
        ("bot_instantiate_solve", 1),
        ("bot_thought_distill", 1),
    ),
    "reflexion_style": (("reflexion_generate", 1), ("reflexion_reflect", 1)),
    "dc_rs": (("dc_rs_generate", 1), ("dc_rs_synthesize", 1)),
}
_SUFFIX_STAGES = {
    "fh_bounded": (("full_history_generate", 50),),
    "rag_frozen": (("rag_generate", 50),),
    "bot_style": (
        ("bot_problem_distill", 50),
        ("bot_instantiate_solve", 50),
        ("bot_thought_distill", 50),
    ),
    "reflexion_style": (("reflexion_generate", 100), ("reflexion_reflect", 100)),
    "dc_rs": (("dc_rs_generate", 50), ("dc_rs_synthesize", 50)),
}
_STAGE_ENVELOPES = {
    "full_history_generate": (9330, 512, 37507, 9880),
    "rag_generate": (344, 512, 830, 5928),
    "bot_problem_distill": (1177, 384, 4732, 7410),
    "bot_instantiate_solve": (1949, 512, 7835, 9880),
    "bot_thought_distill": (2545, 384, 10231, 7410),
    "reflexion_generate": (2282, 512, 18302, 19710),
    "reflexion_reflect": (3349, 384, 26859, 14783),
    "dc_rs_generate": (9212, 512, 37033, 9880),
    "dc_rs_synthesize": (13521, 8192, 54355, 158073),
    "no_memory_generate": (1160, 512, 1160, 2458),
}


@dataclass(frozen=True, slots=True)
class ProductionObject:
    sequence: int
    unit_id: str
    kind: UnitKind
    seed: int
    task: str
    memory_baseline: str | None
    arm: str
    prefix_unit_id: str | None
    projected_cost_krw: int
    execution_template_id: str | None = None
    ordered_sample_ids_sha256: str | None = None
    registration_packet_sha256: str | None = None
    checkpoint_registry_sha256: str | None = None


def build_production_objects(
    package: MainExecutionFreeze,
    ordered_sample_ids_sha256: dict[tuple[str, int], str] | None = None,
) -> tuple[ProductionObject, ...]:
    pairs_by_task = {
        task: tuple(
            baseline
            for pair_task, baseline in package.active_cells.included_task_baseline_pairs
            if pair_task == task
        )
        for task in package.dispatch.task_order
    }
    objects: list[ProductionObject] = []
    for seed_rank, seed in enumerate(package.dispatch.concrete_seed_ids):
        arms = package.arm_order.sequences[
            package.arm_order.seed_sequence_indices[seed_rank]
        ].arms
        for task in package.dispatch.task_order:
            for baseline in pairs_by_task[task]:
                prefix = _object(
                    len(objects), "CLEAN_PREFIX", seed, task, baseline, "NOT_APPLICABLE", None
                )
                objects.append(prefix)
                objects.extend(
                    _object(
                        len(objects),
                        "MEMORY_BEARING",
                        seed,
                        task,
                        baseline,
                        arm,
                        prefix.unit_id,
                    )
                    for arm in arms
                )
            objects.append(
                _object(
                    len(objects),
                    "NO_MEMORY_SINGLETON",
                    seed,
                    task,
                    None,
                    "NOT_APPLICABLE",
                    None,
                )
            )
    expected = package.active_cells.attempted_trajectory_count + sum(
        len(pairs) for pairs in pairs_by_task.values()
    ) * len(package.dispatch.concrete_seed_ids)
    if len(objects) != expected or len({item.unit_id for item in objects}) != expected:
        raise ValueError("MAIN_RUN_UNIT_DOMAIN_INVALID")
    objects_with_cost = _attribute_projected_cost(tuple(objects), package.cost_guard.cmax_main_krw)
    checkpoint_registry_sha256 = next(
        row.sha256 for row in package.artifacts if row.role == "common_checkpoint_registry"
    )
    return tuple(
        replace(
            item,
            execution_template_id=_execution_template_id(item),
            ordered_sample_ids_sha256=(
                None
                if ordered_sample_ids_sha256 is None
                else ordered_sample_ids_sha256[(item.task, item.seed)]
            ),
            registration_packet_sha256=package.observability.packet_sha256,
            checkpoint_registry_sha256=checkpoint_registry_sha256,
        )
        for item in objects_with_cost
    )


def units_sha256(units: tuple[ProductionObject, ...]) -> str:
    rows = [
        [
            unit.sequence,
            unit.unit_id,
            unit.kind,
            unit.seed,
            unit.task,
            unit.memory_baseline,
            unit.arm,
            unit.prefix_unit_id,
            unit.projected_cost_krw,
        ]
        for unit in units
    ]
    return hashlib.sha256(_canonical(rows)).hexdigest()


def prefix_stage_call_counts(units: tuple[ProductionObject, ...]) -> dict[str, int]:
    counts = Counter(
        stage
        for unit in units
        if unit.kind == "CLEAN_PREFIX"
        for stage, count in _stages(unit)
        for _ in range(count)
    )
    return dict(sorted(counts.items()))


def _object(
    sequence: int,
    kind: UnitKind,
    seed: int,
    task: str,
    baseline: str | None,
    arm: str,
    prefix_unit_id: str | None,
) -> ProductionObject:
    identity = [UNIT_IDENTITY_LAW_ID, kind, seed, task, baseline, arm]
    return ProductionObject(
        sequence=sequence,
        unit_id=hashlib.sha256(_canonical(identity)).hexdigest(),
        kind=kind,
        seed=seed,
        task=task,
        memory_baseline=baseline,
        arm=arm,
        prefix_unit_id=prefix_unit_id,
        projected_cost_krw=0,
    )


def _attribute_projected_cost(
    objects: tuple[ProductionObject, ...], expected_total: int
) -> tuple[ProductionObject, ...]:
    stage_counts = Counter(
        stage
        for item in objects
        for stage, count in _stages(item)
        for _ in range(count)
    )
    positions: Counter[str] = Counter()
    attributed: list[ProductionObject] = []
    for item in objects:
        projected = 0
        for stage, count in _stages(item):
            _, _, input_krw, output_krw = _STAGE_ENVELOPES[stage]
            stage_krw = input_krw + output_krw
            for _ in range(count):
                positions[stage] += 1
                position = positions[stage]
                projected += _ceil_fraction(position * stage_krw, stage_counts[stage])
                projected -= _ceil_fraction(
                    (position - 1) * stage_krw, stage_counts[stage]
                )
        attributed.append(replace(item, projected_cost_krw=projected))
    component_totals = {
        stage: (
            _ceil_fraction(count * envelope[0], 2500),
            _ceil_fraction(count * envelope[1] * 24, 12500),
        )
        for stage, count in stage_counts.items()
        for envelope in (_STAGE_ENVELOPES[stage],)
    }
    expected_components = {
        stage: envelope[2:] for stage, envelope in _STAGE_ENVELOPES.items()
    }
    if (
        positions != stage_counts
        or component_totals != expected_components
        or sum(item.projected_cost_krw for item in attributed) != expected_total
    ):
        raise ValueError("MAIN_RUN_COST_PROJECTION_INVALID")
    return tuple(attributed)


def _stages(item: ProductionObject) -> tuple[tuple[str, int], ...]:
    if item.kind == "NO_MEMORY_SINGLETON":
        return (("no_memory_generate", 50),)
    assert item.memory_baseline is not None
    return (
        _PREFIX_STAGES[item.memory_baseline]
        if item.kind == "CLEAN_PREFIX"
        else _SUFFIX_STAGES[item.memory_baseline]
    )


def _execution_template_id(item: ProductionObject) -> str:
    if item.kind == "NO_MEMORY_SINGLETON":
        return f"{item.task}|nomem"
    assert item.memory_baseline is not None
    suffix = "prefix" if item.kind == "CLEAN_PREFIX" else item.arm
    return f"{item.task}|{item.memory_baseline}|{suffix}"


def _ceil_fraction(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _canonical(value: list) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


__all__ = [
    "ProductionObject",
    "UNIT_IDENTITY_LAW_ID",
    "UnitKind",
    "build_production_objects",
    "prefix_stage_call_counts",
    "units_sha256",
]
