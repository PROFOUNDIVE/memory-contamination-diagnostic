from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from memcontam.readiness.phase13_authority_files import read_regular_nofollow
from memcontam.readiness.phase13_cost_policy_models import StageEnvelopeRegistry
from memcontam.readiness.phase13_main_execution_models import MainExecutionFreeze
from .phase13_v3_cost_binding import attribute_v3_projected_cost as attribute_v3_projected_cost


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
    registry_binding = next(row for row in package.artifacts if row.role == "stage_envelope_registry")
    expected_path = {
        "phase13_main_execution_freeze_v1": "data/phase13/main/cost_envelope_v2/stage_envelope_registry_v1.json",
        "phase13_main_execution_freeze_v2": "data/phase13/main/cost_envelope_v2/stage_envelope_registry_corrected_v2.json",
    }[package.schema_version]
    if registry_binding.path != expected_path:
        raise ValueError("MAIN_RUN_COST_PROJECTION_INVALID")
    registry_raw = read_regular_nofollow(Path(__file__).resolve().parents[3] / expected_path)
    if hashlib.sha256(registry_raw).hexdigest() != registry_binding.sha256:
        raise ValueError("MAIN_RUN_COST_PROJECTION_INVALID")
    registry = StageEnvelopeRegistry.model_validate_json(registry_raw)
    objects_with_cost = _attribute_projected_cost(
        tuple(objects), package.cost_guard.cmax_main_krw, registry,
    )
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
    objects: tuple[ProductionObject, ...], expected_total: int, registry: StageEnvelopeRegistry,
) -> tuple[ProductionObject, ...]:
    stage_counts = Counter(
        stage
        for item in objects
        for stage, count in _stages(item)
        for _ in range(count)
    )
    components = {
        stage.semantic_stage_id: (
            _ceil_fraction(stage.calls * stage.maximum_input_tokens, 2500),
            _ceil_fraction(stage.calls * stage.maximum_output_tokens * 24, 12500),
        )
        for stage in registry.stages
    }
    if stage_counts != {stage.semantic_stage_id: stage.calls for stage in registry.stages}:
        raise ValueError("MAIN_RUN_COST_PROJECTION_INVALID")
    positions: Counter[str] = Counter()
    attributed: list[ProductionObject] = []
    for item in objects:
        projected = 0
        for stage, count in _stages(item):
            input_krw, output_krw = components[stage]
            stage_krw = input_krw + output_krw
            for _ in range(count):
                positions[stage] += 1
                position = positions[stage]
                projected += _ceil_fraction(position * stage_krw, stage_counts[stage])
                projected -= _ceil_fraction(
                    (position - 1) * stage_krw, stage_counts[stage]
                )
        attributed.append(replace(item, projected_cost_krw=projected))
    if (
        positions != stage_counts
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
    "attribute_v3_projected_cost",
    "prefix_stage_call_counts",
    "units_sha256",
]
