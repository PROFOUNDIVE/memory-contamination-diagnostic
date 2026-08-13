from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from random import Random
from statistics import mean
from typing import Final, Literal

from memcontam.manifests.phase13 import PrefixDerivationArtifact
from memcontam.readiness.phase13_analysis_contract import load_analysis_registry
from memcontam.readiness.phase13_calibration_v2_runtime_models import (
    CompletedTrajectory,
    TrajectoryRequest,
)
from memcontam.readiness.phase13_support_authority import authenticate_conformance


ANALYSIS_PATH: Final = Path("data/phase13/authority/analysis_registry_v1.json")
ANALYSIS_SHA256: Final = "b58e6aec8acc040fb934e9b25842eb68c702d098a08b41ba0eab9502a198a0f3"
BASELINES: Final = ("fh_bounded", "rag_frozen", "bot_style", "reflexion_style")
PAIRS: Final = ((0, 1), (0, 2), (0, 3))


class InferenceError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ArmTrajectory:
    baseline: str
    clean: tuple[Literal[0, 1], ...]
    contam: tuple[Literal[0, 1], ...]


@dataclass(frozen=True, slots=True)
class SeedBundle:
    request: TrajectoryRequest
    source: CompletedTrajectory
    prefix: PrefixDerivationArtifact
    arms: tuple[ArmTrajectory, ...]


@dataclass(frozen=True, slots=True)
class SlotSupport:
    support_population_id: str
    supported: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class TaskInferenceInput:
    task: str
    seeds: tuple[SeedBundle, ...]
    support: tuple[SlotSupport, ...]


@dataclass(frozen=True, slots=True)
class InferenceSlot:
    order: int
    estimand_id: str
    support_population_id: str
    status: Literal["ESTIMABLE", "NOT_ESTIMABLE"]
    reason: str | None
    estimate: float | str
    lower: float | str
    upper: float | str
    p_value: float
    holm_adjusted_p: float
    reject_null: bool
    decision: Literal["REJECTED", "NOT_REJECTED"]


@dataclass(frozen=True, slots=True)
class PrimaryInference:
    task: str
    family_id: str
    interval_id: str
    interval_method: str
    replicates: int
    rng_seed: int
    slots: tuple[InferenceSlot, ...]


@dataclass(frozen=True, slots=True)
class NonPrimaryEstimate:
    analysis_window_id: str
    inference_status: Literal["estimation_only"]
    description: str
    interval_id: str
    interval_method: str
    estimate: float | str
    lower: float | str
    upper: float | str


def _authority(root: Path):  # noqa: ANN202
    path = root / ANALYSIS_PATH
    if hashlib.sha256(path.read_bytes()).hexdigest() != ANALYSIS_SHA256:
        raise InferenceError("INFERENCE_AUTHORITY_INVALID")
    return load_analysis_registry(path, root)


def _validate(request: TaskInferenceInput, root: Path):  # noqa: ANN202
    analysis = _authority(root)
    family = next((row for row in analysis.inference.families if row.task == request.task), None)
    if family is None:
        raise InferenceError("INFERENCE_TASK_UNREGISTERED")
    populations = (
        *(row.support_population_id for row in analysis.support.level_1),
        *(row.support_population_id for row in analysis.support.level_2 if row.route_gating),
    )
    supplied = tuple(row.support_population_id for row in request.support)
    if len(supplied) != 7:
        raise InferenceError("INFERENCE_SLOT_INVENTORY_INVALID")
    if supplied != populations:
        raise InferenceError("INFERENCE_SLOT_ORDER_INVALID")
    if len(request.seeds) < 2:
        raise InferenceError("INCOMPLETE_SEED_DENOMINATOR")
    seed_ids = tuple(row.request.seed_id for row in request.seeds)
    if len(set(seed_ids)) != len(seed_ids):
        raise InferenceError("DUPLICATE_TRAJECTORY_SEED")
    for bundle in request.seeds:
        if bundle.request.task != request.task:
            raise InferenceError("INFERENCE_TASK_SOURCE_MISMATCH")
        try:
            authenticate_conformance(bundle.prefix, bundle.request, bundle.source)
        except ValueError as error:
            raise InferenceError("INFERENCE_SOURCE_AUTHORITY_INVALID") from error
        if tuple(row.baseline for row in bundle.arms) != BASELINES:
            raise InferenceError("INCOMPLETE_BASELINE_PANEL")
        if any(len(row.clean) != 5 or len(row.contam) != 5 for row in bundle.arms):
            raise InferenceError("INCOMPLETE_H5_TRAJECTORY")
    return analysis, family


def _effects(request: TaskInferenceInput) -> tuple[tuple[float, ...], ...]:
    baseline = tuple(
        tuple(mean(arm.clean) - mean(arm.contam) for arm in seed.arms)
        for seed in request.seeds
    )
    columns = tuple(tuple(row[index] for row in baseline) for index in range(4))
    return (*columns, *(tuple(columns[left][seed] - columns[right][seed] for seed in range(len(baseline))) for left, right in PAIRS))


def _statistics(effects: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, float, float, float], ...]:
    observed = tuple(mean(values) for values in effects)
    bootstrap = [[] for _ in effects]
    null = [[] for _ in effects]
    generator = Random(13)
    count = len(effects[0])
    for _ in range(20_000):
        indices = tuple(generator.randrange(count) for _ in range(count))
        for slot, values in enumerate(effects):
            bootstrap[slot].append(mean(values[index] for index in indices))
            null[slot].append(mean(values[index] - observed[slot] for index in indices))
    result = []
    for slot, estimate in enumerate(observed):
        ordered = sorted(bootstrap[slot])
        p_value = (1 + sum(abs(value) >= abs(estimate) for value in null[slot])) / 20_001
        result.append((estimate, ordered[500], ordered[19_499], p_value))
    return tuple(result)


def _holm(rows: tuple[InferenceSlot, ...]) -> tuple[InferenceSlot, ...]:
    ranked = sorted(enumerate(rows), key=lambda item: (item[1].p_value, item[0]))
    adjusted = [0.0] * 7
    rejected = [False] * 7
    running = 0.0
    active = True
    for rank, (index, row) in enumerate(ranked):
        running = max(running, min(1.0, (7 - rank) * row.p_value))
        adjusted[index] = running
        active = active and row.p_value <= 0.05 / (7 - rank)
        rejected[index] = active
    return tuple(
        replace(row, holm_adjusted_p=adjusted[index], reject_null=rejected[index],
                decision="REJECTED" if rejected[index] else "NOT_REJECTED")
        for index, row in enumerate(rows)
    )


def infer_primary(request: TaskInferenceInput, root: Path) -> PrimaryInference:
    analysis, family = _validate(request, root)
    statistics = _statistics(_effects(request))
    rows = []
    for support, slot, values in zip(request.support, family.slots, statistics, strict=True):
        if support.supported:
            estimate, lower, upper, p_value = values
            rows.append(InferenceSlot(slot.order, slot.estimand_id, support.support_population_id,
                                      "ESTIMABLE", None, estimate, lower, upper, p_value,
                                      0.0, False, "NOT_REJECTED"))
        else:
            if not support.reason:
                raise InferenceError("NOT_ESTIMABLE_REASON_REQUIRED")
            rows.append(InferenceSlot(slot.order, slot.estimand_id, support.support_population_id,
                                      "NOT_ESTIMABLE", support.reason, "NOT_ESTIMABLE",
                                      "NOT_ESTIMABLE", "NOT_ESTIMABLE", 1.0, 1.0, False,
                                      "NOT_REJECTED"))
    return PrimaryInference(request.task, family.family_id, analysis.inference.interval_id,
                            analysis.inference.interval_method, 20_000, 13, _holm(tuple(rows)))


def estimate_non_primary(
    analysis_window_id: str, seed_estimates: tuple[float, ...], root: Path
) -> NonPrimaryEstimate:
    analysis = _authority(root)
    windows = {row.analysis_window_id for row in analysis.non_primary_windows}
    if analysis_window_id not in windows:
        raise InferenceError("NON_PRIMARY_WINDOW_UNREGISTERED")
    if len(seed_estimates) < 2:
        values: tuple[float | str, float | str, float | str] = (
            "NOT_ESTIMABLE", "NOT_ESTIMABLE", "NOT_ESTIMABLE",
        )
    else:
        estimate, lower, upper, _ = _statistics((seed_estimates,))[0]
        values = estimate, lower, upper
    return NonPrimaryEstimate(
        analysis_window_id, "estimation_only", "seed-level estimation interval only",
        analysis.inference.interval_id, analysis.inference.interval_method, *values,
    )


__all__ = (
    "ArmTrajectory", "InferenceError", "NonPrimaryEstimate", "PrimaryInference", "SeedBundle",
    "SlotSupport", "TaskInferenceInput", "estimate_non_primary", "infer_primary",
)
