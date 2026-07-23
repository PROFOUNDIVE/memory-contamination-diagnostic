from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Literal

from memcontam.evaluation.phase12_aggregate import ValidatedRun
from memcontam.experiment.phase12.contracts import (
    CodeMatrixPlan,
    ValidatedExploratoryActivation,
    canonical_json_hash,
)
from memcontam.logging.schema_v3 import ScientificExploratoryCodeRunMetadata, ToolEvent
from memcontam.tools.base import ToolInfrastructureError, ToolPolicyError
from memcontam.tools.policy import load_tool_runtime_contract


_BASELINES = frozenset({"nomem", "bot_style", "dc_rs"})
_MODES = frozenset({"text_only", "python_sandbox"})


class CodeMatrixError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class CodeMatrixRun:
    run: ValidatedRun
    tool_mode: Literal["text_only", "python_sandbox"]
    suffix_id: str
    artifact_id: str
    tool_events: tuple[ToolEvent, ...] = ()


@dataclass(frozen=True)
class CodeMatrixDiagnostic:
    baseline_condition_id: str
    seed_score_deltas: Mapping[int, float]
    mean_score_delta: float
    nomem_adjusted_mean_score_delta: float | None
    tool_event_count: int


@dataclass(frozen=True)
class CodeMatrixAggregate:
    plan_id: str
    plan_hash: str
    activation_manifest_id: str | None
    diagnostics: tuple[CodeMatrixDiagnostic, ...]


def build_code_matrix(config: Mapping[str, object]) -> CodeMatrixPlan:
    if config.get("protocol_version") != "phase12_code_exploratory_v1":
        raise CodeMatrixError("CODE_MATRIX_PROTOCOL_REQUIRED")
    if config.get("activation_status") != "inactive":
        raise CodeMatrixError("EXPLORATORY_ACTIVATION_FORBIDDEN")
    if config.get("task_family") != "game24":
        raise CodeMatrixError("GAME24_ANCHOR_REQUIRED")
    if set(_strings(config.get("baseline_condition_ids"))) != _BASELINES:
        raise CodeMatrixError("CODE_MATRIX_DOMAIN_INCOMPLETE")

    registry_id = _string(config.get("exploratory_run_template_registry_id"))
    registry_hash = _string(config.get("exploratory_run_template_registry_hash"))
    slots = _strings(config.get("abstract_slots"))
    calls = config.get("estimated_exploratory_calls")
    if not registry_id or not registry_hash or not slots or len(set(slots)) != len(slots):
        raise CodeMatrixError("CODE_MATRIX_PLAN_INVALID")
    if type(calls) is not int or calls < 0:
        raise CodeMatrixError("CODE_MATRIX_PLAN_INVALID")
    if any(not slot.startswith("game24|exploratory|") for slot in slots):
        raise CodeMatrixError("GAME24_ANCHOR_REQUIRED")
    _validate_oci_contract(config.get("oci_contract_path"))

    payload = {
        "exploratory_run_template_registry_id": registry_id,
        "exploratory_run_template_registry_hash": registry_hash,
        "abstract_slots": slots,
        "estimated_exploratory_calls": calls,
    }
    artifact_hash = canonical_json_hash(payload)
    return CodeMatrixPlan(
        plan_id=f"code-matrix-{artifact_hash[:12]}",
        artifact_hash=artifact_hash,
        **payload,
    )


def aggregate_code_matrix(
    runs: Sequence[CodeMatrixRun],
    plan: CodeMatrixPlan,
    activation: ValidatedExploratoryActivation | None,
    *,
    claim_kind: str = "paired_diagnostic",
) -> CodeMatrixAggregate:
    if claim_kind != "paired_diagnostic":
        raise CodeMatrixError("CROSS_TOOL_SUPERIORITY_CLAIM_FORBIDDEN")
    if not runs:
        raise CodeMatrixError("EMPTY_CODE_MATRIX")
    if activation is None:
        raise CodeMatrixError("EXPLORATORY_ACTIVATION_REQUIRED")
    _validate_activation(plan, activation)

    artifacts: set[str] = set()
    pairs: dict[tuple[str, str, int, str], dict[str, CodeMatrixRun]] = defaultdict(dict)
    slots_by_baseline_mode: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in runs:
        metadata = item.run.metadata
        if not isinstance(metadata, ScientificExploratoryCodeRunMetadata):
            raise CodeMatrixError("EXPLORATORY_ACTIVATION_REQUIRED")
        if (
            not item.artifact_id
            or item.artifact_id in artifacts
            or not item.suffix_id
            or item.tool_mode not in _MODES
        ):
            raise CodeMatrixError("TOOL_MODE_POOLING_FORBIDDEN")
        artifacts.add(item.artifact_id)
        if metadata.run_template_registry_version != plan.exploratory_run_template_registry_hash:
            raise CodeMatrixError("STALE_EXPLORATORY_REGISTRY")
        if (
            metadata.protocol_version != "phase12_code_exploratory_v1"
            or metadata.task_family != "game24"
            or metadata.baseline_condition_id not in _BASELINES
        ):
            raise CodeMatrixError("TOOL_MODE_POOLING_FORBIDDEN")
        slot = metadata.abstract_seed_slot_or_none
        if (
            metadata.exploratory_activation_manifest_id != activation.exploratory_activation_manifest_id
            or metadata.source_route_selection_manifest_id != activation.route_selection_manifest_id
            or metadata.source_seed_allocation_manifest_id != activation.seed_allocation_manifest_id
            or slot is None
            or activation.exploratory_slot_to_seed.get(slot) != metadata.trajectory_seed
        ):
            raise CodeMatrixError("EXPLORATORY_SEED_ASSIGNMENT_MISMATCH")
        key = (metadata.baseline_condition_id, slot, metadata.trajectory_seed, item.suffix_id)
        if item.tool_mode in pairs[key]:
            raise CodeMatrixError("TOOL_MODE_POOLING_FORBIDDEN")
        pairs[key][item.tool_mode] = item
        slots_by_baseline_mode[(metadata.baseline_condition_id, item.tool_mode)].add(slot)

    if any(
        slots_by_baseline_mode[(baseline, mode)] != set(plan.abstract_slots)
        for baseline in _BASELINES
        for mode in _MODES
    ):
        raise CodeMatrixError("UNPAIRED_SUFFIX")

    deltas: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    tool_counts: dict[str, int] = defaultdict(int)
    for (baseline, _, seed, _), pair in pairs.items():
        if set(pair) != _MODES:
            raise CodeMatrixError("UNPAIRED_SUFFIX")
        text, code = pair["text_only"], pair["python_sandbox"]
        if (
            text.artifact_id == code.artifact_id
            or text.run.metadata.run_template_id == code.run.metadata.run_template_id
            or text.run.metadata.tool_contract_hash == code.run.metadata.tool_contract_hash
        ):
            raise CodeMatrixError("TOOL_MODE_POOLING_FORBIDDEN")
        deltas[baseline][seed].append(_score(code.run) - _score(text.run))
        tool_counts[baseline] += len(code.tool_events)

    if set(deltas) != _BASELINES:
        raise CodeMatrixError("CODE_MATRIX_DOMAIN_INCOMPLETE")
    seed_deltas = {
        baseline: {seed: mean(values) for seed, values in seeds.items()}
        for baseline, seeds in deltas.items()
    }
    nomem_seeds = set(seed_deltas["nomem"])
    if any(set(values) != nomem_seeds for values in seed_deltas.values()):
        raise CodeMatrixError("UNPAIRED_SUFFIX")
    mean_deltas = {baseline: mean(values.values()) for baseline, values in seed_deltas.items()}
    return CodeMatrixAggregate(
        plan_id=plan.plan_id,
        plan_hash=plan.artifact_hash,
        activation_manifest_id=activation.exploratory_activation_manifest_id,
        diagnostics=tuple(
            CodeMatrixDiagnostic(
                baseline_condition_id=baseline,
                seed_score_deltas=seed_deltas[baseline],
                mean_score_delta=mean_deltas[baseline],
                nomem_adjusted_mean_score_delta=(
                    None if baseline == "nomem" else mean_deltas[baseline] - mean_deltas["nomem"]
                ),
                tool_event_count=tool_counts[baseline],
            )
            for baseline in sorted(_BASELINES)
        ),
    )


def _validate_oci_contract(value: object) -> None:
    if not isinstance(value, (str, Path)):
        raise CodeMatrixError("OCI_CONTRACT_UNAVAILABLE")
    try:
        load_tool_runtime_contract(Path(value), scientific=False)
    except (ToolInfrastructureError, ToolPolicyError) as error:
        raise CodeMatrixError("OCI_CONTRACT_UNAVAILABLE") from error


def _validate_activation(plan: CodeMatrixPlan, activation: ValidatedExploratoryActivation) -> None:
    if not activation.resource_manifest_id or not activation.resource_manifest_hash:
        raise CodeMatrixError("EXPLORATORY_RESOURCE_RESERVATION_REQUIRED")
    if activation.exploratory_plan_id != plan.plan_id or activation.exploratory_plan_hash != plan.artifact_hash:
        raise CodeMatrixError("STALE_EXPLORATORY_PLAN")
    if (
        activation.exploratory_run_template_registry_id
        != plan.exploratory_run_template_registry_id
        or activation.exploratory_run_template_registry_hash
        != plan.exploratory_run_template_registry_hash
    ):
        raise CodeMatrixError("STALE_EXPLORATORY_REGISTRY")
    if set(activation.exploratory_slot_to_seed) != set(plan.abstract_slots):
        raise CodeMatrixError("EXPLORATORY_SEED_ASSIGNMENT_MISMATCH")
    if plan.estimated_exploratory_calls > activation.exploratory_call_budget:
        raise CodeMatrixError("EXPLORATORY_BUDGET_INSUFFICIENT")
    if (
        activation.exploratory_call_budget + activation.reproducibility_reserve
        > activation.remaining_call_capacity
    ):
        raise CodeMatrixError("REPRODUCIBILITY_RESERVE_INSUFFICIENT")


def _score(run: ValidatedRun) -> float:
    values = [
        trial.verified_score
        for trial in run.trials
        if trial.analysis_inclusion == "included"
        and trial.execution_status == "completed"
        and trial.verified_score in {0, 1}
    ]
    if not values:
        raise CodeMatrixError("UNPAIRED_SUFFIX")
    return mean(values)


def _string(value: object) -> str:
    return value if isinstance(value, str) else ""


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or any(not isinstance(item, str) or not item for item in value):
        return ()
    return tuple(value)


__all__ = [
    "CodeMatrixAggregate",
    "CodeMatrixDiagnostic",
    "CodeMatrixError",
    "CodeMatrixRun",
    "aggregate_code_matrix",
    "build_code_matrix",
]
