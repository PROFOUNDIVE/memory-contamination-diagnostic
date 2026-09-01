from __future__ import annotations

from collections.abc import Callable, Sequence
from decimal import Decimal, ROUND_CEILING
from typing import Final, assert_never

from pydantic import TypeAdapter, ValidationError

from memcontam.experiment.phase12.runtime_registry import RuntimeTrialResult
from memcontam.experiment.phase13_ordinary_runtime import OrdinaryTask
from memcontam.logging.schema import MethodCall, VerifierResult
from memcontam.readiness.phase13_core_bundle import CoreTask
from memcontam.readiness.phase13_legacy_rag_models import FeasibleTaskName
from memcontam.readiness.phase13_main_live_dispatch import MainUnitDispatchOutput
from memcontam.readiness.phase13_main_production import ProductionObject
from memcontam.readiness.phase13_production_observability import ProviderRequestRecord
from memcontam.readiness.phase13_production_runtime_models import ProductionOrdinaryRunIdentity
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.multiple_choice import verify_answer
from memcontam.verifiers.game24 import verify_expression
from memcontam.verifiers.math_equation_balancer import verify_rhs_completion_answer
from memcontam.verifiers.word_sorting import verify_words


class MainLiveRuntimeError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


_TASK_ADAPTER: Final = TypeAdapter(OrdinaryTask)
_CORE_TASK_ADAPTER: Final = TypeAdapter(CoreTask)
_LEGACY_TASK_ADAPTER: Final = TypeAdapter(FeasibleTaskName)


def task_name(task: str) -> OrdinaryTask:
    try:
        return _TASK_ADAPTER.validate_python(task, strict=True)
    except ValidationError as error:
        raise MainLiveRuntimeError("MAIN_RUNTIME_TASK_INVALID") from error


def core_task_name(task: str) -> CoreTask:
    try:
        return _CORE_TASK_ADAPTER.validate_python(task, strict=True)
    except ValidationError as error:
        raise MainLiveRuntimeError("MAIN_RUNTIME_TASK_INVALID") from error


def legacy_task_name(task: str) -> FeasibleTaskName:
    try:
        return _LEGACY_TASK_ADAPTER.validate_python(task, strict=True)
    except ValidationError as error:
        raise MainLiveRuntimeError("MAIN_RUNTIME_TASK_INVALID") from error


def verifier(task: OrdinaryTask) -> Callable[[str, TaskInstance], bool | VerifierResult]:
    match task:
        case "game24":
            return lambda answer, seen: verify_expression(
                answer, seen.input["numbers"], seen.verifier_spec["target"]
            ).is_correct
        case "math_equation_balancer":
            return lambda answer, seen: verify_rhs_completion_answer(
                answer, seen.verifier_spec
            ).is_correct
        case "word_sorting":
            return lambda answer, seen: verify_words(
                answer.split(), seen.verifier_spec["sorted_words"]
            ).is_correct
        case "mmlu_pro_engineering" | "mmlu_pro_physics":
            return verify_answer
        case unreachable:
            assert_never(unreachable)


def production_identity(unit: ProductionObject) -> ProductionOrdinaryRunIdentity:
    execution_template_id = unit.execution_template_id
    ordered_sample_ids_sha256 = unit.ordered_sample_ids_sha256
    registration_packet_sha256 = unit.registration_packet_sha256
    checkpoint_registry_sha256 = unit.checkpoint_registry_sha256
    if (
        execution_template_id is None
        or ordered_sample_ids_sha256 is None
        or registration_packet_sha256 is None
        or checkpoint_registry_sha256 is None
    ):
        raise MainLiveRuntimeError("MAIN_RUNTIME_IDENTITY_INVALID")
    return ProductionOrdinaryRunIdentity(
        execution_template_id=execution_template_id,
        trajectory_seed=unit.seed,
        concrete_seed_id=str(unit.seed),
        ordered_sample_ids_sha256=ordered_sample_ids_sha256,
        registration_packet_sha256=registration_packet_sha256,
        scientific_result=False,
        checkpoint_registry_sha256=checkpoint_registry_sha256,
    )


def dispatch_output(
    unit: ProductionObject,
    trials: Sequence[RuntimeTrialResult],
    identity: ProductionOrdinaryRunIdentity,
) -> MainUnitDispatchOutput:
    calls = tuple(
        call
        for trial in trials
        for call in trial.outcome.method_calls
        if isinstance(call, MethodCall)
    )
    realized = sum(
        int(
            (Decimal(str(call.provider_cost_usd)) * 1600).to_integral_value(
                rounding=ROUND_CEILING
            )
        )
        for call in calls
    )
    return MainUnitDispatchOutput(
        evidence={
            "unit_id": unit.unit_id,
            "task": unit.task,
            "seed": unit.seed,
            "memory_baseline": unit.memory_baseline,
            "arm": unit.arm,
            "production_identity": identity.model_dump(mode="json"),
            "observability_registration_packet_sha256": identity.registration_packet_sha256,
            "request": ProviderRequestRecord(
                api="OpenAI Responses API",
                model="gpt-5.6-luna",
                service_tier="default",
                reasoning_mode="standard",
                reasoning_effort="none",
                reasoning_context="current_turn",
                previous_response_id=None,
                store=False,
                timeout_seconds=180,
                retries_after_initial_attempt=0,
                semantic_invalid_generic_retry=False,
            ).model_dump(mode="json"),
        },
        provider_calls=calls,
        realized_cost_krw=realized,
    )


__all__ = [
    "MainLiveRuntimeError",
    "core_task_name",
    "dispatch_output",
    "legacy_task_name",
    "production_identity",
    "task_name",
    "verifier",
]
