from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Final, Literal, TypeAlias, assert_never

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

from memcontam.logging.schema import MethodCall
from memcontam.readiness.phase13_cost_policy import load_cost_policy_bundle
from memcontam.readiness.phase13_cost_policy_models import Sha256
from memcontam.readiness.phase13_main_production import ProductionObject
from memcontam.readiness.phase13_main_runner_models import InFlightEvidence
from memcontam.readiness.phase13_production_observability import (
    ProductionObservabilityArchive,
    ProviderRequestRecord,
)
from memcontam.readiness.phase13_production_runtime_models import ProductionOrdinaryRunIdentity
from memcontam.readiness.phase13_readiness0_evidence_models import (
    ProviderAuthorityContract,
    ProviderRequestContract,
)
from memcontam.readiness.phase13_v3_cost_actual import reconcile_actual
from memcontam.readiness.phase13_v3_cost_models import ActivatedPolicyV3, ProviderCostEvidence
from memcontam.readiness.phase13_v3_retry import AUTHORITY_TO_STAGE


class MainEvidenceValidationError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class MainMethodCall(MethodCall):
    dispatch_id: Sha256


RecordedMethodCall: TypeAlias = Annotated[
    MainMethodCall | MethodCall,
    Field(union_mode="left_to_right"),
]


class MainRuntimeEvidence(_FrozenModel):
    unit_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    task: str = Field(min_length=1)
    seed: int = Field(ge=0)
    memory_baseline: str | None
    arm: str = Field(min_length=1)
    production_identity: ProductionOrdinaryRunIdentity
    observability_registration_packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verifier_result: bool | None = None
    request: ProviderRequestRecord
    production_observability_archive: ProductionObservabilityArchive | None = None

    @model_validator(mode="after")
    def _observability_identity(self) -> MainRuntimeEvidence:
        if (
            self.observability_registration_packet_sha256
            != self.production_identity.registration_packet_sha256
        ):
            raise MainEvidenceValidationError("MAIN_UNIT_RUNTIME_IDENTITY_INVALID")
        return self


class PrefixCheckpointState(_FrozenModel):
    schema_version: Literal["phase13_main_prefix_checkpoint_v1"]
    baseline: str = Field(min_length=1)
    checkpoint_id: str = Field(min_length=1)
    checkpoint_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_state_utf8: str
    checkpoint_index: int = Field(ge=0)


class PrefixUnitEvidence(_FrozenModel):
    evidence_kind: Literal["CLEAN_PREFIX"]
    prefix_unit_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint: PrefixCheckpointState
    runtime_evidence: MainRuntimeEvidence


class MemoryUnitEvidence(_FrozenModel):
    evidence_kind: Literal["MEMORY_BEARING"]
    prefix_unit_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    consumed_checkpoint_id: str = Field(min_length=1)
    consumed_checkpoint_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    consumed_checkpoint_canonical_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_evidence: MainRuntimeEvidence


class NoMemUnitEvidence(_FrozenModel):
    evidence_kind: Literal["NO_MEMORY_SINGLETON"]
    internal_baseline: Literal["nomem"]
    internal_arm: Literal["clean"]
    scientific_arm: Literal["NOT_APPLICABLE"]
    runtime_evidence: MainRuntimeEvidence


UnitEvidence: TypeAlias = Annotated[
    PrefixUnitEvidence | MemoryUnitEvidence | NoMemUnitEvidence,
    Field(discriminator="evidence_kind"),
]


class MainUnitEvidence(_FrozenModel):
    schema_version: Literal["phase13_main_unit_evidence_v1"]
    sequence: int = Field(ge=0)
    unit_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: Literal["CLEAN_PREFIX", "MEMORY_BEARING", "NO_MEMORY_SINGLETON"]
    seed: int = Field(ge=0)
    task: str = Field(min_length=1)
    memory_baseline: str | None
    arm: str = Field(min_length=1)
    evidence: UnitEvidence
    provider_calls: tuple[RecordedMethodCall, ...]
    realized_cost_krw: int = Field(ge=0)


class MainReconciliationEvidence(_FrozenModel):
    schema_version: Literal["phase13_main_reconciliation_evidence_v1"]
    disposition: Literal["NO_PROVIDER_REQUEST", "TERMINAL_FAILURE"]
    unit_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    intent_event_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    authorization_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    realized_cost_krw: int = Field(ge=0)
    failure_code: str | None


@dataclass(frozen=True, slots=True)
class DispatchEvidenceInput:
    evidence: JsonValue
    provider_calls: tuple[MethodCall, ...]
    claimed_cost_krw: int
    current_policy: ActivatedPolicyV3 | None = None
    attempt_costs: tuple[tuple[str, tuple[ProviderCostEvidence, ...]], ...] = ()
    count_cost_krw: int = 0


class CurrentProviderAuthorityContract(_FrozenModel):
    maximum_input_tokens: int = Field(gt=0)
    maximum_output_tokens: int = Field(gt=0)
    execution_envelope_id: str = Field(min_length=1)
    execution_envelope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    failure_contract_id: str = Field(min_length=1)
    failure_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    terminal_failure_contract_id: str = Field(min_length=1)
    terminal_failure_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rate_card_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def current_stage_authority(policy: ActivatedPolicyV3, stage: str) -> tuple[int, dict[str, int | str]]:
    authority = policy.authority
    stage_id = next(name for name, native in AUTHORITY_TO_STAGE.items() if native == stage)
    envelope = next(row for row in authority.registry.stages if row.stage_id == stage_id)
    rate_hash = hashlib.sha256(json.dumps(
        policy.rate_card.model_dump(mode="json"), sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    return envelope.maximum_output_tokens, {
        "maximum_input_tokens": envelope.maximum_input_tokens,
        "maximum_output_tokens": envelope.maximum_output_tokens,
        "execution_envelope_id": authority.registry.registry_id,
        "execution_envelope_sha256": authority.registry.sha256,
        "failure_contract_id": authority.transport.contract_id,
        "failure_contract_sha256": authority.transport.sha256,
        "terminal_failure_contract_id": authority.terminal.contract_id,
        "terminal_failure_contract_sha256": authority.terminal.sha256,
        "rate_card_sha256": rate_hash,
    }


_BASELINE_STAGES: Final = MappingProxyType(
    {
        "fh_bounded": ("full_history_generate",),
        "rag_frozen": ("rag_generate",),
        "bot_style": (
            "bot_problem_distill",
            "bot_instantiate_solve",
            "bot_thought_distill",
        ),
        "reflexion_style": ("reflexion_generate", "reflexion_reflect"),
        "dc_rs": ("dc_rs_synthesize", "dc_rs_generate"),
    }
)
_ROOT: Final = Path(__file__).resolve().parents[3]


def validate_dispatch_evidence(
    unit: ProductionObject,
    supplied: DispatchEvidenceInput,
) -> tuple[UnitEvidence, int]:
    evidence = _parse_evidence(unit, supplied.evidence)
    calls = supplied.provider_calls
    call_ids = tuple(call.call_id for call in calls)
    if (
        any(not call_id for call_id in call_ids)
        or len(call_ids) != len(set(call_ids))
        or not _stages_valid(unit, calls, evidence)
        or any(not _completed_call(call, evidence.runtime_evidence.request, supplied.current_policy) for call in calls)
        or any(not _reconciled_cost(call) for call in calls)
    ):
        raise MainEvidenceValidationError("MAIN_UNIT_PROVIDER_CALLS_INVALID")
    if supplied.current_policy is not None:
        by_id = dict(supplied.attempt_costs)
        if (len(by_id) != len(calls)
            or any(not isinstance(call, MainMethodCall) or call.dispatch_id not in by_id
                   or not 1 <= len(by_id[call.dispatch_id]) <= 2 for call in calls)):
            raise MainEvidenceValidationError("MAIN_UNIT_PROVIDER_CALLS_INVALID")
        realized = sum(reconcile_actual(cost).realized_krw for costs in by_id.values() for cost in costs)
        if type(supplied.count_cost_krw) is not int or supplied.count_cost_krw < 0:
            raise MainEvidenceValidationError("MAIN_UNIT_REALIZED_COST_MISMATCH")
        realized += supplied.count_cost_krw
    else:
        realized = sum(int((Decimal(str(call.provider_cost_usd)) * Decimal(1600)).to_integral_value(
            rounding=ROUND_CEILING)) for call in calls)
    if supplied.claimed_cost_krw != realized:
        raise MainEvidenceValidationError("MAIN_UNIT_REALIZED_COST_MISMATCH")
    return evidence, realized


def _stages_valid(
    unit: ProductionObject,
    calls: tuple[MethodCall, ...],
    evidence: UnitEvidence,
) -> bool:
    archive = evidence.runtime_evidence.production_observability_archive
    terminal_records = archive.records if (
        archive is not None and unit.kind != "CLEAN_PREFIX"
        and 0 < len(archive.records) <= 50
        and archive.records[-1].evidence.trial.execution_status == "failed"
    ) else ()
    if terminal_records:
        archived_calls = tuple(call for record in terminal_records for call in record.method_calls)
        if tuple(call.call_id for call in calls) != tuple(call.call_id for call in archived_calls):
            return False
    if unit.memory_baseline == "reflexion_style":
        if terminal_records:
            completed = terminal_records[:-1]
            failed = terminal_records[-1]
            return (
                _reflexion_stage_sequences_valid(
                    tuple(call for row in completed for call in row.method_calls), len(completed), unit.kind,
                    tuple(row.evidence.trial_id for row in completed))
                and tuple(call.stage for call in failed.method_calls) in {
                    ("reflexion_generate",),
                    ("reflexion_generate", "reflexion_reflect"),
                    ("reflexion_generate", "reflexion_reflect", "reflexion_generate"),
                    ("reflexion_generate", "reflexion_reflect", "reflexion_generate", "reflexion_reflect"),
                }
            )
        expected_trials = 1 if unit.kind == "CLEAN_PREFIX" else (len(terminal_records) if terminal_records else 50)
        trial_ids = (
            None
            if archive is None
            else tuple(record.evidence.trial_id for record in archive.records)
        )
        return _reflexion_stage_sequences_valid(
            calls, expected_trials, unit.kind, trial_ids
        )
    stages = tuple(call.stage for call in calls)
    if terminal_records:
        expected = ("no_memory_generate",) if unit.memory_baseline is None else _BASELINE_STAGES[unit.memory_baseline]
        return (all(tuple(call.stage for call in record.method_calls) == expected for record in terminal_records[:-1])
                and 0 < len(terminal_records[-1].method_calls) <= len(expected)
                and tuple(call.stage for call in terminal_records[-1].method_calls)
                == expected[:len(terminal_records[-1].method_calls)])
    return Counter(stages) == _expected_stages(unit)


def _reflexion_stage_sequences_valid(
    calls: tuple[MethodCall, ...], expected_trials: int, kind: str,
    expected_trial_ids: tuple[str, ...] | None = None,
) -> bool:
    groups: list[list[str]] = []
    trial_ids: list[str] = []
    for call in calls:
        if call.call_id is None or ":call:" not in call.call_id:
            return False
        trial_id = call.call_id.rsplit(":call:", 1)[0]
        if not trial_ids or trial_ids[-1] != trial_id:
            if trial_id in trial_ids:
                return False
            trial_ids.append(trial_id)
            groups.append([])
        if call.call_id != f"{trial_id}:call:{len(groups[-1]) + 1}":
            return False
        groups[-1].append(call.stage)
    allowed = (
        {("reflexion_generate",), ("reflexion_generate", "reflexion_reflect")}
        if kind == "CLEAN_PREFIX"
        else {
            ("reflexion_generate",),
            ("reflexion_generate", "reflexion_reflect", "reflexion_generate"),
            (
            "reflexion_generate",
            "reflexion_reflect",
            "reflexion_generate",
            "reflexion_reflect",
            ),
        }
    )
    return (
        len(groups) == expected_trials
        and (expected_trial_ids is None or tuple(trial_ids) == expected_trial_ids)
        and all(tuple(group) in allowed for group in groups)
    )


def load_durable_unit_evidence(
    path: Path,
    unit: ProductionObject,
    evidence_sha256: str,
    realized_cost_krw: int,
) -> MainUnitEvidence:
    try:
        raw = path.read_bytes()
        record = MainUnitEvidence.model_validate_json(raw)
    except (OSError, ValidationError) as error:
        raise MainEvidenceValidationError("MAIN_UNIT_EVIDENCE_INVALID") from error
    if (
        hashlib.sha256(raw).hexdigest() != evidence_sha256
        or record.realized_cost_krw != realized_cost_krw
        or not _record_identity_valid(record, unit)
    ):
        raise MainEvidenceValidationError("MAIN_UNIT_EVIDENCE_JOIN_INVALID")
    validated, validated_cost = validate_dispatch_evidence(
        unit,
        DispatchEvidenceInput(
            record.evidence.model_dump(mode="json"),
            record.provider_calls,
            record.realized_cost_krw,
        ),
    )
    if validated != record.evidence or validated_cost != realized_cost_krw:
        raise MainEvidenceValidationError("MAIN_UNIT_EVIDENCE_JOIN_INVALID")
    return record


def load_durable_reconciliation_evidence(
    path: Path,
    evidence: InFlightEvidence,
) -> MainReconciliationEvidence:
    try:
        raw = path.read_bytes()
        record = MainReconciliationEvidence.model_validate_json(raw)
    except (OSError, ValidationError) as error:
        raise MainEvidenceValidationError("MAIN_RECONCILIATION_EVIDENCE_INVALID") from error
    context = evidence.context
    if (
        hashlib.sha256(raw).hexdigest() != evidence.evidence_sha256
        or record.disposition != evidence.disposition
        or record.unit_id != context.unit_id
        or record.intent_event_hash != context.intent_event_hash
        or record.package_sha256 != context.package_sha256
        or record.authorization_sha256 != context.authorization_sha256
        or record.realized_cost_krw != evidence.realized_cost_krw
        or record.failure_code != evidence.failure_code
        or (
            record.disposition == "NO_PROVIDER_REQUEST"
            and (record.realized_cost_krw != 0 or record.failure_code is not None)
        )
        or (record.disposition == "TERMINAL_FAILURE" and not record.failure_code)
    ):
        raise MainEvidenceValidationError("MAIN_RECONCILIATION_EVIDENCE_INVALID")
    return record


def _completed_call(call: MethodCall, runtime_request: ProviderRequestRecord,
                    current_policy: ActivatedPolicyV3 | None) -> bool:
    authority: ProviderAuthorityContract | CurrentProviderAuthorityContract
    expected_authority: Mapping[str, int | str]
    try:
        request = ProviderRequestContract.model_validate(call.provider_request_contract)
        if current_policy is None:
            authority = ProviderAuthorityContract.model_validate(call.provider_authority_contract)
            bundle = load_cost_policy_bundle(_ROOT)
            stage = next(row for row in bundle.registry.stages if row.semantic_stage_id == call.stage)
            maximum_output_tokens = stage.maximum_output_tokens
            expected_authority = {
                "maximum_input_tokens": stage.maximum_input_tokens,
                "maximum_output_tokens": stage.maximum_output_tokens,
                "execution_envelope_id": bundle.registry.registry_id,
                "execution_envelope_sha256": bundle.registry.registry_hash,
                "failure_contract_id": bundle.retry.contract_id,
                "failure_contract_sha256": bundle.retry.contract_hash,
                "terminal_failure_contract_id": bundle.retry.terminal_failure_contract_id,
                "terminal_failure_contract_sha256": bundle.retry.terminal_failure_contract_sha256,
                "rate_card_sha256": hashlib.sha256(json.dumps(
                    bundle.proof.rate_card.model_dump(mode="json"), sort_keys=True, separators=(",", ":"),
                ).encode()).hexdigest(),
            }
        else:
            authority = CurrentProviderAuthorityContract.model_validate(call.provider_authority_contract)
            maximum_output_tokens, expected_authority = current_stage_authority(current_policy, call.stage)
    except (StopIteration, ValidationError, ValueError):
        return False
    expected_input_sha256 = hashlib.sha256(
        json.dumps(
            call.messages,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    return (
        call.provider_status == "completed"
        and call.provider_response_status == "completed"
        and call.model == runtime_request.model
        and call.provider_returned_model == runtime_request.model
        and call.provider_service_tier == runtime_request.service_tier
        and call.transport_attempts == 1
        and call.retry_count == 0
        and bool(call.provider_response_id)
        and bool(call.provider_usage)
        and bool(call.token_usage)
        and call.error_type is None
        and call.failure_code is None
        and call.provider_incomplete_reason is None
        and request.model == runtime_request.model
        and request.input_sha256 == expected_input_sha256
        and request.temperature == call.temperature
        and call.temperature == 0.0
        and request.top_p == call.top_p
        and call.top_p == 1.0
        and request.reasoning.mode == runtime_request.reasoning_mode
        and request.reasoning.effort == runtime_request.reasoning_effort
        and request.reasoning.context == runtime_request.reasoning_context
        and request.previous_response_id == runtime_request.previous_response_id
        and request.service_tier == runtime_request.service_tier
        and request.store == runtime_request.store
        and tuple(request.tools) == runtime_request.tools
        and request.max_output_tokens == maximum_output_tokens
        and authority.model_dump(mode="json") == expected_authority
    )


def _reconciled_cost(call: MethodCall) -> bool:
    selected = call.provider_cost_usd
    derived = call.derived_cost_usd
    authoritative = call.authoritative_provider_cost_usd
    if selected is None or derived is None:
        return False
    if authoritative is None:
        return call.provider_cost_source == "DERIVED_FROM_PROVIDER_USAGE" and Decimal(
            str(selected)
        ) == Decimal(str(derived))
    return call.provider_cost_source == "AUTHORITATIVE_PROVIDER" and Decimal(
        str(selected)
    ) == Decimal(str(authoritative))


def _parse_evidence(unit: ProductionObject, value: JsonValue) -> UnitEvidence:
    evidence: UnitEvidence
    try:
            match unit.kind:
                case "CLEAN_PREFIX":
                    evidence = PrefixUnitEvidence.model_validate_json(json.dumps(value))
                case "MEMORY_BEARING":
                    evidence = MemoryUnitEvidence.model_validate_json(json.dumps(value))
                case "NO_MEMORY_SINGLETON":
                    evidence = NoMemUnitEvidence.model_validate_json(json.dumps(value))
                case unreachable:
                    assert_never(unreachable)
    except ValidationError as error:
        raise MainEvidenceValidationError("MAIN_UNIT_EVIDENCE_JOIN_INVALID") from error
    if not _evidence_join_valid(evidence, unit):
        raise MainEvidenceValidationError("MAIN_UNIT_EVIDENCE_JOIN_INVALID")
    if not _evidence_identity_valid(evidence, unit):
        raise MainEvidenceValidationError("MAIN_UNIT_RUNTIME_IDENTITY_INVALID")
    return evidence


def _record_identity_valid(record: MainUnitEvidence, unit: ProductionObject) -> bool:
    return (
        record.sequence == unit.sequence
        and record.unit_id == unit.unit_id
        and record.kind == unit.kind
        and record.seed == unit.seed
        and record.task == unit.task
        and record.memory_baseline == unit.memory_baseline
        and record.arm == unit.arm
    )


def _evidence_join_valid(evidence: UnitEvidence, unit: ProductionObject) -> bool:
    match unit.kind:
        case "CLEAN_PREFIX":
            return (
                isinstance(evidence, PrefixUnitEvidence)
                and evidence.prefix_unit_id == unit.unit_id
                and evidence.checkpoint.baseline == unit.memory_baseline
            )
        case "MEMORY_BEARING":
            return (
                isinstance(evidence, MemoryUnitEvidence)
                and evidence.prefix_unit_id == unit.prefix_unit_id
            )
        case "NO_MEMORY_SINGLETON":
            return isinstance(evidence, NoMemUnitEvidence) and unit.prefix_unit_id is None
        case unreachable:
            assert_never(unreachable)


def _evidence_identity_valid(evidence: UnitEvidence, unit: ProductionObject) -> bool:
    runtime = evidence.runtime_evidence
    identity = runtime.production_identity
    return (
        unit.execution_template_id is not None
        and unit.ordered_sample_ids_sha256 is not None
        and unit.registration_packet_sha256 is not None
        and unit.checkpoint_registry_sha256 is not None
        and runtime.unit_id == unit.unit_id
        and runtime.task == unit.task
        and runtime.seed == unit.seed
        and runtime.memory_baseline == unit.memory_baseline
        and runtime.arm == unit.arm
        and identity.execution_template_id == unit.execution_template_id
        and identity.trajectory_seed == unit.seed
        and identity.concrete_seed_id == str(unit.seed)
        and identity.ordered_sample_ids_sha256 == unit.ordered_sample_ids_sha256
        and identity.registration_packet_sha256 == unit.registration_packet_sha256
        and identity.checkpoint_registry_sha256 == unit.checkpoint_registry_sha256
        and runtime.observability_registration_packet_sha256
        == unit.registration_packet_sha256
    )


def _expected_stages(unit: ProductionObject) -> Counter[str]:
    match unit.kind:
        case "NO_MEMORY_SINGLETON":
            return Counter({"no_memory_generate": 50})
        case "CLEAN_PREFIX" | "MEMORY_BEARING" as kind:
            baseline = unit.memory_baseline
            if baseline is None:
                raise MainEvidenceValidationError("MAIN_UNIT_PROVIDER_CALLS_INVALID")
            try:
                stages = _BASELINE_STAGES[baseline]
            except KeyError as error:
                raise MainEvidenceValidationError("MAIN_UNIT_PROVIDER_CALLS_INVALID") from error
            repetitions = 1 if kind == "CLEAN_PREFIX" else (100 if baseline == "reflexion_style" else 50)
            return Counter({stage: repetitions for stage in stages})
        case unreachable:
            assert_never(unreachable)


__all__ = [
    "DispatchEvidenceInput",
    "MainEvidenceValidationError",
    "MainReconciliationEvidence",
    "MainRuntimeEvidence",
    "MainUnitEvidence",
    "MemoryUnitEvidence",
    "NoMemUnitEvidence",
    "PrefixCheckpointState",
    "PrefixUnitEvidence",
    "UnitEvidence",
    "load_durable_reconciliation_evidence",
    "load_durable_unit_evidence",
    "validate_dispatch_evidence",
]
