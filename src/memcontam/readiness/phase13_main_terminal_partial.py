from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from memcontam.logging.schema import MethodCall
from memcontam.evaluation.phase13_observability_models import Phase13TrialEvidence
from memcontam.readiness.phase13_cost_policy_models import Sha256
from memcontam.readiness.phase13_main_live_evidence import (
    MainMethodCall,
    _BASELINE_STAGES,
    _completed_call,
    _reflexion_stage_sequences_valid,
)
from memcontam.readiness.phase13_v3_retry import schedule_unit
from memcontam.readiness.phase13_main_request_recovery import RequestIdentityReceiptV3
from memcontam.readiness.phase13_main_request_dispatch import DispatchTechnicalFailureV3
from memcontam.readiness.phase13_production_observability import (
    ProductionObservabilityArchive,
    ProductionTrialRecord,
    validate_production_archive,
)
from memcontam.readiness.phase13_v3_cost_actual import reconcile_actual
from memcontam.readiness.phase13_v3_count import count_costs_krw
from memcontam.readiness.phase13_v3_cost_models import CostError, digest
from memcontam.readiness.phase13_v3_request import PackageBindingV3, RequestKeyV3
from memcontam.readiness.phase13_v3_terminal_models import (
    DispatchIntentV3,
    ProviderFailureV3,
    RetryableAttemptFailureV3,
    TerminalEvidenceError,
    parse_event,
)

if TYPE_CHECKING:
    from memcontam.readiness.phase13_main_production import ProductionObject
    from memcontam.readiness.phase13_main_v3_runner import V3MainRun


@dataclass(frozen=True, slots=True)
class TerminalPartialDispatch:
    archive: ProductionObservabilityArchive
    observed_calls: tuple[MethodCall, ...]
    request_keys: tuple[RequestKeyV3, ...]
    terminal_sample_id: str
    failure: DispatchTechnicalFailureV3


class TerminalPartialArchive(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["phase13_main_terminal_partial_archive_v1"]
    registration_packet_sha256: Sha256
    records: tuple[ProductionTrialRecord, ...]


class TerminalPartialParent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["phase13_main_terminal_partial_parent_v1"]
    identity: JsonValue
    package_sha256: Sha256
    authorization_sha256: Sha256
    unit_id: Sha256
    archive: TerminalPartialArchive
    observed_calls: tuple[MainMethodCall, ...]
    observation_cost_krw: int = Field(ge=0)
    whole_unit_cost_krw: int | None = Field(default=None, ge=0)
    terminal_sample_id: str = Field(min_length=1)
    terminal_key: RequestKeyV3
    terminal_event_hash: Sha256
    interrupted_keys: tuple[RequestKeyV3, ...] = Field(min_length=1)


def validate_terminal_partial(runner: V3MainRun, record: TerminalPartialParent,
                              unit: ProductionObject) -> None:
    ledger = runner.ledger
    from memcontam.readiness.phase13_main_preloaded_resources import PreloadedMainResources

    loaded = PreloadedMainResources(runner.selected)
    resources = runner.selected.package
    seed_order = loaded.checkpoint_registry.tasks[unit.task].seeds[unit.seed].suffix_sample_ids
    archive = ProductionObservabilityArchive(
        schema_version="phase13_production_observability_archive_v2",
        registration_packet_sha256=record.archive.registration_packet_sha256,
        u_t_status="NOT_REGISTERED_FOR_CURRENT_MAIN", records=record.archive.records,
    )
    observed_ids = tuple(call.dispatch_id for call in record.observed_calls)
    interrupted_ids = tuple(key.dispatch_id for key in record.interrupted_keys)
    binding = PackageBindingV3(identity=resources.identity,
        package_sha256=runner.selected.package_sha256,
        authorization_sha256=runner.selected.authorization_sha256)
    parents = runner.dispatcher().parents
    cost_unit = next(row for row in runner.selected.costs.resources.phase4.base.units
                     if row.unit_id == unit.unit_id)
    schedule = tuple(key for key, _ in schedule_unit(unit, cost_unit))
    slots_per_trial = len(schedule) // len(seed_order)
    completed_keys = tuple(
        key for index, row in enumerate(archive.records)
        for key in schedule[index * slots_per_trial:index * slots_per_trial + len(row.method_calls)]
    )
    next_slot = schedule[len(archive.records) * slots_per_trial:(len(archive.records) + 1) * slots_per_trial]
    if (
        record.identity != resources.identity.model_dump(mode="json")
        or record.package_sha256 != runner.selected.package_sha256
        or record.authorization_sha256 != runner.selected.authorization_sha256
        or record.unit_id != unit.unit_id or unit.kind == "CLEAN_PREFIX"
        or record.archive.registration_packet_sha256 != unit.registration_packet_sha256
        or len(archive.records) >= len(seed_order)
        or record.terminal_sample_id != seed_order[len(archive.records)]
        or not record.interrupted_keys or record.terminal_key != record.interrupted_keys[-1]
        or len(set((*observed_ids, *interrupted_ids))) != len(observed_ids) + len(interrupted_ids)
        or any(key.parent_id != unit.unit_id for key in record.interrupted_keys)
        or tuple(call.dispatch_id for call in record.observed_calls) != tuple(
            key.dispatch_id for key in completed_keys)
        or any(tuple(call.stage for call in row.method_calls) != tuple(
            key.stage for key in schedule[index * slots_per_trial:
                                          index * slots_per_trial + len(row.method_calls)])
            for index, row in enumerate(archive.records))
        or record.interrupted_keys != next_slot[:len(record.interrupted_keys)]
        or not _archived_trial_identity_valid(archive.records, unit, seed_order)
        or tuple(row.task_instance.sample_id if row.task_instance is not None else None
                 for row in archive.records) != seed_order[:len(archive.records)]
        or any(row.evidence.trial.execution_status != "completed"
               or row.execution_template_id != unit.execution_template_id
               or row.run_id != f"main-a-{unit.prefix_unit_id or unit.unit_id}"
               or row.ordered_sample_ids_sha256 != unit.ordered_sample_ids_sha256
               or row.session_id != f"{row.evidence.trial_id}:session"
               for row in archive.records)
    ):
        raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    if archive.records:
        validate_production_archive(archive, loaded.packet,
            record.archive.registration_packet_sha256, frozen_tasks=loaded.tasks(unit.task))
    if unit.prefix_unit_id is not None:
        runner.checkpoint(unit)
    enriched_fields = {
        "dispatch_id", "provider_cost_usd", "authoritative_provider_cost_usd",
        "derived_cost_usd", "provider_cost_source", "provider_request_contract",
        "provider_authority_contract",
    }
    archive_calls = tuple(call for row in archive.records for call in row.method_calls)
    archive_requests = tuple(row.request for row in archive.records for _ in row.method_calls)
    if tuple(call.model_dump(mode="json", exclude=enriched_fields) for call in record.observed_calls) != (
        tuple(call.model_dump(mode="json", exclude=enriched_fields) for call in archive_calls)
    ) or len(record.observed_calls) != len(archive_requests) or any(
        not _completed_call(call, request, runner.selected.costs.resources.phase4.policy)
        for call, request in zip(record.observed_calls, archive_requests, strict=True)
    ):
        raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    runner._validate_parent_calls(unit.unit_id, record.observed_calls)
    if unit.memory_baseline == "reflexion_style":
        if not _reflexion_stage_sequences_valid(tuple(archive_calls), len(archive.records),
            unit.kind, tuple(row.evidence.trial_id for row in archive.records)) and archive.records:
            raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    else:
        expected = (("no_memory_generate",) if unit.memory_baseline is None
                    else _BASELINE_STAGES[unit.memory_baseline])
        if any(tuple(call.stage for call in row.method_calls) != expected
               for row in archive.records):
            raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    actual_keys = tuple((event.unit_id, RequestIdentityReceiptV3.model_validate_json(
        ledger.read_record(f"{event.unit_id}.identity.json")
    )) for raw in ledger.rows() if isinstance(event := parse_event(raw), DispatchIntentV3)
        and event.unit_id in ledger.binding.unit_ids)
    if any(dispatch_id != row.key.dispatch_id for dispatch_id, row in actual_keys):
        raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    unit_receipts = tuple(row for _, row in actual_keys if row.key.parent_id == unit.unit_id)
    if (tuple(row.key.dispatch_id for row in unit_receipts) != (*observed_ids, *interrupted_ids)
        or any(row.binding != binding or row.parents != parents for row in unit_receipts)):
        raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    terminal_events = tuple(parse_event(raw) for raw in ledger.rows())
    terminal_event = next((event for event in terminal_events
                           if isinstance(event, ProviderFailureV3)
                           and digest(event) == record.terminal_event_hash), None)
    if not isinstance(terminal_event, ProviderFailureV3):
        raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    total = count_costs_krw(ledger, tuple(row.key.dispatch_id for row in unit_receipts))
    known = True
    for receipt in unit_receipts:
        key = receipt.key
        state = ledger.state(key.dispatch_id)
        if key.dispatch_id in interrupted_ids:
            compiled = json.loads(ledger.read_record(f"{key.dispatch_id}.compiled.json"))
            if (state.compiled is None or compiled.get("binding") != binding.model_dump(mode="json")
                or compiled.get("key") != key.model_dump(mode="json")
                or compiled.get("parents") != [row.model_dump(mode="json") for row in parents]
                or compiled.get("compiled") != state.compiled.model_dump(mode="json")
                or hashlib.sha256(bytes.fromhex(compiled["request_hex"])).hexdigest()
                   != state.compiled.compiled_request_hash
                or hashlib.sha256(bytes.fromhex(compiled["input_hex"])).hexdigest()
                   != state.compiled.immutable_input_hash
                or hashlib.sha256(bytes.fromhex(compiled["native_state_hex"])).hexdigest()
                   != state.compiled.native_state_hash
                or state.kind != ("ATTEMPTED_PROVIDER_FAILURE" if key == record.terminal_key else "COMPLETED")):
                raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
        costs = ((*(event.cost for event in terminal_events
                    if isinstance(event, RetryableAttemptFailureV3) and event.unit_id == key.dispatch_id),
                  terminal_event.cost) if key == record.terminal_key else state.attempt_costs)
        for cost in costs:
            try:
                total += reconcile_actual(cost).realized_krw
            except CostError as error:
                if error.code != "MAIN_TERMINAL_COST_UNKNOWN":
                    raise
                known = False
    observed = sum(reconcile_actual(cost).realized_krw
                   for call in record.observed_calls
                   for cost in ledger.state(call.dispatch_id).attempt_costs)
    observed += count_costs_krw(ledger, observed_ids)
    if (
        terminal_event.unit_id != record.terminal_key.dispatch_id
        or terminal_event.compiled != ledger.state(record.terminal_key.dispatch_id).compiled
        or hashlib.sha256(ledger.read_record(
            f"{record.terminal_key.dispatch_id}.observation.json"
        )).hexdigest() != terminal_event.observation_hash
        or record.observation_cost_krw != observed
        or record.whole_unit_cost_krw != (total if known else None)
    ):
        raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    for event in terminal_events:
        if isinstance(event, RetryableAttemptFailureV3) and event.unit_id in interrupted_ids:
            if hashlib.sha256(ledger.read_record(
                f"{event.unit_id}.retry-observation-0.json"
            )).hexdigest() != event.observation_hash:
                raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")


def _archived_trial_identity_valid(
    records: tuple[ProductionTrialRecord, ...], unit: ProductionObject, seed_order: tuple[str, ...],
    *, allow_terminal: bool = False,
) -> bool:
    baseline = "nomem" if unit.memory_baseline is None else unit.memory_baseline
    run_id = f"main-a-{unit.prefix_unit_id or unit.unit_id}"
    arm = "clean" if unit.memory_baseline is None else unit.arm
    for index, row in enumerate(records, start=1):
        evidence = row.evidence
        expected_id = f"{run_id}{'' if arm == 'clean' else ':' + arm}:trial:{index}:{seed_order[index - 1]}"
        order = index if unit.memory_baseline is None else index + 1
        if (evidence.task != unit.task or evidence.baseline != baseline
            or evidence.trajectory_seed != unit.seed or evidence.concrete_seed_id != str(unit.seed)
            or evidence.analysis_window_id != "core_prefix_50"
            or evidence.order_key != order or evidence.trial.absolute_trial_index != order
            or evidence.trial.event_time != index - 1 or evidence.trial_id != expected_id
            or evidence.trial.analysis_inclusion != (
                "excluded" if allow_terminal and evidence.trial.execution_status == "failed" else "included")
            or evidence.trial.inclusion_reason != (
                "terminal_technical_missingness" if allow_terminal and evidence.trial.execution_status == "failed"
                else "production_runtime_join")
            or row.session_id != f"{expected_id}:session"):
            return False
        if unit.memory_baseline is not None:
            if not isinstance(evidence, Phase13TrialEvidence):
                return False
            if (evidence.trial.branch_id != arm or evidence.trial.execution_key.arm != arm
                or evidence.trial.checkpoint_index != 1):
                return False
    return True
