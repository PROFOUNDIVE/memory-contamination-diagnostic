"""Read-only joins of canonical reservations to committed native call evidence.

This module has no runner, recovery, provider or publication dependency.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from memcontam.memory.checkpoint_v3 import NATIVE_ENTRY_V1, NativeEntry, NativeState, Phase12Checkpoint, serialize_checkpoint, append_native_entry
from memcontam.experiment.phase12.game24_runner import RuntimeIdentities
from memcontam.tasks.base import TaskInstance

from .phase13_main_live_evidence import MainMethodCall
from .phase13_main_production import ProductionObject
from .phase13_main_request_recovery import RequestIdentityReceiptV3
from .phase13_production_observability import ProductionTrialRecord
from .phase13_v3_cost_actual import reconcile_actual
from .phase13_v3_cost_models import CostError, CostUnit, digest
from .phase13_v3_count import count_costs_krw
from .phase13_v3_request import PackageBindingV3, RequestKeyV3
from .phase13_v3_request import ParentTrajectoryV3
from .phase13_v3_entrypoint import SelectedExecutionV3
from .phase13_v3_retry import schedule_unit
from .phase13_v3_terminal_ledger import TerminalLedgerV3
from .phase13_v3_terminal_models import CompletedV3, ProviderFailureV3, RetryableAttemptFailureV3, TerminalEvidenceError, parse_event

if TYPE_CHECKING:
    from .phase13_main_v3_runner import DurableParentRecordV3


@dataclass(frozen=True, slots=True)
class _RestoreContext:
    baseline_configs: dict[str, dict[str, int]]
    task: TaskInstance
    identities: RuntimeIdentities


def scheduled_prefix_keys(unit: ProductionObject, cost_unit: CostUnit,
                          records: tuple[ProductionTrialRecord, ...]) -> tuple[RequestKeyV3, ...] | None:
    schedule = tuple(key for key, _ in schedule_unit(unit, cost_unit))
    if cost_unit.unit_id != unit.unit_id or not schedule or len(schedule) % 50 or len(records) > 50:
        return None
    stride = len(schedule) // 50
    selected: list[RequestKeyV3] = []
    for index, row in enumerate(records):
        slots = schedule[index * stride:(index + 1) * stride]
        stages = tuple(call.stage for call in row.method_calls)
        if not stages or len(stages) > stride or stages != tuple(key.stage for key in slots[:len(stages)]):
            return None
        if unit.memory_baseline == "reflexion_style" and row.evidence.trial.execution_status != "failed":
            from .phase13_main_live_evidence import _reflexion_stage_sequences_valid
            if not _reflexion_stage_sequences_valid(row.method_calls, 1, unit.kind,
                                                    (row.evidence.trial_id,)):
                return None
        selected.extend(slots[:len(stages)])
    return tuple(selected)


def validate_parent_calls(ledger: TerminalLedgerV3, binding: PackageBindingV3,
                          parent_id: str, calls: tuple[MainMethodCall, ...], *,
                          expected_parents: tuple[ParentTrajectoryV3, ...],
                          expected_keys: tuple[RequestKeyV3, ...] | None = None,
                          terminal_call_id: str | None = None) -> None:
    seen: set[str] = set()
    if (binding.identity, binding.package_sha256, binding.authorization_sha256) != (
        ledger.binding.identity, ledger.binding.package_sha256, ledger.binding.authorization_sha256
    ) or not any(parent.parent_id == parent_id for parent in expected_parents):
        raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    if expected_keys is not None and len(expected_keys) != len(calls):
        raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    events = tuple(parse_event(raw) for raw in ledger.rows())
    for index, call in enumerate(calls):
        receipt = RequestIdentityReceiptV3.model_validate_json(
            ledger.read_record(f"{call.dispatch_id}.identity.json"))
        key = receipt.key
        state = ledger.state(call.dispatch_id)
        compiled = json.loads(ledger.read_record(f"{call.dispatch_id}.compiled.json"))
        terminal = state.kind == "ATTEMPTED_PROVIDER_FAILURE" and call.call_id == terminal_call_id
        matches = tuple(event for event in events if event.unit_id == call.dispatch_id
                        and digest(event) == (state.event_hash if terminal else state.completion_hash))
        if len(matches) != 1 or state.compiled is None or state.attempted_cost is None:
            raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
        event = matches[0]
        response = call.raw_response
        response_hash = None if response is None else hashlib.sha256(response.encode()).hexdigest()
        valid_event = (isinstance(event, CompletedV3) and not terminal
                       and event.result_hash == response_hash)
        if terminal and isinstance(event, ProviderFailureV3):
            observation = ledger.read_record(f"{call.dispatch_id}.observation.json")
            valid_event = (event.failure_code == "MAIN_SEMANTIC_RESULT_UNAVAILABLE"
                and hashlib.sha256(observation).hexdigest() == event.observation_hash
                and observation == json.dumps({"response_sha256": response_hash,
                    "semantic_success": False, "failure_code": event.failure_code}, sort_keys=True).encode())
        actual = reconcile_actual(state.attempted_cost)
        input_bytes = json.dumps(call.messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        if (call.dispatch_id in seen or call.dispatch_id not in ledger.binding.unit_ids
            or receipt.binding != binding or receipt.parents != expected_parents
            or not any(parent.parent_id == parent_id for parent in receipt.parents)
            or key.parent_id != parent_id or key.dispatch_id != call.dispatch_id or key.stage != call.stage
            or (expected_keys is not None and key != expected_keys[index])
            or (state.kind != "COMPLETED" and not terminal) or not valid_event or response is None
            or event.compiled != state.compiled
            or compiled.get("binding") != binding.model_dump(mode="json")
            or compiled.get("key") != key.model_dump(mode="json")
            or compiled.get("parents") != [parent.model_dump(mode="json") for parent in receipt.parents]
            or compiled.get("compiled") != state.compiled.model_dump(mode="json")
            or bytes.fromhex(compiled["input_hex"]) != input_bytes
            or hashlib.sha256(input_bytes).hexdigest() != state.compiled.immutable_input_hash
            or hashlib.sha256(bytes.fromhex(compiled["request_hex"])).hexdigest() != state.compiled.compiled_request_hash
            or hashlib.sha256(bytes.fromhex(compiled["native_state_hex"])).hexdigest() != state.compiled.native_state_hash
            or call.provider_cost_usd != float(actual.selected_usd)
            or call.provider_cost_source != actual.source
            or call.authoritative_provider_cost_usd != (None if state.attempted_cost.monetary_cost is None
                                                       else float(state.attempted_cost.monetary_cost))
            or call.derived_cost_usd != (None if actual.derived_usd is None else float(actual.derived_usd))):
            raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
        seen.add(call.dispatch_id)


def validate_partial_costs(ledger: TerminalLedgerV3, calls: tuple[MainMethodCall, ...],
                           interrupted: tuple[RequestKeyV3, ...], terminal_key: RequestKeyV3,
                           terminal_event_hash: str, observation_cost_krw: int,
                           whole_unit_cost_krw: int | None, *,
                           expected_parents: tuple[ParentTrajectoryV3, ...]) -> None:
    events = tuple(parse_event(raw) for raw in ledger.rows())
    terminal = next((event for event in events if isinstance(event, ProviderFailureV3)
                     and digest(event) == terminal_event_hash), None)
    if (terminal is None or not interrupted or interrupted[-1] != terminal_key
        or terminal.unit_id != terminal_key.dispatch_id
        or terminal.compiled != ledger.state(terminal_key.dispatch_id).compiled
        or hashlib.sha256(ledger.read_record(f"{terminal_key.dispatch_id}.observation.json")).hexdigest()
           != terminal.observation_hash):
        raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    observed_ids = tuple(call.dispatch_id for call in calls)
    interrupted_ids = tuple(key.dispatch_id for key in interrupted)
    terminal_receipt = RequestIdentityReceiptV3.model_validate_json(
        ledger.read_record(f"{terminal_key.dispatch_id}.identity.json"))
    binding = PackageBindingV3(identity=ledger.binding.identity,
        package_sha256=ledger.binding.package_sha256,
        authorization_sha256=ledger.binding.authorization_sha256)
    for key in interrupted:
        receipt = RequestIdentityReceiptV3.model_validate_json(
            ledger.read_record(f"{key.dispatch_id}.identity.json"))
        state = ledger.state(key.dispatch_id)
        compiled = json.loads(ledger.read_record(f"{key.dispatch_id}.compiled.json"))
        if (receipt.key != key or key.parent_id != terminal_key.parent_id
            or receipt.binding != binding or receipt.parents != expected_parents
            or terminal_receipt.parents != expected_parents
            or not any(parent.parent_id == key.parent_id for parent in receipt.parents)
            or state.compiled is None
            or state.kind != ("ATTEMPTED_PROVIDER_FAILURE" if key == terminal_key else "COMPLETED")
            or compiled.get("binding") != binding.model_dump(mode="json")
            or compiled.get("key") != key.model_dump(mode="json")
            or compiled.get("parents") != [parent.model_dump(mode="json") for parent in receipt.parents]
            or compiled.get("compiled") != state.compiled.model_dump(mode="json")
            or hashlib.sha256(bytes.fromhex(compiled["request_hex"])).hexdigest() != state.compiled.compiled_request_hash
            or hashlib.sha256(bytes.fromhex(compiled["input_hex"])).hexdigest() != state.compiled.immutable_input_hash
            or hashlib.sha256(bytes.fromhex(compiled["native_state_hex"])).hexdigest() != state.compiled.native_state_hash):
            raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    observed = count_costs_krw(ledger, observed_ids) + sum(
        reconcile_actual(cost).realized_krw for dispatch_id in observed_ids
        for cost in ledger.state(dispatch_id).attempt_costs)
    total = count_costs_krw(ledger, (*observed_ids, *interrupted_ids))
    known = True
    for dispatch_id in (*observed_ids, *interrupted_ids):
        costs = ((*(event.cost for event in events if isinstance(event, RetryableAttemptFailureV3)
                    and event.unit_id == dispatch_id), terminal.cost)
                 if dispatch_id == terminal_key.dispatch_id else ledger.state(dispatch_id).attempt_costs)
        for cost in costs:
            try:
                total += reconcile_actual(cost).realized_krw
            except CostError as error:
                if error.code != "MAIN_TERMINAL_COST_UNKNOWN":
                    raise
                known = False
    for event in events:
        if isinstance(event, RetryableAttemptFailureV3) and event.unit_id in interrupted_ids:
            if hashlib.sha256(ledger.read_record(
                f"{event.unit_id}.retry-observation-0.json")).hexdigest() != event.observation_hash:
                raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    if observation_cost_krw != observed or whole_unit_cost_krw != (total if known else None):
        raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")


def validated_parent_context(selected: SelectedExecutionV3, ledger: TerminalLedgerV3,
                             unit: ProductionObject, cost_unit: CostUnit
                             ) -> tuple[PackageBindingV3, tuple[ParentTrajectoryV3, ...]]:
    """Derive expected joins from the admitted package, never submitted receipts.

    The selected resources and LiveCosts were qualified by the entrypoint. This
    seam only reads immutable package values and ledger identity; it creates no
    dispatcher and performs no recovery, status, publication or provider work.
    """
    binding = PackageBindingV3(identity=selected.package.identity,
        package_sha256=selected.package_sha256,
        authorization_sha256=selected.authorization_sha256)
    expected_unit = next((row for row in selected.package.production if row.unit_id == unit.unit_id), None)
    if (expected_unit is None
        or replace(unit, projected_cost_krw=expected_unit.projected_cost_krw) != expected_unit
        or unit.projected_cost_krw not in (0, expected_unit.projected_cost_krw)
        or selected.costs.package != selected.package
        or cost_unit not in selected.costs.resources.phase4.base.units
        or cost_unit.unit_id != unit.unit_id
        or (ledger.binding.identity, ledger.binding.package_sha256,
            ledger.binding.authorization_sha256) != (
                binding.identity, binding.package_sha256, binding.authorization_sha256)):
        raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    parents = tuple(ParentTrajectoryV3(parent_id=row.unit_id, kind=row.kind,
        prefix_parent_id=row.prefix_unit_id) for row in selected.package.production)
    return binding, parents


def validate_durable_parent_data(selected: SelectedExecutionV3, ledger: TerminalLedgerV3,
                                 record: DurableParentRecordV3, unit: ProductionObject) -> Phase12Checkpoint | None:
    from .phase13_main_live_evidence import (
        DispatchEvidenceInput, MemoryUnitEvidence, PrefixUnitEvidence, validate_dispatch_evidence,
    )
    from .phase13_main_preloaded_resources import PreloadedMainResources
    from .phase13_main_terminal_partial import _archived_trial_identity_valid
    from .phase13_production_observability import validate_production_archive
    from .phase13_v3_terminal_models import DispatchIntentV3

    cost_unit = next(row for row in selected.costs.resources.phase4.base.units if row.unit_id == unit.unit_id)
    binding, parents = validated_parent_context(selected, ledger, unit, cost_unit)
    calls = tuple(call for call in record.unit_evidence.provider_calls if isinstance(call, MainMethodCall))
    evidence = record.unit_evidence.evidence
    validated, realized = validate_dispatch_evidence(unit, DispatchEvidenceInput(
        evidence.model_dump(mode="json"), calls, record.unit_evidence.realized_cost_krw,
        selected.costs.resources.phase4.policy,
        tuple((call.dispatch_id, ledger.state(call.dispatch_id).attempt_costs) for call in calls),
        count_costs_krw(ledger, tuple(call.dispatch_id for call in calls))))
    if (len(calls) != len(record.unit_evidence.provider_calls) or validated != evidence
        or realized != record.unit_evidence.realized_cost_krw
        or record.identity != binding.identity.model_dump(mode="json")
        or record.package_sha256 != binding.package_sha256
        or record.authorization_sha256 != binding.authorization_sha256
        or (record.unit_evidence.unit_id, record.unit_evidence.sequence, record.unit_evidence.kind,
            record.unit_evidence.seed, record.unit_evidence.task, record.unit_evidence.memory_baseline,
            record.unit_evidence.arm) != (unit.unit_id, unit.sequence, unit.kind, unit.seed,
                                        unit.task, unit.memory_baseline, unit.arm)):
        raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    if unit.kind == "CLEAN_PREFIX":
        if not isinstance(validated, PrefixUnitEvidence):
            raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
        keys = tuple(key for key, _ in schedule_unit(unit, cost_unit))[:len(calls)]
        validate_parent_calls(ledger, binding, unit.unit_id, calls,
                              expected_keys=keys, expected_parents=parents)
        checkpoint = serialize_checkpoint(NativeState.from_mapping(
            json.loads(validated.checkpoint.canonical_state_utf8)),
            checkpoint_index=validated.checkpoint.checkpoint_index)
        if (validated.checkpoint.checkpoint_index != 1
            or validated.checkpoint.baseline != unit.memory_baseline
            or validated.checkpoint.checkpoint_id != checkpoint.identity.checkpoint_id
            or validated.checkpoint.checkpoint_identity_sha256 != checkpoint.identity.sha256
            or validated.checkpoint.canonical_sha256 != checkpoint.canonical_sha256):
            raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
        return checkpoint
    archive = validated.runtime_evidence.production_observability_archive
    if archive is None or unit.registration_packet_sha256 is None:
        raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    loaded = PreloadedMainResources(selected)
    seed_order = loaded.checkpoint_registry.tasks[unit.task].seeds[unit.seed].suffix_sample_ids
    validate_production_archive(archive, loaded.packet, unit.registration_packet_sha256,
                                frozen_tasks=loaded.tasks(unit.task))
    keys = scheduled_prefix_keys(unit, cost_unit, archive.records)
    enriched = {"dispatch_id", "provider_cost_usd", "authoritative_provider_cost_usd",
                "derived_cost_usd", "provider_cost_source", "provider_request_contract",
                "provider_authority_contract"}
    archived_calls = tuple(call for row in archive.records for call in row.method_calls)
    if (keys is None or not keys or not archive.records
        or archive.schema_version != ("phase13_production_observability_archive_v2"
            if validated.runtime_evidence.production_identity.scientific_result
            else "phase13_production_observability_archive_v1")
        or len(archive.records) > len(seed_order)
        or (len(archive.records) != len(seed_order)
            and archive.records[-1].evidence.trial.execution_status != "failed")
        or not _archived_trial_identity_valid(archive.records, unit, seed_order, allow_terminal=True)
        or tuple(row.task_instance.sample_id for row in archive.records if row.task_instance is not None)
           != seed_order[:len(archive.records)]
        or archive.u_t_status != "NOT_REGISTERED_FOR_CURRENT_MAIN"
        or any(row.request != validated.runtime_evidence.request
               or row.execution_template_id != unit.execution_template_id
               or row.run_id != f"main-a-{unit.prefix_unit_id or unit.unit_id}"
               or row.ordered_sample_ids_sha256 != unit.ordered_sample_ids_sha256
               for row in archive.records)
        or tuple(call.model_dump(mode="json", exclude=enriched) for call in calls)
           != tuple(call.model_dump(mode="json", exclude=enriched) for call in archived_calls)):
        raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    terminal = archive.records[-1].terminal_method_call
    validate_parent_calls(ledger, binding, unit.unit_id, calls, expected_parents=parents,
        expected_keys=keys, terminal_call_id=None if terminal is None else terminal.call_id)
    intents = tuple(event.unit_id for raw in ledger.rows()
        if isinstance(event := parse_event(raw), DispatchIntentV3)
        and RequestIdentityReceiptV3.model_validate_json(ledger.read_record(
            f"{event.unit_id}.identity.json")).key.parent_id == unit.unit_id)
    if intents != tuple(key.dispatch_id for key in keys):
        raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    prefix = validated_prefix_checkpoint(selected, ledger, unit)
    if isinstance(validated, MemoryUnitEvidence) and (prefix is None or (
        validated.consumed_checkpoint_id, validated.consumed_checkpoint_identity_sha256,
        validated.consumed_checkpoint_canonical_sha256) != (
            prefix.identity.checkpoint_id, prefix.identity.sha256, prefix.canonical_sha256)):
        raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    validate_branch_join(selected, ledger, unit, archive.records, keys, prefix)
    return None


def validate_branch_join(selected: SelectedExecutionV3, ledger: TerminalLedgerV3,
                         unit: ProductionObject, records: tuple[ProductionTrialRecord, ...],
                         keys: tuple[RequestKeyV3, ...], prefix: Phase12Checkpoint | None) -> None:
    """Authenticate the intervention and first native input without executing it.

    Materialized checkpoint identities and restored runtime representations are
    distinct contracts. Compare the former to archive identities and the latter
    to the first committed native input, including zero-observation interruptions.
    No later retention/eviction is inferred to be checkpoint non-support.
    """
    if prefix is None:
        if unit.memory_baseline is not None:
            raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
        return
    from memcontam.experiment.phase12.branching import BranchSet, build_matched_branches
    from memcontam.evaluation.phase13_observability_models import Phase13TrialEvidence
    from memcontam.experiment.phase12.live_branch import _inject_root
    from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
    from memcontam.memory.admission import AdmissionContext
    from memcontam.memory.cards_v3 import canonical_content_hash
    from .phase13_main_new_mcq_runtime import new_mcq_native_entries, _inject_root as inject_mcq, _baseline_state
    from .phase13_main_live_runtime_support import core_task_name
    from .phase13_main_preloaded_resources import PreloadedMainResources

    loaded = PreloadedMainResources(selected)
    root: NativeEntry | None = None
    triplet_id: str | None = None
    checkpoint = prefix
    new_mcq = unit.task in ("mmlu_pro_engineering", "mmlu_pro_physics")
    if new_mcq:
        task = core_task_name(unit.task)
        registry = loaded.new_mcq_registry
        roots = new_mcq_native_entries(task, prefix.state.baseline, registry)
        if unit.arm != "clean":
            if unit.arm not in ("correct", "irrelevant", "contam"):
                raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
            root = roots[unit.arm]
            checkpoint = append_native_entry(prefix, root)
            triplet_id = f"phase13-new-mcq::{task}::{registry.tasks[task].selected_candidate_id}"
    else:
        triplets = tuple(row for row in loaded.candidate_registry.triplets if row.task == unit.task)
        if len(triplets) != 1:
            raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
        branches = build_matched_branches(prefix, triplets[0], loaded.legacy_dc_rs_renderers,
                                          AdmissionContext())
        if not isinstance(branches, BranchSet):
            raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
        branch = next(row for row in branches.materialized if row.arm == unit.arm)
        checkpoint = branch.checkpoint
        if branch.intervention is not None:
            last = checkpoint.state.entries[-1]
            if not isinstance(last, NativeEntry):
                raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
            root = last
            triplet_id = branch.intervention.candidate_triplet_id
    for row in records:
        evidence = row.evidence
        if not isinstance(evidence, Phase13TrialEvidence) or (
            evidence.trial.prefix_run_id != prefix.identity.checkpoint_id
            or evidence.trial.checkpoint_id != checkpoint.identity.checkpoint_id
            or evidence.trial.candidate_triplet_id_or_none != triplet_id
            or evidence.trial.native_render_id_or_none != (None if root is None else root.render_id)
            or evidence.target_set.target_entry_ids != (
                (root.entry_id,) if unit.arm == "contam" and root is not None else ())
            or evidence.trial.intervention_event_id_or_none != (
                None if root is None else f"{evidence.trial_id}:intervention")):
            raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    if not keys:
        raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    runtime = PHASE13_CORE_BASELINE_REGISTRY[prefix.state.baseline]
    if prefix.state.baseline == "rag_frozen":
        arm = unit.arm
        if arm not in ("clean", "correct", "irrelevant", "contam"):
            raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
        corpus, index = loaded.legacy_bundles(unit.task)
        corpus_row, index_row = corpus.branches[arm], index.branches[arm]
        expected = NativeState("rag_frozen", tuple(NativeEntry(
            entry_id=doc.id, semantic_kind="rag_document", schema_version=NATIVE_ENTRY_V1,
            native_component="corpus", content=doc.text, content_hash=canonical_content_hash(doc.text))
            for doc in corpus_row.documents if doc.id in corpus_row.active_document_ids),
            {"branch": unit.arm, "corpus_id": corpus_row.serialization_id,
             "index_id": index_row.index_serialization_id, "read_only": True})
    else:
        first_sample = loaded.checkpoint_registry.tasks[unit.task].seeds[unit.seed].suffix_sample_ids[0]
        task_row = next(row for row in loaded.tasks(unit.task) if row.sample_id == first_sample)
        run_id = f"main-a-{unit.prefix_unit_id}"
        state = runtime.restore_state(prefix.state, _RestoreContext(
            {"dc_rs": {"serialized_cheatsheet_budget_tokens": 8192}}, task_row,
            RuntimeIdentities(run_id, f"{run_id}:trial:2:clean:{first_sample}", 2)))
        if root is not None:
            if new_mcq:
                inject_mcq(_baseline_state(state), root)
            else:
                _inject_root(state, root)
        from memcontam.baselines.full_history_phase12 import FullHistoryStateV3, verify_complete_fit
        from .phase13_route_capacity import FH_CAPACITY_CONFIG
        if isinstance(state, FullHistoryStateV3) and state.injected_root_id is not None:
            fit = verify_complete_fit(task_row, state.records, FH_CAPACITY_CONFIG,
                                      requested_fh_mode="bounded")
            if state.injected_root_id not in fit.context.post_record_ids:
                raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
            state.injected_root_was_visible = True
        expected = runtime.serialize_state(state)
    compiled = json.loads(ledger.read_record(f"{keys[0].dispatch_id}.compiled.json"))
    actual = NativeState.from_mapping(json.loads(bytes.fromhex(compiled["native_state_hex"])))
    if not isinstance(expected, NativeState) or actual != expected:
        raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    if records:
        first = records[0].evidence
        if not isinstance(first, Phase13TrialEvidence) or first.memory_before_ids != tuple(
            entry.entry_id if isinstance(entry, NativeEntry) else entry for entry in expected.entries):
            raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")


def validated_prefix_checkpoint(selected: SelectedExecutionV3, ledger: TerminalLedgerV3,
                                 unit: ProductionObject) -> Phase12Checkpoint | None:
    from .phase13_main_v3_runner import DurableParentRecordV3

    if unit.prefix_unit_id is None:
        return None
    prefix_unit = next(row for row in selected.package.production if row.unit_id == unit.prefix_unit_id)
    if (prefix_unit.kind != "CLEAN_PREFIX" or prefix_unit.prefix_unit_id is not None
        or (prefix_unit.task, prefix_unit.seed, prefix_unit.memory_baseline) != (
            unit.task, unit.seed, unit.memory_baseline)):
        raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    raw = ledger.read_record(f"{unit.prefix_unit_id}.parent.json")
    record = DurableParentRecordV3.model_validate_json(raw)
    if raw != json.dumps(record.model_dump(mode="json"), sort_keys=True, allow_nan=False).encode():
        raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    return validate_durable_parent_data(selected, ledger, record, prefix_unit)
