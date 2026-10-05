from __future__ import annotations

import fcntl
import hashlib
import json
import sqlite3
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from memcontam.logging.schema import MethodCall
from memcontam.memory.checkpoint_v3 import NativeState, Phase12Checkpoint, serialize_checkpoint
from memcontam.readiness.phase13_v3_retry import AUTHORITY_TO_STAGE

from .phase13_main_live_evidence import (
    DispatchEvidenceInput,
    MainEvidenceValidationError,
    MainMethodCall,
    MainUnitEvidence,
    MemoryUnitEvidence,
    PrefixUnitEvidence,
    current_stage_authority,
    validate_dispatch_evidence,
)
from .phase13_main_live_runtime import ProductionMainRuntime
from .phase13_main_live_runtime_support import MainLiveRuntimeError, MainUnitDispatchOutput
from .phase13_main_preloaded_resources import PreloadedMainResources
from .phase13_main_production import ProductionObject
from .phase13_main_production_backend import OrdinaryRuntimeRequest, _memory_baseline, _ordinary_arm
from .phase13_main_request_client import MainRequestClientV3
from .phase13_main_request_dispatch import (
    CompiledProvider,
    DispatchTechnicalFailureV3,
    ProductionRequestDispatcherV3,
    production_provider,
)
from .phase13_main_request_recovery import require_known_costs
from .phase13_main_terminal_partial import (
    TerminalPartialArchive,
    TerminalPartialDispatch,
    TerminalPartialParent,
    validate_terminal_partial,
    _archived_trial_identity_valid,
)
from .phase13_main_run_journal import ReconstructionFailureV3, RunJournalV3, RunPauseV3
from .phase13_production_observability import (
    ProductionObservabilityError,
    validate_production_archive,
)
from .phase13_v3_cost_actual import reconcile_actual
from .phase13_v3_count import count_costs_krw
from .phase13_v3_cost_models import CostError, ProviderCostEvidence
from .phase13_v3_entrypoint import EntrypointError, SelectedExecutionV3
from .phase13_v3_entrypoint_paths import PrivateLedger, private_ledger
from .phase13_v3_request import PackageBindingV3, ParentTrajectoryV3, RequestKeyV3, Stage
from .phase13_v3_terminal_ledger import TerminalLedgerV3
from .phase13_v3_terminal_models import (
    CompletedV3,
    LedgerBindingV3,
    ProviderFailureV3,
    TerminalEvidenceError,
    parse_event,
)

STAGE_NAMES: dict[str, Stage] = AUTHORITY_TO_STAGE
_STAGE_ADAPTER: TypeAdapter[Stage] = TypeAdapter(Stage)


class DurableParentRecordV3(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["phase13_main_parent_record_v3"]
    identity: JsonValue
    package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    authorization_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    unit_evidence: MainUnitEvidence


@dataclass(frozen=True, slots=True)
class V3RunStatus:
    session_state: str
    completed_count: int
    terminal_technical_missing_count: int
    pending_count: int
    provider_calls_issued: int


@dataclass(frozen=True, slots=True)
class V3MainRun:
    selected: SelectedExecutionV3
    private: PrivateLedger
    ledger: TerminalLedgerV3
    lease: ExitStack
    seed: int

    def close(self) -> None:
        self.lease.close()

    @classmethod
    def open(cls, selected: SelectedExecutionV3, directory: Path, *, create: bool, seed: int) -> V3MainRun:
        with ExitStack() as lease:
            lease.callback(selected.close)
            selected.preflight(selected.repository_root)
            if seed not in {unit.seed for unit in selected.package.production}:
                raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
            private = lease.enter_context(private_ledger(directory, create=create))
            try:
                fcntl.flock(private.directory_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise EntrypointError("MAIN_RUN_ALREADY_OWNED") from None
            unit_ids: list[str] = []
            for unit in selected.costs.resources.phase4.base.units:
                ordinals: dict[Stage, int] = {}
                for group in unit.stages:
                    stage = STAGE_NAMES[group.stage_id]
                    start = ordinals.get(stage, 0)
                    unit_ids.extend(RequestKeyV3(parent_id=unit.unit_id, stage=stage, ordinal=ordinal).dispatch_id
                                    for ordinal in range(start, start + group.calls))
                    ordinals[stage] = start + group.calls
            binding = LedgerBindingV3(schema_version="phase13_main_run_ledger_v3", unit_ids=tuple(unit_ids),
                identity=selected.package.identity,
                package_sha256=selected.package_sha256, authorization_sha256=selected.authorization_sha256)
            ledger = (TerminalLedgerV3.create_guarded(private, binding.model_dump(mode="json")) if create
                      else TerminalLedgerV3.open_guarded(private, binding))
            lease.callback(ledger.close)
            with private.connect() as connection:
                if create:
                    connection.execute("CREATE TABLE run_journal (sequence INTEGER PRIMARY KEY, raw BLOB NOT NULL, sha256 TEXT NOT NULL)")
                    connection.execute("CREATE TABLE parents (unit_id TEXT PRIMARY KEY, raw BLOB, sha256 TEXT)")
                    connection.executemany("INSERT INTO parents VALUES (?, NULL, NULL)",
                        ((unit.unit_id,) for unit in selected.package.production))
                if {row[0] for row in connection.execute("SELECT unit_id FROM parents")} != set(selected.package.final_order.unit_ids):
                    raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
            RunJournalV3(ledger, selected.package.production).rows()
            run = cls(selected, private, ledger, lease.pop_all(), seed)
            run.status()
            return run

    def dispatcher(self, factory: Callable[[PackageBindingV3], CompiledProvider] = production_provider) -> ProductionRequestDispatcherV3:
        def checked_factory(binding: PackageBindingV3) -> CompiledProvider:
            self.selected.preflight(self.selected.repository_root)
            self.private.check()
            return factory(binding)
        return ProductionRequestDispatcherV3(self.ledger,
            PackageBindingV3(identity=self.selected.package.identity,
                package_sha256=self.selected.package_sha256, authorization_sha256=self.selected.authorization_sha256),
            tuple(ParentTrajectoryV3(parent_id=unit.unit_id, kind=unit.kind, prefix_parent_id=unit.prefix_unit_id)
                  for unit in self.selected.package.production), provider_factory=checked_factory,
            retry_entitlements=frozenset(
                row.dispatch_id for row in self.selected.costs.resources.phase4.base.retry_reservations
            ))

    def status(self) -> V3RunStatus:
        failed = self.dispatcher().terminal_parents
        with self.private.connect() as connection:
            parent_rows = tuple(connection.execute("SELECT * FROM parents"))
        receipts = self.private.parent_receipts({f"{unit_id}.parent.json" for unit_id, _, _ in parent_rows})
        published: set[str] = set()
        for unit_id, raw, checksum in parent_rows:
            name = f"{unit_id}.parent.json"
            receipt = receipts.get(name)
            exists = self.private.record_exists(name, receipt=receipt)
            if raw is None:
                if checksum is not None:
                    raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
                if exists:
                    raw = self.private.read_record(name, receipt=receipt)
                    checksum = hashlib.sha256(raw).hexdigest()
                    self._load_parent(unit_id, raw, checksum, receipt=receipt)
                    if receipt is not None:
                        self.private.finish_parent_receipt(name, receipt)
                    with self.private.connect() as connection:
                        connection.execute(
                            "UPDATE parents SET raw=?, sha256=? "
                            "WHERE unit_id=? AND raw IS NULL AND sha256 IS NULL",
                            (raw, checksum, unit_id),
                        )
                    published.add(unit_id)
                continue
            if checksum is None or not exists:
                raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
            self._load_parent(unit_id, raw, checksum, receipt=receipt)
            if receipt is not None:
                self.private.finish_parent_receipt(name, receipt)
            published.add(unit_id)
        attempts = sum(json.loads(raw)["kind"] == "ATTEMPT_STARTED" for raw in self.ledger.rows())
        completed = len(published - failed)
        pending = len(self.selected.package.production) - len(published | failed)
        with self.private.connect() as connection:
            journal_rows = tuple(connection.execute("SELECT raw FROM run_journal ORDER BY sequence"))
        session = "COMPLETED" if pending == 0 else "READY"
        if journal_rows:
            events = RunJournalV3(self.ledger, self.selected.package.production).rows()
            if pending:
                match events[-1]:
                    case ReconstructionFailureV3():
                        session = "RECONSTRUCTION_FAILED"
                    case RunPauseV3():
                        session = events[-1].kind
        return V3RunStatus(session, completed, len(failed), pending, attempts)

    def execute(self, cache: Path, *, max_units: int | None, tranche_ceiling_krw: int,
                provider_factory: Callable[[PackageBindingV3], CompiledProvider] = production_provider) -> V3RunStatus:
        self.selected.preflight(self.selected.repository_root)
        dispatcher = self.dispatcher(provider_factory)
        self.status()
        dispatcher.recover()
        preceding = {unit.unit_id for unit in self.selected.package.production if unit.seed < self.seed}
        if preceding:
            with self.private.connect() as connection:
                completed = {row[0] for row in connection.execute(
                    "SELECT unit_id FROM parents WHERE raw IS NOT NULL"
                )}
            if preceding - completed - dispatcher.terminal_parents:
                raise EntrypointError("MAIN_TRANCHE_ORDER_MISMATCH")
        attempted = 0
        for unit in (unit for unit in self.selected.package.production if unit.seed == self.seed):
            with self.private.connect() as connection:
                complete = connection.execute("SELECT raw FROM parents WHERE unit_id=?", (unit.unit_id,)).fetchone()[0]
            if complete is not None or unit.unit_id in dispatcher.terminal_parents:
                continue
            if max_units is not None and attempted >= max_units:
                break
            journal = RunJournalV3(self.ledger, self.selected.package.production)
            try:
                require_known_costs(self.ledger)
            except CostError as error:
                if error.code != "MAIN_TERMINAL_COST_UNKNOWN":
                    raise
                journal.pause(unit, "MAIN_TERMINAL_COST_UNKNOWN")
                break
            for cost_unit in self.selected.costs.resources.phase4.base.units:
                if cost_unit.unit_id != unit.unit_id:
                    continue
                for group in cost_unit.stages:
                    for ordinal in range(group.calls):
                        key = RequestKeyV3(parent_id=unit.unit_id, stage=STAGE_NAMES[group.stage_id], ordinal=ordinal)
                        if self.ledger.state(key.dispatch_id).kind == "COMPLETED":
                            raise EntrypointError("MAIN_RUN_IN_FLIGHT_RECONCILIATION_REQUIRED")
            realized = self.ledger.realized_cost_krw()
            if realized + unit.projected_cost_krw > tranche_ceiling_krw:
                journal.pause(unit, "MAIN_TRANCHE_CEILING_EXCEEDED")
                break
            if realized + unit.projected_cost_krw > 450000:
                journal.pause(unit, "MAIN_AUTHORIZATION_CEILING_EXCEEDED")
                break
            journal.pause(unit, "MAIN_DISPATCH_ADMITTED")
            client = MainRequestClientV3(dispatcher, unit.unit_id, lambda: self.selected.preflight(self.selected.repository_root))
            runtime = ProductionMainRuntime(self.selected.repository_root, cache, client=client,
                                            resources=PreloadedMainResources(self.selected))
            runtime.preflight((unit,))
            try:
                dispatch: MainUnitDispatchOutput | TerminalPartialDispatch
                if unit.kind == "CLEAN_PREFIX":
                    output = runtime.execute_prefix(unit)
                    dispatch = output.dispatch
                    state = NativeState.from_mapping(
                        json.loads(output.checkpoint.canonical_bytes)
                    )
                    evidence_payload: JsonValue = {
                        "evidence_kind": "CLEAN_PREFIX",
                        "prefix_unit_id": unit.unit_id,
                        "checkpoint": {
                            "schema_version": "phase13_main_prefix_checkpoint_v1",
                            "baseline": state.baseline,
                            "checkpoint_id": output.checkpoint.identity.checkpoint_id,
                            "checkpoint_identity_sha256": output.checkpoint.identity.sha256,
                            "canonical_sha256": output.checkpoint.canonical_sha256,
                            "canonical_state_utf8": output.checkpoint.canonical_bytes.decode(),
                            "checkpoint_index": output.checkpoint.checkpoint_index,
                        },
                        "runtime_evidence": dispatch.evidence,
                    }
                else:
                    checkpoint = self.checkpoint(unit)
                    dispatch = runtime.execute_ordinary(OrdinaryRuntimeRequest(unit,
                        "nomem" if unit.memory_baseline is None else _memory_baseline(unit.memory_baseline),
                        "clean" if unit.memory_baseline is None else _ordinary_arm(unit.arm), unit.arm,
                        unit.prefix_unit_id, checkpoint))
                    if isinstance(dispatch, TerminalPartialDispatch):
                        observed = self._strict_calls(client, dispatch.observed_calls,
                            keys=dispatch.request_keys[:len(dispatch.observed_calls)])
                        observed_cost = sum(reconcile_actual(cost).realized_krw
                            for call in observed for cost in self.ledger.state(call.dispatch_id).attempt_costs)
                        observed_cost += count_costs_krw(self.ledger, tuple(call.dispatch_id for call in observed))
                        all_costs = tuple(cost for key in dispatch.request_keys
                            for cost in self.ledger.state(key.dispatch_id).attempt_costs)
                        try:
                            whole_cost = sum(reconcile_actual(cost).realized_krw for cost in all_costs)
                            whole_cost += count_costs_krw(self.ledger,
                                tuple(key.dispatch_id for key in dispatch.request_keys))
                        except CostError as error:
                            if error.code != "MAIN_TERMINAL_COST_UNKNOWN":
                                raise
                            whole_cost = None
                        if dispatch.failure.evidence_sha256 is None:
                            raise TerminalEvidenceError("MAIN_TERMINAL_EVIDENCE_CONFLICT")
                        partial_record = TerminalPartialParent(
                            schema_version="phase13_main_terminal_partial_parent_v1",
                            identity=self.selected.package.identity.model_dump(mode="json"),
                            package_sha256=self.selected.package_sha256,
                            authorization_sha256=self.selected.authorization_sha256,
                            unit_id=unit.unit_id,
                            archive=TerminalPartialArchive(
                                schema_version="phase13_main_terminal_partial_archive_v1",
                                registration_packet_sha256=dispatch.archive.registration_packet_sha256,
                                records=dispatch.archive.records,
                            ),
                            observed_calls=observed, observation_cost_krw=observed_cost,
                            whole_unit_cost_krw=whole_cost,
                            terminal_sample_id=dispatch.terminal_sample_id,
                            terminal_key=dispatch.request_keys[-1],
                            terminal_event_hash=dispatch.failure.evidence_sha256,
                            interrupted_keys=dispatch.request_keys[len(observed):],
                        )
                        validate_terminal_partial(self, partial_record, unit)
                        raw = json.dumps(partial_record.model_dump(mode="json"), sort_keys=True, allow_nan=False).encode()
                        self.private.publish_record(f"{unit.unit_id}.parent.json", raw)
                        with self.private.connect() as connection:
                            connection.execute("UPDATE parents SET raw=?, sha256=? WHERE unit_id=? AND raw IS NULL",
                                (raw, hashlib.sha256(raw).hexdigest(), unit.unit_id))
                        attempted += 1
                        continue
                    if checkpoint is None:
                        evidence_payload = {
                            "evidence_kind": "NO_MEMORY_SINGLETON",
                            "internal_baseline": "nomem",
                            "internal_arm": "clean",
                            "scientific_arm": "NOT_APPLICABLE",
                            "runtime_evidence": dispatch.evidence,
                        }
                    else:
                        evidence_payload = {
                            "evidence_kind": "MEMORY_BEARING",
                            "prefix_unit_id": unit.prefix_unit_id,
                            "consumed_checkpoint_id": checkpoint.identity.checkpoint_id,
                            "consumed_checkpoint_identity_sha256": checkpoint.identity.sha256,
                            "consumed_checkpoint_canonical_sha256": checkpoint.canonical_sha256,
                            "runtime_evidence": dispatch.evidence,
                        }
                strict_calls = self._strict_calls(client, dispatch.provider_calls)
                evidence, realized_cost = validate_dispatch_evidence(
                    unit,
                    DispatchEvidenceInput(
                        evidence_payload,
                        strict_calls,
                        dispatch.realized_cost_krw,
                        self.selected.costs.resources.phase4.policy,
                        self._attempt_costs(strict_calls),
                        count_costs_krw(self.ledger, tuple(call.dispatch_id for call in strict_calls)),
                    ),
                )
                unit_evidence = MainUnitEvidence(
                    schema_version="phase13_main_unit_evidence_v1",
                    sequence=unit.sequence,
                    unit_id=unit.unit_id,
                    kind=unit.kind,
                    seed=unit.seed,
                    task=unit.task,
                    memory_baseline=unit.memory_baseline,
                    arm=unit.arm,
                    evidence=evidence,
                    provider_calls=strict_calls,
                    realized_cost_krw=realized_cost,
                )
                durable_record = DurableParentRecordV3(
                    schema_version="phase13_main_parent_record_v3",
                    identity=self.selected.package.identity.model_dump(mode="json"),
                    package_sha256=self.selected.package_sha256,
                    authorization_sha256=self.selected.authorization_sha256,
                    unit_evidence=unit_evidence,
                )
                raw = json.dumps(
                    durable_record.model_dump(mode="json"),
                    sort_keys=True,
                    allow_nan=False,
                ).encode()
                self.private.publish_record(
                    f"{unit.unit_id}.parent.json", raw
                )
                with self.private.connect() as connection:
                    connection.execute("UPDATE parents SET raw=?, sha256=? WHERE unit_id=? AND raw IS NULL",
                                       (raw, hashlib.sha256(raw).hexdigest(), unit.unit_id))
            except DispatchTechnicalFailureV3:
                attempted += 1
                continue
            except ProductionObservabilityError as error:
                cause = (
                    error.__cause__
                    if error.code == "PRODUCTION_RECONSTRUCTION_FAILED"
                    else error
                )
                journal.reconstruction(unit, cause)
                if error.code == "PRODUCTION_RECONSTRUCTION_FAILED":
                    raise
                raise ProductionObservabilityError(
                    "PRODUCTION_RECONSTRUCTION_FAILED"
                ) from error
            except (
                EntrypointError,
                MainLiveRuntimeError,
                MainEvidenceValidationError,
                TerminalEvidenceError,
                ValidationError,
                CostError,
                sqlite3.Error,
            ) as error:
                cause = (
                    error
                    if isinstance(getattr(error, "code", None), str)
                    else EntrypointError("MAIN_PARENT_FINALIZATION_FAILED")
                )
                journal.reconstruction(unit, cause)
                raise ProductionObservabilityError(
                    "PRODUCTION_RECONSTRUCTION_FAILED"
                ) from error
            attempted += 1
        return self.status()

    def checkpoint(self, unit: ProductionObject) -> Phase12Checkpoint | None:
        if unit.prefix_unit_id is None:
            return None
        with self.private.connect() as connection:
            raw, expected = connection.execute("SELECT raw, sha256 FROM parents WHERE unit_id=?", (unit.prefix_unit_id,)).fetchone()
        if raw is None:
            raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
        record = self._load_parent(unit.prefix_unit_id, raw, expected)
        if not isinstance(record, DurableParentRecordV3):
            raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
        evidence = record.unit_evidence.evidence
        if not isinstance(evidence, PrefixUnitEvidence):
            raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
        return self._validated_checkpoint(unit, evidence)

    def _validated_checkpoint(
        self, unit: ProductionObject, evidence: PrefixUnitEvidence
    ) -> Phase12Checkpoint:
        try:
            checkpoint = serialize_checkpoint(
                NativeState.from_mapping(
                    json.loads(evidence.checkpoint.canonical_state_utf8)
                ),
                checkpoint_index=evidence.checkpoint.checkpoint_index,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH") from error
        if (
            evidence.checkpoint.checkpoint_index != 1
            or evidence.checkpoint.baseline != unit.memory_baseline
            or evidence.checkpoint.checkpoint_id != checkpoint.identity.checkpoint_id
            or evidence.checkpoint.checkpoint_identity_sha256
            != checkpoint.identity.sha256
            or evidence.checkpoint.canonical_sha256 != checkpoint.canonical_sha256
        ):
            raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
        return checkpoint

    def _load_parent(
        self, unit_id: str, raw: bytes, checksum: str, *, receipt: str | None = None
    ) -> DurableParentRecordV3 | TerminalPartialParent:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and parsed.get("schema_version") == "phase13_main_terminal_partial_parent_v1":
                partial = TerminalPartialParent.model_validate_json(raw)
                unit = next(row for row in self.selected.package.production if row.unit_id == unit_id)
                validate_terminal_partial(self, partial, unit)
                if (hashlib.sha256(raw).hexdigest() != checksum
                    or raw != json.dumps(partial.model_dump(mode="json"), sort_keys=True, allow_nan=False).encode()
                    or self.private.read_record(f"{unit_id}.parent.json", receipt=receipt) != raw):
                    raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
                return partial
            record = DurableParentRecordV3.model_validate_json(raw)
            unit = next(
                candidate
                for candidate in self.selected.package.production
                if candidate.unit_id == unit_id
            )
            calls = tuple(
                call
                for call in record.unit_evidence.provider_calls
                if isinstance(call, MainMethodCall)
            )
            if len(calls) != len(record.unit_evidence.provider_calls):
                raise MainEvidenceValidationError("MAIN_UNIT_EVIDENCE_JOIN_INVALID")
            validated, realized_cost = validate_dispatch_evidence(
                unit,
                DispatchEvidenceInput(
                    record.unit_evidence.evidence.model_dump(mode="json"),
                    calls,
                    record.unit_evidence.realized_cost_krw,
                    self.selected.costs.resources.phase4.policy,
                    self._attempt_costs(calls),
                    count_costs_krw(self.ledger, tuple(call.dispatch_id for call in calls)),
                ),
            )
            archive = validated.runtime_evidence.production_observability_archive
            terminal_call_id = (
                archive.records[-1].terminal_method_call.call_id
                if archive is not None and archive.records and archive.records[-1].terminal_method_call is not None
                else None
            )
            self._validate_parent_calls(unit_id, calls, terminal_call_id=terminal_call_id)
            if self.private.read_record(f"{unit_id}.parent.json", receipt=receipt) != raw:
                raise MainEvidenceValidationError("MAIN_UNIT_EVIDENCE_JOIN_INVALID")
            if isinstance(validated, PrefixUnitEvidence):
                self._validated_checkpoint(unit, validated)
            if unit.kind != "CLEAN_PREFIX":
                if archive is None:
                    raise MainEvidenceValidationError(
                        "MAIN_UNIT_EVIDENCE_JOIN_INVALID"
                    )
                preloaded = PreloadedMainResources(self.selected)
                validate_production_archive(
                    archive,
                    preloaded.packet,
                    validated.runtime_evidence.production_identity.registration_packet_sha256,
                    frozen_tasks=preloaded.tasks(unit.task),
                )
                seed_order = preloaded.checkpoint_registry.tasks[unit.task].seeds[unit.seed].suffix_sample_ids
                archived_calls = tuple(call for row in archive.records for call in row.method_calls)
                enriched_fields = {
                    "dispatch_id", "provider_cost_usd", "authoritative_provider_cost_usd",
                    "derived_cost_usd", "provider_cost_source", "provider_request_contract",
                    "provider_authority_contract",
                }
                if (
                    archive.schema_version != "phase13_production_observability_archive_v2"
                    or archive.u_t_status != "NOT_REGISTERED_FOR_CURRENT_MAIN"
                    or len(archive.records) > len(seed_order)
                    or (len(archive.records) != len(seed_order) and (
                        not archive.records or archive.records[-1].evidence.trial.execution_status != "failed"
                    ))
                    or tuple(row.task_instance.sample_id for row in archive.records if row.task_instance is not None)
                    != seed_order[:len(archive.records)]
                    or not _archived_trial_identity_valid(archive.records, unit, seed_order,
                                                          allow_terminal=True)
                    or any(
                        row.execution_template_id != unit.execution_template_id
                        or row.run_id != f"main-a-{unit.prefix_unit_id or unit.unit_id}"
                        or row.session_id != f"{row.evidence.trial_id}:session"
                        or row.ordered_sample_ids_sha256 != unit.ordered_sample_ids_sha256
                        or row.request != validated.runtime_evidence.request
                        for row in archive.records
                    )
                    or tuple(call.model_dump(mode="json", exclude=enriched_fields) for call in archived_calls)
                    != tuple(call.model_dump(mode="json", exclude=enriched_fields) for call in calls)
                ):
                    raise MainEvidenceValidationError("MAIN_UNIT_EVIDENCE_JOIN_INVALID")
                if isinstance(validated, MemoryUnitEvidence):
                    prefix_id = validated.prefix_unit_id
                    with self.private.connect() as connection:
                        prefix_row = connection.execute(
                            "SELECT raw, sha256 FROM parents WHERE unit_id=?", (prefix_id,)
                        ).fetchone()
                    if prefix_row is None or prefix_row[0] is None:
                        raise MainEvidenceValidationError("MAIN_UNIT_EVIDENCE_JOIN_INVALID")
                    prefix_record = self._load_parent(prefix_id, prefix_row[0], prefix_row[1])
                    if not isinstance(prefix_record, DurableParentRecordV3):
                        raise MainEvidenceValidationError("MAIN_UNIT_EVIDENCE_JOIN_INVALID")
                    prefix = prefix_record.unit_evidence.evidence
                    if not isinstance(prefix, PrefixUnitEvidence) or (
                        validated.consumed_checkpoint_id,
                        validated.consumed_checkpoint_identity_sha256,
                        validated.consumed_checkpoint_canonical_sha256,
                    ) != (
                        prefix.checkpoint.checkpoint_id,
                        prefix.checkpoint.checkpoint_identity_sha256,
                        prefix.checkpoint.canonical_sha256,
                    ):
                        raise MainEvidenceValidationError("MAIN_UNIT_EVIDENCE_JOIN_INVALID")
        except (
            MainEvidenceValidationError,
            ProductionObservabilityError,
            StopIteration,
            TerminalEvidenceError,
            ValidationError,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
        ) as error:
            raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH") from error
        if (
            hashlib.sha256(raw).hexdigest() != checksum
            or raw != json.dumps(record.model_dump(mode="json"), sort_keys=True, allow_nan=False).encode()
            or record.identity != self.selected.package.identity.model_dump(mode="json")
            or record.package_sha256 != self.selected.package_sha256
            or record.authorization_sha256 != self.selected.authorization_sha256
            or record.unit_evidence.unit_id != unit_id
            or record.unit_evidence.sequence != unit.sequence
            or record.unit_evidence.kind != unit.kind
            or record.unit_evidence.seed != unit.seed
            or record.unit_evidence.task != unit.task
            or record.unit_evidence.memory_baseline != unit.memory_baseline
            or record.unit_evidence.arm != unit.arm
            or validated != record.unit_evidence.evidence
            or realized_cost != record.unit_evidence.realized_cost_krw
        ):
            raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
        return record

    def _attempt_costs(self, calls: tuple[MainMethodCall, ...]) -> tuple[tuple[str, tuple[ProviderCostEvidence, ...]], ...]:
        return tuple((call.dispatch_id, self.ledger.state(call.dispatch_id).attempt_costs)
                     for call in calls)

    def _validate_parent_calls(
        self, parent_id: str, calls: tuple[MainMethodCall, ...], *, terminal_call_id: str | None = None
    ) -> None:
        expected_binding = PackageBindingV3(
            identity=self.selected.package.identity,
            package_sha256=self.selected.package_sha256,
            authorization_sha256=self.selected.authorization_sha256,
        ).model_dump(mode="json")
        for call in calls:
            stage = _STAGE_ADAPTER.validate_python(call.stage, strict=True)
            state = self.ledger.state(call.dispatch_id)
            receipt = json.loads(
                self.ledger.read_record(f"{call.dispatch_id}.compiled.json")
            )
            request_key = RequestKeyV3.model_validate(receipt["key"])
            request_bytes = bytes.fromhex(receipt["request_hex"])
            input_bytes = bytes.fromhex(receipt["input_hex"])
            native_state = bytes.fromhex(receipt["native_state_hex"])
            expected_input = json.dumps(
                call.messages,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
            terminal = state.kind == "ATTEMPTED_PROVIDER_FAILURE" and call.call_id == terminal_call_id
            with self.ledger.connection() as connection:
                event_row = connection.execute(
                    "SELECT raw FROM events WHERE event_hash=?",
                    (state.event_hash if terminal else state.completion_hash,),
                ).fetchone()
            completion = None if event_row is None else parse_event(event_row[0])
            terminal_observation_valid = False
            if terminal and isinstance(completion, ProviderFailureV3):
                observation = self.ledger.read_record(f"{call.dispatch_id}.observation.json")
                terminal_observation_valid = (
                    completion.failure_code == "MAIN_SEMANTIC_RESULT_UNAVAILABLE"
                    and hashlib.sha256(observation).hexdigest() == completion.observation_hash
                    and observation == json.dumps({"response": call.raw_response, "semantic_success": False},
                                                  sort_keys=True).encode()
                )
            cost_evidence = state.attempted_cost
            terminal_cost = None if cost_evidence is None else reconcile_actual(cost_evidence)
            if (
                (state.kind != "COMPLETED" and not terminal)
                or state.compiled is None
                or (not isinstance(completion, CompletedV3) and not terminal_observation_valid)
                or terminal_cost is None
                or call.provider_cost_usd != float(terminal_cost.selected_usd)
                or call.provider_cost_source != terminal_cost.source
                or call.authoritative_provider_cost_usd != (
                    None if cost_evidence is None or cost_evidence.monetary_cost is None
                    else float(cost_evidence.monetary_cost)
                )
                or call.derived_cost_usd != (
                    None if terminal_cost.derived_usd is None else float(terminal_cost.derived_usd)
                )
                or call.raw_response is None
                or (not terminal and call.raw_response is not None and isinstance(completion, CompletedV3)
                    and hashlib.sha256(call.raw_response.encode()).hexdigest() != completion.result_hash)
                or receipt.get("binding") != expected_binding
                or request_key.parent_id != parent_id
                or request_key.stage != stage
                or request_key.dispatch_id != call.dispatch_id
                or receipt.get("key") != request_key.model_dump(mode="json")
                or receipt.get("compiled") != state.compiled.model_dump(mode="json")
                or input_bytes != expected_input
                or hashlib.sha256(request_bytes).hexdigest()
                != state.compiled.compiled_request_hash
                or hashlib.sha256(input_bytes).hexdigest()
                != state.compiled.immutable_input_hash
                or hashlib.sha256(native_state).hexdigest()
                != state.compiled.native_state_hash
            ):
                raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")

    def _strict_calls(
        self, client: MainRequestClientV3, calls: tuple[MethodCall, ...],
        *, keys: tuple[RequestKeyV3, ...] | None = None,
    ) -> tuple[MainMethodCall, ...]:
        policy = self.selected.costs.resources.phase4.policy
        enriched: list[MainMethodCall] = []
        for call, key in zip(calls, client.request_keys if keys is None else keys, strict=True):
            stage = _STAGE_ADAPTER.validate_python(call.stage, strict=True)
            if key.parent_id != client.parent_id or key.stage != stage:
                raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
            compiled = client.dispatcher.compiled_request(key)
            cost = self.ledger.state(key.dispatch_id).attempted_cost
            if cost is None:
                raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
            actual = reconcile_actual(cost)
            maximum_output_tokens, authority_contract = current_stage_authority(policy, call.stage)
            enriched.append(MainMethodCall.model_validate(call.model_copy(update={
                "provider_cost_usd": float(actual.selected_usd),
                "authoritative_provider_cost_usd": (
                    None if cost.monetary_cost is None else float(cost.monetary_cost)
                ),
                "derived_cost_usd": (
                    None if actual.derived_usd is None else float(actual.derived_usd)
                ),
                "provider_cost_source": actual.source,
                "provider_request_contract": {
                    "model": "gpt-5.6-luna",
                    "input_sha256": hashlib.sha256(
                        json.dumps(
                            call.messages,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                        ).encode()
                    ).hexdigest(),
                    "temperature": compiled.material.temperature,
                    "top_p": compiled.material.top_p,
                    "reasoning": {
                        "mode": "standard",
                        "effort": "none",
                        "context": "current_turn",
                    },
                    "previous_response_id": None,
                    "service_tier": "default",
                    "store": False,
                    "tools": [],
                    "max_output_tokens": maximum_output_tokens,
                },
                "provider_authority_contract": authority_contract,
            }).model_dump(mode="json") | {"dispatch_id": key.dispatch_id}))
        return tuple(enriched)
