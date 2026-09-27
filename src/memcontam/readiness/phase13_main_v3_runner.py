from __future__ import annotations

import hashlib
import json
import fcntl
import sqlite3
from collections import Counter
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError
from memcontam.logging.schema import MethodCall
from memcontam.memory.checkpoint_v3 import NativeState, Phase12Checkpoint, serialize_checkpoint

from .phase13_cost_policy import load_cost_policy_bundle
from .phase13_main_live_evidence import (
    DispatchEvidenceInput,
    MainEvidenceValidationError,
    MainUnitEvidence,
    MemoryUnitEvidence,
    PrefixUnitEvidence,
    validate_dispatch_evidence,
)
from .phase13_main_preloaded_resources import PreloadedMainResources
from .phase13_main_live_runtime import ProductionMainRuntime
from .phase13_main_live_runtime_support import MainLiveRuntimeError
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
from .phase13_main_run_journal import ReconstructionFailureV3, RunJournalV3, RunPauseV3
from .phase13_production_observability import (
    ProductionObservabilityError,
    validate_production_archive,
)
from .phase13_v3_cost_models import CostError
from .phase13_v3_cost_actual import reconcile_actual
from .phase13_v3_entrypoint import EntrypointError, SelectedExecutionV3
from .phase13_v3_entrypoint_paths import PrivateLedger, private_ledger
from .phase13_v3_request import PackageBindingV3, ParentTrajectoryV3, RequestKeyV3, Stage
from memcontam.readiness.phase13_v3_retry import AUTHORITY_TO_STAGE, allocate_retry_entitlements
from .phase13_v3_terminal_ledger import TerminalLedgerV3
from .phase13_v3_terminal_models import (
    CompletedV3,
    LedgerBindingV3,
    TerminalEvidenceError,
    parse_event,
)

STAGE_NAMES: dict[str, Stage] = AUTHORITY_TO_STAGE
_STAGE_ADAPTER: TypeAdapter[Stage] = TypeAdapter(Stage)
_ROOT = Path(__file__).resolve().parents[3]


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
            retry_entitlements=allocate_retry_entitlements(
                self.selected.package.production, self.selected.costs.resources.phase4.base,
            ))

    def status(self) -> V3RunStatus:
        failed = self.dispatcher().terminal_parents
        with self.private.connect() as connection:
            parent_rows = tuple(connection.execute("SELECT * FROM parents"))
        completed = 0
        for unit_id, raw, checksum in parent_rows:
            exists = self.private.record_exists(f"{unit_id}.parent.json")
            if raw is None:
                if checksum is not None:
                    raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
                if exists:
                    raw = self.private.read_record(f"{unit_id}.parent.json")
                    checksum = hashlib.sha256(raw).hexdigest()
                    self._load_parent(unit_id, raw, checksum)
                    with self.private.connect() as connection:
                        connection.execute(
                            "UPDATE parents SET raw=?, sha256=? "
                            "WHERE unit_id=? AND raw IS NULL AND sha256 IS NULL",
                            (raw, checksum, unit_id),
                        )
                    completed += 1
                continue
            if checksum is None or not exists:
                raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
            self._load_parent(unit_id, raw, checksum)
            completed += 1
        attempts = sum(json.loads(raw)["kind"] == "ATTEMPT_STARTED" for raw in self.ledger.rows())
        pending = len(self.selected.package.production) - completed - len(failed)
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
                record = DurableParentRecordV3(
                    schema_version="phase13_main_parent_record_v3",
                    identity=self.selected.package.identity.model_dump(mode="json"),
                    package_sha256=self.selected.package_sha256,
                    authorization_sha256=self.selected.authorization_sha256,
                    unit_evidence=unit_evidence,
                )
                raw = json.dumps(
                    record.model_dump(mode="json"),
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
        self, unit_id: str, raw: bytes, checksum: str
    ) -> DurableParentRecordV3:
        try:
            record = DurableParentRecordV3.model_validate_json(raw)
            unit = next(
                candidate
                for candidate in self.selected.package.production
                if candidate.unit_id == unit_id
            )
            validated, realized_cost = validate_dispatch_evidence(
                unit,
                DispatchEvidenceInput(
                    record.unit_evidence.evidence.model_dump(mode="json"),
                    record.unit_evidence.provider_calls,
                    record.unit_evidence.realized_cost_krw,
                ),
            )
            self._validate_parent_calls(unit_id, record.unit_evidence.provider_calls)
            if self.private.read_record(f"{unit_id}.parent.json") != raw:
                raise MainEvidenceValidationError("MAIN_UNIT_EVIDENCE_JOIN_INVALID")
            if isinstance(validated, PrefixUnitEvidence):
                self._validated_checkpoint(unit, validated)
            archive = validated.runtime_evidence.production_observability_archive
            if unit.kind != "CLEAN_PREFIX":
                if archive is None:
                    raise MainEvidenceValidationError(
                        "MAIN_UNIT_EVIDENCE_JOIN_INVALID"
                    )
                validate_production_archive(
                    archive,
                    PreloadedMainResources(self.selected).packet,
                    validated.runtime_evidence.production_identity.registration_packet_sha256,
                )
                if isinstance(validated, MemoryUnitEvidence):
                    prefix_id = validated.prefix_unit_id
                    with self.private.connect() as connection:
                        prefix_row = connection.execute(
                            "SELECT raw, sha256 FROM parents WHERE unit_id=?", (prefix_id,)
                        ).fetchone()
                    if prefix_row is None or prefix_row[0] is None:
                        raise MainEvidenceValidationError("MAIN_UNIT_EVIDENCE_JOIN_INVALID")
                    prefix = self._load_parent(prefix_id, prefix_row[0], prefix_row[1]).unit_evidence.evidence
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
        ) as error:
            raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH") from error
        if (
            hashlib.sha256(raw).hexdigest() != checksum
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

    def _validate_parent_calls(
        self, parent_id: str, calls: tuple[MethodCall, ...]
    ) -> None:
        ordinals: Counter[str] = Counter()
        expected_binding = PackageBindingV3(
            identity=self.selected.package.identity,
            package_sha256=self.selected.package_sha256,
            authorization_sha256=self.selected.authorization_sha256,
        ).model_dump(mode="json")
        for call in calls:
            stage = _STAGE_ADAPTER.validate_python(call.stage, strict=True)
            request_key = RequestKeyV3(
                parent_id=parent_id,
                stage=stage,
                ordinal=ordinals[stage],
            )
            ordinals[stage] += 1
            state = self.ledger.state(request_key.dispatch_id)
            receipt = json.loads(
                self.ledger.read_record(f"{request_key.dispatch_id}.compiled.json")
            )
            request_bytes = bytes.fromhex(receipt["request_hex"])
            input_bytes = bytes.fromhex(receipt["input_hex"])
            native_state = bytes.fromhex(receipt["native_state_hex"])
            expected_input = json.dumps(
                call.messages,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
            with self.ledger.connection() as connection:
                event_row = connection.execute(
                    "SELECT raw FROM events WHERE event_hash=?", (state.event_hash,)
                ).fetchone()
            completion = None if event_row is None else parse_event(event_row[0])
            cost_evidence = state.attempted_cost
            terminal_cost = None if cost_evidence is None else reconcile_actual(cost_evidence)
            if (
                state.kind != "COMPLETED"
                or state.compiled is None
                or not isinstance(completion, CompletedV3)
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
                or hashlib.sha256(call.raw_response.encode()).hexdigest()
                != completion.result_hash
                or receipt.get("binding") != expected_binding
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
        self, client: MainRequestClientV3, calls: tuple[MethodCall, ...]
    ) -> tuple[MethodCall, ...]:
        bundle = load_cost_policy_bundle(_ROOT)
        rate_card_sha256 = hashlib.sha256(
            json.dumps(
                bundle.proof.rate_card.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        ordinals: Counter[str] = Counter()
        enriched: list[MethodCall] = []
        for call in calls:
            stage = _STAGE_ADAPTER.validate_python(call.stage, strict=True)
            key = RequestKeyV3(
                parent_id=client.parent_id,
                stage=stage,
                ordinal=ordinals[stage],
            )
            ordinals[stage] += 1
            compiled = client.dispatcher.compiled_request(key)
            cost = self.ledger.state(key.dispatch_id).attempted_cost
            if cost is None:
                raise EntrypointError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
            actual = reconcile_actual(cost)
            authority_stage = next(
                row
                for row in bundle.registry.stages
                if row.semantic_stage_id == call.stage
            )
            enriched.append(call.model_copy(update={
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
                    "max_output_tokens": authority_stage.maximum_output_tokens,
                },
                "provider_authority_contract": {
                    "maximum_input_tokens": authority_stage.maximum_input_tokens,
                    "maximum_output_tokens": authority_stage.maximum_output_tokens,
                    "execution_envelope_id": bundle.registry.registry_id,
                    "execution_envelope_sha256": bundle.registry.registry_hash,
                    "failure_contract_id": bundle.retry.contract_id,
                    "failure_contract_sha256": bundle.retry.contract_hash,
                    "terminal_failure_contract_id": bundle.retry.terminal_failure_contract_id,
                    "terminal_failure_contract_sha256": bundle.retry.terminal_failure_contract_sha256,
                    "rate_card_sha256": rate_card_sha256,
                },
            }))
        return tuple(enriched)
