from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from .phase13_cost_policy_models import Sha256
from .phase13_main_production import ProductionObject
from .phase13_main_request_recovery import RequestIdentityReceiptV3
from .phase13_v3_authority_models import FrozenModel, IdentityComponent
from .phase13_v3_cost_models import canonical_bytes, digest
from .phase13_v3_request import RequestKeyV3
from .phase13_v3_terminal_ledger import TerminalLedgerV3
from .phase13_v3_terminal_models import TerminalEvidenceError

InnerCode = Literal[
    "ORDINARY_SEQUENCE_CONTINUITY_MISMATCH", "SEQUENCE_EVIDENCE_MISMATCH",
    "FIXTURE_EXACTLY_ONE_REGISTERED_ROOT_REQUIRED", "EXACT_LINEAGE_REQUIRED",
    "FABRICATED_LINEAGE", "LINEAGE_CYCLE", "PROPAGATION_REQUIRES_EXPOSURE",
    "FINAL_CONTEXT_EVIDENCE_REQUIRED", "NONCONTAM_TARGET_EVIDENCE",
    "FILTER_NOT_CURRENT_MAIN_ARM", "EXCLUDED_CURRENT_MAIN_CELL",
    "TRIAL_CELL_IDENTITY_MISMATCH", "TRIAL_EVENT_IDENTITY_MISMATCH",
    "CONTEXT_EVENT_IDENTITY_MISMATCH", "RETRIEVAL_EVENT_IDENTITY_MISMATCH",
    "WRITER_EVENT_IDENTITY_MISMATCH", "EVENT_ORDER_MISMATCH", "RUN_EVENT_IDENTITY_MISMATCH",
    "ANSWER_CALL_IDENTITY_MISMATCH", "TARGET_SET_IDENTITY_MISMATCH",
    "MUTATION_CONTEXT_REQUIRED", "WRITER_EVENT_REQUIRED", "MEMORY_MUTATION_SET_MISMATCH",
    "UNKNOWN_TASK_FAILURE_CLASS", "CORRECT_RESPONSE_HAS_FAILURE_CLASS",
    "OBSERVABILITY_REGISTRATION_PACKET_STALE", "UNREGISTERED_RECONSTRUCTION_CAUSE",
    "PRODUCTION_CONCRETE_SEED_MISMATCH", "PRODUCTION_PROVIDER_MODEL_REQUIRED",
    "PRODUCTION_CHECKPOINT_REQUIRED", "PRODUCTION_NOMEM_CHECKPOINT_FORBIDDEN",
    "PRODUCTION_SCIENTIFIC_RESULT_REQUIRED", "PRODUCTION_TRAJECTORY_SEED_MISMATCH",
    "PRODUCTION_SAMPLE_ORDER_MISMATCH", "PRODUCTION_RESULT_IDENTITY_MISMATCH",
    "PRODUCTION_CHECKPOINT_INDEX_REQUIRED", "PRODUCTION_TERMINAL_CALL_REQUIRED",
    "PRODUCTION_RETRIEVAL_EVENT_INVALID", "PRODUCTION_CONTEXT_EVENT_INVALID",
    "PRODUCTION_WRITER_ORIGIN_MISSING", "PRODUCTION_LINEAGE_PARENT_MISSING",
    "PRODUCTION_LINEAGE_CYCLE", "PRODUCTION_MUTATION_CONTEXT_REQUIRED",
    "MAIN_UNIT_EVIDENCE_INVALID", "MAIN_UNIT_EVIDENCE_JOIN_INVALID",
    "MAIN_UNIT_PROVIDER_CALLS_INVALID", "MAIN_UNIT_REALIZED_COST_MISMATCH",
    "MAIN_UNIT_RUNTIME_IDENTITY_INVALID", "MAIN_AUTHORIZATION_BINDING_MISMATCH",
    "MAIN_TERMINAL_EVIDENCE_CONFLICT", "MAIN_TERMINAL_COST_UNKNOWN",
    "MAIN_PARENT_FINALIZATION_FAILED",
]
_INNER = TypeAdapter(InnerCode)


class CompletionReferenceV3(FrozenModel):
    key: RequestKeyV3
    event_hash: Sha256


class RunJournalBaseV3(FrozenModel):
    schema_version: Literal["phase13_main_run_journal_v3"] = "phase13_main_run_journal_v3"
    run_id: IdentityComponent
    binding_hash: Sha256
    parent_id: Sha256
    task: str
    baseline: str | None
    arm: str
    order: Annotated[int, Field(ge=0)]
    previous_hash: Sha256


class RunPauseV3(RunJournalBaseV3):
    kind: Literal["PAUSED_BEFORE_DISPATCH", "READY"]
    stage: Literal["pre_dispatch"] = "pre_dispatch"
    outer_code: Literal["MAIN_TRANCHE_CEILING_EXCEEDED", "MAIN_AUTHORIZATION_CEILING_EXCEEDED",
                        "MAIN_TERMINAL_COST_UNKNOWN", "MAIN_DISPATCH_ADMITTED"]


class ReconstructionFailureV3(RunJournalBaseV3):
    kind: Literal["RECONSTRUCTION_FAILED"] = "RECONSTRUCTION_FAILED"
    stage: Literal["archive_reconstruction"] = "archive_reconstruction"
    outer_code: Literal["PRODUCTION_RECONSTRUCTION_FAILED"] = "PRODUCTION_RECONSTRUCTION_FAILED"
    inner_code: InnerCode
    provider_completed: Literal[True] = True
    completions: Annotated[tuple[CompletionReferenceV3, ...], Field(min_length=1)]


JournalEvent = Annotated[RunPauseV3 | ReconstructionFailureV3, Field(discriminator="kind")]
_EVENT = TypeAdapter(JournalEvent)


@dataclass(frozen=True, slots=True)
class RunJournalV3:
    ledger: TerminalLedgerV3
    units: tuple[ProductionObject, ...]

    def rows(self) -> tuple[RunPauseV3 | ReconstructionFailureV3, ...]:
        with self.ledger.connection() as connection:
            rows = connection.execute("SELECT sequence, raw, sha256 FROM run_journal ORDER BY sequence").fetchall()
        previous = digest(self.ledger.binding)
        events: list[RunPauseV3 | ReconstructionFailureV3] = []
        units = {unit.unit_id: unit for unit in self.units}
        for sequence, raw, checksum in rows:
            event = _EVENT.validate_json(raw)
            unit = units.get(event.parent_id)
            if (sequence != len(events) + 1 or raw != canonical_bytes(event) or checksum != digest(event)
                or event.previous_hash != previous or event.binding_hash != digest(self.ledger.binding)
                or event.run_id != self.ledger.binding.identity.run_id or unit is None
                or (event.task, event.baseline, event.arm, event.order) !=
                   (unit.task, unit.memory_baseline, unit.arm, unit.sequence)):
                raise TerminalEvidenceError()
            if isinstance(event, ReconstructionFailureV3):
                with self.ledger.connection() as connection:
                    for reference in event.completions:
                        matched = connection.execute("SELECT raw FROM events WHERE event_hash=?",
                                                     (reference.event_hash,)).fetchone()
                        if (reference.key.parent_id != event.parent_id or matched is None):
                            raise TerminalEvidenceError()
                        from .phase13_v3_terminal_models import parse_event
                        completion = parse_event(matched[0])
                        if completion.kind != "COMPLETED" or completion.unit_id != reference.key.dispatch_id:
                            raise TerminalEvidenceError()
            events.append(event)
            previous = checksum
        return tuple(events)

    def append(self, event: RunPauseV3 | ReconstructionFailureV3) -> None:
        rows = self.rows()
        previous = digest(rows[-1]) if rows else digest(self.ledger.binding)
        bound = event.model_copy(update={"previous_hash": previous})
        with self.ledger.connection() as connection:
            connection.execute("INSERT INTO run_journal(raw, sha256) VALUES (?, ?)",
                               (canonical_bytes(bound), digest(bound)))

    def pause(self, unit: ProductionObject, code: Literal["MAIN_TRANCHE_CEILING_EXCEEDED",
              "MAIN_AUTHORIZATION_CEILING_EXCEEDED", "MAIN_TERMINAL_COST_UNKNOWN", "MAIN_DISPATCH_ADMITTED"]) -> None:
        self.append(RunPauseV3(run_id=self.ledger.binding.identity.run_id, binding_hash=digest(self.ledger.binding),
            parent_id=unit.unit_id, task=unit.task, baseline=unit.memory_baseline, arm=unit.arm, order=unit.sequence,
            previous_hash=digest(self.ledger.binding), outer_code=code,
            kind="READY" if code == "MAIN_DISPATCH_ADMITTED" else "PAUSED_BEFORE_DISPATCH"))

    def reconstruction(self, unit: ProductionObject, cause: BaseException | None) -> None:
        references: list[CompletionReferenceV3] = []
        for request_id, state in self.ledger.states().items():
            if state.kind != "COMPLETED":
                continue
            receipt = RequestIdentityReceiptV3.model_validate_json(self.ledger.read_record(f"{request_id}.identity.json"))
            if receipt.key.parent_id == unit.unit_id:
                references.append(CompletionReferenceV3(key=receipt.key, event_hash=state.event_hash))
        if not references:
            return
        inner: InnerCode = "UNREGISTERED_RECONSTRUCTION_CAUSE"
        code = getattr(cause, "code", None)
        if isinstance(code, str) and code in _INNER.json_schema()["enum"]:
            inner = _INNER.validate_python(code)
        self.append(ReconstructionFailureV3(run_id=self.ledger.binding.identity.run_id,
            binding_hash=digest(self.ledger.binding), parent_id=unit.unit_id, task=unit.task,
            baseline=unit.memory_baseline, arm=unit.arm, order=unit.sequence,
            previous_hash=digest(self.ledger.binding), inner_code=inner, completions=tuple(references)))
