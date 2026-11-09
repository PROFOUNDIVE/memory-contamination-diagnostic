from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from inspect import getfile
from pathlib import Path
from typing import Literal

import httpx

from .phase13_diagnostic_exchange import DiagnosticWireReply, rates
from .phase13_diagnostic_runtime_contract import DiagnosticExecutionContext
from .phase13_live_diagnostic_models import DiagnosticRequestIntent, LiveDiagnosticError
from .phase13_main_runner_store import connect
from .phase13_shared_budget_models import BudgetOperation
from .phase13_shared_provider_budget import SharedProviderBudget


@dataclass(frozen=True, slots=True)
class DiagnosticLaunch:
    """Trusted launcher callback must authenticate eligibility, source/resources and exact-ID approval."""
    child_execution_id: str
    api_key: str = field(repr=False)
    verify_admission: Callable[[DiagnosticExecutionContext, str, str], None] = field(repr=False)


class DiagnosticHttpTransport:
    provenance: Literal["PROVIDER"] = "PROVIDER"

    def __init__(self, context: DiagnosticExecutionContext, launch: DiagnosticLaunch) -> None:
        self.context = context
        self.client: httpx.Client | None = None
        self.launch = launch
        self._sent_attempts: set[str] = set()

    def open(self) -> None:
        self.verify_admission()
        if self.client is not None:
            raise LiveDiagnosticError("DIAGNOSTIC_TRANSPORT_ALREADY_OPEN")
        self.client = httpx.Client(timeout=self.context.contract.timeout_seconds, follow_redirects=False, trust_env=False,
            transport=httpx.HTTPTransport(retries=0), headers={"Authorization": "Bearer " + self.launch.api_key})

    def verify_admission(self) -> None:
        _verify_launch(self.context, self.launch)

    def send(self, operation: BudgetOperation, raw: bytes, before_egress: Callable[[], None]) -> DiagnosticWireReply:
        self.verify_admission()
        if self.client is None:
            raise LiveDiagnosticError("DIAGNOSTIC_TRANSPORT_NOT_OPEN")
        if operation.endpoint not in {self.context.contract.endpoint + "/responses",
                                      self.context.contract.endpoint + "/responses/input_tokens"}:
            raise LiveDiagnosticError("DIAGNOSTIC_ENDPOINT_MISMATCH")
        observation = next((row for row in self.context.binding.observations if row.run_id == operation.run_id), None)
        if (observation is None or observation.child_execution_id != self.launch.child_execution_id
            or operation.execution_binding_sha256 != self.context.binding.sha256()
            or operation.request_sha256 != hashlib.sha256(raw).hexdigest()
            or operation.attempt_id in self._sent_attempts):
            raise LiveDiagnosticError("DIAGNOSTIC_TRANSPORT_INTENT_MISMATCH")
        ledger_path = Path(self.context.binding.ledger_path)
        if not ledger_path.is_file():
            raise LiveDiagnosticError("DIAGNOSTIC_TRANSPORT_INTENT_MISMATCH")
        with connect(ledger_path) as connection:
            row = connection.execute("SELECT reason FROM events WHERE unit_id=? AND kind='REQUEST_INTENT' ORDER BY event_sequence DESC LIMIT 1",
                                     (operation.unit_id,)).fetchone()
            receipted = connection.execute("SELECT 1 FROM events WHERE kind='REQUEST_RECEIPT' AND json_extract(reason,'$.intent.request_id')=?",
                                           (operation.request_id,)).fetchone()
        if row is None or receipted is not None:
            raise LiveDiagnosticError("DIAGNOSTIC_TRANSPORT_INTENT_MISMATCH")
        intent = DiagnosticRequestIntent.model_validate_json(row[0])
        budget = SharedProviderBudget.open(Path(self.context.contract.budget_binding.path))
        with connect(budget.path) as connection:
            row = connection.execute("SELECT operation,state FROM reservations WHERE reservation_id=?", (intent.reservation_id,)).fetchone()
        if (operation.intent(intent.reservation_id) != intent or row is None
            or row != (operation.model_dump_json(), "intent")):
            raise LiveDiagnosticError("DIAGNOSTIC_TRANSPORT_FINANCIAL_INTENT_MISMATCH")
        self._sent_attempts.add(operation.attempt_id)
        before_egress()
        response = self.client.post(operation.endpoint, content=raw, headers={"Content-Type": "application/json"})
        return DiagnosticWireReply(response.status_code, response.content, response.headers.get("x-request-id"))

    def close(self) -> None:
        if self.client is not None:
            self.client.close()


def _verify_launch(context: DiagnosticExecutionContext, launch: DiagnosticLaunch | None) -> None:
    if type(context) is not DiagnosticExecutionContext:
        raise LiveDiagnosticError("DIAGNOSTIC_LIVE_ADMISSION_UNESTABLISHED")
    context.verify()
    if (context.contract.qualification != "LIVE_REQUIRES_ADMISSION" or type(launch) is not DiagnosticLaunch
        or not launch.api_key or launch.child_execution_id not in {
            row.child_execution_id for row in context.binding.observations}):
        raise LiveDiagnosticError("DIAGNOSTIC_LIVE_ADMISSION_UNESTABLISHED")
    from memcontam.experiment.phase13_live_diagnostic_runtime import runtime_source_paths
    from memcontam.experiment.phase13_snapshot_diagnostic import snapshot_source_paths

    from .phase13_main_live_evidence import _BASELINE_STAGES
    required = runtime_source_paths()
    for observation in context.binding.observations:
        stages = [request.stage for request in observation.requests if request.operation == "generation"]
        if tuple(stages) not in (_BASELINE_STAGES["bot_style"], _BASELINE_STAGES["dc_rs"]):
            raise LiveDiagnosticError("DIAGNOSTIC_LIVE_ADMISSION_SCHEDULE_INVALID")
        required.update(snapshot_source_paths("bot_style" if stages[0].startswith("bot_") else "dc_rs"))
    if (not required <= {Path(pin.path) for pin in context.binding.source_files}
        or context.contract.embedding_metadata_sha256 is None):
        raise LiveDiagnosticError("DIAGNOSTIC_LIVE_ADMISSION_RESOURCES_UNESTABLISHED")
    rates(context)
    verifier_source = Path(getfile(launch.verify_admission)).resolve()
    if str(verifier_source) not in {pin.path for pin in context.binding.source_files}:
        raise LiveDiagnosticError("DIAGNOSTIC_LIVE_ADMISSION_VERIFIER_UNPINNED")
    launch.verify_admission(context, launch.child_execution_id, launch.api_key)
    context.verify()
    budget = SharedProviderBudget.open(Path(context.contract.budget_binding.path))
    if not budget.status().paid_eligible:
        raise LiveDiagnosticError("DIAGNOSTIC_LIVE_ADMISSION_BUDGET_HELD")


def open_live_transport(context: DiagnosticExecutionContext, launch: DiagnosticLaunch | None) -> DiagnosticHttpTransport:
    _verify_launch(context, launch)
    assert launch is not None
    transport = DiagnosticHttpTransport(context, launch)
    transport.open()
    return transport
