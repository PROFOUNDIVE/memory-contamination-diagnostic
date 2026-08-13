from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import time
from uuid import uuid4

from pydantic import ValidationError

from memcontam.clients.base import LLMClient, LLMResponse
from memcontam.readiness.phase13_analysis_contract import load_analysis_registry
from memcontam.readiness.phase13_execution_contract import load_execution_registry
from memcontam.readiness.phase13_authority import JsonValue
from memcontam.readiness.phase13_provider_models import (
    AccountingReport,
    AccountingTotals,
    ExecutionTemplateIdentity,
    JsonScalar,
    OfflineAccounting,
    ProviderAccountingError,
    ProviderDispatchFailure,
    ProviderDispatchPayload,
    ProviderTotals,
    SettledCall,
    TransportAttempt,
)
from memcontam.readiness.phase13_provider_normalization import (
    normalize_exception,
    normalize_malformed_provider_failure,
    normalize_openai_response,
    sum_attempts,
)
from memcontam.readiness.phase13_provider_payload import provider_config, validate_messages


class OwnedProviderAccounting:
    def __init__(
        self,
        client: LLMClient,
        root: Path,
        intended_template: ExecutionTemplateIdentity,
    ) -> None:
        execution = load_execution_registry(
            root / "data/phase13/authority/execution_registry_v1.json", root
        )
        analysis = load_analysis_registry(
            root / "data/phase13/authority/analysis_registry_v1.json", root
        )
        self._client = client
        self._execution_owner_id = execution.execution_owner_id
        matches = tuple(
            row.template_id
            for row in execution.execution_templates
            if (row.task, row.baseline, row.arm_key)
            == (intended_template.task, intended_template.baseline, intended_template.arm_key)
        )
        if len(matches) != 1:
            raise ProviderAccountingError("UNKNOWN_EXECUTION_TEMPLATE")
        self._intended_template_id = matches[0]
        self._offline_rows = tuple(
            OfflineAccounting(
                operation=row.operation,
                owner_id=row.owner_id,
                provider_calls=0,
                cost_microusd=0,
            )
            for row in analysis.offline_compute.rows
        )
        self._planned_ids: set[str] = set()
        self._calls: list[SettledCall] = []
        self._dispatches = 0
        self._dispatched_payloads: list[ProviderDispatchPayload] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        config: dict,
    ) -> LLMResponse:
        self._validate_dispatch(config)
        dispatch_config = provider_config(config)
        validate_messages(messages)
        semantic_call_id = str(uuid4())
        if semantic_call_id in self._planned_ids:
            raise ProviderAccountingError("DUPLICATE_SEMANTIC_CALL_ID")
        self._planned_ids.add(semantic_call_id)
        self._dispatches += 1
        dispatch_config["execution_owner_id"] = self._execution_owner_id
        dispatch_config["semantic_call_id"] = semantic_call_id
        dispatch_config["execution_template_id"] = self._intended_template_id
        self._dispatched_payloads.append(
            ProviderDispatchPayload(
                messages=tuple(messages),
                model=model,
                config=dispatch_config,
                session_id=str(dispatch_config.get("session_id", "")),
                intervention_id=(
                    str(dispatch_config["intervention_id"])
                    if dispatch_config.get("intervention_id") is not None
                    else None
                ),
            )
        )
        started = time.perf_counter()
        try:
            response = self._client.chat(messages, model, dispatch_config)
        except ProviderDispatchFailure as failure:
            try:
                call = self._settle(
                    semantic_call_id,
                    failure.provider_attempts,
                    failure.provider_totals,
                    failure.provider_error,
                )
            except ProviderAccountingError:
                attempts, totals = normalize_malformed_provider_failure(
                    semantic_call_id,
                    self._execution_owner_id,
                    self._intended_template_id,
                    failure,
                    failure.provider_attempts,
                    started,
                )
                call = self._settle(
                    semantic_call_id,
                    [row.model_dump() for row in attempts],
                    totals.model_dump(),
                    failure.provider_error,
                )
            self._calls.append(call)
            raise ProviderAccountingError("PROVIDER_DISPATCH_FAILED") from failure
        except Exception as error:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK
            self._calls.append(self._settle_exception(semantic_call_id, error, started))
            raise
        try:
            call = self._settle_from_response(semantic_call_id, response)
        except ProviderAccountingError as error:
            self._calls.append(self._settle_exception(semantic_call_id, error, started))
            raise
        self._calls.append(call)
        return response

    def reconcile(self) -> AccountingReport:
        actual_ids = [call.semantic_call_id for call in self._calls]
        if len(set(actual_ids)) != len(actual_ids):
            raise ProviderAccountingError("DUPLICATE_SEMANTIC_CALL_ID")
        if set(actual_ids) != self._planned_ids:
            raise ProviderAccountingError("PLANNED_ACTUAL_SETTLEMENT_MISMATCH")
        totals = self._aggregate(self._calls)
        return AccountingReport(calls=tuple(self._calls), offline=self._offline_rows, totals=totals)

    @property
    def dispatched_payloads(self) -> tuple[ProviderDispatchPayload, ...]:
        return tuple(self._dispatched_payloads)

    @property
    def execution_template_id(self) -> str:
        return self._intended_template_id

    @property
    def execution_owner_id(self) -> str:
        return self._execution_owner_id


    def _validate_dispatch(self, config: dict) -> None:
        claimed_owner = config.get("execution_owner_id")
        claimed_template = config.get("execution_template_id")
        if claimed_owner in {row.owner_id for row in self._offline_rows}:
            raise ProviderAccountingError("OFFLINE_OWNER_FORBIDDEN")
        if claimed_owner is not None and claimed_owner != self._execution_owner_id:
            raise ProviderAccountingError("UNKNOWN_EXECUTION_OWNER")
        if claimed_template is not None and claimed_template != self._intended_template_id:
            raise ProviderAccountingError("EXECUTION_TEMPLATE_MISMATCH")

    def _settle_from_response(
        self,
        call_id: str,
        response: LLMResponse,
    ) -> SettledCall:
        attempts = response.raw.get("provider_attempts")
        totals = response.raw.get("provider_totals")
        if isinstance(attempts, (list, tuple)) and isinstance(totals, Mapping):
            call = self._settle(call_id, attempts, totals, None)
        else:
            call = self._settle_openai_response(call_id, response)
        if (
            response.token_usage.get("prompt_tokens") != call.totals.input_tokens
            or response.token_usage.get("completion_tokens") != call.totals.output_tokens
            or response.latency_ms != call.totals.latency_ms
        ):
            raise ProviderAccountingError("OWNER_TOTAL_MISMATCH")
        return call

    def _settle(
        self,
        call_id: str,
        raw_attempts: tuple[Mapping[str, JsonValue], ...] | list[Mapping[str, JsonValue]],
        raw_totals: Mapping[str, int] | Mapping[str, JsonScalar],
        provider_error: str | None,
    ) -> SettledCall:
        try:
            attempts = tuple(TransportAttempt.model_validate(row) for row in raw_attempts)
            totals = ProviderTotals.model_validate(raw_totals)
        except (ValidationError, TypeError) as error:
            raise ProviderAccountingError("PROVIDER_ACCOUNTING_REQUIRED") from error
        if not attempts:
            raise ProviderAccountingError("PROVIDER_ACCOUNTING_REQUIRED")
        if any(row.semantic_call_id != call_id for row in attempts):
            raise ProviderAccountingError("SEMANTIC_CALL_ID_MISMATCH")
        if any(row.execution_template_id != self._intended_template_id for row in attempts):
            raise ProviderAccountingError("EXECUTION_TEMPLATE_MISMATCH")
        owners = {row.execution_owner_id for row in attempts}
        if owners & {row.owner_id for row in self._offline_rows}:
            raise ProviderAccountingError("OFFLINE_OWNER_FORBIDDEN")
        if owners != {self._execution_owner_id}:
            raise ProviderAccountingError("UNKNOWN_EXECUTION_OWNER")
        if len({row.attempt_id for row in attempts}) != len(attempts):
            raise ProviderAccountingError("DUPLICATE_ATTEMPT_ID")
        if tuple(row.attempt_number for row in attempts) != tuple(range(1, len(attempts) + 1)):
            raise ProviderAccountingError("TRANSPORT_RETRY_SEQUENCE_INVALID")
        expected = sum_attempts(attempts)
        if totals != expected:
            raise ProviderAccountingError("OWNER_TOTAL_MISMATCH")
        return SettledCall(
            semantic_call_id=call_id,
            execution_owner_id=self._execution_owner_id,
            execution_template_id=self._intended_template_id,
            provider_error=provider_error,
            attempts=attempts,
            totals=totals,
        )

    def _aggregate(self, calls: list[SettledCall]) -> AccountingTotals:
        fields = ProviderTotals.model_fields
        sums = {name: sum(getattr(call.totals, name) for call in calls) for name in fields}
        return AccountingTotals(
            semantic_calls=len(calls),
            dispatches=self._dispatches,
            **sums,
        )

    def _settle_openai_response(self, call_id: str, response: LLMResponse) -> SettledCall:
        rows, totals = normalize_openai_response(
            call_id,
            self._execution_owner_id,
            self._intended_template_id,
            response,
        )
        return self._settle(call_id, [row.model_dump() for row in rows], totals.model_dump(), None)

    def _settle_exception(
        self, call_id: str, error: Exception, started: float
    ) -> SettledCall:
        rows, totals = normalize_exception(
            call_id,
            self._execution_owner_id,
            self._intended_template_id,
            error,
            started,
        )
        return self._settle(call_id, [row.model_dump() for row in rows], totals.model_dump(), str(error))


def build_owned_provider_client(
    client: LLMClient,
    root: Path,
    intended_template: ExecutionTemplateIdentity,
) -> OwnedProviderAccounting:
    return OwnedProviderAccounting(client, root, intended_template)
